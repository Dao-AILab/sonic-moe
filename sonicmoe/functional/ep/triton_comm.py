# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
#
# Triton + symm-mem collectives for SonicMoE EP without overlapping.
#
# Stateless: no global caches, no autograd. Every function takes tensors in,
# launches kernels, returns tensors out. The caller owns all buffer
# allocation and barrier placement.
#
# Autotuning notes:
#   - All non-trivial kernels are autotuned over BLOCK_SIZE / num_warps /
#     num_stages. Configs that would produce grid_y > 65535 for the given
#     problem size are pruned at autotune time via prune_configs_by.
#   - EVEN_K is computed via @triton.heuristics, so problem sizes that happen
#     to be block-aligned automatically take the unmasked fast path. We do
#     NOT pad source buffers up to a block multiple — that would require
#     out-of-bounds loads from peer symm-mem buffers, which is unsafe.
#
# Naming convention used throughout:
#   T_local     tokens per rank
#   K           top-K experts per token
#   TK_local    T_local * K
#   W           EP world size
#   TK_global   W * TK_local
#   E_local     experts per rank
# ********************************************************************************

from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch.distributed import _symmetric_memory as symm_mem


_CUDA_MAX_GRID_Y = 65535

# AG and RS use different best warp choices, so keep their autotune
# spaces separate instead of sharing one _IO_BLOCK_CONFIGS list.
#
# AG is mostly peer memcpy: 4/8 warps are often enough, with 16 kept only
# as a large-tile fallback.
_AG_BLOCK_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE": 16384}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE": 16384}, num_warps=16, num_stages=3),
    triton.Config({"BLOCK_SIZE": 32768}, num_warps=16, num_stages=3),
]


def _prune_by_grid_y(numel_key: str):
    """Return a dict suitable for triton.autotune's `prune_configs_by` arg.
    Drops configs whose grid_y along the tile axis would exceed CUDA's 65535
    limit for the given problem size.

    Note: Triton's autotuner passes positional kernel args via `named_args`
    (a dict mapping arg name → value) and keyword kernel args via `**kwargs`.
    Wrappers in this file pass `numel_per_rank=...` etc. as kwargs, so the
    callback must consult both."""

    def _prune(configs, named_args, **kwargs):
        if numel_key in kwargs:
            numel = kwargs[numel_key]
        elif numel_key in named_args:
            numel = named_args[numel_key]
        else:
            # Don't know the value — keep all configs.
            return list(configs)
        kept = []
        for cfg in configs:
            bs = cfg.kwargs["BLOCK_SIZE"]
            if triton.cdiv(numel, bs) <= _CUDA_MAX_GRID_Y:
                kept.append(cfg)
        if not kept:
            kept = [max(configs, key=lambda c: c.kwargs["BLOCK_SIZE"])]
        return kept

    return {"early_config_prune": _prune}


def _prune_block_d_vs_d(configs, named_args, **kwargs):
    d = kwargs.get("d", named_args.get("d"))
    if d is None:
        return list(configs)
    kept = []
    for cfg in configs:
        bd = cfg.kwargs["BLOCK_D"]
        if bd > max(d, 1):
            continue  # BLOCK_D > d wastes lanes
        if triton.cdiv(d, bd) > _CUDA_MAX_GRID_Y:
            continue  # CUDA grid_y limit
        kept.append(cfg)
    if not kept:
        # Fall back to the largest BLOCK_D ≤ d if everything got dropped.
        valid_for_d = [c for c in configs if c.kwargs["BLOCK_D"] <= max(d, 1)]
        kept = [max(valid_for_d or configs, key=lambda c: c.kwargs["BLOCK_D"])]
    return kept


@triton.autotune(
    configs=_AG_BLOCK_CONFIGS,
    key=["numel_per_rank", "world_size"],
    prune_configs_by=_prune_by_grid_y("numel_per_rank"),
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["numel_per_rank"] % args["BLOCK_SIZE"] == 0,
    }
)
@triton.jit
def _all_gather_kernel(
    buf_tuple,
    output_ptr,
    numel_per_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """output[r*N:(r+1)*N] ← buf_tuple[r][0:N]"""
    pid_rank = tl.program_id(0)
    pid_tile = tl.program_id(1)
    offs = pid_tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    for i in tl.static_range(world_size):
        if pid_rank == i:
            if EVEN_K:
                data = tl.load(buf_tuple[i] + offs)
                tl.store(output_ptr + i * numel_per_rank + offs, data)
            else:
                mask = offs < numel_per_rank
                data = tl.load(buf_tuple[i] + offs, mask=mask)
                tl.store(output_ptr + i * numel_per_rank + offs, data, mask=mask)


# ============================================================================
# Reduce-scatter (sum) — symmetric to all_gather. Each rank reads peers'
# my_rank-th chunk via NVLink and accumulates locally. fp32 accumulation,
# implicit cast on store. No NCCL.
#
# Shape contract: x_symm is (W*T_local, ...) — i.e. the same total-size
# input that NCCL's reduce_scatter_tensor expects. Output is (T_local, ...).
#
# Determinism: summation order is rank 0 → W-1, fixed at compile time via
# tl.static_range. Bitwise reproducible across runs for the same inputs;
# NCCL ring RS is not, because the algorithm reorders depending on topology.
# ============================================================================

_RS_BLOCK_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE": 16384}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE": 16384}, num_warps=16, num_stages=3),
    triton.Config({"BLOCK_SIZE": 32768}, num_warps=16, num_stages=3),
]


@triton.autotune(
    configs=_RS_BLOCK_CONFIGS,
    key=["numel_per_rank", "world_size"],
    prune_configs_by=_prune_by_grid_y("numel_per_rank"),
)
@triton.heuristics({"EVEN_K": lambda args: args["numel_per_rank"] % args["BLOCK_SIZE"] == 0})
@triton.jit
def _reduce_scatter_kernel(
    buf_tuple,
    output_ptr,
    numel_per_rank: tl.constexpr,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """output[0:N] ← Σ_{r in 0..W} buf_tuple[r][my_rank*N : (my_rank+1)*N]

    AG fans out: 1 program reads 1 peer chunk, writes 1 output chunk.
    RS fans in:  1 program reads W peer chunks, writes 1 local chunk.
    fp32 accumulation; tl.store implicitly casts to output dtype.
    """
    pid_tile = tl.program_id(0)
    offs = pid_tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    base = my_rank * numel_per_rank
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    if EVEN_K:
        for i in tl.static_range(world_size):
            data = tl.load(buf_tuple[i] + base + offs)
            acc += data.to(tl.float32)
        tl.store(output_ptr + offs, acc)
    else:
        mask = offs < numel_per_rank
        for i in tl.static_range(world_size):
            data = tl.load(buf_tuple[i] + base + offs, mask=mask, other=0.0)
            acc += data.to(tl.float32)
        tl.store(output_ptr + offs, acc, mask=mask)


def reduce_scatter(x_symm, group, out=None):
    """Sum-reduce-scatter via Triton + symm-mem (no NCCL).

    Equivalent to:
        dist.reduce_scatter_tensor(out, x_symm, op=ReduceOp.SUM, group=group)
    up to fp32-accumulation order. Output dtype matches x_symm.

    Args:
        x_symm: symm-mem tensor, shape (W*T_local, ...). The leading dim must
            be divisible by world_size.
        group: process group used at rendezvous time.
        out: optional pre-allocated output of shape (T_local, ...) and matching
            dtype/device. If None, allocated here.

    Caller contract: x_symm has been written and a barrier has been issued
    before the call (peers read this rank's bytes via NVLink).
    """
    hdl = rendezvous(x_symm, group)
    W = hdl.world_size
    rank = hdl.rank
    assert x_symm.shape[0] % W == 0, f"reduce_scatter: x_symm.shape[0]={x_symm.shape[0]} not divisible by W={W}"
    T_local = x_symm.shape[0] // W
    out_shape = (T_local,) + tuple(x_symm.shape[1:])
    if out is None:
        out = torch.empty(out_shape, dtype=x_symm.dtype, device=x_symm.device)

    numel_per_rank = out.numel()
    buf_tuple = tuple(hdl.get_buffer(r, tuple(x_symm.shape), x_symm.dtype) for r in range(W))
    grid = lambda META: (triton.cdiv(numel_per_rank, META["BLOCK_SIZE"]),)
    _reduce_scatter_kernel[grid](
        buf_tuple,
        out,
        numel_per_rank=numel_per_rank,
        my_rank=rank,
        world_size=W,
    )
    return out


_A2A_PULL_CONFIGS = [
    triton.Config({"BLOCK_D": BLOCK_D}, num_warps=nw, num_stages=ns)
    for BLOCK_D in [512, 1024, 2048, 4096]
    for nw in [4, 8]
    for ns in [3, 4]
    if BLOCK_D // nw >= 32  # at least 1 elem/thread
]


# ============================================================================
# Fused A2A dispatch (pull with permute) — replaces the
# zero_() + scatter + barrier + all_to_all + barrier sequence with one kernel.
# ============================================================================
@triton.autotune(
    configs=_A2A_PULL_CONFIGS,
    key=["d", "world_size"],
    prune_configs_by={"early_config_prune": _prune_block_d_vs_d},
)
@triton.heuristics(
    {
        "EVEN_D": lambda args: args["d"] % args["BLOCK_D"] == 0,
    }
)
@triton.jit
def _a2a_dispatch_pull_kernel(
    x_peer_tuple,  # tuple[(T_local, d) tensor, ...] for each peer
    dst_rank_flat_ptr,  # (TK_global,) int32
    slot_flat_per_rank_ptr,  # (TK_global,) int32
    recv_ptr,  # (W * TK_local, d) flat output
    TK_local,  # = T_local * K
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    K: tl.constexpr,
    d: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    pid_orig = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1)

    # Peer-interleaved decomposition: adjacent programs target distinct peers.
    src_rank = (pid_orig % world_size).to(tl.int32)
    pid_tk = pid_orig // world_size  # ∈ [0, TK_local)

    # Metadata layout is unchanged (src_rank * TK_local + tk). We read it
    # at the strided offset corresponding to this program's (src_rank, tk).
    orig_idx = src_rank.to(tl.int64) * TK_local + pid_tk

    dst = tl.load(dst_rank_flat_ptr + orig_idx)
    if dst != my_rank:
        return  # invalid lane: no-op for this program

    slot = tl.load(slot_flat_per_rank_ptr + orig_idx).to(tl.int64)
    t_local = pid_tk // K

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    src_offs = t_local * d + offs_d
    dst_offs = (src_rank.to(tl.int64) * TK_local + slot) * d + offs_d

    # Static dispatch over peers; only the matching i actually loads/stores.
    for i in tl.static_range(world_size):
        if src_rank == i:
            if EVEN_D:
                row = tl.load(x_peer_tuple[i] + src_offs)
                tl.store(recv_ptr + dst_offs, row)
            else:
                d_mask = offs_d < d
                row = tl.load(x_peer_tuple[i] + src_offs, mask=d_mask)
                tl.store(recv_ptr + dst_offs, row, mask=d_mask)


# ============================================================================
# Combine kernel — single-pass with inline s_reverse resolve, no atomics.
# ============================================================================
#
# Design (one kernel):
#   grid (T_local, cdiv(d, BLOCK_D)). One program per (t, BLOCK_D-tile).
#   Inside, the K-loop is statically unrolled. For each k in 0..K:
#     1. Load peer = src_dst_rank[t, k], pos = dispatch_pos[t, k], score.
#     2. Load peer.s_reverse[pos] via NVLink to get the row in peer.y_symm
#        (this is the work the prior _resolve_peer_rows pre-pass did, folded
#        inline — saves a kernel launch and a barrier; the extra NVLink
#        bytes are <0.1% of gather traffic and pipeline freely).
#     3. Load peer.y_symm[s_peer, d-tile] via NVLink, multiply by score,
#        accumulate into an fp32 register block.
#   After the K-loop, store the register block (cast to dtype) into
#   out[t, d-tile]. One contiguous non-atomic store; bitwise deterministic.
# ============================================================================

_GATHER_AGGREGATION_CONFIGS = [
    triton.Config({"BLOCK_D": BLOCK_D}, num_warps=nw, num_stages=4)
    for BLOCK_D in [128, 256, 512, 1024, 2048, 4096]
    for nw in [2, 4, 8]
    if BLOCK_D // nw >= 32
]


@triton.autotune(
    configs=_GATHER_AGGREGATION_CONFIGS,
    key=["K", "d", "world_size"],
    prune_configs_by={"early_config_prune": _prune_block_d_vs_d},
)
@triton.heuristics({"EVEN_D": lambda args: args["d"] % args["BLOCK_D"] == 0})
@triton.jit
def _gather_aggregation_kernel(
    peer_y_tuple,
    peer_s_reverse_tuple,
    src_dst_rank_ptr,  # (TK_local,) int32, peer rank for each (t, k)
    dispatch_pos_ptr,  # (TK_local,) int32, global flat slot index
    scores_ptr,  # (TK_local,) dtype, topk score per (t, k)
    y_local_ptr,  # (T_local, d) output, same dtype as y_symm
    K: tl.constexpr,
    d: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    """One program per (t, BLOCK_D-tile). K-serial inner loop with inline
    s_reverse resolve, register fp32 accumulator, single non-atomic store."""
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    base = pid_t * K

    for k in tl.static_range(K):
        peer = tl.load(src_dst_rank_ptr + base + k)
        pos = tl.load(dispatch_pos_ptr + base + k).to(tl.int64)
        score = tl.load(scores_ptr + base + k).to(tl.float32)
        for i in tl.static_range(world_size):
            if peer == i:
                # Inline resolve: peer.s_reverse[pos] → row in peer.y_symm.
                s_peer = tl.load(peer_s_reverse_tuple[i] + pos).to(tl.int64)
                row_offs = s_peer * d + offs_d
                if EVEN_D:
                    row = tl.load(peer_y_tuple[i] + row_offs).to(tl.float32)
                else:
                    d_mask = offs_d < d
                    row = tl.load(peer_y_tuple[i] + row_offs, mask=d_mask, other=0.0).to(tl.float32)
                acc += row * score

    out_offs = pid_t * d + offs_d
    if EVEN_D:
        tl.store(y_local_ptr + out_offs, acc)
    else:
        d_mask = offs_d < d
        tl.store(y_local_ptr + out_offs, acc, mask=d_mask)


# ============================================================================
# Dispatch metadata — three-phase parallel scan, all-Triton (CUDA-graph safe).
# ----------------------------------------------------------------------------
# Three Triton kernels share `tile_count`, a (W, n_tiles, W) histogram of
# "slots in source rank r, tile t_idx, heading to peer p":
#
#   Phase 1 (`_metadata_phase1_reduce_kernel`, grid (W, n_tiles)):
#     Each program owns one (src_rank, tile) and processes BLOCK_TK slots
#     in parallel. Writes:
#       - dst_rank_flat[orig_idx]            = expert_global // E_local
#       - within_tile_slot[orig_idx]         = exclusive count within this
#                                              tile of slots heading to dst
#       - tile_count[src_rank, t_idx, p]     = #slots in this tile heading
#                                              to peer p
#       - my_dst_rank, my_expert_local       (only if src_rank == my_rank)
#     For W=8 T_local=32k K=8 with BLOCK_TK=512, n_tiles=512 — total
#     programs 4096, ~30 H100 waves.
#
#   Phase 2 (`_metadata_phase2_scan_kernel`, grid (W,)):
#     Each program owns one source rank, loads the (n_tiles, W) slice of
#     tile_count into registers, and computes:
#       - tile_prefix[r, t, p]      = exclusive cumsum of tile_count
#                                     along the n_tiles axis
#       - peer_count_per_rank[r, p] = sum of tile_count along the n_tiles
#                                     axis
#     One Triton kernel — replaces three torch ops (sum, cumsum, subtract)
#     that historically had CUDA-graph capture issues. tile_count fits in
#     registers comfortably for the typical n_tiles ≤ 512: at W=8 that's
#     a (512, 8) = 4096-int = 16KB tile, well within the register file.
#
#   Phase 3 (`_metadata_phase3_emit_kernel`, grid (W, n_tiles)):
#     Same grid as Phase 1. Each program reloads dst_rank_flat and
#     within_tile_slot for its tile, computes early_count[pid_r, :] inline
#     from peer_count_per_rank (cumsum + masked-row-sum, same idiom as the
#     prior _slot_global_kernel), and writes:
#       - slot_per_rank = within_tile_slot + tile_prefix[r, t, dst]
#       - slot_global   = slot_per_rank   + early_count[r, dst]
#
# Computing early_count inline in Phase 3 (rather than materializing it
# in Phase 2) saves a (W, W) write+read round-trip and folds naturally
# into the existing per-program one-hot-mul-sum gather.
#
# Correctness vs the prior two-kernel version: the within-tile +
# cross-tile split of slot_per_rank, and the cross-rank prefix for
# slot_global, are both reorganizations of the same exclusive cumsums.
# Tested bit-exact in test_metadata_phase2.py.
# ============================================================================


@triton.jit
def _metadata_phase1_reduce_kernel(
    topk_idx_g_ptr,  # (W, T_local, K) int32, contiguous
    out_dst_rank_flat_ptr,  # (TK_global,) int32
    out_within_tile_slot_ptr,  # (TK_global,) int32
    out_tile_count_ptr,  # (W, n_tiles, W) int32
    out_my_dst_rank_ptr,  # (T_local, K) int32
    out_my_expert_local_ptr,  # (T_local, K) int32
    out_expert_local_padded_ptr,  # (TK_global,) int32
    n_tiles,  # runtime stride for tile_count
    my_rank: tl.constexpr,
    W: tl.constexpr,
    TK_local: tl.constexpr,
    E_local: tl.constexpr,
    BLOCK_TK: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_tile = tl.program_id(1)
    is_mine = pid_r == my_rank

    tile_offs = pid_tile * BLOCK_TK + tl.arange(0, BLOCK_TK)
    valid = tile_offs < TK_local
    flat_offs = pid_r * TK_local + tile_offs

    expert_global = tl.load(topk_idx_g_ptr + flat_offs, mask=valid, other=0)
    dst = expert_global // E_local

    peer_axis = tl.arange(0, W)
    one_hot = (dst[:, None] == peer_axis[None, :]).to(tl.int32)
    one_hot = tl.where(valid[:, None], one_hot, 0)

    cumsum = tl.cumsum(one_hot, axis=0)
    within_pos = tl.sum(cumsum * one_hot, axis=1) - 1

    tile_count_p = tl.sum(one_hot, axis=0)

    # expert_local_padded: real local expert when dst == my_rank,
    # sentinel (flat_offs % E_local) otherwise. Absorbs 3 torch ops
    # from ep.py (_moe_ep_forward_inner).
    local_expert = expert_global - dst * E_local
    sentinel = flat_offs % E_local
    padded = tl.where(dst == my_rank, local_expert, sentinel)

    # Writes.
    tl.store(out_dst_rank_flat_ptr + flat_offs, dst, mask=valid)
    tl.store(out_within_tile_slot_ptr + flat_offs, within_pos, mask=valid)
    tl.store(
        out_tile_count_ptr + pid_r * n_tiles * W + pid_tile * W + peer_axis,
        tile_count_p,
    )
    tl.store(out_expert_local_padded_ptr + flat_offs, padded, mask=valid)
    if is_mine:
        tl.store(out_my_dst_rank_ptr + tile_offs, dst, mask=valid)
        tl.store(out_my_expert_local_ptr + tile_offs, local_expert, mask=valid)


@triton.jit
def _metadata_phase2_scan_kernel(
    tile_count_ptr,  # (W, n_tiles, W) int32, contiguous
    out_tile_prefix_ptr,  # (W, n_tiles, W) int32, contiguous
    out_peer_count_per_rank_ptr,  # (W, W) int32, contiguous
    n_tiles,  # runtime: stride and mask bound
    W: tl.constexpr,
    BLOCK_NTILES: tl.constexpr,  # next_pow2(n_tiles), >= 2
):
    """Per-source-rank cumsum + sum along the n_tiles axis.

    Grid (W,). Each program loads tile_count[pid_r, :, :] of shape
    (BLOCK_NTILES, W) into registers (masked beyond actual n_tiles), runs
    tl.cumsum on axis 0, and writes back tile_prefix and peer_count_per_rank
    for that source rank.

    Replaces the three torch ops (sum, cumsum, subtract) the prior version
    used. Single Triton launch is unconditionally CUDA-graph capturable,
    whereas torch.cumsum has historically had issues with graph capture
    across PyTorch versions (workspace allocations, multi-launch CUB
    fallbacks, etc.).
    """
    pid_r = tl.program_id(0)

    t_offs = tl.arange(0, BLOCK_NTILES)
    p_offs = tl.arange(0, W)
    t_mask = t_offs < n_tiles

    # Load tile_count[pid_r, :, :] of shape (BLOCK_NTILES, W).
    addr = tile_count_ptr + pid_r * n_tiles * W + t_offs[:, None] * W + p_offs[None, :]
    tc = tl.load(addr, mask=t_mask[:, None], other=0)  # (BLOCK_NTILES, W) int32

    # Inclusive cumsum along the n_tiles axis, then subtract for exclusive.
    incl = tl.cumsum(tc, axis=0)
    excl = incl - tc  # (BLOCK_NTILES, W)

    # Per-rank totals = sum over n_tiles.
    peer_count = tl.sum(tc, axis=0)  # (W,) int32

    # Stores.
    tl.store(
        out_tile_prefix_ptr + pid_r * n_tiles * W + t_offs[:, None] * W + p_offs[None, :],
        excl,
        mask=t_mask[:, None],
    )
    tl.store(out_peer_count_per_rank_ptr + pid_r * W + p_offs, peer_count)


@triton.jit
def _metadata_phase3_emit_kernel(
    dst_rank_flat_ptr,  # (TK_global,) int32
    within_tile_slot_ptr,  # (TK_global,) int32
    tile_prefix_ptr,  # (W, n_tiles, W) int32, contiguous
    peer_count_per_rank_ptr,  # (W, W) int32, contiguous
    out_slot_per_rank_ptr,  # (TK_global,) int32
    out_slot_global_ptr,  # (TK_global,) int32
    out_a2a_token_indices_ptr,  # (TK_global,) int32
    n_tiles,  # runtime stride
    W: tl.constexpr,
    TK_local: tl.constexpr,
    BLOCK_TK: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_tile = tl.program_id(1)

    rs = tl.arange(0, W)[:, None]
    ps = tl.arange(0, W)[None, :]
    pc = tl.load(peer_count_per_rank_ptr + rs * W + ps)
    excl_pc = tl.cumsum(pc, axis=0) - pc
    row_mask = tl.arange(0, W) == pid_r
    ec_row = tl.sum(tl.where(row_mask[:, None], excl_pc, 0), axis=0)

    tile_offs = pid_tile * BLOCK_TK + tl.arange(0, BLOCK_TK)
    valid = tile_offs < TK_local
    flat_offs = pid_r * TK_local + tile_offs

    dst = tl.load(dst_rank_flat_ptr + flat_offs, mask=valid, other=0)
    within_pos = tl.load(within_tile_slot_ptr + flat_offs, mask=valid, other=0)

    p_offs = tl.arange(0, W)
    tp_row = tl.load(tile_prefix_ptr + pid_r * n_tiles * W + pid_tile * W + p_offs)

    one_hot = (dst[:, None] == p_offs[None, :]).to(tl.int32)
    tp_at_dst = tl.sum(one_hot * tp_row[None, :], axis=1)
    ec_at_dst = tl.sum(one_hot * ec_row[None, :], axis=1)

    slot_pr = tp_at_dst + within_pos
    slot_gl = slot_pr + ec_at_dst

    # a2a_token_indices: src_rank * TK_local + slot_per_rank.
    # Absorbs 3 torch ops from ep.py (A2A mode token_indices_padded).
    a2a_ti = pid_r * TK_local + slot_pr

    tl.store(out_slot_per_rank_ptr + flat_offs, slot_pr, mask=valid)
    tl.store(out_slot_global_ptr + flat_offs, slot_gl, mask=valid)
    tl.store(out_a2a_token_indices_ptr + flat_offs, a2a_ti, mask=valid)


def compute_dispatch_metadata(
    topk_idx_global: torch.Tensor,
    my_rank: int,
    E_local: int,
):
    """Three-phase parallel dispatch metadata, all-Triton (CUDA-graph safe).

    Phase 1 emits dst_rank_flat, within_tile_slot, tile_count,
    my_dst_rank, my_expert_local, and expert_local_padded. Phase 2
    computes tile_prefix and peer_count_per_rank. Phase 3 emits
    slot_per_rank, slot_global, and a2a_token_indices.

    expert_local_padded and a2a_token_indices absorb torch ops that were
    previously computed in ep.py's _moe_ep_forward_inner, eliminating
    3 + 3 torch kernel launches and two cached workspace patterns
    (invalid_lane_expert, src_rank_pattern)."""
    W, T_local, K = topk_idx_global.shape
    TK_local = T_local * K
    TK_global = W * TK_local
    device = topk_idx_global.device

    BLOCK_TK = max(triton.next_power_of_2(TK_local), 64) if TK_local <= 256 else 512
    n_tiles = (TK_local + BLOCK_TK - 1) // BLOCK_TK

    dst_rank_flat = torch.empty(TK_global, dtype=torch.int32, device=device)
    within_tile_slot = torch.empty(TK_global, dtype=torch.int32, device=device)
    tile_count = torch.empty((W, n_tiles, W), dtype=torch.int32, device=device)
    my_dst_rank = torch.empty((T_local, K), dtype=torch.int32, device=device)
    my_expert_local = torch.empty((T_local, K), dtype=torch.int32, device=device)
    expert_local_padded = torch.empty(TK_global, dtype=torch.int32, device=device)

    _metadata_phase1_reduce_kernel[(W, n_tiles)](
        topk_idx_global,
        dst_rank_flat,
        within_tile_slot,
        tile_count,
        my_dst_rank,
        my_expert_local,
        expert_local_padded,
        n_tiles,
        my_rank=my_rank,
        W=W,
        TK_local=TK_local,
        E_local=E_local,
        BLOCK_TK=BLOCK_TK,
    )

    tile_prefix = torch.empty_like(tile_count)
    peer_count_per_rank = torch.empty((W, W), dtype=torch.int32, device=device)
    BLOCK_NTILES = max(triton.next_power_of_2(n_tiles), 2)
    _metadata_phase2_scan_kernel[(W,)](
        tile_count,
        tile_prefix,
        peer_count_per_rank,
        n_tiles,
        W=W,
        BLOCK_NTILES=BLOCK_NTILES,
    )

    slot_per_rank = torch.empty(TK_global, dtype=torch.int32, device=device)
    slot_global = torch.empty(TK_global, dtype=torch.int32, device=device)
    a2a_token_indices = torch.empty(TK_global, dtype=torch.int32, device=device)
    _metadata_phase3_emit_kernel[(W, n_tiles)](
        dst_rank_flat,
        within_tile_slot,
        tile_prefix,
        peer_count_per_rank,
        slot_per_rank,
        slot_global,
        a2a_token_indices,
        n_tiles,
        W=W,
        TK_local=TK_local,
        BLOCK_TK=BLOCK_TK,
    )

    my_pos_on_peer = slot_global[my_rank * TK_local : (my_rank + 1) * TK_local].view(T_local, K)
    my_pos_per_rank = slot_per_rank[my_rank * TK_local : (my_rank + 1) * TK_local].view(T_local, K)

    return {
        "dst_rank_flat": dst_rank_flat,
        "slot_flat_per_rank": slot_per_rank,
        "slot_flat_global": slot_global,
        "my_dst_rank": my_dst_rank,
        "my_pos_on_peer": my_pos_on_peer,
        "my_pos_per_rank": my_pos_per_rank,
        "my_expert_local": my_expert_local,
        "peer_count_per_rank": peer_count_per_rank,
        "expert_local_padded": expert_local_padded,
        "a2a_token_indices": a2a_token_indices,
    }


# ============================================================================
# Utilities
# ============================================================================


def rendezvous(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    return symm_mem.rendezvous(tensor, group=group)


def barrier(tensor: torch.Tensor, group: dist.ProcessGroup):
    rendezvous(tensor, group).barrier()


def safe_block_size(chunk_numel: int, requested: int = 4096) -> int:
    block = requested
    while triton.cdiv(chunk_numel, block) > _CUDA_MAX_GRID_Y:
        block *= 2
    return block


# ============================================================================
# Python wrappers
# ============================================================================


def all_gather(x_symm, group, out=None):
    hdl = rendezvous(x_symm, group)
    W = hdl.world_size
    numel = x_symm.numel()
    if out is None:
        out = torch.empty((W * x_symm.shape[0],) + tuple(x_symm.shape[1:]), dtype=x_symm.dtype, device=x_symm.device)
    buf_tuple = tuple(hdl.get_buffer(r, tuple(x_symm.shape), x_symm.dtype) for r in range(W))
    grid = lambda META: (W, triton.cdiv(numel, META["BLOCK_SIZE"]))
    _all_gather_kernel[grid](
        buf_tuple,
        out,
        numel_per_rank=numel,
        world_size=W,
    )
    return out


def a2a_dispatch_pull(
    x_symm: torch.Tensor,
    dst_rank_flat: torch.Tensor,
    slot_flat_per_rank: torch.Tensor,
    recv: torch.Tensor,
    K: int,
    group,
):
    """Fused A2A dispatch via NVLink reads from peer x_symm.

    Args:
        x_symm: this rank's x in symm-mem, shape (T_local, d).
        dst_rank_flat: (TK_global,) int32. Destination peer for each global slot.
        slot_flat_per_rank: (TK_global,) int32. Per-source-rank slot for each
            global slot.
        recv: local output buffer, shape (W, TK_local, d) or (W*TK_local, d).
            Populated with the rows this rank consumes; unwritten positions
            retain prior contents.
        K: top-K experts per token.
        group: process group.
    """
    hdl = rendezvous(x_symm, group)
    W = hdl.world_size
    T_local, d = x_symm.shape
    TK_local = T_local * K
    TK_global = W * TK_local

    x_peer_tuple = tuple(hdl.get_buffer(r, (T_local, d), x_symm.dtype) for r in range(W))
    recv_flat = recv.view(W * TK_local, d)

    grid = lambda META: (TK_global, triton.cdiv(d, META["BLOCK_D"]))
    _a2a_dispatch_pull_kernel[grid](
        x_peer_tuple,
        dst_rank_flat,
        slot_flat_per_rank,
        recv_flat,
        TK_local=TK_local,
        my_rank=hdl.rank,
        world_size=W,
        K=K,
        d=d,
    )
    return recv


def gather_aggregation(y_symm, s_reverse_symm, src_dst_rank, dispatch_pos, topk_scores, out, K, group):
    """gather + weighted-accumulate.

    Drop-in replacement for the prior two-pass version. Same caller
    signature; semantics unchanged.

    Caller contract:
      * Barrier on y_symm AND s_reverse_symm has been issued before this
        call (the kernel reads both via NVLink).
      * y_symm shape (TK_global, d), s_reverse_symm shape (TK_global,) —
        both symm-mem tensors allocated by SymmMemManager.
      * src_dst_rank (T_local, K): peer rank for each (t, k) assignment.
      * dispatch_pos (T_local, K): the GLOBAL flat slot index for (t, k) —
        i.e. my_rank * TK_local + t * K + k. Used to index peer.s_reverse.
      * topk_scores (T_local, K): score per (t, k).
      * out (T_local, d): output, same dtype as y_symm. Written directly
        (cast from in-register fp32 acc).

    No atomics: the K-fold sum lives in a Triton register block, and the
    final store is a single non-atomic write per (t, BLOCK_D-tile). This
    is bitwise deterministic — repeated calls with identical inputs
    produce identical outputs to the bit. See the combine module-level
    docstring for why this design beat the atomic-add restructure on H100
    and is preferred for determinism even where atomic-add helps (B300).
    """
    hdl_y = rendezvous(y_symm, group)
    hdl_s = rendezvous(s_reverse_symm, group)
    W = hdl_y.world_size
    d = y_symm.shape[1]
    T_local = src_dst_rank.shape[0]
    TK_global = W * T_local * K

    src_flat = src_dst_rank.view(-1)
    pos_flat = dispatch_pos.view(-1)
    scores_flat = topk_scores.view(-1)
    s_buf = tuple(hdl_s.get_buffer(r, (TK_global,), s_reverse_symm.dtype) for r in range(W))
    y_buf = tuple(hdl_y.get_buffer(r, (TK_global, d), y_symm.dtype) for r in range(W))

    grid = lambda META: (T_local, triton.cdiv(d, META["BLOCK_D"]))
    _gather_aggregation_kernel[grid](
        y_buf,
        s_buf,
        src_flat,
        pos_flat,
        scores_flat,
        out,
        K=K,
        d=d,
        world_size=W,
    )


# ============================================================================
# RS aggregation — single-store-per-row.
# ----------------------------------------------------------------------------
#
# Design (one kernel):
#   grid (W * T_local, cdiv(d, BLOCK_D)). One program per (home_rank,
#   home_t, BLOCK_D-tile). Inside, a tl.static_range(K) loop walks the K
#   expert slots for this (home_rank, home_t) token.
#
#   For each k: if dst_rank_flat[f] == my_rank, loads score and
#   y_symm[s_reverse[f], :], accumulates score * row in fp32 registers.
#   If dst != my_rank, a masked load returns 0 (other=0.0), so score * 0 = 0
#   and the accumulator is unaffected — no conditional branch, no Triton SSA
#   miscompile risk from `if` inside tl.static_range.
#
#   After the K-loop, stores the register block ONCE into
#   rs_buf[home_rank * T_local + home_t, :]. No atomic_add. No zero_()
#   required — the kernel writes every row exactly once.
# ============================================================================

_RS_AGGREGATION_CONFIGS = [
    triton.Config({"BLOCK_D": BD}, num_warps=nw, num_stages=4)
    for BD in [128, 256, 512, 1024, 2048, 4096]
    for nw in [2, 4, 8]
    if BD // nw >= 32
]


@triton.autotune(
    configs=_RS_AGGREGATION_CONFIGS,
    key=["d", "world_size", "K"],
    prune_configs_by={"early_config_prune": _prune_block_d_vs_d},
)
@triton.heuristics({"EVEN_D": lambda args: args["d"] % args["BLOCK_D"] == 0})
@triton.jit
def _rs_aggregation_kernel(
    y_symm_ptr,  # (TK_global, d) expert output, flat
    s_reverse_ptr,  # (TK_global,) int32, dispatch slot -> row index
    dst_rank_flat_ptr,  # (TK_global,) int32, destination rank per slot
    scores_ag_ptr,  # (TK_global,) float32, score per slot
    rs_buf_ptr,  # (W * T_local, d) float32 output, flat
    T_local,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    K: tl.constexpr,
    d: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    """One program per (home_rank, home_t, d-tile). Walks K expert slots,
    accumulates score*row in registers for those routed to my_rank,
    stores once. No atomics, no zero_().

    KEY DESIGN POINT -- no `if dst == my_rank` block. Triton SSA merging
    for `if` inside tl.static_range can miscompile (acc updates from
    earlier iterations don't propagate forward). Instead we do a masked
    load where the mask folds in `is_mine`. When is_mine is False the load
    returns 0 (other=0.0), so score*row = 0 and acc is unaffected -- same
    semantics, no conditional update."""
    pid_ht = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1)

    base = pid_ht * K  # = (home_rank * T_local + home_t) * K

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = offs_d < d  # const-True for EVEN_D

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for k in tl.static_range(K):
        f = base + k
        dst = tl.load(dst_rank_flat_ptr + f)
        is_mine = dst == my_rank  # scalar bool

        score = tl.load(scores_ag_ptr + f).to(tl.float32)
        row_idx = tl.load(s_reverse_ptr + f).to(tl.int64)
        row_offs = row_idx * d + offs_d

        # Vector mask: True where (valid d lane) AND (routed to my_rank).
        m = is_mine & d_mask
        row = tl.load(y_symm_ptr + row_offs, mask=m, other=0.0).to(tl.float32)

        acc += score * row

    out_offs = pid_ht * d + offs_d
    if EVEN_D:
        tl.store(rs_buf_ptr + out_offs, acc)
    else:
        tl.store(rs_buf_ptr + out_offs, acc, mask=d_mask)


def rs_aggregation(
    y_symm: torch.Tensor,
    s_reverse: torch.Tensor,
    dst_rank_flat: torch.Tensor,
    scores_ag: torch.Tensor,
    rs_buf: torch.Tensor,
    K: int,
    T_local: int,
    group: dist.ProcessGroup,
) -> None:
    """Writes rs_buf[home_rank * T_local + home_t, :] =
        sum_{k where dst[f] == my_rank} score[f] * y_symm[s_reverse[f], :]
    for each (home_rank, home_t) pair, where f = (home_rank*T_local + home_t)*K + k.

    No atomic_add. No zero_() needed — the kernel writes every output row
    exactly once (zero accumulator → zero store when no k matched).

    Intended call pattern in the EP forward (when rs combine is wired up):
        topk_idx_g, topk_scores_g = _ag_routing_decision(
            ws.idx_symm, topk_idx_l, grp,
            scores_symm=ws.scores_symm, topk_scores_l=topk_scores_l,
        )                                          # topk_scores_g: (W*T_local, K) fp32
        ...
        rs_aggregation(y_symm, s_reverse, meta["dst_rank_flat"],
                       topk_scores_g, rs_buf, K, T_local, group=grp)
        barrier(rs_buf, grp)
        out = reduce_scatter(rs_buf, grp)

    Args:
        y_symm: (TK_global, d) expert output. Reads are local — symm-mem
            allocation only matters for the subsequent reduce_scatter.
        s_reverse: (TK_global,) int32, dispatch slot → row in y_symm.
        dst_rank_flat: (TK_global,) int32, destination rank per slot.
            From compute_dispatch_metadata.
        scores_ag: GLOBAL all-gathered scores, fp32. Accepted shapes:
            (TK_global,) flat, or (W*T_local, K) — the natural output of
            all_gather(scores_symm). Flattened internally; no copy when
            already contiguous.
        rs_buf: (W*T_local, d) fp32 output buffer. Should be symm-mem so it
            can feed reduce_scatter directly. Written in-place.
        K: top-K experts per token.
        T_local: tokens per rank.
        group: the EP process group. Used to query my_rank consistently
            with the rest of the EP collective stack.
    """
    assert rs_buf.ndim == 2, f"rs_buf must be 2D (W*T_local, d), got shape {tuple(rs_buf.shape)}."
    WT, d = rs_buf.shape
    W = dist.get_world_size(group)
    my_rank = dist.get_rank(group)
    assert WT == W * T_local, f"rs_buf.shape[0]={WT} != W*T_local={W*T_local} (W from group)"

    grid = lambda META: (W * T_local, triton.cdiv(d, META["BLOCK_D"]))
    _rs_aggregation_kernel[grid](
        y_symm,
        s_reverse,
        dst_rank_flat,
        scores_ag.view(-1),
        rs_buf,
        T_local=T_local,
        my_rank=my_rank,
        world_size=W,
        K=K,
        d=d,
    )
