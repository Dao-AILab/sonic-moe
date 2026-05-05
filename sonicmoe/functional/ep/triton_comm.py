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
from cuda.bindings import driver
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


def reduce_scatter_triton(x_symm, group, out=None, hdl=None):
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
    if hdl is None:
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
# ----------------------------------------------------------------------------
# Generic recv layout: the kernel takes a `recv_pos` tensor of length
# TK_global. For each global slot f where dst_rank_flat[f] == my_rank, the
# kernel writes the source row (peer x_symm[t_local]) into recv at row
# `recv_pos[f]`. The caller chooses what layout `recv_pos` encodes:
#
#   * Legacy per-rank-slot layout (recv shape (W, TK_local, d)):
#       recv_pos[f] = src_rank * TK_local + slot_per_rank[f]
#     i.e. meta["a2a_token_indices"] from compute_dispatch_metadata.
#
#   * Expert-sorted (nogather) layout (recv shape (TK_global, d)):
#       recv_pos[f] = s_reverse_local[f]
#     where s_reverse_local comes from general_routing_router_metadata_triton
#     and gives each slot's row in the expert-sorted x_compute tensor.
#
# Slots where dst != my_rank are no-ops; their recv positions retain whatever
# was there before the call. Downstream GEMM tolerates garbage at sentinel
# rows because combine reads outputs only at rows where dst == my_rank.
# ============================================================================
@triton.autotune(
    configs=_A2A_PULL_CONFIGS,
    key=["d", "world_size"],
    prune_configs_by={"early_config_prune": _prune_block_d_vs_d},
)
@triton.heuristics({"EVEN_D": lambda args: args["d"] % args["BLOCK_D"] == 0})
@triton.jit
def _a2a_dispatch_pull_kernel(
    x_peer_tuple,  # tuple[(T_local, d) tensor, ...] for each peer
    dst_rank_flat_ptr,  # (TK_global,) int32
    recv_pos_ptr,  # (TK_global,) int32 — destination row in recv per global slot
    recv_ptr,  # flat (>= TK_global, d) output
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

    # Generic recv layout: caller-supplied per-slot destination row.
    pos = tl.load(recv_pos_ptr + orig_idx).to(tl.int64)
    t_local = pid_tk // K

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    src_offs = t_local * d + offs_d
    dst_offs = pos * d + offs_d

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
#     1. Load peer = src_dst_rank[t, k], score.
#        pos = my_rank * TK_local + t * K + k is computed inline (each rank
#        owns the contiguous [my_rank * TK_local, (my_rank+1) * TK_local)
#        slice of every peer's TK_global-sized buffer; no per-(t, k) lookup
#        needed).
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
    src_dst_rank_ptr,
    scores_ptr,  # only read when WITH_SCORES is True
    y_local_ptr,
    my_rank_offset,  # int64 scalar: my_rank * TK_local
    K: tl.constexpr,
    d: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
    WITH_SCORES: tl.constexpr,
):
    """One program per (t, BLOCK_D-tile). K-serial inner loop with inline
    s_reverse resolve, register fp32 accumulator, single non-atomic store.
    """
    pid_t = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = offs_d < d  # const-True for EVEN_D
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    base = pid_t * K
    pos_base = my_rank_offset.to(tl.int64) + base  # = my_rank*TK_local + t*K

    # Branch-free pattern: fold peer-match into load mask, accumulate
    # unconditionally. Triton's SSA optimizer can miscompile `if peer == i:`
    # inside tl.static_range when the body has multiple loads (s_reverse +
    # y_row), producing nondeterministic results. See _rs_aggregation_kernel
    # for the same pattern.
    for k in tl.static_range(K):
        peer = tl.load(src_dst_rank_ptr + base + k)
        pos = pos_base + k
        if WITH_SCORES:
            score = tl.load(scores_ptr + base + k).to(tl.float32)
        for i in tl.static_range(world_size):
            is_match = peer == i  # scalar bool
            # When peer != i, s_peer load is masked → 0; subsequent y_row
            # load reads peer_y_tuple[i] row 0 but is also masked → 0.
            # The accumulator gets 0 contribution, equivalent to skipping.
            s_peer = tl.load(peer_s_reverse_tuple[i] + pos, mask=is_match, other=0).to(tl.int64)
            row_offs = s_peer * d + offs_d
            m = is_match & d_mask
            row = tl.load(peer_y_tuple[i] + row_offs, mask=m, other=0.0).to(tl.float32)
            if WITH_SCORES:
                acc += row * score
            else:
                acc += row

    out_offs = pid_t * d + offs_d
    if EVEN_D:
        tl.store(y_local_ptr + out_offs, acc)
    else:
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
#       - tile_count[src_rank, t_idx, p]     = #slots in this tile heading
#                                              to peer p
#       - my_dst_rank, my_expert_local       (only if src_rank == my_rank)
#       - expert_local_padded                (sentinel for invalid lanes)
#     Note: within_tile_slot is NOT materialized — phase 3 recomputes it
#     from dst_rank_flat in registers, saving an 8 MB HBM round-trip at
#     T=32k K=8 W=8.
#
#   Phase 2 (`_metadata_phase2_scan_kernel`, grid (W, W)):
#     Each program owns one (src_rank, peer) pair and does a 1-D cumsum
#     of tile_count[src_rank, :, peer] of length n_tiles. Writes:
#       - tile_prefix[r, t, p]      = exclusive cumsum along n_tiles axis
#       - peer_count_per_rank[r, p] = sum along n_tiles axis (scalar per prog)
#     Grid is (W, W) instead of (W,) for ~8× better SM utilization.
#     Strided loads of stride W are L2-friendly for W ≤ 16.
#
#   Phase 3 (`_metadata_phase3_emit_kernel`, grid (W, n_tiles)):
#     Same grid as Phase 1. Each program reloads dst_rank_flat for its
#     tile, RECOMPUTES within_tile_slot via the same one-hot cumsum that
#     phase 1 used (register-only), and emits:
#       - slot_per_rank = within_tile_slot + tile_prefix[r, t, dst]
#       - slot_global   = slot_per_rank   + early_count[r, dst]
#       - a2a_token_indices = pid_r * TK_local + slot_per_rank
#     `early_count[pid_r, :]` is computed inline from peer_count_per_rank
#     (cumsum + masked-row-sum), avoiding a (W, W) materialization.
# ============================================================================


@triton.jit
def _metadata_phase1_reduce_kernel(
    topk_idx_g_ptr,  # (W, T_local, K) int32, contiguous
    out_dst_rank_flat_ptr,  # (TK_global,) int32
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

    tile_count_p = tl.sum(one_hot, axis=0)

    local_expert = expert_global - dst * E_local
    padded = tl.where(dst == my_rank, local_expert, E_local)

    tl.store(out_dst_rank_flat_ptr + flat_offs, dst, mask=valid)
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
    """1-D cumsum over n_tiles for one (src_rank, peer) pair.

    Grid (W, W). Each program loads tile_count[pid_r, :, pid_p] of length
    n_tiles into registers (masked beyond n_tiles), runs tl.cumsum, and
    writes back tile_prefix[pid_r, :, pid_p] (exclusive) and a single
    scalar peer_count_per_rank[pid_r, pid_p].

    Strided loads of stride W (typically 4–16) — L2-friendly. Compared
    to the prior grid-(W,) version, this gives W× better SM utilization
    at the cost of a strided memory pattern, which the H100 L2 absorbs.
    """
    pid_r = tl.program_id(0)
    pid_p = tl.program_id(1)

    t_offs = tl.arange(0, BLOCK_NTILES)
    t_mask = t_offs < n_tiles

    # Strided 1-D load: tile_count[pid_r, :, pid_p].
    addr = tile_count_ptr + pid_r * n_tiles * W + t_offs * W + pid_p
    tc = tl.load(addr, mask=t_mask, other=0)  # (BLOCK_NTILES,) int32

    # Inclusive cumsum, then subtract for exclusive.
    incl = tl.cumsum(tc, axis=0)
    excl = incl - tc  # (BLOCK_NTILES,)

    # Per-(rank, peer) total = sum over n_tiles.
    peer_count = tl.sum(tc, axis=0)  # scalar

    # Stores.
    tl.store(
        out_tile_prefix_ptr + pid_r * n_tiles * W + t_offs * W + pid_p,
        excl,
        mask=t_mask,
    )
    tl.store(out_peer_count_per_rank_ptr + pid_r * W + pid_p, peer_count)


@triton.jit
def _metadata_phase3_emit_kernel(
    dst_rank_flat_ptr,  # (TK_global,) int32
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

    # early_count[pid_r, :] = exclusive cumsum of peer_count_per_rank
    # along source-rank axis, gathered for this rank's row.
    rs = tl.arange(0, W)[:, None]
    ps = tl.arange(0, W)[None, :]
    pc = tl.load(peer_count_per_rank_ptr + rs * W + ps)
    excl_pc = tl.cumsum(pc, axis=0) - pc
    row_mask = tl.arange(0, W) == pid_r
    ec_row = tl.sum(tl.where(row_mask[:, None], excl_pc, 0), axis=0)

    tile_offs = pid_tile * BLOCK_TK + tl.arange(0, BLOCK_TK)
    valid = tile_offs < TK_local
    flat_offs = pid_r * TK_local + tile_offs

    # Reload dst for this tile.
    dst = tl.load(dst_rank_flat_ptr + flat_offs, mask=valid, other=0)

    # Recompute within_tile_slot from dst (same idiom as phase 1).
    # No HBM intermediate — pure register work.
    p_offs = tl.arange(0, W)
    one_hot = (dst[:, None] == p_offs[None, :]).to(tl.int32)
    one_hot = tl.where(valid[:, None], one_hot, 0)
    cumsum = tl.cumsum(one_hot, axis=0)
    within_pos = tl.sum(cumsum * one_hot, axis=1) - 1

    # Tile prefix for this (pid_r, pid_tile), gathered at dst.
    tp_row = tl.load(tile_prefix_ptr + pid_r * n_tiles * W + pid_tile * W + p_offs)

    tp_at_dst = tl.sum(one_hot * tp_row[None, :], axis=1)
    ec_at_dst = tl.sum(one_hot * ec_row[None, :], axis=1)

    slot_pr = tp_at_dst + within_pos
    slot_gl = slot_pr + ec_at_dst

    # a2a_token_indices: src_rank * TK_local + slot_per_rank.
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

    Phase 1 emits dst_rank_flat, tile_count, my_dst_rank, my_expert_local,
    and expert_local_padded. Phase 2 computes tile_prefix and
    peer_count_per_rank with a (W, W) grid. Phase 3 recomputes
    within_tile_slot from dst_rank_flat in registers and emits
    slot_per_rank, slot_global, and a2a_token_indices.

    Optimization log:
      - within_tile_slot is no longer materialized (recomputed in phase 3
        — saves an 8 MB HBM round-trip at T=32k K=8 W=8).
      - Phase 2 grid is (W, W) instead of (W,) for ~W× better SM
        utilization.
      - expert_local_padded and a2a_token_indices absorb torch ops that
        were previously in ep.py's _moe_ep_forward_inner."""
    W, T_local, K = topk_idx_global.shape
    TK_local = T_local * K
    TK_global = W * TK_local
    device = topk_idx_global.device

    BLOCK_TK = max(triton.next_power_of_2(TK_local), 64) if TK_local <= 256 else 512
    n_tiles = (TK_local + BLOCK_TK - 1) // BLOCK_TK

    # Phase 1 outputs (within_tile_slot intentionally absent).
    dst_rank_flat = torch.empty(TK_global, dtype=torch.int32, device=device)
    tile_count = torch.empty((W, n_tiles, W), dtype=torch.int32, device=device)
    my_dst_rank = torch.empty((T_local, K), dtype=torch.int32, device=device)
    my_expert_local = torch.empty((T_local, K), dtype=torch.int32, device=device)
    expert_local_padded = torch.empty(TK_global, dtype=torch.int32, device=device)

    _metadata_phase1_reduce_kernel[(W, n_tiles)](
        topk_idx_global,
        dst_rank_flat,
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

    # Phase 2 outputs.
    tile_prefix = torch.empty_like(tile_count)
    peer_count_per_rank = torch.empty((W, W), dtype=torch.int32, device=device)
    BLOCK_NTILES = max(triton.next_power_of_2(n_tiles), 2)
    _metadata_phase2_scan_kernel[(W, W)](
        tile_count,
        tile_prefix,
        peer_count_per_rank,
        n_tiles,
        W=W,
        BLOCK_NTILES=BLOCK_NTILES,
    )

    # Phase 3 outputs.
    slot_per_rank = torch.empty(TK_global, dtype=torch.int32, device=device)
    slot_global = torch.empty(TK_global, dtype=torch.int32, device=device)
    a2a_token_indices = torch.empty(TK_global, dtype=torch.int32, device=device)
    _metadata_phase3_emit_kernel[(W, n_tiles)](
        dst_rank_flat,
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


def all_gather_triton(x_symm, group, out=None, hdl=None):
    if hdl is None:
        hdl = rendezvous(x_symm, group)
    W = hdl.world_size
    numel = x_symm.numel()
    if out is None:
        out = torch.empty(
            (W * x_symm.shape[0],) + tuple(x_symm.shape[1:]),
            dtype=x_symm.dtype,
            device=x_symm.device,
        )
    buf_tuple = tuple(hdl.get_buffer(r, tuple(x_symm.shape), x_symm.dtype) for r in range(W))
    grid = lambda META: (W, triton.cdiv(numel, META["BLOCK_SIZE"]))
    _all_gather_kernel[grid](
        buf_tuple,
        out,
        numel_per_rank=numel,
        world_size=W,
    )
    return out


# ============================================================================
# AllGather via Copy Engine (cuMemcpyAsync direct).
# ----------------------------------------------------------------------------
# Routes peer copies through the GPU's Copy Engines instead of a SM-side
# copy kernel. The driver dispatches intra-device P2P cuMemcpyAsync to a
# CE; torch.Tensor.copy_ falls through to a SM kernel even on P2P-mapped
# tensors, which is why the prior implementation of this function failed
# to actually use the CE despite the name. We bypass torch.copy_ by issuing
# the copies through cuda-python driver bindings directly.
#
# Caller contract: x_symm has been written AND a barrier (NCCL or symm-mem)
# has been issued before the call, so peer reads observe valid data.
# Same contract as all_gather_triton.
# ============================================================================


def _check(result):
    """Unwrap cuda-python driver returns; raise on error."""
    if not isinstance(result, tuple):
        return result
    err, *rest = result
    if err != driver.CUresult.CUDA_SUCCESS:
        _, name = driver.cuGetErrorName(err)
        _, msg = driver.cuGetErrorString(err)
        decode = lambda b: b.decode() if isinstance(b, bytes) else b
        raise RuntimeError(f"CUDA error {decode(name)}: {decode(msg)}")
    if len(rest) == 0:
        return None
    if len(rest) == 1:
        return rest[0]
    return tuple(rest)


class CEHandle:
    __slots__ = ("out", "_streams")

    def __init__(self, out, streams):
        self.out = out
        self._streams = streams

    def wait(self):
        main = torch.cuda.current_stream()
        for s in self._streams:
            main.wait_stream(s)
        return self.out

    def __call__(self):
        return self.wait()


_CE_STREAM_POOL: dict[tuple, list[torch.cuda.Stream]] = {}


def _get_ce_streams(device, n):
    """Return n cached high-priority CUDA streams for CE copies."""
    key = (str(device), n)
    pool = _CE_STREAM_POOL.get(key)
    if pool is not None:
        return pool
    pool = [torch.cuda.Stream(device=device, priority=-1) for _ in range(n)]
    _CE_STREAM_POOL[key] = pool
    return pool


def all_gather_copy_engine_async(x_symm, group, out=None, num_streams=2):
    """All-gather routed through the Copy Engines via cuMemcpyAsync.
 
    For each remote rank r, issues
        cuMemcpyAsync(out[r], peer[r].x_symm, ...)
    on a dedicated CUDA stream. The local chunk is filled via cuMemcpyAsync
    on the torch stream (intra-device, no NVLink). Going through the driver
    bindings rather than torch.Tensor.copy_ ensures the driver picks the
    Copy Engine path; torch.copy_ launches a SM copy kernel even on
    P2P-mapped tensors. To verify CE usage in practice:
        ncu --metrics lts__t_sectors_srcunit_l1ces_op_read.sum,\\
                     lts__t_sectors_srcunit_tex_op_read.sum
    The L1ces counter should account for the (W-1) cross-rank chunks; tex
    should be near-zero for those copies.
 
    Caller contract:
        x_symm has been written AND a barrier (NCCL or symm-mem) has been
        issued before the call. The CE streams take a wait_stream
        dependency on the torch stream, so peer reads observe the
        barrier's completion.
 
    Args:
        x_symm:      symm-mem tensor, this rank's data, shape (T_local, ...).
                     Must be contiguous.
        group:       process group used at rendezvous.
        out:         optional pre-allocated output (W*T_local, ...) and
                     matching dtype/device. If None, allocated here.
        num_streams: number of CE streams. Default 2. More streams = more
                     concurrent peer copies; past 4 the host overhead
                     usually outweighs the parallelism on a single NVLink
                     island.
 
    Returns:
        CEHandle. Use .wait() to sync, then access .out (or call() it).
 
    Example:
        h = all_gather_copy_engine_async(x_symm, group)
        y = some_gemm(...)              # overlaps with CE copies
        x_global = h.wait()
    """
    hdl = rendezvous(x_symm, group)
    W = hdl.world_size
    rank = hdl.rank

    if out is None:
        out = torch.empty(
            (W * x_symm.shape[0],) + tuple(x_symm.shape[1:]),
            dtype=x_symm.dtype,
            device=x_symm.device,
        )

    num_streams = max(1, min(num_streams, W))
    streams = _get_ce_streams(x_symm.device, num_streams)
    chunks = out.chunk(W)
    buf_tuple = tuple(hdl.get_buffer(r, tuple(x_symm.shape), x_symm.dtype) for r in range(W))
    bytes_per_chunk = x_symm.numel() * x_symm.element_size()

    main = torch.cuda.current_stream()
    # CE streams wait on whatever was last enqueued on the torch stream
    # (typically the caller's barrier), so peer reads observe valid data.
    for s in streams:
        s.wait_stream(main)

    # Self-copy on the torch stream — intra-device, no NVLink. Goes
    # through the driver here too, for consistency. The cross-stream
    # parallelism is enough that this isn't on the critical path.
    _check(
        driver.cuMemcpyAsync(
            chunks[rank].data_ptr(),
            x_symm.data_ptr(),
            bytes_per_chunk,
            main.cuda_stream,
        )
    )

    # Cross-rank pulls via cuMemcpyAsync on dedicated CE streams. Round-
    # robin across the pool. Self-skipped (handled above). Iteration
    # order starts at (rank+1) to spread the initial requests across
    # peers — at large W this avoids transient hot-spotting on rank 0.
    peers = [(rank + i) % W for i in range(1, W)]
    for idx, r in enumerate(peers):
        s = streams[idx % num_streams]
        _check(
            driver.cuMemcpyAsync(
                chunks[r].data_ptr(),
                buf_tuple[r].data_ptr(),
                bytes_per_chunk,
                s.cuda_stream,
            )
        )

    return CEHandle(out, streams)


def a2a_dispatch_pull_triton(
    x_symm: torch.Tensor,
    dst_rank_flat: torch.Tensor,
    recv_pos: torch.Tensor,
    recv: torch.Tensor,
    K: int,
    group,
    hdl=None,
):
    """Fused A2A dispatch via NVLink reads from peer x_symm.

    For each global slot f where dst_rank_flat[f] == my_rank, the kernel
    reads peer.x_symm[t_local] (where t_local = (f % TK_local) // K and the
    peer is the source rank f // TK_local) and writes it into recv at row
    `recv_pos[f]`. Slots where dst != my_rank are no-ops; their recv rows
    retain prior contents.

    Args:
        x_symm: this rank's x in symm-mem, shape (T_local, d).
        dst_rank_flat: (TK_global,) int32. Destination peer per global slot,
            from compute_dispatch_metadata.
        recv_pos: (TK_global,) int32. Caller-supplied destination row in recv
            for each global slot. The kernel uses recv_pos[f] only when
            dst_rank_flat[f] == my_rank; entries elsewhere are unread.
            Common choices:
              * meta["a2a_token_indices"]   → legacy (W, TK_local, d) layout
              * metadata["s_reverse_local"] → expert-sorted (TK_global, d)
                                              layout for nogather GEMM
        recv: local output buffer. Any shape whose flat (rows, d) view has
            >= TK_global rows; the kernel writes recv.view(-1, d)[recv_pos[f]]
            for each f routed to my_rank.
        K: top-K experts per token.
        group: process group.
    """
    if hdl is None:
        hdl = rendezvous(x_symm, group)
    W = hdl.world_size
    T_local, d = x_symm.shape
    TK_local = T_local * K
    TK_global = W * TK_local

    x_peer_tuple = tuple(hdl.get_buffer(r, (T_local, d), x_symm.dtype) for r in range(W))
    recv_flat = recv.view(-1, d)

    grid = lambda META: (TK_global, triton.cdiv(d, META["BLOCK_D"]))
    _a2a_dispatch_pull_kernel[grid](
        x_peer_tuple,
        dst_rank_flat,
        recv_pos,
        recv_flat,
        TK_local=TK_local,
        my_rank=hdl.rank,
        world_size=W,
        K=K,
        d=d,
    )
    return recv


# ============================================================================
# gather_aggregation_triton — NVLink combine of K (peer.y_symm, peer.s_reverse)
# rows per local token, with optional per-(t,k) scoring.
# ============================================================================
#
# Each rank consumes the contiguous [my_rank * TK_local, (my_rank+1) * TK_local)
# slice of every peer's TK_global-sized s_reverse / score buffer. The
# kernel computes that slot index inline as `my_rank*TK_local + pid_t*K + k`
# from `pid_t` and a single int64 scalar (`my_rank_offset = my_rank*TK_local`),
# so callers no longer need to materialize a per-(t, k) `dispatch_pos`
# tensor (which was just `arange(TK_local) + my_rank*TK_local` reshaped).
# ============================================================================
def gather_aggregation_triton(
    y_symm: torch.Tensor,
    s_reverse_symm: torch.Tensor,
    src_dst_rank: torch.Tensor,
    topk_scores: Optional[torch.Tensor],
    out: torch.Tensor,
    K,
    group,
    hdl_y=None,
    hdl_s=None,
):
    """gather + (optionally weighted) accumulate.

    Forward combine passes per-(t, k) scores; backward dx combine passes
    None (score-less). The kernel branches on WITH_SCORES at compile time.
    """
    if hdl_y is None:
        hdl_y = rendezvous(y_symm, group)
    if hdl_s is None:
        hdl_s = rendezvous(s_reverse_symm, group)
    W = hdl_y.world_size
    my_rank = dist.get_rank(group)
    d = y_symm.shape[1]
    T_local = src_dst_rank.shape[0]
    TK_global = W * T_local * K
    my_rank_offset = my_rank * T_local * K  # int64 in kernel

    src_flat = src_dst_rank.view(-1)

    with_scores = topk_scores is not None
    if with_scores:
        scores_flat = topk_scores.view(-1)
    else:
        scores_flat = src_flat  # unused when WITH_SCORES is False

    s_buf = tuple(hdl_s.get_buffer(r, (TK_global,), s_reverse_symm.dtype) for r in range(W))
    y_buf = tuple(hdl_y.get_buffer(r, (TK_global, d), y_symm.dtype) for r in range(W))

    grid = lambda META: (T_local, triton.cdiv(d, META["BLOCK_D"]))
    _gather_aggregation_kernel[grid](
        y_buf,
        s_buf,
        src_flat,
        scores_flat,
        out,
        my_rank_offset,
        K=K,
        d=d,
        world_size=W,
        WITH_SCORES=with_scores,
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
    scores_ag_ptr,  # only read when WITH_SCORES is True
    rs_buf_ptr,  # (W * T_local, d) float32 output, flat
    T_local,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    K: tl.constexpr,
    d: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
    WITH_SCORES: tl.constexpr,
):
    """One program per (home_rank, home_t, d-tile). Walks K expert slots,
    accumulates row contributions in registers for those routed to my_rank,
    stores once. No atomics, no zero_().

    WITH_SCORES=True  → score-weighted sum (forward RS combine).
    WITH_SCORES=False → identity-weighted sum (backward dx RS combine);
                        the score load and multiply are elided at compile
                        time.

    KEY DESIGN POINT — the dst-vs-my_rank check is ALWAYS folded into the
    load mask (m = is_mine & d_mask, other=0.0), independent of
    WITH_SCORES. This is what makes the K-loop branch-free and avoids
    Triton SSA merging issues with `if` inside tl.static_range. Score-less
    mode just drops the multiply, not the mask.
    """
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

        if WITH_SCORES:
            score = tl.load(scores_ag_ptr + f).to(tl.float32)
        row_idx = tl.load(s_reverse_ptr + f).to(tl.int64)
        row_offs = row_idx * d + offs_d

        # Vector mask: True where (valid d lane) AND (routed to my_rank).
        # When is_mine is False the load returns 0, so the contribution is
        # 0 regardless of WITH_SCORES — no conditional update.
        m = is_mine & d_mask
        row = tl.load(y_symm_ptr + row_offs, mask=m, other=0.0).to(tl.float32)

        if WITH_SCORES:
            acc += score * row
        else:
            acc += row

    out_offs = pid_ht * d + offs_d
    if EVEN_D:
        tl.store(rs_buf_ptr + out_offs, acc)
    else:
        tl.store(rs_buf_ptr + out_offs, acc, mask=d_mask)


def rs_aggregation(
    y_symm: torch.Tensor,
    s_reverse: torch.Tensor,
    dst_rank_flat: torch.Tensor,
    scores_ag: Optional[torch.Tensor],
    rs_buf: torch.Tensor,
    K: int,
    T_local: int,
    group: dist.ProcessGroup,
) -> None:
    """Writes
        rs_buf[home_rank * T_local + home_t, :] =
            sum_{k where dst[f] == my_rank}
                w[f] * y_symm[s_reverse[f], :]
    for each (home_rank, home_t) pair, where
        f    = (home_rank*T_local + home_t)*K + k
        w[f] = scores_ag[f]   if scores_ag is not None     (forward combine)
             = 1              otherwise                    (backward dx combine)

    No atomic_add. No zero_() needed — the kernel writes every output row
    exactly once.

    Args:
        y_symm: (TK_global, d) expert output.
        s_reverse: (TK_global,) int32, dispatch slot → row in y_symm.
        dst_rank_flat: (TK_global,) int32, destination rank per slot.
        scores_ag: GLOBAL all-gathered scores OR None for score-less mode.
            Accepted shapes when not None: (TK_global,) flat, or
            (W*T_local, K) — the natural output of all_gather(scores_symm).
            Flattened internally; no copy when already contiguous.
        rs_buf: (W*T_local, d) fp32 output buffer. Should be symm-mem so it
            can feed reduce_scatter directly. Written in-place.
        K: top-K experts per token.
        T_local: tokens per rank.
        group: the EP process group.
    """
    assert rs_buf.ndim == 2, f"rs_buf must be 2D (W*T_local, d), got shape {tuple(rs_buf.shape)}."
    WT, d = rs_buf.shape
    W = dist.get_world_size(group)
    my_rank = dist.get_rank(group)
    assert WT == W * T_local, f"rs_buf.shape[0]={WT} != W*T_local={W*T_local} (W from group)"

    with_scores = scores_ag is not None
    if with_scores:
        scores_flat = scores_ag.view(-1)
    else:
        # Triton needs a concrete pointer arg even though the kernel never
        # reads it when WITH_SCORES=False. Reuse `dst_rank_flat` — its
        # type and content are irrelevant since the load is compiled out.
        scores_flat = dst_rank_flat

    grid = lambda META: (W * T_local, triton.cdiv(d, META["BLOCK_D"]))
    _rs_aggregation_kernel[grid](
        y_symm,
        s_reverse,
        dst_rank_flat,
        scores_flat,
        rs_buf,
        T_local=T_local,
        my_rank=my_rank,
        world_size=W,
        K=K,
        d=d,
        WITH_SCORES=with_scores,
    )
