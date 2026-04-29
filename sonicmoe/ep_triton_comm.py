# ********************************************************************************
# Copyright (c) 2025 Sonic-MoE contributors
#
# Triton + symm-mem collectives for SonicMoE EP.
#
# Stateless: no global caches, no autograd. Every function takes tensors in,
# launches kernels, returns tensors out. The caller (ep.py) owns all buffer
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
#
# ----------------------------------------------------------------------------
# Triton autotune knob reference
# ----------------------------------------------------------------------------
# BLOCK_SIZE / BLOCK_D
#   Elements per program tile. Bigger blocks → fewer programs, fewer launch
#   slots, more registers + SMEM per program. Sweep range chosen to span
#   small problems (where small tiles avoid under-utilization) and large
#   problems (where big tiles minimize launch overhead).
#
# num_warps
#   Threads per program / 32. Rules of thumb:
#     * BLOCK_SIZE / num_warps ≥ 32 (one element per thread minimum).
#     * num_warps=4 (128 threads) — safe default.
#     * num_warps=8 (256 threads) — needs a tile big enough to feed them.
#     * num_warps=16 (512 threads) — only for very large tiles (≥ 16K).
#
# num_stages
#   Software pipeline depth — how many tile iterations the compiler tries
#   to overlap via shared-memory N-buffering.
#     * num_stages=2 — double buffer (load tile k+1 while store tile k);
#       captures most of the win for IO-bound kernels.
#     * num_stages=3 — triple buffer; small additional gain on Hopper+ when
#       the load-store pipeline is uneven.
#     * num_stages=4 — quad buffer; pays off most when the kernel has an
#       inner loop with compute (e.g. _gather_combine_kernel's K *
#       world_size accumulation), at the cost of ~2× more shared memory
#       per program. The autotuner falls back to a smaller value when the
#       chosen config fails to compile under SMEM pressure.
#   We do not include num_stages≥5: gains are vanishing past 4 and SMEM
#   pressure starts dominating.
# ********************************************************************************

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed import _symmetric_memory as symm_mem
import triton
import triton.language as tl


# ============================================================================
# Autotune configuration shared across IO kernels (AG, A2A, RS)
# ----------------------------------------------------------------------------
# These kernels are pure memcpy / reduce-and-store: load, optionally accumulate,
# store. Deeper pipelines pay off less than for kernels with inner-loop
# compute. We still expand to num_stages=4 so the autotuner can pick the
# 1-3% wins on Hopper+, and the autotuner drops configs that would exceed
# CUDA's 65535 grid_y limit.
# ============================================================================

_CUDA_MAX_GRID_Y = 65535

_IO_BLOCK_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 2048},  num_warps=4,  num_stages=3),
    triton.Config({"BLOCK_SIZE": 4096},  num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_SIZE": 8192},  num_warps=8,  num_stages=3),
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


# ============================================================================
# Generic IO kernels (still useful as building blocks; no longer used for EP
# A2A dispatch, which is now done by the fused pull kernel below).
# ============================================================================

@triton.autotune(
    configs=_IO_BLOCK_CONFIGS,
    key=["numel_per_rank", "world_size"],
    prune_configs_by=_prune_by_grid_y("numel_per_rank"),
)
@triton.heuristics({
    "EVEN_K": lambda args: args["numel_per_rank"] % args["BLOCK_SIZE"] == 0,
})
@triton.jit
def _all_gather_kernel(
    buf_tuple, output_ptr,
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


@triton.autotune(
    configs=_IO_BLOCK_CONFIGS,
    key=["chunk_numel", "world_size"],
    prune_configs_by=_prune_by_grid_y("chunk_numel"),
)
@triton.heuristics({
    "EVEN_K": lambda args: args["chunk_numel"] % args["BLOCK_SIZE"] == 0,
})
@triton.jit
def _all_to_all_kernel(
    buf_tuple, output_ptr,
    chunk_numel: tl.constexpr,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """recv[r*chunk:(r+1)*chunk] ← peer_r.send[my_rank*chunk:...]"""
    pid_rank = tl.program_id(0)
    pid_tile = tl.program_id(1)
    offs = pid_tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    for i in tl.static_range(world_size):
        if pid_rank == i:
            src_offs = my_rank * chunk_numel + offs
            if EVEN_K:
                data = tl.load(buf_tuple[i] + src_offs)
                tl.store(output_ptr + i * chunk_numel + offs, data)
            else:
                mask = offs < chunk_numel
                data = tl.load(buf_tuple[i] + src_offs, mask=mask)
                tl.store(output_ptr + i * chunk_numel + offs, data, mask=mask)


@triton.autotune(
    configs=_IO_BLOCK_CONFIGS,
    key=["numel_per_rank", "world_size"],
    prune_configs_by=_prune_by_grid_y("numel_per_rank"),
)
@triton.heuristics({
    "EVEN_K": lambda args: args["numel_per_rank"] % args["BLOCK_SIZE"] == 0,
})
@triton.jit
def _reduce_scatter_kernel(
    buf_tuple, output_ptr,
    numel_per_rank: tl.constexpr,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """output[k] = sum_r(peer_r[my_rank*N + k]). fp32 accumulation."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    src_offs = my_rank * numel_per_rank + offs
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    if EVEN_K:
        for i in tl.static_range(world_size):
            x = tl.load(buf_tuple[i] + src_offs).to(tl.float32)
            acc += x
        tl.store(output_ptr + offs, acc)
    else:
        mask = offs < numel_per_rank
        for i in tl.static_range(world_size):
            x = tl.load(buf_tuple[i] + src_offs, mask=mask, other=0.0).to(tl.float32)
            acc += x
        tl.store(output_ptr + offs, acc, mask=mask)


# ============================================================================
# Fused A2A dispatch (pull with permute) — replaces the
# zero_() + scatter + barrier + all_to_all + barrier sequence with one kernel.
# ----------------------------------------------------------------------------
#
# Each program handles one orig_i ∈ [0, TK_global). If
# dst_rank_flat[orig_i] == my_rank, the program reads
# peer_{src_rank}.x_symm[t_local] directly via NVLink and writes to
# recv[src_rank * TK_local + slot, :], where:
#   src_rank = orig_i // (T_local * K)
#   t_local  = orig_i // K - src_rank * T_local
#   slot     = slot_flat_per_rank[orig_i]
#
# Programs with dst_rank_flat[orig_i] != my_rank early-return; their cost is
# one load + compare + exit. The W× over-launch is the price for a
# fully-async, sync-free dispatch path.
#
# vs. the producer-permute + all_to_all approach this replaces:
#   * No a2a_send buffer. Saves W * TK_local * d * dtype_bytes per workspace.
#   * No zero_() of the send buffer.
#   * NVLink traffic is exactly (#valid lanes) * d * dtype_bytes (only the
#     data this rank actually consumes), not TK_global * d. ~W× less in
#     balanced cases.
#   * Two fewer barriers per A2A forward (Ba1 and Ba2 are gone). A2A now has
#     the same barrier count as AG mode.
# ============================================================================

# Pull kernel sweeps a wider range than the AG/A2A IO kernels because:
#  * BLOCK_D operates over the d (channel) axis, so small d models (256)
#    benefit from BLOCK_D=128 to keep CTAs busy, while large-d models
#    (4096) want BLOCK_D=2048+ to amortize launch overhead.
#  * The kernel has no inner loop — it's a single permuted memcpy — so
#    num_stages=4 wins are smaller than for the combine kernel below, but
#    we still include them for the few configurations where they help.
_A2A_PULL_CONFIGS = [
    triton.Config({"BLOCK_D": 512},  num_warps=4,  num_stages=3),  
    triton.Config({"BLOCK_D": 1024}, num_warps=4,  num_stages=3),
    triton.Config({"BLOCK_D": 2048}, num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_D": 4096}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_D": 4096}, num_warps=16, num_stages=3),
]


def _prune_pull_by_grid_y(configs, named_args, **kwargs):
    """Drop pull configs whose grid_y = cdiv(d, BLOCK_D) exceeds 65535.
    Wrapper passes `d` as a kwarg, so check kwargs first."""
    if "d" in kwargs:
        d = kwargs["d"]
    elif "d" in named_args:
        d = named_args["d"]
    else:
        return list(configs)
    kept = [cfg for cfg in configs
            if triton.cdiv(d, cfg.kwargs["BLOCK_D"]) <= _CUDA_MAX_GRID_Y]
    if not kept:
        kept = [max(configs, key=lambda c: c.kwargs["BLOCK_D"])]
    return kept


@triton.autotune(
    configs=_A2A_PULL_CONFIGS,
    key=["d", "world_size"],
    prune_configs_by={"early_config_prune": _prune_pull_by_grid_y},
)
@triton.heuristics({
    "EVEN_D": lambda args: args["d"] % args["BLOCK_D"] == 0,
})
@triton.jit
def _a2a_dispatch_pull_kernel(
    x_peer_tuple,                  # tuple[(T_local, d) tensor, ...] for each peer
    dst_rank_flat_ptr,             # (TK_global,) int32
    slot_flat_per_rank_ptr,        # (TK_global,) int32
    recv_ptr,                      # (W * TK_local, d) flat output
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    T_local: tl.constexpr,
    K: tl.constexpr,
    TK_local: tl.constexpr,        # = T_local * K
    d: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    pid_orig = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1)

    dst = tl.load(dst_rank_flat_ptr + pid_orig)
    if dst != my_rank:
        return  # invalid lane: no-op for this program

    src_rank = (pid_orig // (T_local * K)).to(tl.int32)
    t_local = (pid_orig // K - src_rank.to(tl.int64) * T_local).to(tl.int64)
    slot = tl.load(slot_flat_per_rank_ptr + pid_orig).to(tl.int64)

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
# Combine kernel (unchanged)
# ============================================================================

@triton.jit
def _resolve_peer_rows_kernel(
    peer_s_reverse_idx_tuple,
    src_dst_rank_ptr, dispatch_pos_on_peer_ptr,
    out_s_on_peer_ptr, n_assignments,
    BLOCK: tl.constexpr, world_size: tl.constexpr,
):
    """Combine pass 1: peer.s_reverse_idx[pos] → row index. Cheap; not autotuned."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_assignments
    peer = tl.load(src_dst_rank_ptr + offs, mask=mask, other=0)
    pos = tl.load(dispatch_pos_on_peer_ptr + offs, mask=mask, other=0)
    s = tl.zeros([BLOCK], dtype=tl.int32)
    for i in tl.static_range(world_size):
        is_peer = (peer == i) & mask
        s_i = tl.load(peer_s_reverse_idx_tuple[i] + pos, mask=is_peer, other=0)
        s = tl.where(is_peer, s_i, s)
    tl.store(out_s_on_peer_ptr + offs, s, mask=mask)


# Combine kernel has an inner K * world_size accumulation loop with FMA, so
# deeper pipelines pay off more reliably than for the pure-memcpy A2A pull
# kernel. We include num_stages=4 across most of the BLOCK_D range.
_GATHER_COMBINE_CONFIGS = [
    triton.Config({"BLOCK_D": 512},  num_warps=4,  num_stages=3),  
    triton.Config({"BLOCK_D": 512},  num_warps=8,  num_stages=3),  
    triton.Config({"BLOCK_D": 1024}, num_warps=4,  num_stages=3),
    triton.Config({"BLOCK_D": 2048}, num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_D": 4096}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_D": 4096}, num_warps=16, num_stages=3),
]


@triton.autotune(configs=_GATHER_COMBINE_CONFIGS, key=["K", "d", "world_size"])
@triton.heuristics({"EVEN_D": lambda args: args["d"] % args["BLOCK_D"] == 0})
@triton.jit
def _gather_combine_kernel(
    peer_y_tuple,
    src_dst_rank_ptr, s_on_peer_ptr, scores_ptr,
    y_local_ptr,
    K: tl.constexpr, d: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_D: tl.constexpr, EVEN_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    base = pid_t * K
    for k in tl.static_range(K):
        peer = tl.load(src_dst_rank_ptr + base + k)
        s_peer = tl.load(s_on_peer_ptr + base + k)
        score = tl.load(scores_ptr + base + k)
        row_offs = s_peer.to(tl.int64) * d + offs_d
        for i in tl.static_range(world_size):
            if peer == i:
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
# Dispatch metadata — two-kernel, sync-free.
# ----------------------------------------------------------------------------
# Phase 1 (`_dispatch_metadata_per_rank_kernel`): grid (W,). Each program owns
#   one source rank, scans its slice of topk_idx_g, emits dst_rank_flat,
#   slot_per_rank (per-source-rank slot), and per-rank totals
#   peer_count_per_rank[r, p] = #slots from r destined for p.
#
# Phase 2 (`_slot_global_kernel`): grid (W, cdiv(TK_local, BLOCK_TK)). Each
#   program owns (src_rank=pid_r, tile of slots in that source rank). It
#   loads the full (W, W) peer_count, computes an in-register exclusive
#   cumsum along axis 0 to get early_count[r, p] = Σ_{r'<r} peer_count[r', p],
#   masks down to row pid_r, and for each slot in the tile writes
#   slot_global = slot_per_rank + early_count[pid_r, dst].
#
# Two kernels (rather than one) because cross-rank dependency on
# peer_count_per_rank precludes intra-launch sync.
#
# Replaces the previous post-kernel torch sequence:
#   peer_count_per_rank.cumsum(dim=0)
#   torch.arange(TK_global) // TK_local
#   * W + dst_rank_flat.to(int64)
#   index_select on flattened early_count
#   add and final +
# i.e. five PyTorch ops collapsed into one kernel launch.
# ============================================================================

@triton.jit
def _dispatch_metadata_per_rank_kernel(
    topk_idx_g_ptr,
    out_dst_rank_flat_ptr,
    out_slot_per_rank_ptr,
    out_my_dst_rank_ptr,
    out_my_expert_local_ptr,
    out_peer_count_per_rank_ptr,
    my_rank: tl.constexpr,
    W: tl.constexpr,
    TK_local: tl.constexpr,
    E_local: tl.constexpr,
    BLOCK_TK: tl.constexpr,
):
    pid_r = tl.program_id(0)
    is_mine = pid_r == my_rank
    base = pid_r * TK_local
    peer_axis = tl.arange(0, W)

    carry = tl.zeros([W], dtype=tl.int32)
    n_tiles = (TK_local + BLOCK_TK - 1) // BLOCK_TK
    for tile in range(n_tiles):
        offs = tile * BLOCK_TK + tl.arange(0, BLOCK_TK)
        valid = offs < TK_local
        expert_global = tl.load(
            topk_idx_g_ptr + base + offs, mask=valid, other=0)
        dst = expert_global // E_local

        one_hot = (dst[:, None] == peer_axis[None, :]).to(tl.int32)
        one_hot = tl.where(valid[:, None], one_hot, 0)

        cumsum = tl.cumsum(one_hot, axis=0) + carry[None, :]
        pos = tl.sum(cumsum * one_hot, axis=1) - 1

        flat_offs = base + offs
        tl.store(out_dst_rank_flat_ptr + flat_offs, dst, mask=valid)
        tl.store(out_slot_per_rank_ptr + flat_offs, pos, mask=valid)
        if is_mine:
            local_expert = expert_global - dst * E_local
            tl.store(out_my_dst_rank_ptr + offs, dst, mask=valid)
            tl.store(out_my_expert_local_ptr + offs, local_expert, mask=valid)

        carry = carry + tl.sum(one_hot, axis=0)

    tl.store(out_peer_count_per_rank_ptr + pid_r * W + peer_axis, carry)


@triton.jit
def _slot_global_kernel(
    dst_rank_flat_ptr,             # (TK_global,) int32
    slot_per_rank_ptr,             # (TK_global,) int32
    peer_count_per_rank_ptr,       # (W, W) int32
    out_slot_global_ptr,           # (TK_global,) int32
    W: tl.constexpr,
    TK_local: tl.constexpr,
    BLOCK_TK: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_tile = tl.program_id(1)

    # Load full (W, W) peer_count, compute exclusive cumsum along axis 0.
    rs = tl.arange(0, W)[:, None]                      # source rank
    ps = tl.arange(0, W)[None, :]                      # destination peer
    pc = tl.load(peer_count_per_rank_ptr + rs * W + ps)
    excl = tl.cumsum(pc, axis=0) - pc                  # (W, W) int32

    # Extract row pid_r as a (W,) early-count vector. (No direct row-indexing
    # in Triton — use mask + sum.)
    row_mask = (tl.arange(0, W) == pid_r)              # (W,) bool
    early_row = tl.sum(tl.where(row_mask[:, None], excl, 0), axis=0)  # (W,) int32

    # Process this tile of slots in source rank pid_r.
    tile = tl.arange(0, BLOCK_TK)
    tk_offs = pid_tile * BLOCK_TK + tile
    valid = tk_offs < TK_local
    flat = pid_r * TK_local + tk_offs

    dst = tl.load(dst_rank_flat_ptr + flat, mask=valid, other=0)
    slot_pr = tl.load(slot_per_rank_ptr + flat, mask=valid, other=0)

    # Gather early_row[dst] via outer-equality + masked sum.
    one_hot_p = (dst[:, None] == tl.arange(0, W)[None, :]).to(tl.int32)
    add = tl.sum(one_hot_p * early_row[None, :], axis=1)

    slot_global = slot_pr + add
    tl.store(out_slot_global_ptr + flat, slot_global, mask=valid)


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


def compute_dispatch_metadata(
    topk_idx_g: torch.Tensor,
    my_rank: int,
    E_local: int,
):
    """Fused producer+consumer dispatch metadata.

    Two kernels, no post-kernel torch ops. All cross-rank arithmetic
    (cumsum + gather of early_count[src_rank, dst]) is folded into
    `_slot_global_kernel`."""
    assert topk_idx_g.dim() == 3
    assert topk_idx_g.dtype == torch.int32
    W, T_local, K = topk_idx_g.shape
    TK_local = T_local * K
    TK_global = W * TK_local
    device = topk_idx_g.device

    dst_rank_flat = torch.empty(TK_global, dtype=torch.int32, device=device)
    slot_per_rank = torch.empty(TK_global, dtype=torch.int32, device=device)
    slot_global = torch.empty(TK_global, dtype=torch.int32, device=device)
    my_dst_rank = torch.empty((T_local, K), dtype=torch.int32, device=device)
    my_expert_local = torch.empty((T_local, K), dtype=torch.int32, device=device)
    peer_count_per_rank = torch.empty((W, W), dtype=torch.int32, device=device)

    target_block = max(256, min(2048, 16384 // max(W, 1)))
    BLOCK_TK = min(_next_pow2(TK_local), _next_pow2(target_block))
    BLOCK_TK = max(BLOCK_TK, 1)

    _dispatch_metadata_per_rank_kernel[(W,)](
        topk_idx_g.contiguous(),
        dst_rank_flat, slot_per_rank,
        my_dst_rank, my_expert_local, peer_count_per_rank,
        my_rank=my_rank, W=W, TK_local=TK_local, E_local=E_local,
        BLOCK_TK=BLOCK_TK,
    )

    _slot_global_kernel[(W, triton.cdiv(TK_local, BLOCK_TK))](
        dst_rank_flat, slot_per_rank, peer_count_per_rank,
        slot_global,
        W=W, TK_local=TK_local, BLOCK_TK=BLOCK_TK,
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
    }


# ============================================================================
# Utilities
# ============================================================================

def rendezvous(tensor, group):
    try:
        return symm_mem.rendezvous(tensor, group=group)
    except TypeError:
        return symm_mem.rendezvous(tensor, group_name=group.group_name)


def barrier(tensor, group):
    rendezvous(tensor, group).barrier()


def safe_block_size(chunk_numel, requested=4096):
    block = requested
    while triton.cdiv(chunk_numel, block) > _CUDA_MAX_GRID_Y:
        block *= 2
    return block


def _product(xs):
    out = 1
    for x in xs:
        out *= int(x)
    return out


# ============================================================================
# Python wrappers
# ============================================================================
# all_gather / all_to_all / reduce_scatter are intentionally shape-generic —
# they're called by non-EP code that doesn't have named (T_local, d, K, W)
# dimensions in scope. The EP-specific wrappers below (a2a_dispatch_pull,
# fused_gather_combine) DO have those dimensions, so they construct
# get_buffer shapes from the named dimensions instead of probing
# tensor.shape.
# ============================================================================

def all_gather(x_symm, group):
    """AG kernel. Caller provides symm input, gets regular output."""
    hdl = rendezvous(x_symm, group)
    W = hdl.world_size
    numel = x_symm.numel()
    output = torch.empty((W * x_symm.shape[0],) + tuple(x_symm.shape[1:]),
                         dtype=x_symm.dtype, device=x_symm.device)
    buf_tuple = tuple(hdl.get_buffer(r, tuple(x_symm.shape), x_symm.dtype) for r in range(W))
    grid = lambda META: (W, triton.cdiv(numel, META["BLOCK_SIZE"]))
    _all_gather_kernel[grid](
        buf_tuple, output,
        numel_per_rank=numel, world_size=W,
    )
    return output


def all_to_all(send_symm, group, recv):
    """Generic A2A kernel — kept as a building block for non-EP callers.
    The EP forward uses a2a_dispatch_pull instead."""
    hdl = rendezvous(send_symm, group)
    W = hdl.world_size
    chunk_numel = _product(send_symm.shape[1:])
    buf_tuple = tuple(hdl.get_buffer(r, tuple(send_symm.shape), send_symm.dtype) for r in range(W))
    grid = lambda META: (W, triton.cdiv(chunk_numel, META["BLOCK_SIZE"]))
    _all_to_all_kernel[grid](
        buf_tuple, recv,
        chunk_numel=chunk_numel, my_rank=hdl.rank, world_size=W,
    )
    return recv


def reduce_scatter(x_symm, group):
    hdl = rendezvous(x_symm, group)
    W = hdl.world_size
    T_local = x_symm.shape[0] // W
    per_rank_shape = (T_local,) + tuple(x_symm.shape[1:])
    numel = _product(per_rank_shape)
    output = torch.empty(per_rank_shape, dtype=x_symm.dtype, device=x_symm.device)
    buf_tuple = tuple(hdl.get_buffer(r, tuple(x_symm.shape), x_symm.dtype) for r in range(W))
    grid = lambda META: (triton.cdiv(numel, META["BLOCK_SIZE"]),)
    _reduce_scatter_kernel[grid](
        buf_tuple, output,
        numel_per_rank=numel, my_rank=hdl.rank, world_size=W,
    )
    return output


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
    assert dst_rank_flat.shape == (TK_global,) and dst_rank_flat.dtype == torch.int32
    assert slot_flat_per_rank.shape == (TK_global,) and slot_flat_per_rank.dtype == torch.int32

    x_peer_tuple = tuple(
        hdl.get_buffer(r, (T_local, d), x_symm.dtype) for r in range(W))
    recv_flat = recv.view(W * TK_local, d)

    grid = lambda META: (TK_global, triton.cdiv(d, META["BLOCK_D"]))
    _a2a_dispatch_pull_kernel[grid](
        x_peer_tuple,
        dst_rank_flat, slot_flat_per_rank,
        recv_flat,
        my_rank=hdl.rank, world_size=W,
        T_local=T_local, K=K, TK_local=TK_local, d=d,
    )
    return recv


def fused_gather_combine(y_symm, s_reverse_symm, src_dst_rank, dispatch_pos,
                       topk_scores, out, K, group, block_resolve=1024):
    """Two-pass combine kernel.

    y_symm has shape (TK_global, d) and s_reverse_symm has shape
    (TK_global,) by the workspace allocation contract. Build get_buffer
    shapes from the named (T_local, K, W, d) instead of probing tensor
    shapes."""
    hdl_y = rendezvous(y_symm, group)
    hdl_s = rendezvous(s_reverse_symm, group)
    W = hdl_y.world_size
    d = y_symm.shape[1]
    T_local = src_dst_rank.shape[0]
    n = T_local * K
    TK_global = W * T_local * K
    if T_local == 0:
        return

    src_flat = src_dst_rank.contiguous().view(-1)
    pos_flat = dispatch_pos.contiguous().view(-1)
    scores_flat = topk_scores.contiguous().view(-1)
    s_buf = tuple(hdl_s.get_buffer(r, (TK_global,), s_reverse_symm.dtype) for r in range(W))
    y_buf = tuple(hdl_y.get_buffer(r, (TK_global, d), y_symm.dtype) for r in range(W))

    s_on_peer = torch.empty(n, dtype=torch.int32, device=out.device)
    _resolve_peer_rows_kernel[(triton.cdiv(n, block_resolve),)](
        s_buf, src_flat, pos_flat, s_on_peer,
        n_assignments=n, BLOCK=block_resolve, world_size=W)

    grid = lambda META: (T_local, triton.cdiv(d, META["BLOCK_D"]))
    _gather_combine_kernel[grid](
        y_buf, src_flat, s_on_peer, scores_flat, out,
        K=K, d=d, world_size=W)
