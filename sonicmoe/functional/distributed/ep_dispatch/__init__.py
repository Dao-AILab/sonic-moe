# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from __future__ import annotations

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from ..collectives import _prune_block_d_vs_d, rendezvous


_A2A_DISPATCH_CONFIGS = [
    triton.Config({"BLOCK_D": BLOCK_D}, num_warps=nw, num_stages=ns)
    for BLOCK_D in [128, 256, 512, 1024, 2048, 4096]
    for nw in [2, 4, 8]
    for ns in [3, 4]
    if BLOCK_D // nw >= 32  # min lanes per warp
    if not (BLOCK_D <= 256 and nw == 8)  # over-paralleled small tile
    if not (BLOCK_D >= 2048 and ns == 4)  # huge memory-bound tile already saturates at ns=3
]


# Private to `_rank_dedup_dispatch_kernel`.
#
# ``BLOCK_SLOT`` is the slot-coarsening factor: each program walks
# ``BLOCK_SLOT`` adjacent global slots in the peer-interleaved
# decomposition (one slot = one (src_rank, src_local_token, k) triple
# in the TK_global enumeration). Coarsening amortizes the per-program
# launch + early-return cost across BLOCK_SLOT slots, which matters
# for rank-dedup because most programs early-return — only canonical
# slots routed to ``my_rank`` produce work, so the dead-program rate
# is high. ``BLOCK_SLOT=1`` is in the autotune sweep as the escape
# hatch for shapes where coarsening hurts.
_RANK_DEDUP_DISPATCH_CONFIGS = [
    triton.Config(
        {"BLOCK_D": BLOCK_D, "BLOCK_SLOT": BLOCK_SLOT},
        num_warps=nw,
        num_stages=ns,
    )
    for BLOCK_D in [128, 256, 512, 1024, 2048, 4096]
    for BLOCK_SLOT in [1, 2, 4]
    for nw in [2, 4, 8]
    for ns in [3, 4]
    if BLOCK_D // nw >= 32
    if not (BLOCK_D <= 256 and nw == 8)
    if not (BLOCK_D >= 2048 and ns == 4)
]


# ============================================================================
# Fused A2A dispatch with permute
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
    configs=_A2A_DISPATCH_CONFIGS,
    key=["d", "world_size"],
    prune_configs_by={"early_config_prune": _prune_block_d_vs_d},
)
@triton.heuristics({"EVEN_D": lambda args: args["d"] % args["BLOCK_D"] == 0})
@triton.jit
def _a2a_dispatch_kernel(
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


def a2a_dispatch_triton(
    x_symm: torch.Tensor,
    dst_rank_flat: torch.Tensor,
    recv_pos: torch.Tensor,
    recv: torch.Tensor,
    K: int,
    group,
    hdl=None,
    peer_bufs=None,
    my_rank=None,
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
    T_local, d = x_symm.shape
    if peer_bufs is None:
        if hdl is None:
            hdl = rendezvous(x_symm, group)
        W = hdl.world_size
        my_rank = hdl.rank if my_rank is None else my_rank
        peer_bufs = tuple(hdl.get_buffer(r, (T_local, d), x_symm.dtype) for r in range(W))
    else:
        W = len(peer_bufs)
        if my_rank is None:
            my_rank = hdl.rank if hdl is not None else dist.get_rank(group)
    TK_local = T_local * K
    TK_global = W * TK_local

    recv_flat = recv.view(-1, d)

    grid = lambda META: (TK_global, triton.cdiv(d, META["BLOCK_D"]))
    _a2a_dispatch_kernel[grid](
        peer_bufs, dst_rank_flat, recv_pos, recv_flat, TK_local=TK_local, my_rank=my_rank, world_size=W, K=K,
        d=d,
    )
    return recv


# ============================================================================
# RANK_DEDUP dispatch — single-pass: peer-pull canonical-only into a packed
# (by source rank) symm-mem buffer. The expert-grouped recv layout is no
# longer materialized; the up-proj GEMM consumes ``recv_packed`` directly
# via an A_idx that maps expert-grouped row → packed row (built by
# ``build_rank_dedup_a_idx`` below).
# ============================================================================


@triton.autotune(
    configs=_RANK_DEDUP_DISPATCH_CONFIGS,
    key=["d", "world_size", "K"],
    prune_configs_by={"early_config_prune": _prune_block_d_vs_d},
)
@triton.heuristics({"EVEN_D": lambda args: args["d"] % args["BLOCK_D"] == 0})
@triton.jit
def _rank_dedup_dispatch_kernel(
    x_peer_tuple,  # tuple[(T_local, d) tensor, ...] for each peer
    pair_present_mask_ptr,  # (TK_global,) int8
    dst_rank_flat_ptr,  # (TK_global,) int32
    rank_dedup_recv_pos_ptr,  # (TK_global,) int32
    recv_packed_ptr,  # flat (>= MAX_PAIR_COUNT, d)
    TK_local,
    TK_global,  # tail bounds check for the BLOCK_SLOT walk
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    K: tl.constexpr,
    d: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_SLOT: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    pid_block = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1)

    base_pid_orig = pid_block * BLOCK_SLOT

    # BLOCK_SLOT-batched: each program walks BLOCK_SLOT adjacent slots in the
    # original peer-interleaved decomposition. Same per-slot work, but
    # the launch + early-return cost amortizes across BLOCK_SLOT slots.
    # The static_range over peers stays inside (still required to
    # materialize a constant peer pointer for tl.load).
    for j in tl.static_range(BLOCK_SLOT):
        pid_orig = base_pid_orig + j
        # Tail guard: TK_global may not divide BLOCK_SLOT evenly.
        if pid_orig < TK_global:
            src_rank = (pid_orig % world_size).to(tl.int32)
            pid_tk = pid_orig // world_size
            orig_idx = src_rank.to(tl.int64) * TK_local + pid_tk

            dst = tl.load(dst_rank_flat_ptr + orig_idx)
            is_canonical = tl.load(pair_present_mask_ptr + orig_idx)
            # Combined predicate: both loads are issued unconditionally
            # (the second is one int8, dominated by the first int32),
            # which folds the original two-stage early-return into a
            # single warp-uniform branch. Saves a divergence point.
            if (dst == my_rank) & (is_canonical != 0):
                pos = tl.load(rank_dedup_recv_pos_ptr + orig_idx).to(tl.int64)
                t_local = pid_tk // K

                offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
                src_offs = t_local * d + offs_d
                dst_offs = pos * d + offs_d

                for i in tl.static_range(world_size):
                    if src_rank == i:
                        if EVEN_D:
                            row = tl.load(x_peer_tuple[i] + src_offs)
                            tl.store(recv_packed_ptr + dst_offs, row)
                        else:
                            d_mask = offs_d < d
                            row = tl.load(x_peer_tuple[i] + src_offs, mask=d_mask)
                            tl.store(recv_packed_ptr + dst_offs, row, mask=d_mask)


@triton.jit
def _build_rank_dedup_a_idx_kernel(
    dst_rank_flat_ptr,  # (TK_global,) int32
    s_reverse_local_ptr,  # (TK_global,) int32 — slot → expert-grouped row (permutation on my-rank slots)
    rank_dedup_recv_pos_ptr,  # (TK_global,) int32 — slot → packed row at my_rank
    a_idx_ptr,  # (MAX_ROWS_PER_RANK_STATIC,) int32 — output
    TK_global,
    my_rank: tl.constexpr,
    BLOCK_SLOT: tl.constexpr,
):
    """Scatter ``rank_dedup_recv_pos[f] → a_idx[s_reverse_local[f]]`` for every
    slot f routed to my_rank. Contention-free: when restricted to my-rank
    slots, ``s_reverse_local`` is a permutation onto
    ``[0, sum_p pair_count[p, my_rank] · K_avg)`` (one unique e per
    canonical slot, K-fanout for shared (src,t)→my_rank triples). All K
    slots of a shared triple share the same ``rank_dedup_recv_pos`` value, so
    the K writes to distinct e's all carry the same packed row index ⇒
    the K stores agree by construction.

    Tail of a_idx (rows beyond #routed_to_my_rank) is left uninitialized —
    the GEMM only reads up to expert_frequency_offset[E_local], matching
    the same convention as x_gather_idx in AG mode.

    PRECONDITION (not enforced locally): for every pair of slots (f1, f2)
    with ``dst_rank_flat[f1] == dst_rank_flat[f2] == my_rank``, the values
    ``s_reverse_local[f1]`` and ``s_reverse_local[f2]`` are equal iff
    ``rank_dedup_recv_pos[f1] == rank_dedup_recv_pos[f2]``. Equivalently:
    restricted to my-rank slots, ``(s_reverse_local, rank_dedup_recv_pos)``
    is consistent — every distinct expert-grouped row e maps to a single
    packed row p, and every K-fanout group of slots sharing one packed
    row p maps to a contiguous block of distinct e's. This is guaranteed
    by ``general_routing_router_metadata_triton`` in
    ``_build_consumer_metadata`` (the histogram-and-bucketed-sort that
    builds ``s_reverse_local``). If violated (duplicate expert-grouped
    rows mapping to *different* packed rows for in-rank slots), the
    scatter degrades into a lost-update race and ``a_idx`` is silently
    corrupted.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SLOT + tl.arange(0, BLOCK_SLOT)
    valid = offs < TK_global

    dst = tl.load(dst_rank_flat_ptr + offs, mask=valid, other=-1)
    is_mine = (dst == my_rank) & valid

    e = tl.load(s_reverse_local_ptr + offs, mask=is_mine, other=0).to(tl.int64)
    p = tl.load(rank_dedup_recv_pos_ptr + offs, mask=is_mine, other=0)
    tl.store(a_idx_ptr + e, p, mask=is_mine)


def rank_dedup_dispatch_triton(
    x_symm: torch.Tensor,
    dst_rank_flat: torch.Tensor,
    pair_present_mask: torch.Tensor,
    rank_dedup_recv_pos: torch.Tensor,
    recv_packed: torch.Tensor,
    K: int,
    group,
    hdl=None,
    peer_bufs=None,
    my_rank=None,
):
    """RANK_DEDUP dispatch: one peer NVLink read per (src, t, my_rank)
    triple with ≥1 routed slot. Output ``recv_packed`` is packed-by-source:
    rows grouped by src rank, stripes given by ``pair_offset[:, my_rank]``,
    within-stripe order = source-side token index.

    Downstream up-proj GEMM consumes ``recv_packed`` via an A_idx that
    maps expert-grouped row → packed row. Build it via
    ``build_rank_dedup_a_idx``.

    Caller contract: x_symm has been written and a barrier issued before
    the call. recv_packed is read only locally by this rank's GEMM, so
    no post-call barrier is required.
    """
    T_local, d = x_symm.shape
    if peer_bufs is None:
        if hdl is None:
            hdl = rendezvous(x_symm, group)
        W = hdl.world_size
        my_rank = hdl.rank if my_rank is None else my_rank
        peer_bufs = tuple(hdl.get_buffer(r, (T_local, d), x_symm.dtype) for r in range(W))
    else:
        W = len(peer_bufs)
        if my_rank is None:
            my_rank = hdl.rank if hdl is not None else dist.get_rank(group)
    TK_local = T_local * K
    TK_global = W * TK_local

    recv_packed_flat = recv_packed.view(-1, d)
    grid = lambda META: (
        triton.cdiv(TK_global, META["BLOCK_SLOT"]),
        triton.cdiv(d, META["BLOCK_D"]),
    )
    _rank_dedup_dispatch_kernel[grid](
        peer_bufs, pair_present_mask, dst_rank_flat, rank_dedup_recv_pos, recv_packed_flat, TK_local=TK_local,
        TK_global=TK_global, my_rank=my_rank, world_size=W, K=K, d=d,
    )
    return recv_packed


def build_rank_dedup_a_idx(
    dst_rank_flat: torch.Tensor,
    s_reverse_local: torch.Tensor,
    rank_dedup_recv_pos: torch.Tensor,
    my_rank: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Build the dedup-mode up-proj A_idx in-place into ``out``.

    For every f with ``dst_rank_flat[f] == my_rank``:
        out[s_reverse_local[f]] = rank_dedup_recv_pos[f]

    A_idx[e] then gives the row in the dedup packed buffer that the
    e-th expert-grouped row should gather from.

    Args:
        dst_rank_flat: (TK_global,) int32 — destination rank per slot.
        s_reverse_local: (TK_global,) int32 — slot → expert-grouped row.
            From _build_consumer_metadata / general_routing_router_metadata_triton.
        rank_dedup_recv_pos: (TK_global,) int32 — slot → packed row at destination.
            From compute_dispatch_metadata's emit_dedup output.
        my_rank: this rank.
        out: (MAX_ROWS_PER_RANK_STATIC,) int32. Written in place. Tail
            beyond #routed_to_my_rank stays uninitialized.

    Returns ``out``.
    """
    TK_global = dst_rank_flat.shape[0]
    BLOCK_SLOT = 1024
    grid = (triton.cdiv(TK_global, BLOCK_SLOT),)
    _build_rank_dedup_a_idx_kernel[grid](
        dst_rank_flat, s_reverse_local, rank_dedup_recv_pos, out, TK_global=TK_global, my_rank=my_rank,
        BLOCK_SLOT=BLOCK_SLOT,
    )
    return out
