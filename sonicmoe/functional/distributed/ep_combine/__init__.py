# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from ..collectives import _CUDA_MAX_GRID_Y, _prune_block_d_vs_d, reduce_scatter_triton, rendezvous


# ============================================================================
# A2A Combine kernel — single-pass with inline s_reverse resolve, no atomics.
# ============================================================================
# Design (one kernel):
#   grid (T_local, cdiv(d, BLOCK_D)). One program per (t, BLOCK_D-tile).
#   Inside, the K-loop is statically unrolled. For each k in 0..K:
#     1. Load peer = src_dst_rank[t, k], score.
#        pos = my_rank * TK_local + t * K + k is computed inline (each rank
#        owns the contiguous [my_rank * TK_local, (my_rank+1) * TK_local)
#        slice of every peer's TK_global-sized buffer.
#     2. Load peer.s_reverse[pos] via NVLink to get the row in peer.y_symm
#        (folded inline — no separate resolve kernel/barrier
#     3. Load peer.y_symm[s_peer, d-tile] via NVLink, multiply by score,
#        accumulate into an fp32 register block.
#   After the K-loop, store the register block (cast to dtype) into out[t, d-tile].
# ============================================================================

_A2A_COMBINE_CONFIGS = [
    triton.Config({"BLOCK_D": BLOCK_D}, num_warps=nw, num_stages=4)
    for BLOCK_D in [128, 256, 512, 1024, 2048, 4096]
    for nw in [2, 4, 8]
    if BLOCK_D // nw >= 32
]


@triton.autotune(
    configs=_A2A_COMBINE_CONFIGS,
    key=["K", "d", "world_size"],
    prune_configs_by={"early_config_prune": _prune_block_d_vs_d},
)
@triton.heuristics({"EVEN_D": lambda args: args["d"] % args["BLOCK_D"] == 0})
@triton.jit
def _a2a_combine_kernel(
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

    # Fold peer-match into load mask, accumulate unconditionally.
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
# a2a_combine_triton — NVLink combine of K (peer.y_symm, peer.s_reverse)
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
def a2a_combine_triton(
    y_symm: torch.Tensor,
    s_reverse_symm: torch.Tensor,
    src_dst_rank: torch.Tensor,
    topk_scores: Optional[torch.Tensor],
    out: torch.Tensor,
    K,
    group,
    hdl_y=None,
    hdl_s=None,
    y_peer_bufs=None,
    s_peer_bufs=None,
    my_rank=None,
):
    """gather + (optionally weighted) accumulate.

    Forward combine passes per-(t, k) scores; backward dx combine passes
    None (score-less). The kernel branches on WITH_SCORES at compile time.
    """
    d = y_symm.shape[1]
    T_local = src_dst_rank.shape[0]
    if y_peer_bufs is None:
        if hdl_y is None:
            hdl_y = rendezvous(y_symm, group)
        W = hdl_y.world_size
        TK_global = W * T_local * K
        y_peer_bufs = tuple(hdl_y.get_buffer(r, (TK_global, d), y_symm.dtype) for r in range(W))
    else:
        W = len(y_peer_bufs)
        TK_global = W * T_local * K
    if s_peer_bufs is None:
        if hdl_s is None:
            hdl_s = rendezvous(s_reverse_symm, group)
        s_peer_bufs = tuple(hdl_s.get_buffer(r, (TK_global,), s_reverse_symm.dtype) for r in range(W))
    if my_rank is None:
        if hdl_y is not None:
            my_rank = hdl_y.rank
        elif hdl_s is not None:
            my_rank = hdl_s.rank
        else:
            my_rank = dist.get_rank(group)
    my_rank_offset = my_rank * T_local * K  # int64 in kernel

    src_flat = src_dst_rank.view(-1)

    with_scores = topk_scores is not None
    if with_scores:
        scores_flat = topk_scores.view(-1)
    else:
        scores_flat = src_flat  # unused when WITH_SCORES is False

    grid = lambda META: (T_local, triton.cdiv(d, META["BLOCK_D"]))
    _a2a_combine_kernel[grid](
        y_peer_bufs,
        s_peer_bufs,
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
# Local combine — local pre-sum producer step. Single-store-per-row.
# ----------------------------------------------------------------------------
#
# Design (one kernel):
#   grid (cdiv(W * T_local, BLOCK_SLOT), cdiv(d, BLOCK_D)). Each program
#   walks ``BLOCK_SLOT`` adjacent (home_rank, home_t) output rows. For
#   each row a ``tl.static_range(K)`` loop walks the K expert slots:
#   if ``dst_rank_flat[f] == my_rank``, loads score and
#   y_symm[s_reverse[f], :] and accumulates score * row in fp32 registers.
#   If dst != my_rank, a masked load returns 0 (other=0.0), so score * 0 = 0
#   and the accumulator is unaffected — no conditional branch, no Triton SSA
#   miscompile risk from `if` inside tl.static_range.
#
#   After the K-loop, stores the register block ONCE into
#   partial_combine_buf[home_rank * T_local + home_t, :]. No atomic_add.
#   No zero_() required.
# ============================================================================

# ``BLOCK_SLOT`` is the row-coarsening factor for ``_local_combine_kernel``.
# Each program walks ``BLOCK_SLOT`` adjacent (home_rank, home_t) output
# rows; the per-row K-loop and store stay the same, but the per-program
# launch + d_mask compute amortizes across BLOCK_SLOT rows. Capped at 4
# to keep the unrolled body (BLOCK_SLOT × K) under control at K=10.
# ``BLOCK_SLOT=1`` is in the sweep as the autotune escape hatch.
_LOCAL_COMBINE_CONFIGS = [
    triton.Config(
        {"BLOCK_D": BD, "BLOCK_SLOT": BS},
        num_warps=nw,
        num_stages=4,
    )
    for BD in [128, 256, 512, 1024, 2048, 4096]
    for BS in [1, 2, 4, 8]
    for nw in [2, 4, 8]
    if 1024 <= BD * BS <= 16384
    if BD // nw >= 32
]


@triton.autotune(
    configs=_LOCAL_COMBINE_CONFIGS,
    key=["d", "world_size", "K", "SKIP_EMPTY", "WITH_OUT_FOR_SELF"],
    prune_configs_by={"early_config_prune": _prune_block_d_vs_d},
)
@triton.heuristics({"EVEN_D": lambda args: args["d"] % args["BLOCK_D"] == 0})
@triton.jit
def _local_combine_kernel(
    y_symm_ptr,  # (TK_global, d) expert output, flat
    s_reverse_ptr,  # (TK_global,) int32, dispatch slot -> row index
    dst_rank_flat_ptr,  # (TK_global,) int32, destination rank per slot
    scores_ag_ptr,  # only read when WITH_SCORES is True
    partial_combine_buf_ptr,  # (W * T_local, d) output, flat — register acc is fp32, store casts to ptr dtype
    out_for_self_ptr,  # (T_local, d) — only used when WITH_OUT_FOR_SELF
    T_local,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    K: tl.constexpr,
    d: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_SLOT: tl.constexpr,
    EVEN_D: tl.constexpr,
    WITH_SCORES: tl.constexpr,
    SKIP_EMPTY: tl.constexpr,
    WITH_OUT_FOR_SELF: tl.constexpr,
):
    """One program per (BLOCK_SLOT adjacent (home_rank, home_t) rows, d-tile).
    Walks BLOCK_SLOT output rows; for each, K expert slots are accumulated
    in registers and stored once. No atomics, no zero_().

    WITH_SCORES=True  → score-weighted sum (forward RS combine).
    WITH_SCORES=False → identity-weighted sum (backward dx RS combine);
                        the score load and multiply are elided at compile
                        time.

    SKIP_EMPTY=True   → if NO slot of (home_rank, home_t) routes to
                        my_rank, the accumulator is structurally zero
                        and we skip the store for that row. Tracked
                        per-row via a scalar `any_mine` register
                        OR-accumulated across the K static_range, gating
                        the final store. With BLOCK_SLOT > 1 the early
                        return is converted into a per-row store guard
                        so subsequent j-iterations within the same
                        program still execute.
    SKIP_EMPTY=False  → store every row unconditionally (zeros for
                        empty contributions). REQUIRED by the dense
                        reduce-scatter path: the RS reducer reads ALL
                        peer rows and would mis-sum stale data otherwise.

    KEY DESIGN POINT — the dst-vs-my_rank check is ALWAYS folded into the
    load mask (m = is_mine & d_mask, other=0.0), independent of
    WITH_SCORES. This is what makes the K-loop branch-free and avoids
    Triton SSA merging issues with `if` inside tl.static_range. Score-less
    mode just drops the multiply, not the mask.

    BLOCK_SLOT coarsens the program-x axis: each program walks BLOCK_SLOT
    adjacent output rows, amortizing launch + per-program setup
    (offs_d / d_mask compute, autotune key dispatch, etc.) across the
    rows. ``home_rank`` and ``is_self_stripe`` are recomputed per j —
    a single program may straddle a home_rank boundary at high BLOCK_SLOT,
    and each j-iteration routes to its own store target independently.
    """
    pid_block = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = offs_d < d  # const-True for EVEN_D

    WT_local = world_size * T_local  # tail-guard bound

    # Walk BLOCK_SLOT adjacent (home_rank, home_t) rows. Tail-guard for
    # the last block when BLOCK_SLOT doesn't divide WT_local evenly.
    for j in tl.static_range(BLOCK_SLOT):
        pid_ht = pid_block * BLOCK_SLOT + j
        if pid_ht < WT_local:
            base = pid_ht * K  # = (home_rank * T_local + home_t) * K

            # Per-row accumulator + ``any_mine`` flag (NOT shared
            # across j-iterations — each row must reset).
            acc = tl.zeros([BLOCK_D], dtype=tl.float32)
            any_mine = False

            for k in tl.static_range(K):
                f = base + k
                dst = tl.load(dst_rank_flat_ptr + f)
                is_mine = dst == my_rank  # scalar bool
                any_mine = any_mine | is_mine

                if WITH_SCORES:
                    score = tl.load(scores_ag_ptr + f).to(tl.float32)
                row_idx = tl.load(s_reverse_ptr + f).to(tl.int64)
                row_offs = row_idx * d + offs_d

                # Vector mask: valid-d AND routed-to-my_rank. When
                # is_mine is False the load returns 0, so the
                # contribution is 0 regardless of WITH_SCORES — no
                # conditional update.
                m = is_mine & d_mask
                row = tl.load(y_symm_ptr + row_offs, mask=m, other=0.0).to(tl.float32)

                if WITH_SCORES:
                    acc += score * row
                else:
                    acc += row

            home_rank = (pid_ht // T_local).to(tl.int32)
            is_self_stripe = home_rank == my_rank

            if WITH_OUT_FOR_SELF and is_self_stripe:
                # Self stripe: ALWAYS store to ``out_for_self``
                # unconditionally. ``rank_dedup_combine_triton``'s
                # consumer reads ``out[t]`` as the initial accumulator
                # value — gating this store on SKIP_EMPTY would leave
                # stale data and silently corrupt the result.
                home_t = (pid_ht - my_rank * T_local).to(tl.int64)
                out_self_offs = home_t * d + offs_d
                if EVEN_D:
                    tl.store(out_for_self_ptr + out_self_offs, acc)
                else:
                    tl.store(out_for_self_ptr + out_self_offs, acc, mask=d_mask)
            else:
                # Non-self stripes (or WITH_OUT_FOR_SELF=False).
                # Per-row store guard: ``return`` would skip the
                # remaining j-iterations under BLOCK_SLOT > 1, so we
                # gate the store with ``(not SKIP_EMPTY) | any_mine``
                # instead. At SKIP_EMPTY=False that's always True
                # (unconditional store, matches the original
                # behavior); at SKIP_EMPTY=True it reduces to
                # ``any_mine`` (matches the original early-return
                # gate, just per-row).
                if (not SKIP_EMPTY) | any_mine:
                    out_offs = pid_ht * d + offs_d
                    if EVEN_D:
                        tl.store(partial_combine_buf_ptr + out_offs, acc)
                    else:
                        tl.store(partial_combine_buf_ptr + out_offs, acc, mask=d_mask)


def local_combine(
    y_symm: torch.Tensor,
    s_reverse: torch.Tensor,
    dst_rank_flat: torch.Tensor,
    scores_ag: Optional[torch.Tensor],
    partial_combine_buf: torch.Tensor,
    K: int,
    T_local: int,
    group: dist.ProcessGroup,
    skip_empty: bool = False,
    out_for_self: Optional[torch.Tensor] = None,
) -> None:
    """Writes
        partial_combine_buf[home_rank * T_local + home_t, :] =
            sum_{k where dst[f] == my_rank}
                w[f] * y_symm[s_reverse[f], :]
    for each (home_rank, home_t) pair, where
        f    = (home_rank*T_local + home_t)*K + k
        w[f] = scores_ag[f]   if scores_ag is not None     (forward combine)
             = 1              otherwise                    (backward dx combine)

    No atomic_add. No zero_() needed when ``skip_empty=False`` — the
    kernel writes every output row exactly once.

    Args:
        y_symm: (TK_global, d) expert output.
        s_reverse: (TK_global,) int32, dispatch slot → row in y_symm.
        dst_rank_flat: (TK_global,) int32, destination rank per slot.
        scores_ag: GLOBAL all-gathered scores OR None for score-less mode.
            Accepted shapes when not None: (TK_global,) flat, or
            (W*T_local, K) — the natural output of all_gather(scores_symm).
            Flattened internally; no copy when already contiguous.
        partial_combine_buf: (W*T_local, d) output buffer (any float dtype). Should be
            symm-mem so it can feed reduce_scatter directly. The kernel
            accumulates in fp32 registers and casts at store time, so the
            buffer dtype controls only NVLink/HBM bytes, not accumulation
            precision. Written in-place.
        K: top-K experts per token.
        T_local: tokens per rank.
        group: the EP process group.
        skip_empty: when True, the kernel skips the partial_combine_buf store for
            (home_rank, home_t) rows that have no slot routed to my_rank.
            Saves the HBM store bandwidth on those rows; partial_combine_buf retains
            stale data there. Safe ONLY when downstream consumers access
            partial_combine_buf via a presence mask that excludes these same rows
            (e.g. ``rank_dedup_combine_triton`` reads partial_combine_buf only at
            rows where ``peer_present_mask`` is set, which is exactly
            the complement of empty rows). MUST be False for the dense
            reduce-scatter path (the RS reducer reads ALL peer rows
            and would mis-sum stale data otherwise). Default False.
        out_for_self: optional (T_local, d) tensor. When given, programs
            with ``home_rank == my_rank`` store the per-(my_rank, home_t)
            accumulator directly into ``out_for_self[home_t]`` and skip
            the ``partial_combine_buf[my_rank*T_local + home_t]`` write entirely.
            Self-stripe stores are unconditional regardless of
            ``skip_empty`` because ``rank_dedup_combine_triton``'s
            consumer reads ``out[t]`` as its initial accumulator value.
            Used by ``rank_dedup_combine_triton`` to elide the
            consumer's ``q == my_rank`` symm-mem self-read.
    """
    assert (
        partial_combine_buf.ndim == 2
    ), f"partial_combine_buf must be 2D (W*T_local, d), got shape {tuple(partial_combine_buf.shape)}."
    WT, d = partial_combine_buf.shape
    W = dist.get_world_size(group)
    my_rank = dist.get_rank(group)
    assert WT == W * T_local, f"partial_combine_buf.shape[0]={WT} != W*T_local={W*T_local} (W from group)"

    with_scores = scores_ag is not None
    if with_scores:
        scores_flat = scores_ag.view(-1)
    else:
        # Triton needs a concrete pointer arg even though the kernel never
        # reads it when WITH_SCORES=False. Reuse `dst_rank_flat` — its
        # type and content are irrelevant since the load is compiled out.
        scores_flat = dst_rank_flat

    with_out_for_self = out_for_self is not None
    if with_out_for_self:
        assert out_for_self.shape == (T_local, d), (
            f"out_for_self shape {tuple(out_for_self.shape)} != " f"(T_local={T_local}, d={d})"
        )
        out_self_arg = out_for_self
    else:
        # Same dummy-pointer trick as scores_flat: kernel never touches
        # this when WITH_OUT_FOR_SELF=False, so the type/content are
        # irrelevant. Reuse dst_rank_flat to keep argument count stable.
        out_self_arg = dst_rank_flat

    WT_local = W * T_local
    grid = lambda META: (
        triton.cdiv(WT_local, META["BLOCK_SLOT"]),
        triton.cdiv(d, META["BLOCK_D"]),
    )
    _local_combine_kernel[grid](
        y_symm,
        s_reverse,
        dst_rank_flat,
        scores_flat,
        partial_combine_buf,
        out_self_arg,
        T_local=T_local,
        my_rank=my_rank,
        world_size=W,
        K=K,
        d=d,
        WITH_SCORES=with_scores,
        SKIP_EMPTY=skip_empty,
        WITH_OUT_FOR_SELF=with_out_for_self,
    )


# ============================================================================
# Local-reduce + per-token sparse gather combine.
# ----------------------------------------------------------------------------
# Combine has *no* row-level deduplication — the K experts on a peer rank
# produce K *different* outputs, not duplicate rows. What we exploit is
# associativity: each rank pre-sums its own K-many contributions into
# per-(home_rank, home_t) partial sums (the local reduce — `_local_combine_kernel`
# above), and home ranks then sum those partial sums across peers
# (`_rank_dedup_combine_kernel` below). The bytes saving vs. plain
# reduce-scatter comes from skipping peer reads where the peer has no
# contribution for a given local token (sparse gather).
#
# `_rank_dedup_combine_kernel` — BLOCK_OUT_ROW-batched sparse gather.
#   grid (cdiv(T_local, BLOCK_OUT_ROW), cdiv(d, BLOCK_D)). Each program owns a
#   ``(BLOCK_OUT_ROW tokens, BLOCK_D d-lanes)`` tile and issues W loads of
#   (BLOCK_OUT_ROW × BLOCK_D) bytes per peer, using the 2-D mask
#   ``present[:, None] & d_mask[None, :]`` to skip rows where the peer
#   has no contribution. Per-peer transaction size scales with BLOCK_OUT_ROW,
#   in-flight request count stays roughly the same, NVLink saturation
#   improves at the same total byte count vs. a (T_local, cdiv(d, BLOCK_D))
#   grid that loads only BLOCK_D bytes at a time.
# ============================================================================

_RANK_DEDUP_COMBINE_CONFIGS = [
    triton.Config({"BLOCK_OUT_ROW": bt, "BLOCK_D": bd}, num_warps=nw, num_stages=ns)
    for bt in [1, 2, 4, 8]
    for bd in [128, 256, 512, 1024, 2048]
    for nw in [2, 4, 8]
    for ns in [3, 4]
    if 1024 <= bt * bd <= 16384  # 2-D tile size in [1024, 16384] lanes
    if bd // nw >= 32  # min lanes per warp
    if not (bd <= 256 and nw == 8)  # over-paralleled small tile
    if not (bd >= 1024 and nw == 2)  # under-paralleled large tile
]


def _prune_block_td_vs_d(configs, named_args, **kwargs):
    """Prune RANK_DEDUP combine configs by BLOCK_D vs problem ``d``
    and the CUDA grid_y limit. Same shape as ``_prune_block_d_vs_d``
    but operates on the (BLOCK_OUT_ROW, BLOCK_D) tile config namespace."""
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
        valid = [c for c in configs if c.kwargs["BLOCK_D"] <= max(d, 1)]
        kept = [max(valid or configs, key=lambda c: c.kwargs["BLOCK_D"])]
    return kept


@triton.autotune(
    configs=_RANK_DEDUP_COMBINE_CONFIGS,
    key=["d", "world_size", "WITH_OUT_FOR_SELF"],
    prune_configs_by={"early_config_prune": _prune_block_td_vs_d},
    restore_value=["out_ptr"],
)
@triton.heuristics(
    {
        "EVEN_D": lambda args: args["d"] % args["BLOCK_D"] == 0,
    }
)
@triton.jit
def _rank_dedup_combine_kernel(
    peer_partial_combine_buf_tuple,  # tuple of W (W*T_local, d) tensors — peers' local-reduce buffers
    peer_present_mask_ptr,  # (W, T_local) int8 — 1 if peer q has any expert for my (t)
    out_ptr,  # (T_local, d) — final aggregated output (RMW under WITH_OUT_FOR_SELF)
    T_local,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    d: tl.constexpr,
    BLOCK_OUT_ROW: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
    WITH_OUT_FOR_SELF: tl.constexpr,
):
    """BLOCK_OUT_ROW-batched per-token sparse gather.

    Each program owns ``(BLOCK_OUT_ROW tokens, BLOCK_D d-lanes)``. For each peer
    q in ``static_range``, loads a ``(BLOCK_OUT_ROW, BLOCK_D)`` tile from
    ``peer_partial_combine_buf[q]`` using the 2-D mask
    ``present[:, None] & d_mask[None, :]``. Bitwise deterministic;
    same accumulation order across runs (peer 0 → W-1).

    Per-peer transaction size scales with BLOCK_OUT_ROW × BLOCK_D × itemsize.

    WITH_OUT_FOR_SELF=False  → ``acc`` starts at zeros; the q == my_rank
        peer read goes through the same symm-mem tuple (peer-of-self maps
        to the local tensor with no P2P overhead).
    WITH_OUT_FOR_SELF=True   → the producer redirected the rank-self
        contribution to ``out`` directly. ``acc`` is initialized by
        loading from ``out_ptr`` (the self contribution), and the
        ``q == my_rank`` static_range iteration is fully elided. The
        autotune decorator declares ``restore_value=["out_ptr"]`` so the
        kernel's RMW on ``out`` doesn't accumulate across autotune
        candidate runs.
    """
    pid_t = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)

    t_offs = pid_t * BLOCK_OUT_ROW + tl.arange(0, BLOCK_OUT_ROW)
    valid_t = t_offs < T_local
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = offs_d < d

    out_offs = t_offs[:, None] * d + offs_d[None, :]
    out_mask = valid_t[:, None] & d_mask[None, :]

    if WITH_OUT_FOR_SELF:
        acc = tl.load(out_ptr + out_offs, mask=out_mask, other=0.0).to(tl.float32)
    else:
        acc = tl.zeros([BLOCK_OUT_ROW, BLOCK_D], dtype=tl.float32)

    # my_rank is constexpr ⇒ the base offset constant-folds.
    row_base = (my_rank * T_local + t_offs)[:, None] * d + offs_d[None, :]

    for q in tl.static_range(world_size):
        # Skip q == my_rank under WITH_OUT_FOR_SELF — the self
        # contribution was already loaded into acc from out_ptr above
        # (producer wrote it directly to ``out``, not to partial_combine_buf at this
        # row). Both predicates are constexpr at compile time, so the
        # iteration folds away cleanly.
        if not (WITH_OUT_FOR_SELF and q == my_rank):
            present = (
                tl.load(
                    peer_present_mask_ptr + q * T_local + t_offs,
                    mask=valid_t,
                    other=0,
                )
                != 0
            )
            m = present[:, None] & d_mask[None, :]
            rows = tl.load(peer_partial_combine_buf_tuple[q] + row_base, mask=m, other=0.0).to(tl.float32)
            acc += rows

    tl.store(out_ptr + out_offs, acc, mask=out_mask)


def rank_dedup_combine_triton(
    y_symm: torch.Tensor,
    s_reverse: torch.Tensor,
    dst_rank_flat: torch.Tensor,
    scores: Optional[torch.Tensor],
    peer_present_mask: torch.Tensor,
    partial_combine_buf: torch.Tensor,
    out: torch.Tensor,
    K: int,
    T_local: int,
    group,
    partial_combine_hdl=None,
    partial_combine_peer_bufs=None,
    my_rank=None,
) -> None:
    """Combine via local reduce + per-token sparse gather.

    Step 1 (``local_combine``): writes per-(home_rank, home_t) partial sums
    into local ``partial_combine_buf`` for non-self stripes, and into ``out`` directly
    for the self stripe (``home_rank == my_rank``). Empty contributions
    on non-self stripes are skipped via the ``skip_empty=True`` path;
    self-stripe stores are unconditional because the consumer reads
    ``out[t]`` as its initial accumulator value. Fully local — no
    barrier needed before this step (same-stream ordering with the
    prior down-proj write to ``y_symm`` is sufficient).

    Step 2 (``_rank_dedup_combine_kernel``): for each local token t,
    peer-reads ``partial_combine_buf[q][my_rank*T_local + t]`` and accumulates,
    skipping peers where ``peer_present_mask[q, t] == 0`` AND skipping
    the ``q == my_rank`` iteration entirely (self contribution comes
    from ``out`` via the in-kernel init load).

    NVLink bytes per rank inbound:
        (W-1) · T_local · (1 - (1-1/W)^K) · H · sizeof(dtype)
    in expectation under uniform routing. Worst case ≤ (W-1) · T_local
    · H · sizeof(dtype), matching plain RS combine. Strictly ≤ A2A
    combine when K > 1.

    Args:
        y_symm: (TK_global, d) or (MAX_ROWS_PER_RANK, d) expert outputs.
        s_reverse: (TK_global,) int32 — dispatch slot → row in y_symm.
        dst_rank_flat: (TK_global,) int32 — destination rank per slot.
        scores: GLOBAL (TK_global,) all-gathered scores OR None for
            score-less mode (backward dx combine). Accepted shapes when
            not None: (TK_global,) flat or (W*T_local, K).
        peer_present_mask: (W, T_local) int8 — 1 if peer q has any
            expert for my (t). From compute_dispatch_metadata's
            emit_dedup output.
        partial_combine_buf: symm-mem buffer (W*T_local, d) (model dtype). Step 1
            writes non-self stripes; step 2 peer-reads them. Same buffer
            the RS_COMBINE path uses — workspaces share one allocation.
            The self stripe (``partial_combine_buf[my_rank*T_local : (my_rank+1)*T_local]``)
            is left unwritten by this path (producer redirects to ``out``;
            consumer skips ``q == my_rank``), so its contents are
            irrelevant to the result.
        out: (T_local, d) local output buffer (model dtype). Written
            in-place by both step 1 (self stripe) and step 2 (final
            accumulated output). fp32 register accumulation casts at
            store time.
        K: top-K experts per token.
        T_local: tokens per rank.
        group: process group.
    """
    # Step 1: local reduce — fully local. ``skip_empty=True`` saves the
    # HBM store on non-self rows where no slot routes to my_rank
    # (consumer's peer_present_mask gates exactly those rows). The
    # ``out_for_self=out`` redirect routes the rank-self stripe directly
    # into ``out`` so the consumer can pick it up via its initial
    # accumulator load and elide the q == my_rank symm-mem self-read.
    local_combine(
        y_symm,
        s_reverse,
        dst_rank_flat,
        scores,
        partial_combine_buf,
        K,
        T_local,
        group=group,
        skip_empty=True,
        out_for_self=out,
    )

    # Step 2: per-token sparse gather (communication only). Internally
    # issues the inter-rank barrier partial_combine_hdl.barrier() before the peer
    # reads. ``init_acc_from_out=True`` matches the producer's
    # ``out_for_self=out`` redirect: the kernel loads its initial acc
    # from ``out`` (the self contribution) and skips the q == my_rank
    # symm-mem self-read.
    _rank_dedup_combine_communication_triton(
        partial_combine_buf,
        peer_present_mask,
        out,
        T_local=T_local,
        group=group,
        partial_combine_hdl=partial_combine_hdl,
        partial_combine_peer_bufs=partial_combine_peer_bufs,
        my_rank=my_rank,
        init_acc_from_out=True,
    )


def _rank_dedup_combine_communication_triton(
    partial_combine_buf: torch.Tensor,
    peer_present_mask: torch.Tensor,
    out: torch.Tensor,
    *,
    T_local: int,
    group,
    partial_combine_hdl=None,
    partial_combine_peer_bufs=None,
    my_rank: Optional[int] = None,
    init_acc_from_out: bool = False,
) -> None:
    """Communication-only step of RANK_DEDUP combine: per-token sparse
    cross-rank gather over peers' pre-populated ``partial_combine_buf``.

    Caller contract: ``partial_combine_buf`` has been written by ``local_combine`` (or
    equivalent) on every rank BEFORE this call. The wrapper issues the
    inter-rank ``partial_combine_hdl.barrier()`` internally so peers' writes are
    visible before the gather kernel reads them.

    Args:
        partial_combine_buf: symm-mem buffer (W*T_local, d). Pre-populated.
        peer_present_mask: (W, T_local) int8 — 1 if peer q has any
            partial sum for my (t).
        out: (T_local, d) local output buffer; written in place.
            When ``init_acc_from_out=True``, ALSO read at the start as
            the initial accumulator (must be pre-populated with the
            self-stripe contribution by ``local_combine(out_for_self=out)``).
        T_local: tokens per rank.
        group: process group.
        partial_combine_hdl / partial_combine_peer_bufs / my_rank: optional cached rendezvous
            handle / peer-buf tuple / my-rank int. Auto-resolved when
            None.
        init_acc_from_out: when True, the kernel initializes its
            accumulator by loading from ``out`` and skips the
            ``q == my_rank`` static_range iteration entirely. Pairs
            with ``local_combine(out_for_self=out)`` on the producer
            side. Default False — kernel zero-inits acc and the
            q == my_rank iteration reads partial_combine_buf at the self stripe
            (which the producer must have populated).
    """
    d = partial_combine_buf.shape[-1]
    if partial_combine_peer_bufs is None:
        if partial_combine_hdl is None:
            partial_combine_hdl = rendezvous(partial_combine_buf, group)
        W = partial_combine_hdl.world_size
        my_rank = partial_combine_hdl.rank if my_rank is None else my_rank
        partial_combine_peer_bufs = tuple(
            partial_combine_hdl.get_buffer(r, tuple(partial_combine_buf.shape), partial_combine_buf.dtype)
            for r in range(W)
        )
    else:
        W = len(partial_combine_peer_bufs)
        if my_rank is None:
            my_rank = partial_combine_hdl.rank if partial_combine_hdl is not None else dist.get_rank(group)

    # Barrier: peers must have finished writing partial_combine_buf before we read.
    if partial_combine_hdl is None:
        partial_combine_hdl = rendezvous(partial_combine_buf, group)
    partial_combine_hdl.barrier()

    grid = lambda META: (
        triton.cdiv(T_local, META["BLOCK_OUT_ROW"]),
        triton.cdiv(d, META["BLOCK_D"]),
    )
    _rank_dedup_combine_kernel[grid](
        partial_combine_peer_bufs,
        peer_present_mask,
        out,
        T_local=T_local,
        my_rank=my_rank,
        world_size=W,
        d=d,
        WITH_OUT_FOR_SELF=init_acc_from_out,
    )


def rs_combine_triton(
    y_symm: torch.Tensor,
    s_reverse: torch.Tensor,
    dst_rank_flat: torch.Tensor,
    scores_ag: Optional[torch.Tensor],
    partial_combine_buf: torch.Tensor,
    out: torch.Tensor,
    K: int,
    T_local: int,
    group,
    partial_combine_hdl=None,
    partial_combine_peer_bufs=None,
    my_rank: Optional[int] = None,
) -> torch.Tensor:
    """Full RS combine: ``local_combine`` (producer) + barrier +
    ``reduce_scatter_triton`` (cross-rank dense reduce).

    Caller contract: ``y_symm`` has been written and a same-stream
    ordering with this call is sufficient (the producer is fully local).
    The wrapper issues an ``partial_combine_hdl.barrier()`` between the producer and
    the reduce-scatter so peers' partial_combine_buf writes are visible before the
    cross-rank reads.

    Args:
        y_symm: (TK_global, d) expert outputs.
        s_reverse: (TK_global,) int32 — slot → row in y_symm.
        dst_rank_flat: (TK_global,) int32 — destination rank per slot.
        scores_ag: GLOBAL all-gathered scores OR None for score-less
            mode (backward dx combine). Accepted shapes when not None:
            (TK_global,) flat or (W*T_local, K).
        partial_combine_buf: symm-mem buffer (W*T_local, d). Producer writes here;
            the reducer peer-reads it.
        out: (T_local, d) local output buffer (model dtype). Written
            in-place — fp32 register accumulation casts at store time.
        K: top-K experts per token.
        T_local: tokens per rank.
        group: process group.
        partial_combine_hdl / partial_combine_peer_bufs / my_rank: optional cached state, same
            semantics as elsewhere in this module.

    Returns ``out`` (for convenience).
    """
    # Producer: writes partial_combine_buf locally — no peer reads, no barrier needed
    # before this kernel; same-stream ordering with the prior write to
    # y_symm is sufficient.
    local_combine(
        y_symm,
        s_reverse,
        dst_rank_flat,
        scores_ag,
        partial_combine_buf,
        K,
        T_local,
        group=group,
    )

    # Barrier before the reduce-scatter peer-reads partial_combine_buf.
    if partial_combine_hdl is None:
        partial_combine_hdl = rendezvous(partial_combine_buf, group)
    partial_combine_hdl.barrier()

    reduce_scatter_triton(
        partial_combine_buf,
        group,
        out=out,
        hdl=partial_combine_hdl,
        peer_bufs=partial_combine_peer_bufs,
        my_rank=my_rank,
    )
    return out
