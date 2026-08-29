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


# A2A combine autotune configs (impl + rationale documented in the banner below).

_A2A_COMBINE_CONFIGS = [
    triton.Config({"BLOCK_D": BLOCK_D}, num_warps=nw, num_stages=4)
    for BLOCK_D in [128, 256, 512, 1024, 2048, 4096]
    for nw in [2, 4, 8]
    if BLOCK_D // nw >= 32
]


# ============================================================================
# A2A combine, RUNTIME PEER ADDRESSING: replaces the `for i in range(W)` masked-load loop with an
# int64 peer-base-address tensor indexed by the runtime peer rank (K direct loads/token, not K x W).
# ============================================================================

# torch dtype -> triton pointer element dtype (for int->ptr casts).
_TL_PTR_DTYPE = {
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.int32: tl.int32,
    torch.int64: tl.int64,
}


@triton.autotune(
    configs=_A2A_COMBINE_CONFIGS,
    key=["K", "d"],
    prune_configs_by={"early_config_prune": _prune_block_d_vs_d},
)
@triton.heuristics({"EVEN_D": lambda args: args["d"] % args["BLOCK_D"] == 0})
@triton.jit
def _a2a_combine_rt_kernel(
    peer_y_base_ptr,  # int64[W]: data_ptr of each peer's y_symm buffer
    peer_s_base_ptr,  # int64[W]: data_ptr of each peer's s_reverse buffer
    src_dst_rank_ptr,
    scores_ptr,  # only read when WITH_SCORES is True
    y_local_ptr,
    my_rank_offset,  # int64 scalar: my_rank * TK_local
    K: tl.constexpr,
    d: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
    WITH_SCORES: tl.constexpr,
    Y_DTYPE: tl.constexpr,
    S_DTYPE: tl.constexpr,
):
    """One program per (t, BLOCK_D-tile). K-serial inner loop, runtime peer addressing (no W-loop).
    Register fp32 accumulator, single non-atomic store."""
    pid_t = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = offs_d < d
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    base = pid_t * K
    pos_base = my_rank_offset.to(tl.int64) + base

    for k in tl.static_range(K):
        peer = tl.load(src_dst_rank_ptr + base + k)  # scalar peer rank
        valid = peer >= 0
        peer_safe = tl.where(valid, peer, 0)  # avoid OOB base index when padded
        pos = pos_base + k
        if WITH_SCORES:
            score = tl.load(scores_ptr + base + k).to(tl.float32)
        # Resolve s_reverse[pos] from the matched peer's buffer (one load).
        s_base = tl.load(peer_s_base_ptr + peer_safe)  # int64 address
        s_ptr = s_base.to(tl.pointer_type(S_DTYPE))
        s_peer = tl.load(s_ptr + pos, mask=valid, other=0).to(tl.int64)
        # Read y_symm[s_peer, d-tile] from the matched peer's buffer (one load).
        y_base = tl.load(peer_y_base_ptr + peer_safe)  # int64 address
        y_ptr = y_base.to(tl.pointer_type(Y_DTYPE))
        row_offs = s_peer * d + offs_d
        m = valid & d_mask
        row = tl.load(y_ptr + row_offs, mask=m, other=0.0).to(tl.float32)
        if WITH_SCORES:
            acc += row * score
        else:
            acc += row

    out_offs = pid_t * d + offs_d
    if EVEN_D:
        tl.store(y_local_ptr + out_offs, acc)
    else:
        tl.store(y_local_ptr + out_offs, acc, mask=d_mask)


def build_a2a_peer_base(y_symm, s_reverse_symm, group, hdl_y=None, hdl_s=None):
    """Builds (peer_y_base, peer_s_base, my_rank) address tensors for a2a_combine_triton. Host op — call
    ONCE at setup; symm-mem addresses are constant for the allocation's lifetime (host-sync-free hot path)."""
    d = y_symm.shape[1]
    if hdl_y is None:
        hdl_y = rendezvous(y_symm, group)
    if hdl_s is None:
        hdl_s = rendezvous(s_reverse_symm, group)
    W = hdl_y.world_size
    TK_global = s_reverse_symm.shape[0]
    y_bufs = [hdl_y.get_buffer(r, (TK_global, d), y_symm.dtype) for r in range(W)]
    s_bufs = [hdl_s.get_buffer(r, (TK_global,), s_reverse_symm.dtype) for r in range(W)]
    peer_y_base = torch.tensor([b.data_ptr() for b in y_bufs], dtype=torch.int64, device=y_symm.device)
    peer_s_base = torch.tensor([b.data_ptr() for b in s_bufs], dtype=torch.int64, device=y_symm.device)
    return peer_y_base, peer_s_base, hdl_y.rank


def a2a_combine_triton(
    y_symm: torch.Tensor,
    s_reverse_symm: torch.Tensor,
    src_dst_rank: torch.Tensor,
    topk_scores: Optional[torch.Tensor],
    out: torch.Tensor,
    K,
    group,
    peer_y_base=None,
    peer_s_base=None,
    my_rank=None,
):
    """NVLink A2A combine (runtime peer addressing): gathers + accumulates K rows/token, weighted by
    scores if given (None -> score-less dx combine). Pass precomputed peer bases to stay host-sync-free."""
    d = y_symm.shape[1]
    T_local = src_dst_rank.shape[0]
    if peer_y_base is None or peer_s_base is None:
        peer_y_base, peer_s_base, my_rank = build_a2a_peer_base(y_symm, s_reverse_symm, group)
    if my_rank is None:
        my_rank = dist.get_rank(group)
    my_rank_offset = my_rank * T_local * K

    src_flat = src_dst_rank.view(-1)
    with_scores = topk_scores is not None
    scores_flat = topk_scores.view(-1) if with_scores else src_flat

    grid = lambda META: (T_local, triton.cdiv(d, META["BLOCK_D"]))
    _a2a_combine_rt_kernel[grid](
        peer_y_base,
        peer_s_base,
        src_flat,
        scores_flat,
        out,
        my_rank_offset,
        K=K,
        d=d,
        WITH_SCORES=with_scores,
        Y_DTYPE=_TL_PTR_DTYPE[y_symm.dtype],
        S_DTYPE=_TL_PTR_DTYPE[s_reverse_symm.dtype],
    )


# ============================================================================
# Local pre-sum producer: each program walks BLOCK_SLOT output rows, accumulates its K expert slots
# in fp32 registers (masked load -> 0 contribution when not routed here, avoiding an in-loop branch), single store.
# ============================================================================

# BLOCK_SLOT batches adjacent output rows per program to amortize launch/setup cost; capped at 4 so
# the unrolled (BLOCK_SLOT x K) body stays bounded at K=10. BLOCK_SLOT=1 is the autotune escape hatch.
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
    row_has_mine_ptr,  # (W * T_local,) int8 — 1 iff row has >=1 slot routed to my_rank
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
    """Accumulates BLOCK_SLOT rows' K slots via masked loads (branch-free; avoids Triton SSA issues
    with `if` inside static_range). SKIP_EMPTY=True must stay False for RS combine — it reads every row and would mis-sum stale data otherwise."""
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

            # has_mine (precomputed in metadata) lets a dead row skip the whole K-loop (no wasted
            # scalar loads); acc stays zero there, matching store logic below (skip/zero-store/self-init).
            acc = tl.zeros([BLOCK_D], dtype=tl.float32)
            has_mine = tl.load(row_has_mine_ptr + pid_ht) != 0  # scalar bool

            if has_mine:
                for k in tl.static_range(K):
                    f = base + k
                    dst = tl.load(dst_rank_flat_ptr + f)
                    is_mine = dst == my_rank  # scalar bool

                    # s_reverse/score masked by is_mine so dead slots skip wasted fetches; row_idx=0
                    # when masked is safe since the y_symm load itself is masked below -> zero contribution.
                    if WITH_SCORES:
                        score = tl.load(scores_ag_ptr + f, mask=is_mine, other=0.0).to(tl.float32)
                    row_idx = tl.load(s_reverse_ptr + f, mask=is_mine, other=0).to(tl.int64)
                    row_offs = row_idx * d + offs_d

                    # Vector mask = valid-d AND is_mine; masked load returns 0 so the contribution is
                    # zero regardless of WITH_SCORES -- no conditional branch needed.
                    m = is_mine & d_mask
                    row = tl.load(y_symm_ptr + row_offs, mask=m, other=0.0).to(tl.float32)

                    if WITH_SCORES:
                        acc += score * row
                    else:
                        acc += row

            home_rank = (pid_ht // T_local).to(tl.int32)
            is_self_stripe = home_rank == my_rank

            if WITH_OUT_FOR_SELF and is_self_stripe:
                # Self stripe always stores unconditionally: rank_dedup_combine_triton's consumer reads
                # out[t] as its initial accumulator, so gating this on SKIP_EMPTY would silently corrupt it.
                home_t = (pid_ht - my_rank * T_local).to(tl.int64)
                out_self_offs = home_t * d + offs_d
                if EVEN_D:
                    tl.store(out_for_self_ptr + out_self_offs, acc)
                else:
                    tl.store(out_for_self_ptr + out_self_offs, acc, mask=d_mask)
            else:
                # Per-row store guard (not `return`, which would skip later j-iterations under
                # BLOCK_SLOT>1): gate = (not SKIP_EMPTY) | has_mine, i.e. unconditional or empty-row-skipped.
                if (not SKIP_EMPTY) | has_mine:
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
    """Writes partial_combine_buf[home_rank*T_local+home_t] = sum_{k: dst[f]==my_rank} w[f]*y_symm[s_reverse[f]]
    (w=1 if scores_ag is None). skip_empty=True is safe ONLY when the consumer gates reads by the same presence mask; MUST be False for dense RS (which reads every row)."""
    assert (
        partial_combine_buf.ndim == 2
    ), f"partial_combine_buf must be 2D (W*T_local, d), got shape {tuple(partial_combine_buf.shape)}."
    WT, d = partial_combine_buf.shape
    W = dist.get_world_size(group)
    my_rank = dist.get_rank(group)
    assert WT == W * T_local, f"partial_combine_buf.shape[0]={WT} != W*T_local={W*T_local} (W from group)"
    local_combine_direct(
        y_symm,
        s_reverse,
        dst_rank_flat,
        scores_ag,
        partial_combine_buf,
        K,
        T_local,
        W,
        my_rank,
        skip_empty=skip_empty,
        out_for_self=out_for_self,
    )


def local_combine_direct(
    y_symm: torch.Tensor,
    s_reverse: torch.Tensor,
    dst_rank_flat: torch.Tensor,
    scores_ag: Optional[torch.Tensor],
    partial_combine_buf: torch.Tensor,
    K: int,
    T_local: int,
    W: int,
    my_rank: int,
    skip_empty: bool = False,
    out_for_self: Optional[torch.Tensor] = None,
) -> None:
    """Dist-free core of local_combine (W/my_rank passed explicitly). Purely local (no peer access),
    so it can be driven on a single GPU for any simulated W/my_rank in dev/bench, exercising the real kernel."""
    WT, d = partial_combine_buf.shape
    assert WT == W * T_local, f"partial_combine_buf.shape[0]={WT} != W*T_local={W * T_local}"

    with_scores = scores_ag is not None
    if with_scores:
        scores_flat = scores_ag.view(-1)
    else:
        # Triton needs a concrete pointer even when unused (WITH_SCORES=False, load compiled out);
        # reuse dst_rank_flat as a dummy — its type/content is irrelevant.
        scores_flat = dst_rank_flat

    with_out_for_self = out_for_self is not None
    if with_out_for_self:
        assert out_for_self.shape == (T_local, d), (
            f"out_for_self shape {tuple(out_for_self.shape)} != " f"(T_local={T_local}, d={d})"
        )
        out_self_arg = out_for_self
    else:
        # Same dummy-pointer trick as scores_flat (unused when WITH_OUT_FOR_SELF=False).
        out_self_arg = dst_rank_flat

    WT_local = W * T_local
    # Precomputed per-row presence lets the kernel skip the whole K-loop on dead rows;
    # cheap (one TK_global int32 read) and host-sync-free.
    row_has_mine = (dst_rank_flat.view(WT_local, K) == my_rank).any(dim=1).to(torch.int8)
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
        row_has_mine,
        T_local=T_local,
        my_rank=my_rank,
        world_size=W,
        K=K,
        d=d,
        WITH_SCORES=with_scores,
        SKIP_EMPTY=skip_empty,
        WITH_OUT_FOR_SELF=with_out_for_self,
    )


_PERSIST_COMBINE_CONFIGS = [
    triton.Config({"BLOCK_D": BD, "NUM_CTA": NC, "BLOCK_SLOT": BS}, num_warps=nw, num_stages=ns)
    for BD in [1024, 2048, 4096]
    for NC in [512, 1024, 2048, 4096]
    for BS in [1, 2, 4, 8]
    for nw in [4, 8]
    for ns in [2, 3, 4]
    if BD // nw >= 64
]


@triton.autotune(
    configs=_PERSIST_COMBINE_CONFIGS,
    key=["d", "C", "world_size", "WITH_OUT_FOR_SELF"],
    prune_configs_by={"early_config_prune": _prune_block_d_vs_d},
)
@triton.heuristics(
    {
        "EVEN_D": lambda args: args["d"] % args["BLOCK_D"] == 0,
        # Grid-stride trip count. BLOCK_SLOT batches independent ~14KB gather-loads per step (cross-row
        # MLP) to hide per-row latency, since the chained-indirect load defeats normal software pipelining.
        "MAX_ITERS": lambda args: (
            args["WT_local"] + args["NUM_CTA"] * args["BLOCK_SLOT"] - 1
        )
        // (args["NUM_CTA"] * args["BLOCK_SLOT"]),
    }
)
@triton.jit
def _local_combine_tight_persistent_kernel(
    y_symm_ptr,
    s_reverse_ptr,  # (TK_global,) int32 — slot f → row in y_symm
    scores_ag_ptr,  # (TK_global,) — only read WITH_SCORES
    work_list_ptr,  # (W*T_local,) int32 — compact live rows g, [0:work_count) valid
    work_count_ptr,  # (1,) int32 device scalar — number of live rows
    mine_slot_idx_ptr,  # (W*T_local, C) int32
    mine_count_ptr,  # (W*T_local,) int32
    partial_combine_buf_ptr,  # (W*T_local, d)
    out_for_self_ptr,  # (T_local, d)
    T_local,
    WT_local,  # runtime = world_size * T_local (heuristic input for MAX_ITERS)
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    C: tl.constexpr,
    d: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_CTA: tl.constexpr,
    BLOCK_SLOT: tl.constexpr,
    MAX_ITERS: tl.constexpr,
    EVEN_D: tl.constexpr,
    WITH_SCORES: tl.constexpr,
    WITH_OUT_FOR_SELF: tl.constexpr,
):
    """Persistent producer: fixed NUM_CTA grid grid-strides the device-resident live work-list (no host
    sync, CUDA-graph safe); BLOCK_SLOT-chunked independent gathers restore the memory-level-parallelism a naive 1-row loop loses to per-row latency."""
    pid_cta = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = offs_d < d

    work_count = tl.load(work_count_ptr)

    for it in tl.range(MAX_ITERS):
        chunk = pid_cta + it * NUM_CTA
        w_base = chunk * BLOCK_SLOT
        for j in tl.static_range(BLOCK_SLOT):
            w = w_base + j
            if w < work_count:
                g = tl.load(work_list_ptr + w).to(tl.int64)
                count = tl.load(mine_count_ptr + g)
                home_rank = (g // T_local).to(tl.int32)
                do_self = WITH_OUT_FOR_SELF and (home_rank == my_rank)

                acc = tl.zeros([BLOCK_D], dtype=tl.float32)
                base = g * C
                for c in tl.range(C):
                    if c < count:
                        f = tl.load(mine_slot_idx_ptr + base + c).to(tl.int64)
                        row_idx = tl.load(s_reverse_ptr + f).to(tl.int64)
                        row = tl.load(y_symm_ptr + row_idx * d + offs_d, mask=d_mask, other=0.0).to(tl.float32)
                        if WITH_SCORES:
                            score = tl.load(scores_ag_ptr + f).to(tl.float32)
                            acc += score * row
                        else:
                            acc += row

                if do_self:
                    home_t = (g - my_rank * T_local).to(tl.int64)
                    out_self_offs = home_t * d + offs_d
                    if EVEN_D:
                        tl.store(out_for_self_ptr + out_self_offs, acc)
                    else:
                        tl.store(out_for_self_ptr + out_self_offs, acc, mask=d_mask)
                else:
                    out_offs = g * d + offs_d
                    if EVEN_D:
                        tl.store(partial_combine_buf_ptr + out_offs, acc)
                    else:
                        tl.store(partial_combine_buf_ptr + out_offs, acc, mask=d_mask)


def local_combine_tight_persistent_direct(
    y_symm: torch.Tensor,
    s_reverse: torch.Tensor,
    scores_ag: Optional[torch.Tensor],
    work_list: torch.Tensor,
    work_count: torch.Tensor,
    mine_slot_idx: torch.Tensor,
    mine_count: torch.Tensor,
    partial_combine_buf: torch.Tensor,
    C: int,
    T_local: int,
    W: int,
    my_rank: int,
    out_for_self: Optional[torch.Tensor] = None,
) -> None:
    """Host-sync-free persistent producer for the rank_dedup combine fast path (skip_empty semantics).
    Keeps s_reverse/scores indirection in-kernel — pre-gathering them was measured net-negative end-to-end."""
    WT_local = W * T_local
    d = partial_combine_buf.shape[-1]
    with_scores = scores_ag is not None
    scores_flat = scores_ag.view(-1) if with_scores else s_reverse
    with_out_for_self = out_for_self is not None
    out_self_arg = out_for_self if with_out_for_self else partial_combine_buf

    grid = lambda META: (META["NUM_CTA"], triton.cdiv(d, META["BLOCK_D"]))
    _local_combine_tight_persistent_kernel[grid](
        y_symm,
        s_reverse,
        scores_flat,
        work_list,
        work_count,
        mine_slot_idx,
        mine_count,
        partial_combine_buf,
        out_self_arg,
        T_local,
        WT_local,
        my_rank=my_rank,
        world_size=W,
        C=C,
        d=d,
        WITH_SCORES=with_scores,
        WITH_OUT_FOR_SELF=with_out_for_self,
    )


# ============================================================================
# Rank-dedup combine: exploits associativity — each rank pre-sums its K contributions per (home_rank,home_t)
# (local_combine), then home ranks sparse-gather only PRESENT peers' partials, saving bytes vs dense RS.
# ============================================================================

_RANK_DEDUP_COMBINE_CONFIGS = [
    triton.Config({"BLOCK_OUT_ROW": bt, "BLOCK_D": bd}, num_warps=nw, num_stages=ns)
    for bt in [1, 2, 4, 8]
    for bd in [128, 256, 512, 1024, 2048, 4096]  # 4096: large-d high-W gather BW (cf. RT-A2A PC, 2026-06-21)
    for nw in [2, 4, 8]
    for ns in [3, 4]
    if 1024 <= bt * bd <= 16384  # 2-D tile size in [1024, 16384] lanes (BD=4096 -> bt<=4)
    if bd // nw >= 32  # min lanes per warp
    if not (bd <= 256 and nw == 8)  # over-paralleled small tile
    if not (bd >= 1024 and nw == 2)  # under-paralleled large tile
]


def _prune_block_td_vs_d(configs, named_args, **kwargs):
    """Same as `_prune_block_d_vs_d` (BLOCK_D<=d, grid_y limit) but for the
    (BLOCK_OUT_ROW, BLOCK_D) tile namespace used by RANK_DEDUP combine configs."""
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
    key=["d", "world_size", "WITH_OUT_FOR_SELF", "SELECTIVE", "WITH_SCORES", "K", "PRECOMP_ROW"],
    prune_configs_by={"early_config_prune": _prune_block_td_vs_d},
    restore_value=["out_ptr"],
)
@triton.jit
def _rank_dedup_combine_kernel(
    peer_partial_combine_buf_tuple,  # tuple of W (W*T_local, d) tensors — peers' local-reduce buffers
    peer_present_mask_ptr,  # (W, T_local) int8 — 1 if peer q has any expert for my (t)
    out_ptr,  # (T_local, d) — final aggregated output (RMW under WITH_OUT_FOR_SELF)
    # --- selective-dedup additions (referenced only when SELECTIVE) ---
    peer_y_tuple,  # tuple of W (rows, d) peer y_symm bufs
    peer_s_reverse_tuple,  # tuple of W (TK_global,) peer s_reverse bufs
    single_k_ptr,  # (W, T_local) int8 — k-slot of the single contributor for (q,t), else -1
    scores_ptr,  # (TK_global,) all-gathered scores; read only when SELECTIVE & WITH_SCORES
    single_row_ptr,  # (W, T_local) int32 — PRECOMP_ROW: peer q's y_symm row for single (q,t)
    my_rank_offset,  # int64 scalar = my_rank * T_local * K
    T_local,
    K: tl.constexpr,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    d: tl.constexpr,
    BLOCK_OUT_ROW: tl.constexpr,
    BLOCK_D: tl.constexpr,
    WITH_OUT_FOR_SELF: tl.constexpr,
    SELECTIVE: tl.constexpr,
    WITH_SCORES: tl.constexpr,
    PRECOMP_ROW: tl.constexpr,  # decomp probe: read precomputed single_row (no s_reverse pointer-chase)
):
    """Sums present peers' partials (SELECTIVE=False), or per peer either the pre-reduced partial (multi)
    or a direct scores-weighted y_symm read (single; disjoint by construction). WITH_OUT_FOR_SELF inits acc from `out` — needs restore_value=["out_ptr"] so autotune doesn't double-accumulate."""
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
        # Skip q == my_rank under WITH_OUT_FOR_SELF — the self contribution was already loaded from
        # out_ptr above (producer wrote it directly to `out`); both predicates are constexpr so this folds away.
        if not (WITH_OUT_FOR_SELF and q == my_rank):
            present = (
                tl.load(
                    peer_present_mask_ptr + q * T_local + t_offs,
                    mask=valid_t,
                    other=0,
                )
                != 0
            )
            if SELECTIVE:
                sk = tl.load(single_k_ptr + q * T_local + t_offs, mask=valid_t, other=-1).to(tl.int32)
                is_single = present & (sk >= 0)
                is_multi = present & (sk < 0)
                # MULTI: pre-reduced partial (masked to multi tokens).
                mm = is_multi[:, None] & d_mask[None, :]
                rows = tl.load(peer_partial_combine_buf_tuple[q] + row_base, mask=mm, other=0.0).to(tl.float32)
                acc += rows
                # SINGLE: peer q's y_symm row directly x score, via peer q's s_reverse[pos] (valid
                # since the slot routed to q). int64 throughout — my_rank_offset can near int32 max.
                pos = my_rank_offset.to(tl.int64) + t_offs.to(tl.int64) * K + sk.to(tl.int64)
                if PRECOMP_ROW:
                    # Decomposition probe: reads the precomputed dense row instead of chasing peer
                    # s_reverse at the scattered pos — isolates pointer-chase cost from the y data-load cost.
                    s_row = tl.load(single_row_ptr + q * T_local + t_offs, mask=is_single, other=0).to(tl.int64)
                else:
                    s_row = tl.load(peer_s_reverse_tuple[q] + pos, mask=is_single, other=0).to(tl.int64)
                y_offs = s_row[:, None] * d + offs_d[None, :]
                sm = is_single[:, None] & d_mask[None, :]
                yv = tl.load(peer_y_tuple[q] + y_offs, mask=sm, other=0.0).to(tl.float32)
                if WITH_SCORES:
                    sc = tl.load(scores_ptr + pos, mask=is_single, other=0.0).to(tl.float32)
                    acc += yv * sc[:, None]
                else:
                    acc += yv
            else:
                m = present[:, None] & d_mask[None, :]
                rows = tl.load(peer_partial_combine_buf_tuple[q] + row_base, mask=m, other=0.0).to(tl.float32)
                acc += rows

    tl.store(out_ptr + out_offs, acc, mask=out_mask)


def rank_dedup_combine_triton(
    y_symm: torch.Tensor,
    s_reverse: torch.Tensor,
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
    mine_slot_idx: Optional[torch.Tensor] = None,
    mine_count: Optional[torch.Tensor] = None,
    combine_contrib_C: Optional[int] = None,
    combine_work_list: Optional[torch.Tensor] = None,
    combine_work_count: Optional[torch.Tensor] = None,
    combine_single_k: Optional[torch.Tensor] = None,
    y_peer_bufs=None,
    s_reverse_peer_bufs=None,
    single_row: Optional[torch.Tensor] = None,
) -> None:
    """Rank-dedup combine: producer pre-reduces multi-contributor (q,t) pairs (single-contributor reads
    y_symm directly, its pre-reduce being a no-op), consumer sparse-gathers only present peers. NVLink bytes <= plain RS combine, strictly < A2A when K>1."""
    # Step 1 (local reduce): skip_empty=True skips the HBM store on non-self rows with no contribution;
    # out_for_self=out redirects the self stripe so the consumer can elide its q == my_rank self-read.
    W = dist.get_world_size(group)
    rk = dist.get_rank(group) if my_rank is None else my_rank
    C = combine_contrib_C if combine_contrib_C is not None else mine_slot_idx.shape[1]
    local_combine_tight_persistent_direct(
        y_symm,
        s_reverse,
        scores,
        combine_work_list,
        combine_work_count,
        mine_slot_idx,
        mine_count,
        partial_combine_buf,
        C,
        T_local,
        W,
        rk,
        out_for_self=out,
    )

    # Step 2 (sparse gather): init_acc_from_out=True mirrors the producer's out_for_self=out redirect
    # (loads initial acc from `out`, skips the q==my_rank self-read); also resolves peer y_symm/s_reverse tuples.
    if y_peer_bufs is None or s_reverse_peer_bufs is None:
        W_ = dist.get_world_size(group)
        hdl_y = rendezvous(y_symm, group)
        hdl_s = rendezvous(s_reverse, group)
        if y_peer_bufs is None:
            y_peer_bufs = tuple(hdl_y.get_buffer(r, tuple(y_symm.shape), y_symm.dtype) for r in range(W_))
        if s_reverse_peer_bufs is None:
            s_reverse_peer_bufs = tuple(
                hdl_s.get_buffer(r, tuple(s_reverse.shape), s_reverse.dtype) for r in range(W_)
            )

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
        y_peer_bufs=y_peer_bufs,
        s_reverse_peer_bufs=s_reverse_peer_bufs,
        combine_single_k=combine_single_k,
        scores=scores,
        K=K,
        single_row=single_row,
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
    # Single contributors read directly from peer y_symm via a precomputed source row (single_row) —
    # this IS the rank-dedup gather, not an optional mode.
    y_peer_bufs=None,  # tuple of W (rows, d) peer y_symm bufs
    s_reverse_peer_bufs=None,  # tuple of W (TK_global,) peer s_reverse bufs (placeholder; precomp row used)
    combine_single_k: Optional[torch.Tensor] = None,  # (W, T_local) int8
    scores: Optional[torch.Tensor] = None,  # (TK_global,) all-gathered scores or None (dx)
    K: int = 1,
    # precomputed single_row (peer y_symm row per single (q,t)) — dense load that
    # replaces the scattered s_reverse pointer-chase. Required.
    single_row: Optional[torch.Tensor] = None,
) -> None:
    """Communication-only step of RANK_DEDUP combine: sparse cross-rank gather over peers'
    pre-populated partial_combine_buf. Issues partial_combine_hdl.barrier() internally before reading, so peers must have written it first."""
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

    # Single-contributor peers read y_symm directly via the precomputed row (pre-reduce is a no-op
    # for them); multi-contributor peers read the pre-reduced partial. single_row is required.
    assert single_row is not None, "rank-dedup combine requires precomputed single_row"
    with_scores = scores is not None
    scores_arg = scores.view(-1) if scores is not None else peer_present_mask
    my_rank_offset = my_rank * T_local * K

    grid = lambda META: (
        triton.cdiv(T_local, META["BLOCK_OUT_ROW"]),
        triton.cdiv(d, META["BLOCK_D"]),
    )
    _rank_dedup_combine_kernel[grid](
        partial_combine_peer_bufs,
        peer_present_mask,
        out,
        y_peer_bufs,
        s_reverse_peer_bufs,
        combine_single_k,
        scores_arg,
        single_row,
        my_rank_offset,
        T_local,
        K=K,
        my_rank=my_rank,
        world_size=W,
        d=d,
        WITH_OUT_FOR_SELF=init_acc_from_out,
        SELECTIVE=True,
        WITH_SCORES=with_scores,
        PRECOMP_ROW=True,
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
    """Full RS combine: local_combine (producer, no barrier needed) + barrier + reduce_scatter_triton
    (dense cross-rank reduce). Barrier ensures peers' partial_combine_buf writes are visible before the reduce."""
    # Producer is purely local (no peer reads) — same-stream ordering with the prior y_symm write suffices.
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


# ============================================================================
# Hierarchical combine (mirror of dispatch): a REDUCTION not an expand (no RDMA float atomics). Local-node
# peers NVLink-reduce; remote nodes' gateway reduces then GIN-puts one dense stripe; stripe index = closed-form n<node? n : n-1 (no payload metadata needed).
# ============================================================================


@triton.jit
def _hier_combine_gateway_reduce_kernel(
    peer_partial_tuple,        # tuple of node_size (W*T_local, d) node-local peers' partial_combine_buf
    present_all_ptr,           # (W, W, T_local) int8: present[R][q][t]
    send_buf_ptr,              # ((num_nodes-1)*T_local, d) — gateway's reduced node_partial staging
    T_local,
    node_size: tl.constexpr,
    W: tl.constexpr,
    my_rank: tl.constexpr,
    d: tl.constexpr,
    BLOCK_OUT_ROW: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Gateway (C=my_rank): for each remote node n, reduces node-local peers' partial_combine_buf for the
    origin R=rank_of(n,local_of(C)) into send_buf[s_send(n)] (n==node_of(C) stores nothing; handled by origin reduce)."""
    n = tl.program_id(0)
    pid_t = tl.program_id(1).to(tl.int64)
    pid_d = tl.program_id(2).to(tl.int64)

    g = my_rank // node_size           # constexpr: my node
    m = my_rank % node_size            # constexpr: my local index
    R = n * node_size + m              # origin rank served for node n
    s_send = tl.where(n < g, n, n - 1)

    t_offs = pid_t * BLOCK_OUT_ROW + tl.arange(0, BLOCK_OUT_ROW)
    valid_t = t_offs < T_local
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = offs_d < d

    acc = tl.zeros([BLOCK_OUT_ROW, BLOCK_D], dtype=tl.float32)
    row_base = (R * T_local + t_offs)[:, None] * d + offs_d[None, :]

    for q_local in tl.static_range(node_size):
        q = g * node_size + q_local
        present = tl.load(present_all_ptr + R * (W * T_local) + q * T_local + t_offs,
                          mask=valid_t, other=0) != 0
        mm = present[:, None] & d_mask[None, :]
        rows = tl.load(peer_partial_tuple[q_local] + row_base, mask=mm, other=0.0).to(tl.float32)
        acc += rows

    out_offs = (s_send * T_local + t_offs)[:, None] * d + offs_d[None, :]
    store_mask = (n != g) & (valid_t[:, None] & d_mask[None, :])
    tl.store(send_buf_ptr + out_offs, acc, mask=store_mask)


@triton.jit
def _hier_combine_origin_reduce_kernel(
    peer_partial_tuple,        # tuple of node_size node-local peers' partial_combine_buf
    present_all_ptr,           # (W, W, T_local) int8: present[R][q][t]
    contrib_node_ptr,          # (W, num_nodes) int8: contrib_node_mask[R][n]
    recv_buf_ptr,              # ((num_nodes-1)*T_local, d) — GIN-delivered remote-node partials
    out_ptr,                   # (T_local, d) final output (model dtype)
    T_local,
    node_size: tl.constexpr,
    num_nodes: tl.constexpr,
    W: tl.constexpr,
    my_rank: tl.constexpr,
    d: tl.constexpr,
    BLOCK_OUT_ROW: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Origin final reduce: out[t] = same-node present peers' partials (NVLink) + present remote nodes'
    recv_buf stripes (GIN-delivered). Deterministic order: local peers by rank, then remote nodes by id."""
    pid_t = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)
    g = my_rank // node_size
    R = my_rank

    t_offs = pid_t * BLOCK_OUT_ROW + tl.arange(0, BLOCK_OUT_ROW)
    valid_t = t_offs < T_local
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = offs_d < d

    acc = tl.zeros([BLOCK_OUT_ROW, BLOCK_D], dtype=tl.float32)
    row_base = (R * T_local + t_offs)[:, None] * d + offs_d[None, :]

    # local-node peers (NVLink), increasing rank
    for q_local in tl.static_range(node_size):
        q = g * node_size + q_local
        present = tl.load(present_all_ptr + R * (W * T_local) + q * T_local + t_offs,
                          mask=valid_t, other=0) != 0
        mm = present[:, None] & d_mask[None, :]
        rows = tl.load(peer_partial_tuple[q_local] + row_base, mask=mm, other=0.0).to(tl.float32)
        acc += rows

    # remote-node stripes (combine_recv_buf, GIN-delivered), increasing node id
    for n in tl.static_range(num_nodes):
        if n != g:
            contrib = tl.load(contrib_node_ptr + R * num_nodes + n) != 0
            s_recv = n if n < g else n - 1            # constexpr (n, g both constexpr)
            recv_offs = (s_recv * T_local + t_offs)[:, None] * d + offs_d[None, :]
            cm = contrib & (valid_t[:, None] & d_mask[None, :])
            rv = tl.load(recv_buf_ptr + recv_offs, mask=cm, other=0.0).to(tl.float32)
            acc += rv

    out_offs = t_offs[:, None] * d + offs_d[None, :]
    tl.store(out_ptr + out_offs, acc, mask=valid_t[:, None] & d_mask[None, :])


def _hier_combine_blocks(T_local, d):
    return min(triton.next_power_of_2(T_local), 16), min(triton.next_power_of_2(d), 1024)


def hier_combine_gateway_reduce_triton(peer_partial_bufs, present_all, send_buf, *,
                                       T_local, node_size, num_nodes, W, my_rank, d):
    """Gateway reduce (reference step 2): reduces node-local peers' partial_combine_buf into send_buf
    (one dense (T_local,d) stripe per remote origin-node). peer_partial_bufs must be ordered by local index."""
    assert len(peer_partial_bufs) == node_size, f"need {node_size} node-local peer bufs, got {len(peer_partial_bufs)}"
    BLOCK_OUT_ROW, BLOCK_D = _hier_combine_blocks(T_local, d)
    grid = (num_nodes, triton.cdiv(T_local, BLOCK_OUT_ROW), triton.cdiv(d, BLOCK_D))
    _hier_combine_gateway_reduce_kernel[grid](
        peer_partial_bufs, present_all, send_buf, T_local,
        node_size=node_size, W=W, my_rank=my_rank, d=d,
        BLOCK_OUT_ROW=BLOCK_OUT_ROW, BLOCK_D=BLOCK_D,
    )


def hier_combine_origin_reduce_triton(peer_partial_bufs, present_all, contrib_node_mask, recv_buf, out, *,
                                      T_local, node_size, num_nodes, W, my_rank, d):
    """Origin final reduce (step 5 of the reference): same-node peers (NVLink, present-gated) + present remote
    stripes (``recv_buf``) → ``out`` (T_local, d). ``peer_partial_bufs`` as in the gateway reduce."""
    assert len(peer_partial_bufs) == node_size, f"need {node_size} node-local peer bufs, got {len(peer_partial_bufs)}"
    BLOCK_OUT_ROW, BLOCK_D = _hier_combine_blocks(T_local, d)
    grid = (triton.cdiv(T_local, BLOCK_OUT_ROW), triton.cdiv(d, BLOCK_D))
    _hier_combine_origin_reduce_kernel[grid](
        peer_partial_bufs, present_all, contrib_node_mask, recv_buf, out, T_local,
        node_size=node_size, num_nodes=num_nodes, W=W, my_rank=my_rank, d=d,
        BLOCK_OUT_ROW=BLOCK_OUT_ROW, BLOCK_D=BLOCK_D,
    )
