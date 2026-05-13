# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
# Expert Parallelism (EP) for SonicMoE.
#
# Dispatch modes (``DispatchMode``):
#   AG_TRITON       — all-gather x via Triton kernel → gather GEMM (with A_idx).
#   A2A_TRITON      — pull-based dispatch into expert-sorted layout → non-gather GEMM.
#   RANK_DEDUP_DISPATCH_TRITON — pull-based dispatch with same-token / same-dest dedup.
#                     Strictly Pareto-dominates both AG and A2A on inbound
#                     NVLink rows under the indicator-wise bound
#                     Dedup_d ≤ min(AG_d, A2A_d) for all routings δ. Same
#                     expert-sorted x_compute layout as A2A → non-gather GEMM.
#
# Combine modes (``CombineMode``):
#   A2A_TRITON              — fused NVLink-read combine kernel.
#   RS_COMBINE_TRITON   — local producer + symm-mem reduce-scatter.
#   RANK_DEDUP_COMBINE_TRITON     — same local producer (local_combine) as RS
#                             but cross-rank step is a per-token sparse
#                             gather guided by ``peer_present_mask``.
#                             Strictly ≤ RS bytes, matches dispatch dedup
#                             at (W-1) · T_local · (1 - (1-1/W)^K) · H in
#                             expectation under uniform routing. Sparse
#                             gather is *not* row-level dedup — partial
#                             sums are distinct; the win is from skipping
#                             zero-contribution peers.
#
# Mode selection: ``NetworkProfiler`` benchmarks all three dispatch modes
# and the three combine modes on the local hardware; pass its returned
# ``RuntimeEPConfig`` via ``ep_config=`` to use the measured winner.
#
# Naming: T_local tokens/rank, K experts/token, W=ep world size,
# TK_local=T_local*K, TK_global=W*TK_local, E_local=E//W.
# ********************************************************************************

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from quack.gemm_interface import gemm, gemm_gated

from ..distributed_utils import (
    CombineMode,
    DispatchMode,
    NetworkProfiler,
    RuntimeEPConfig,
    SymmMemManager,
    _EPWorkspace,
    _is_a2a_combine_mode,
    _is_a2a_dispatch_mode,
    _is_ag_dispatch_mode,
    _is_rank_dedup_combine_mode,
    _is_rank_dedup_dispatch_mode,
    _is_rs_combine_mode,
)
from ..enums import ActivationType, is_glu
from . import TC_Softmax_Topk_Router_Function
from .backward import _down_projection_backward_act, _up_projection_backward_act
from .distributed import (
    a2a_combine_triton,
    a2a_dispatch_triton,
    all_gather_copy_engine_async,
    all_gather_triton,
    build_rank_dedup_a_idx,
    compute_dispatch_metadata,
    rank_dedup_combine_triton,
    rank_dedup_dispatch_triton,
    rs_combine_triton,
)
from .metadata import general_routing_router_metadata_triton


__all__ = [
    "CombineMode",
    "DispatchMode",
    "EP_Router_Replicated_Across_Ranks",
    "NetworkProfiler",
    "RuntimeEPConfig",
    "SymmMemManager",
    "moe_ep_TC_softmax_topk_forward",
    "moe_ep_general_routing_forward",
]


class EP_Router_Replicated_Across_Ranks(torch.autograd.Function):
    """``F.linear(x, router_w)`` with EP-aware drouter_w all-reduce.

    In EP, each rank holds ``T_local = T / W`` tokens of the global
    batch and every rank has a replica of ``router_w``. 
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, router_w: torch.Tensor, ep_group):
        ctx.save_for_backward(x, router_w)
        ctx.ep_group = ep_group
        return F.linear(x, router_w)

    @staticmethod
    def backward(ctx, dlogits: torch.Tensor):
        x, router_w = ctx.saved_tensors
        ep_group = ctx.ep_group
        dx = dlogits @ router_w
        drouter_w_local = dlogits.transpose(-2, -1) @ x
        dist.all_reduce(drouter_w_local, op=dist.ReduceOp.SUM, group=ep_group)
        return dx, drouter_w_local, None


def _normalize_activation(activation_type) -> ActivationType:
    if isinstance(activation_type, str):
        return ActivationType[activation_type.upper()]
    return activation_type


def _do_dispatch(
    src_symm: torch.Tensor,
    out_buf: torch.Tensor,
    mode: DispatchMode,
    *,
    dst_rank_flat: Optional[torch.Tensor],
    recv_pos: Optional[torch.Tensor],
    K: int,
    group: dist.ProcessGroup,
    H: int,
    src_peer_bufs: Tuple[torch.Tensor, ...],
    my_rank: int,
    # RANK_DEDUP_DISPATCH_TRITON-only:
    pair_present_mask: Optional[torch.Tensor] = None,
    rank_dedup_recv_pos: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if _is_ag_dispatch_mode(mode):
        return all_gather_triton(src_symm, group, out=out_buf, peer_bufs=src_peer_bufs)
    elif _is_rank_dedup_dispatch_mode(mode):
        rank_dedup_dispatch_triton(
            x_symm=src_symm, dst_rank_flat=dst_rank_flat, pair_present_mask=pair_present_mask,
            rank_dedup_recv_pos=rank_dedup_recv_pos, recv_packed=out_buf, K=K, group=group,
            peer_bufs=src_peer_bufs, my_rank=my_rank,
        )
        return out_buf.view(-1, H)
    elif _is_a2a_dispatch_mode(mode):
        a2a_dispatch_triton(
            x_symm=src_symm, dst_rank_flat=dst_rank_flat, recv_pos=recv_pos, recv=out_buf, K=K, group=group,
            peer_bufs=src_peer_bufs, my_rank=my_rank,
        )
        # out_buf may be (W, TK_local, H) symm-mem or flat
        # (rows, H); collapse to (rows, H) regardless.
        return out_buf.view(-1, H)
    else:
        raise NotImplementedError()


def _all_gather_topk_scores(
    topk_scores_local: torch.Tensor,
    group: dist.ProcessGroup,
    W: int,
    T_local: int,
    K: int,
) -> torch.Tensor:
    scores_global = torch.empty(
        W * T_local * K,
        dtype=topk_scores_local.dtype,
        device=topk_scores_local.device,
    )
    dist.all_gather_into_tensor(
        scores_global,
        topk_scores_local.view(-1).contiguous(),
        group=group,
    )
    return scores_global


def _do_combine(
    ep_ws: "_EPWorkspace",
    *,
    my_dst_rank: torch.Tensor,
    dst_rank_flat: Optional[torch.Tensor],
    topk_scores_local: Optional[torch.Tensor],
    scores_global: Optional[torch.Tensor],
    K: int,
    T_local: int,
    H: int,
    out_dtype: torch.dtype,
    agg_mode: CombineMode,
    # RANK_DEDUP_COMBINE_TRITON-only:
    peer_present_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    out = torch.empty(T_local, H, dtype=out_dtype, device=ep_ws.y_symm.device)
    if _is_a2a_combine_mode(agg_mode):
        ep_ws.o_hdl.barrier()
        a2a_combine_triton(
            ep_ws.y_symm, ep_ws.s_rev_symm, my_dst_rank, topk_scores_local, out, K=K, group=ep_ws.ep_group,
            y_peer_bufs=ep_ws.y_peer_bufs, s_peer_bufs=ep_ws.s_rev_peer_bufs, my_rank=ep_ws.my_rank,
        )
        return out
    elif _is_rank_dedup_combine_mode(agg_mode):
        ep_ws._ensure_partial_combine_buf()
        rank_dedup_combine_triton(
            ep_ws.y_symm, ep_ws.s_rev_symm, dst_rank_flat, scores_global, peer_present_mask,
            ep_ws.partial_combine_buf, out, K=K, T_local=T_local, group=ep_ws.ep_group,
            partial_combine_hdl=ep_ws.partial_combine_hdl,
            partial_combine_peer_bufs=ep_ws.partial_combine_peer_bufs, my_rank=ep_ws.my_rank,
        )
        return out
    elif _is_rs_combine_mode(agg_mode):
        ep_ws._ensure_partial_combine_buf()
        rs_combine_triton(
            ep_ws.y_symm, ep_ws.s_rev_symm, dst_rank_flat, scores_global, ep_ws.partial_combine_buf, out, K,
            T_local, group=ep_ws.ep_group, partial_combine_hdl=ep_ws.partial_combine_hdl,
            partial_combine_peer_bufs=ep_ws.partial_combine_peer_bufs, my_rank=ep_ws.my_rank,
        )
        return out
    else:
        raise NotImplementedError()


class _MoeEPFunction(torch.autograd.Function):
    """Merged EP forward + backward for one MoE layer.

    Forward: dispatch x → up-proj GEMM (gemm_gated) → down-proj GEMM →
    NVLink-gather combine → y_local.

    Backward: dispatch do (configured mode) → start AG-CE x redispatch
    on a side stream if ``redispatch_x_in_backward`` else reuse saved
    x_compute → all-gather scores → gemm_dgated (→ dh, ds, a_prime)
    → dW2 → reduce-scatter ds → up-proj-backward (writes dx_expanded
    into y_symm) → cross-rank gather to dx_local → join CE → dW1.

    Caches across the autograd boundary:
      - h: (total_m, H_act), where ``total_m = max_rows_per_rank_runtime``
        (= synced ``valid_rows`` under ``CPU_sync_on_runtime``,
        ``MAX_ROWS_PER_RANK_STATIC`` otherwise). gemm_gated / gemm_dgated
        strict-check ``preact.shape[0] == total_m``.
      - x_compute: replaced with x_local (T_local, H) under
        ``redispatch_x_in_backward``; otherwise (W·T_local, H) for AG /
        RANK_DEDUP and (total_m, H) for A2A.
    """

    @staticmethod
    def forward(
        ctx,
        x_local: torch.Tensor,
        w1: torch.Tensor,
        b1: Optional[torch.Tensor],
        w2: torch.Tensor,
        b2: Optional[torch.Tensor],
        topk_scores_local: torch.Tensor,
        cfg: RuntimeEPConfig,
        meta: dict,
        activation_type: ActivationType,
        is_inference_mode_enabled: bool,
        concat_layout: bool,
        redispatch_x_in_backward: bool,
        CPU_sync_on_runtime: bool,
        ep_ws,
    ) -> torch.Tensor:
        # x_local: (T_local, H)
        H = ep_ws.d
        device, x_dtype = x_local.device, x_local.dtype

        # Unpack the layer-static + workload-level cfg.
        I = cfg.I
        is_glu_act = cfg.is_glu_act
        E_local = cfg.E_local
        MAX_ROWS_PER_RANK_STATIC = cfg.MAX_ROWS_PER_RANK_STATIC
        mode = cfg.mode
        K = cfg.K
        T_local = ep_ws.T_local

        # Unpack the metadata bundle.
        expert_frequency_offset = meta["expert_frequency_offset"]
        x_gather_idx = meta["x_gather_idx"]
        my_dst_rank = meta["my_dst_rank"]
        recv_pos = meta["recv_pos"]
        dst_rank_flat = meta["dst_rank_flat"]
        pair_present_mask = meta.get("pair_present_mask")
        rank_dedup_recv_pos = meta.get("rank_dedup_recv_pos")
        peer_present_mask = meta.get("peer_present_mask")
        a_idx_rank_dedup = meta.get("a_idx_rank_dedup")

        H_act = 2 * I if is_glu_act else I

        if is_inference_mode_enabled:
            redispatch_x_in_backward = False
            CPU_sync_on_runtime = False

        # Row count every per-call allocation in this forward uses for
        # GEMM-output sizing (h, a, the A2A dispatch recv buffer, the
        # down-proj y_symm slice). Under ``CPU_sync_on_runtime`` it's
        # the synced populated count from
        # ``expert_frequency_offset[E_local]`` (one D2H ``.item()``);
        # otherwise the structural ceiling. 
        max_rows_per_rank_runtime = MAX_ROWS_PER_RANK_STATIC
        if CPU_sync_on_runtime:
            max_rows_per_rank_runtime = expert_frequency_offset[E_local].item()

        # ====================================================================
        # 1. Dispatch x → x_compute
        # ====================================================================
        if is_inference_mode_enabled or redispatch_x_in_backward:
            # x_compute isn't saved-for-backward on these paths
            # (inference saves nothing; redispatch saves x_local), so
            # reuse the workspace recv buffer where available. 
            if _is_ag_dispatch_mode(mode):
                ws_buf = ep_ws.ag_compute
            elif _is_a2a_dispatch_mode(mode):
                ws_buf = ep_ws.a2a_recv
            else:
                ws_buf = torch.empty(ep_ws.world_size * T_local, H, dtype=x_dtype, device=device)
            x_compute = _do_dispatch(
                ep_ws.x_symm, ws_buf, mode, dst_rank_flat=dst_rank_flat, recv_pos=recv_pos, K=K,
                group=ep_ws.ep_group, H=H, src_peer_bufs=ep_ws.x_peer_bufs, my_rank=ep_ws.my_rank,
                pair_present_mask=pair_present_mask, rank_dedup_recv_pos=rank_dedup_recv_pos,
            )
            # A2A's recv is consumed by gemm_gated WITHOUT an A_idx,
            # so the kernel reads ``total_m = A.shape[0]`` directly —
            # trim x_compute to the runtime row count for the strict-
            # check ``A.shape[0] == h.shape[0]`` to pass. AG and
            # RANK_DEDUP both feed the GEMM via A_idx (so total_m comes
            # from A_idx.shape[0], independent of x_compute's shape) and
            # need no trim.
            if _is_a2a_dispatch_mode(mode):
                x_compute = x_compute[:max_rows_per_rank_runtime]
        else:
            # Per-call recv buffer for the cache path. The fresh
            # ``torch.empty`` decouples the saved x_compute from any
            # workspace, so backward step 1's dout dispatch can't
            # clobber it. Mode-aware sizing:
            #   AG_TRITON   — per-token recv at W·T_local rows.
            #   RANK_DEDUP  — packed-by-source recv at W·T_local rows
            #                 (each home rank contributes ≤ T_local
            #                 distinct tokens; the GEMM gathers via
            #                 A_idx and ignores any tail).
            #   A2A_TRITON  — expert-grouped recv at
            #                 ``max_rows_per_rank_runtime`` (synced
            #                 ``valid_rows`` under
            #                 ``CPU_sync_on_runtime``, otherwise the
            #                 structural ceiling
            #                 T_local·W·min(K, E_local)).
            if _is_ag_dispatch_mode(mode) or _is_rank_dedup_dispatch_mode(mode):
                fresh = torch.empty(ep_ws.world_size * T_local, H, dtype=x_dtype, device=device)
            else:
                fresh = torch.empty(max_rows_per_rank_runtime, H, dtype=x_dtype, device=device)
            x_compute = _do_dispatch(
                ep_ws.x_symm, fresh, mode, dst_rank_flat=dst_rank_flat, recv_pos=recv_pos, K=K,
                group=ep_ws.ep_group, H=H, src_peer_bufs=ep_ws.x_peer_bufs, my_rank=ep_ws.my_rank,
                pair_present_mask=pair_present_mask, rank_dedup_recv_pos=rank_dedup_recv_pos,
            )

        # ====================================================================
        # 2. Up-proj GEMM with fused gated activation: x_compute @ w1 → (h, a)
        # ====================================================================
        a = torch.empty(max_rows_per_rank_runtime, I, dtype=x_dtype, device=device)
        h = torch.empty(max_rows_per_rank_runtime, H_act, dtype=x_dtype, device=device)

        # Three-way A_idx selection for the up-proj GEMM:
        #   A2A_TRITON      — recv already expert-grouped, A_idx=None.
        #   RANK_DEDUP_DISPATCH_TRITON — recv packed-by-source, A_idx maps each
        #                     expert-grouped row → its packed row.
        #   AG_TRITON       — recv is per-token; A_idx=x_gather_idx maps
        #                     expert-grouped row → token row.
        if _is_a2a_dispatch_mode(mode):
            a_idx_for_up = None
        elif _is_rank_dedup_dispatch_mode(mode):
            a_idx_for_up = a_idx_rank_dedup[:max_rows_per_rank_runtime]
        else:
            a_idx_for_up = x_gather_idx[:max_rows_per_rank_runtime]

        assert activation_type.value in (
            "swiglu",
            "geglu",
        ), f"gemm_gated only supports glu activations, got {activation_type.value}"
        gemm_gated(
            x_compute,
            w1.permute(2, 1, 0),
            activation=activation_type.value,
            cu_seqlens_m=expert_frequency_offset,
            A_idx=a_idx_for_up,
            preact_out=h,
            postact_out=a,
            store_preact=(not is_inference_mode_enabled),
            bias=b1,
            concat_layout=((("B", "bias") if b1 is not None else ("B",)) if concat_layout else None),
        )

        # ====================================================================
        # 3. Down-proj GEMM: a @ w2 → y_symm
        # ====================================================================
        # ep_ws.y_symm is (MAX_ROWS_PER_RANK_STATIC, H) symm-mem; the
        # kernel writes [0, offset[E_local]) ≤ MAX_ROWS_PER_RANK_STATIC
        # and peers gather from those positions via s_rev_symm. 
        gemm(
            a,
            w2,
            out=ep_ws.y_symm[:max_rows_per_rank_runtime],
            cu_seqlens_m=expert_frequency_offset,
            bias=b2,
            dynamic_scheduler=False,
        )
        del a

        # ====================================================================
        # 4. NVLink combine → o_local
        # ====================================================================
        # Mode-dispatched by ``_do_combine``; barrier placement is internal to each branch there. 
        scores_global: Optional[torch.Tensor] = None
        if cfg.agg_mode in (CombineMode.RS_COMBINE_TRITON, CombineMode.RANK_DEDUP_COMBINE_TRITON):
            scores_global = _all_gather_topk_scores(topk_scores_local, ep_ws.ep_group, ep_ws.world_size, T_local, K)
        o_local = _do_combine(
            ep_ws, my_dst_rank=my_dst_rank, dst_rank_flat=dst_rank_flat, topk_scores_local=topk_scores_local,
            scores_global=scores_global, K=K, T_local=T_local, H=H, out_dtype=x_dtype, agg_mode=cfg.agg_mode,
            peer_present_mask=peer_present_mask,
        )

        # ====================================================================
        # 5. Save state for backward (training only)
        # ====================================================================
        if not is_inference_mode_enabled:
            # h, a are alloc'd fresh at the runtime row count (step 2);
            # cache-path x_compute is alloc'd fresh in step 1; redispatch
            # path saves x_local instead. 
            ctx.save_for_backward(
                x_local if redispatch_x_in_backward else x_compute,
                w1,
                b1,
                w2,
                b2,
                h,
                topk_scores_local,
            )
            ctx.cfg = cfg
            ctx.meta = meta
            ctx.activation_type = activation_type
            ctx.concat_layout = concat_layout
            ctx.redispatch_x_in_backward = redispatch_x_in_backward
            ctx.CPU_sync_on_runtime = CPU_sync_on_runtime
            ctx.max_rows_per_rank_runtime = max_rows_per_rank_runtime
            ctx.ep_ws = ep_ws
            # Cached AG of topk_scores from the RS- or RANK_DEDUP-combine
            # forward path; backward step 3 reuses this when present
            # (avoids a duplicate AG).
            ctx.scores_global = scores_global
            ctx.set_materialize_grads(False)

        ep_ws.o_hdl.barrier()

        return o_local

    @staticmethod
    def backward(ctx, dout_local: torch.Tensor):
        # 14 forward inputs → 14 grads (only the first 6 are tensor inputs).
        (
            x_local_or_compute, w1, b1, w2, b2, h, topk_scores_local,
        ) = ctx.saved_tensors
        cfg: RuntimeEPConfig = ctx.cfg
        meta = ctx.meta
        is_glu_act = cfg.is_glu_act
        K = cfg.K
        W = cfg.W
        T_local = ctx.ep_ws.T_local
        TK_global = W * T_local * K
        activation_type = ctx.activation_type
        concat_layout = ctx.concat_layout
        redispatch = ctx.redispatch_x_in_backward
        cpu_synced = ctx.CPU_sync_on_runtime
        # Metadata tensors (no autograd flow).
        expert_frequency_offset = meta["expert_frequency_offset"]
        x_gather_idx = meta["x_gather_idx"]
        s_scatter_idx = meta["s_scatter_idx"]
        my_dst_rank = meta["my_dst_rank"]
        recv_pos = meta["recv_pos"]
        dst_rank_flat = meta["dst_rank_flat"]
        x_gather_idx_ag_for_dw1 = meta["x_gather_idx_ag_for_dw1"]
        pair_present_mask = meta.get("pair_present_mask")
        rank_dedup_recv_pos = meta.get("rank_dedup_recv_pos")
        peer_present_mask = meta.get("peer_present_mask")
        a_idx_rank_dedup = meta.get("a_idx_rank_dedup")

        max_rows_per_rank_runtime = ctx.max_rows_per_rank_runtime
        ep_ws = ctx.ep_ws
        mode = ep_ws.mode

        H = ep_ws.d
        I = w2.shape[1]
        device, dtype = dout_local.device, dout_local.dtype

        # ====================================================================
        # 1. Dispatch do_local → dout_dispatched
        # ====================================================================
        if redispatch:
            ep_ws._ensure_do_symm()
            do_buf, do_hdl, do_peer_bufs = ep_ws.do_symm, ep_ws.do_hdl, ep_ws.do_peer_bufs
        else:
            do_buf, do_hdl, do_peer_bufs = ep_ws.x_symm, ep_ws.x_hdl, ep_ws.x_peer_bufs
        do_buf.copy_(dout_local)
        do_hdl.barrier()

        if _is_rank_dedup_dispatch_mode(mode):
            do_recv_buf = torch.empty(ep_ws.world_size * T_local, H, dtype=dtype, device=device)
        elif _is_ag_dispatch_mode(mode):
            do_recv_buf = ep_ws.ag_compute
        else:
            do_recv_buf = ep_ws.a2a_recv
        dout_dispatched = _do_dispatch(
            do_buf, do_recv_buf, mode, dst_rank_flat=dst_rank_flat, recv_pos=recv_pos, K=K,
            group=ep_ws.ep_group, H=H, src_peer_bufs=do_peer_bufs, my_rank=ep_ws.my_rank,
            pair_present_mask=pair_present_mask, rank_dedup_recv_pos=rank_dedup_recv_pos,
        )

        # ====================================================================
        # 2. Recover x_compute at the shape dW1 will consume
        # ====================================================================
        # Two mutually-exclusive paths, joined right before dW1:
        #   (a) redispatch=True: AG-CE all-gather of saved x_local on a Copy-Engine stream 
        #   (b) cache otherwise: saved x_compute is already dW1-ready
        # Started here (right after do dispatch) so (a) overlaps with the rest of the backward.
        ce_handle = None
        if redispatch:
            x_local = x_local_or_compute  # (T_local, H)
            ce_handle = all_gather_copy_engine_async(
                x_local, peer_bufs=ep_ws.x_peer_bufs, my_rank=ep_ws.my_rank, out=ep_ws._ag_redispatch_buf,
            )
            x_compute = ce_handle.out
        else:
            x_compute = x_local_or_compute

        # ====================================================================
        # 3. All-gather topk scores (or reuse the forward's cached AG)
        # ====================================================================
        # Forward step 4's RS- or RANK_DEDUP-combine path already AGs
        # scores; if so it cached the result on ``ctx.scores_global``.
        # Reuse to avoid a redundant NCCL collective. A2A_TRITON-combine
        # forward leaves ``ctx.scores_global=None``; we fall back to a fresh AG.
        if ctx.scores_global is not None:
            topk_scores_global = ctx.scores_global
        else:
            topk_scores_global = torch.empty(
                ep_ws.world_size * T_local * K,
                dtype=topk_scores_local.dtype,
                device=topk_scores_local.device,
            )
            dist.all_gather_into_tensor(
                topk_scores_global,
                topk_scores_local.view(-1),
                group=ep_ws.ep_group,
            )

        # ====================================================================
        # 4. Down-proj backward act (gemm_dgated): dh, ds, a_prime
        # ====================================================================
        dh = torch.empty_like(h)
        ds = torch.zeros(TK_global, dtype=topk_scores_global.dtype, device=device)
        a_prime = torch.empty(max_rows_per_rank_runtime, I, dtype=h.dtype, device=device)
        db2 = None if b2 is None else torch.empty_like(b2)

        # Three-way A_idx for the down-proj backward / dW2 GEMM:
        #   A2A_TRITON      — expert-grouped; A_idx=None, slice dout to total_m.
        #   RANK_DEDUP_DISPATCH_TRITON — packed-by-source; A_idx maps expert-grouped row
        #                     → packed row (same a_idx_rank_dedup the up-proj used
        #                     in forward; identical metadata for both passes
        #                     since dst_rank_flat / s_reverse_local are layer-
        #                     and-pass invariant). No slice — all packed rows
        #                     are valid.
        #   AG_TRITON       — A_idx=x_gather_idx[:total_m], no slice.
        if _is_a2a_dispatch_mode(mode):
            a_idx_for_dout = None
            dout_for_kernel = dout_dispatched[:max_rows_per_rank_runtime]
        elif _is_rank_dedup_dispatch_mode(mode):
            a_idx_for_dout = a_idx_rank_dedup[:max_rows_per_rank_runtime]
            dout_for_kernel = dout_dispatched
        else:
            a_idx_for_dout = x_gather_idx[:max_rows_per_rank_runtime]
            dout_for_kernel = dout_dispatched
        s_scatter_idx_local = s_scatter_idx[:max_rows_per_rank_runtime]
        # ds-scatter sentinel mask: only needed when the iteration
        # range can include sentinel slots — i.e., when
        # max_rows_per_rank_runtime > real_total. Under
        # ``CPU_sync_on_runtime`` we synced runtime down to
        # real_total, so s_scatter_idx_local contains only local-
        # routed slots and the mask is dead weight in the kernel.
        # Pass dst_rank_flat=None there to skip the per-slot
        # ``tl.load(dst_rank_flat[slot])`` and the conditional write.
        dst_rank_flat_for_scatter = None if cpu_synced else dst_rank_flat
        _down_projection_backward_act(
            dout=dout_for_kernel,
            h=h,
            w2=w2.permute(2, 1, 0),
            dh=dh,
            ds=ds,
            b2=b2,
            db2=db2,
            a_prime=a_prime,
            topk_scores=topk_scores_global,
            expert_frequency_offset=expert_frequency_offset,
            x_gather_idx=a_idx_for_dout,
            s_scatter_idx=s_scatter_idx_local,
            activation_type=activation_type.value,
            dst_rank_flat=dst_rank_flat_for_scatter,
            my_rank=ep_ws.my_rank,
        )

        # ====================================================================
        # 5. dW2 GEMM
        # ====================================================================
        dw2 = torch.empty_like(w2)
        gemm(
            dout_for_kernel.T,
            a_prime,
            out=dw2.permute(0, 2, 1),
            cu_seqlens_k=expert_frequency_offset,
            A_idx=a_idx_for_dout,
            batch_idx_permute=None,
            dynamic_scheduler=False,
        )
        del dout_dispatched, dout_for_kernel, a_prime, h, topk_scores_global

        # ====================================================================
        # 6. Reduce-scatter ds → ds_local
        # ====================================================================
        ds_local = torch.empty(T_local * K, dtype=ds.dtype, device=ds.device)
        dist.reduce_scatter_tensor(
            ds_local, ds, op=dist.ReduceOp.SUM, group=ep_ws.ep_group,
        )
        ds_local = ds_local.view(T_local, K)

        # ====================================================================
        # 7. Up-proj backward act: dh → dx_expanded (in y_symm), db1
        # ====================================================================
        dw1 = torch.empty_like(w1)
        db1 = None if b1 is None else torch.empty_like(b1)
        _up_projection_backward_act(
            w1=w1,
            dx_expanded=ep_ws.y_symm[:max_rows_per_rank_runtime],
            dh=dh,
            db1=db1,
            expert_frequency_offset=expert_frequency_offset,
            is_glu_activation=is_glu_act,
            concat_layout=concat_layout,
        )

        # ====================================================================
        # 8. Cross-rank combine of dx_expanded → dx_local
        # ====================================================================
        dx_local = _do_combine(
            ep_ws, my_dst_rank=my_dst_rank, dst_rank_flat=dst_rank_flat, topk_scores_local=None,
            scores_global=None, K=K, T_local=T_local, H=H, out_dtype=dtype, agg_mode=cfg.agg_mode,
            peer_present_mask=peer_present_mask,
        )

        # ====================================================================
        # 9. Join the x_compute recovery (step 2) before dW1
        # ====================================================================
        if ce_handle is not None:
            ce_handle.wait()

        # ====================================================================
        # 10. dW1 GEMM
        # ====================================================================
        # K dim is the slot count, bounded to total_m by cu_seqlens_k.
        # Three-way A_idx selection (analogous to up-proj) when the
        # cached path is in use; redispatch overrides because AG-CE
        # produces an AG-shaped x_compute regardless of forward mode.
        #   redispatch       — AG layout; A_idx=x_gather_idx_ag_for_dw1.
        #   A2A_TRITON       — expert-grouped layout; A_idx=None.
        #   RANK_DEDUP_DISPATCH_TRITON — packed-by-source layout; A_idx=a_idx_rank_dedup.
        #   AG_TRITON        — per-token layout; A_idx=x_gather_idx.
        if redispatch:
            a_idx_for_dw1 = x_gather_idx_ag_for_dw1[:max_rows_per_rank_runtime]
        elif _is_a2a_dispatch_mode(mode):
            a_idx_for_dw1 = None
        elif _is_rank_dedup_dispatch_mode(mode):
            a_idx_for_dw1 = a_idx_rank_dedup[:max_rows_per_rank_runtime]
        else:
            a_idx_for_dw1 = x_gather_idx[:max_rows_per_rank_runtime]
        gemm(
            x_compute.T,
            dh,
            out=dw1.permute(2, 1, 0),
            cu_seqlens_k=expert_frequency_offset,
            A_idx=a_idx_for_dw1,
            batch_idx_permute=None,
            dynamic_scheduler=False,
            concat_layout=(("out",) if concat_layout else None),
        )

        ep_ws.o_hdl.barrier()

        ctx.ep_ws = None

        return (
            dx_local,
            dw1,
            db1,
            dw2,
            db2,
            ds_local,
            *([None] * 8),
        )


def _build_consumer_metadata(
    expert_indices: torch.Tensor,
    token_indices: torch.Tensor,
    TK: int,
    E_local: int,
    s_reverse_idx_symm: torch.Tensor,
):
    """Per-expert metadata. Sentinels (slots routed to peer ranks) are
    bucketed into expert id E_local; we slice the offset to the first
    E_local+1 entries so downstream kernels iterate only real experts."""
    device = expert_indices.device
    E_total = E_local + 1

    expert_frequency_offset = torch.empty(E_total + 1, dtype=torch.int32, device=device)
    s_reverse_local = s_reverse_idx_symm[:TK]
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    expert_frequency = torch.empty(E_total, dtype=torch.int32, device=device)

    general_routing_router_metadata_triton(
        token_indices,
        expert_indices,
        TK,
        E_total,
        expert_frequency,
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_local,
        None, # num_activated_expert_per_token_offset
    )

    expert_frequency_offset = expert_frequency_offset[: E_local + 1]

    return {
        "expert_frequency_offset": expert_frequency_offset,
        "s_reverse_local": s_reverse_local,
        "x_gather_idx": x_gather_idx,
        "s_scatter_idx": s_scatter_idx,
    }


def _moe_ep_forward_inner(
    x_local: torch.Tensor,
    topk_idx_global: torch.Tensor,
    topk_scores_local: torch.Tensor,
    w1: torch.Tensor,
    b1: Optional[torch.Tensor],
    w2: torch.Tensor,
    b2: Optional[torch.Tensor],
    ep_ws: _EPWorkspace,
    activation_type: ActivationType,
    is_inference_mode_enabled: bool,
    concat_layout: bool,
    redispatch_x_in_backward: bool,
    CPU_sync_on_runtime: bool,
    agg_mode: CombineMode = CombineMode.A2A_TRITON,
) -> torch.Tensor:
    W, my_rank = ep_ws.world_size, ep_ws.my_rank
    T_local, d, K, E_local = ep_ws.T_local, ep_ws.d, ep_ws.K, ep_ws.E_local
    TK_global = W * T_local * K
    mode = ep_ws.mode

    meta = compute_dispatch_metadata(topk_idx_global, my_rank=my_rank, E_local=E_local)
    dst_rank_flat = meta["dst_rank_flat"]
    my_dst_rank = meta["my_dst_rank"]
    expert_local_padded = meta["expert_local_padded"]
    is_grouped_dispatch = _is_a2a_dispatch_mode(mode) or _is_rank_dedup_dispatch_mode(mode)
    a2a_token_indices = meta["a2a_token_indices"] if is_grouped_dispatch else None

    if _is_ag_dispatch_mode(mode):
        token_indices = ep_ws.t_global_pattern
    else:
        token_indices = a2a_token_indices

    metadata = _build_consumer_metadata(
        expert_indices=expert_local_padded, token_indices=token_indices, TK=TK_global, E_local=E_local,
        s_reverse_idx_symm=ep_ws.s_rev_symm,
    )

    # A2A and RANK_DEDUP_DISPATCH_TRITON both need the expert-grouped recv_pos
    # (s_reverse_local). RANK_DEDUP_DISPATCH_TRITON additionally consumes pair_present_mask
    # / rank_dedup_recv_pos for the canonical-only peer pull.
    recv_pos = metadata["s_reverse_local"] if is_grouped_dispatch else None
    pair_present_mask = meta.get("pair_present_mask") if _is_rank_dedup_dispatch_mode(mode) else None
    rank_dedup_recv_pos = meta.get("rank_dedup_recv_pos") if _is_rank_dedup_dispatch_mode(mode) else None

    # Backward X redispatch always uses AG-CE (independent of forward mode), so dW1 needs an AG-style x_gather_idx.
    # In grouped-dispatch modes (A2A_TRITON or RANK_DEDUP_DISPATCH_TRITON)
    # we build a separate x_gather_idx using the AG token-id pattern.
    x_gather_idx_ag_for_dw1: Optional[torch.Tensor] = None
    if redispatch_x_in_backward:
        if ep_ws._ag_redispatch_buf is None:
            ep_ws._ag_redispatch_buf = torch.empty((W * T_local, d), dtype=x_local.dtype, device=x_local.device)
        if is_grouped_dispatch:  # A2A or RANK_DEDUP_DISPATCH_TRITON — built consumer meta from a2a_token_indices
            if ep_ws.t_global_pattern is None:
                ep_ws.t_global_pattern = torch.arange(TK_global, device=x_local.device, dtype=torch.int32) // K
            ag_metadata = _build_consumer_metadata(
                expert_indices=expert_local_padded,
                token_indices=ep_ws.t_global_pattern,
                TK=TK_global,
                E_local=E_local,
                # Throwaway s_rev — we only need x_gather_idx out of this call.
                # The configured-mode s_rev in ep_ws.s_rev_symm must stay intact
                # since A2A_combine reads through peer NVLink mappings.
                s_reverse_idx_symm=torch.empty_like(ep_ws.s_rev_symm),
            )
            x_gather_idx_ag_for_dw1 = ag_metadata["x_gather_idx"]
        else:
            x_gather_idx_ag_for_dw1 = metadata["x_gather_idx"]

    _I = w1.shape[0]
    _is_glu_act = is_glu(activation_type)
    if _is_glu_act:
        _I //= 2
    _E_local = w1.shape[2]
    _W = ep_ws.world_size
    cfg = RuntimeEPConfig(
        mode=ep_ws.mode,
        W=_W,
        K=K,
        agg_mode=agg_mode if agg_mode is not None else CombineMode.A2A_TRITON,
        I=_I,
        is_glu_act=_is_glu_act,
        E_local=_E_local,
        MAX_ROWS_PER_RANK_STATIC=T_local * _W * min(K, _E_local),
    )

    # peer_present_mask is only consumed by the rank-dedup combine
    # gather kernel; pass None for the other agg modes to keep the
    # bundle small.
    needs_dedup_combine = _is_rank_dedup_combine_mode(agg_mode)
    peer_present_mask = meta.get("peer_present_mask") if needs_dedup_combine else None

    # RANK_DEDUP_DISPATCH_TRITON: build the up-proj A_idx that gathers expert-grouped rows from the packed dispatch buffer.
    a_idx_rank_dedup: Optional[torch.Tensor] = None
    if _is_rank_dedup_dispatch_mode(mode):
        a_idx_rank_dedup = build_rank_dedup_a_idx(
            dst_rank_flat=dst_rank_flat,
            s_reverse_local=metadata["s_reverse_local"],
            rank_dedup_recv_pos=meta["rank_dedup_recv_pos"],
            my_rank=my_rank,
            out=ep_ws.a_idx_rank_dedup_buf,
        )

    meta_bundle = {
        "expert_frequency_offset": metadata["expert_frequency_offset"],
        "x_gather_idx": metadata["x_gather_idx"],
        "s_scatter_idx": metadata["s_scatter_idx"],
        "my_dst_rank": my_dst_rank,
        "recv_pos": recv_pos,
        "dst_rank_flat": dst_rank_flat,
        "x_gather_idx_ag_for_dw1": x_gather_idx_ag_for_dw1,
        # RANK_DEDUP_DISPATCH_TRITON-only — None elsewhere.
        "pair_present_mask": pair_present_mask,
        "rank_dedup_recv_pos": rank_dedup_recv_pos,
        "a_idx_rank_dedup": a_idx_rank_dedup,
        # RANK_DEDUP_COMBINE_TRITON-only — None elsewhere.
        "peer_present_mask": peer_present_mask,
    }
    return _MoeEPFunction.apply(
        x_local, w1, b1, w2, b2, topk_scores_local, cfg, meta_bundle, activation_type,
        is_inference_mode_enabled, concat_layout, redispatch_x_in_backward, CPU_sync_on_runtime, ep_ws,
    )


def _validate_and_resolve(
    mgr: "SymmMemManager",
    mode,
    E: int,
) -> Tuple[int, int, DispatchMode]:
    W = mgr.world_size
    if isinstance(mode, DispatchMode):
        resolved = mode
    elif isinstance(mode, str):
        resolved = DispatchMode(mode.upper())
    else:
        raise TypeError(f"mode must be a DispatchMode or str; got {type(mode).__name__}")
    if E % W != 0:
        raise ValueError(f"E ({E}) must be divisible by EP world size ({W}).")
    return W, E // W, resolved


def _validate_runtime_ep_config(cfg: RuntimeEPConfig, W: int, K: int) -> None:
    if cfg.W != W:
        raise ValueError(
            f"ep_config.W={cfg.W} does not match mgr.world_size={W}; "
            "rebuild the RuntimeEPConfig with NetworkProfiler on the "
            "current process group."
        )
    if cfg.K != K:
        raise ValueError(f"ep_config.K={cfg.K} does not match the call's K={K}; " "RuntimeEPConfig is per-(W, K).")


def _ag_routing_decision(
    ep_ws: _EPWorkspace,
    topk_idx_l: torch.Tensor,
) -> torch.Tensor:
    W = ep_ws.world_size
    T_local, K = topk_idx_l.shape
    out = torch.empty(W * T_local, K, dtype=topk_idx_l.dtype, device=topk_idx_l.device)
    dist.all_gather_into_tensor(out, topk_idx_l.contiguous(), group=ep_ws.ep_group)
    return out.view(W, T_local, K)


def moe_ep_TC_softmax_topk_forward(
    x: torch.Tensor,
    router_w: torch.Tensor,
    w1: torch.Tensor,
    b1: Optional[torch.Tensor],
    w2: torch.Tensor,
    b2: Optional[torch.Tensor],
    K: int,
    E: int,
    mgr: SymmMemManager,
    *,
    activation_type: ActivationType = ActivationType.SWIGLU,
    is_inference_mode_enabled: bool = False,
    is_softmax_over_topk: bool = True,
    norm_topk_probs: bool = False,
    concat_layout: bool = False,
    mode: str = "auto",
    ep_config: Optional[RuntimeEPConfig] = None,
    redispatch_x_in_backward: bool = False,
    CPU_sync_on_runtime: bool = False,
) -> torch.Tensor:
    """EP forward with TC softmax-topk router.

    Uses :class:`EP_Router_Replicated_Across_Ranks` for the router projection so the backward
    automatically all-reduces ``drouter_w`` across the EP group — each
    rank's ``router_w`` accumulates the global-batch gradient.

    ``ep_config`` carries a runtime dispatch decision produced by
    :class:`NetworkProfiler`. When ``None`` (default), the ``mode``
    argument is converted to a :class:`DispatchMode` directly (string
    values must match an enum member exactly, e.g. ``"AG_TRITON"``).
    When given, ``ep_config.mode`` overrides the ``mode`` argument, and
    ``ep_config.W`` / ``ep_config.K`` are validated against the
    actual workload.
    """
    agg_mode = CombineMode.A2A_TRITON
    if ep_config is not None:
        _validate_runtime_ep_config(ep_config, mgr.world_size, K)
        mode = ep_config.mode
        agg_mode = ep_config.agg_mode
    W, E_local, mode = _validate_and_resolve(mgr, mode, E)
    activation_type = _normalize_activation(activation_type)
    T_local, d = x.shape

    ws = mgr._get_or_alloc(T_local, d, K, E_local, x.dtype, mode)

    # Router projection with EP-aware drouter_w all-reduce.
    router_logits = EP_Router_Replicated_Across_Ranks.apply(x, router_w, ws.ep_group)
    topk_scores_l, topk_idx_l = TC_Softmax_Topk_Router_Function.apply(
        router_logits, W * E_local, K, is_softmax_over_topk, norm_topk_probs
    )

    # Publish x to peers (forward x dispatch in _moe_ep_forward_inner
    # reads peer x_symm), then collect topk_idx across ranks.
    ws.x_symm.copy_(x)
    topk_idx_g = _ag_routing_decision(ws, topk_idx_l)
    ws.x_hdl.barrier()

    return _moe_ep_forward_inner(
        x_local=x, topk_idx_global=topk_idx_g, topk_scores_local=topk_scores_l, w1=w1, b1=b1, w2=w2, b2=b2,
        ep_ws=ws, activation_type=activation_type, is_inference_mode_enabled=is_inference_mode_enabled,
        concat_layout=concat_layout, redispatch_x_in_backward=redispatch_x_in_backward,
        CPU_sync_on_runtime=CPU_sync_on_runtime, agg_mode=agg_mode,
    )


def moe_ep_general_routing_forward(
    x: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
    w1: torch.Tensor,
    b1: Optional[torch.Tensor],
    w2: torch.Tensor,
    b2: Optional[torch.Tensor],
    E: int,
    mgr: SymmMemManager,
    *,
    activation_type: ActivationType = ActivationType.SWIGLU,
    is_inference_mode_enabled: bool = False,
    concat_layout: bool = False,
    mode: str = "auto",
    ep_config: Optional[RuntimeEPConfig] = None,
    redispatch_x_in_backward: bool = False,
    CPU_sync_on_runtime: bool = False,
) -> torch.Tensor:
    K = topk_indices.shape[1]
    agg_mode = CombineMode.A2A_TRITON
    if ep_config is not None:
        _validate_runtime_ep_config(ep_config, mgr.world_size, K)
        mode = ep_config.mode
        agg_mode = ep_config.agg_mode
    W, E_local, mode = _validate_and_resolve(mgr, mode, E)
    activation_type = _normalize_activation(activation_type)
    T_local, d = x.shape

    ep_ws = mgr._get_or_alloc(T_local, d, K, E_local, x.dtype, mode)
    ep_ws.x_symm.copy_(x)

    topk_idx_g = _ag_routing_decision(ep_ws, topk_indices.to(torch.int32))
    ep_ws.x_hdl.barrier()

    return _moe_ep_forward_inner(
        x_local=x, topk_idx_global=topk_idx_g, topk_scores_local=topk_scores, w1=w1, b1=b1, w2=w2, b2=b2,
        ep_ws=ep_ws, activation_type=activation_type, is_inference_mode_enabled=is_inference_mode_enabled,
        concat_layout=concat_layout, redispatch_x_in_backward=redispatch_x_in_backward,
        CPU_sync_on_runtime=CPU_sync_on_runtime, agg_mode=agg_mode,
    )
