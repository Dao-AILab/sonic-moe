# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
# Expert Parallelism (EP) for SonicMoE.
#
# Dispatch modes (``DispatchMode``):
#   AG_DISPATCH_TRITON       — all-gather x via Triton kernel → gather GEMM (with A_idx).
#   A2A_DISPATCH_TRITON      — pull-based dispatch into expert-sorted layout → non-gather GEMM.
#   RANK_DEDUP_DISPATCH_TRITON — pull-based dispatch with same-token / same-dest dedup.
#                     Strictly Pareto-dominates both AG and A2A on inbound
#                     NVLink rows under the indicator-wise bound
#                     Dedup_d ≤ min(AG_d, A2A_d) for all routings δ. Same
#                     expert-sorted x_compute layout as A2A → non-gather GEMM.
#
# Combine modes (``CombineMode``):
#   A2A_COMBINE_TRITON              — fused NVLink-read combine kernel.
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
    clear_ep_cache,
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
    "clear_ep_cache",
    "moe_ep_TC_softmax_topk_forward",
    "moe_ep_general_routing_forward",
]


class EP_Router_Replicated_Across_Ranks(torch.autograd.Function):
    """``F.linear(x, router_w)`` with EP-aware drouter_w all-reduce.

    In EP, each rank holds ``T_local`` tokens of the global batch
    (``T_global = T_local * W``) and every rank has a replica of
    ``router_w``.
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
    dispatch_mode: DispatchMode,
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
    if _is_ag_dispatch_mode(dispatch_mode):
        return all_gather_triton(src_symm, group, out=out_buf, peer_bufs=src_peer_bufs)
    elif _is_rank_dedup_dispatch_mode(dispatch_mode):
        rank_dedup_dispatch_triton(
            x_symm=src_symm,
            dst_rank_flat=dst_rank_flat,
            pair_present_mask=pair_present_mask,
            rank_dedup_recv_pos=rank_dedup_recv_pos,
            recv_packed=out_buf,
            K=K,
            group=group,
            peer_bufs=src_peer_bufs,
            my_rank=my_rank,
        )
        return out_buf.view(-1, H)
    elif _is_a2a_dispatch_mode(dispatch_mode):
        a2a_dispatch_triton(
            x_symm=src_symm,
            dst_rank_flat=dst_rank_flat,
            recv_pos=recv_pos,
            recv=out_buf,
            K=K,
            group=group,
            peer_bufs=src_peer_bufs,
            my_rank=my_rank,
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
    combine_mode: CombineMode,
    # RANK_DEDUP_COMBINE_TRITON-only:
    peer_present_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    out = torch.empty(T_local, H, dtype=out_dtype, device=ep_ws.y_symm.device)
    if _is_a2a_combine_mode(combine_mode):
        ep_ws.o_hdl.barrier()
        a2a_combine_triton(
            ep_ws.y_symm,
            ep_ws.s_rev_symm,
            my_dst_rank,
            topk_scores_local,
            out,
            K=K,
            group=ep_ws.ep_group,
            y_peer_bufs=ep_ws.y_peer_bufs,
            s_peer_bufs=ep_ws.s_rev_peer_bufs,
            my_rank=ep_ws.my_rank,
        )
        return out
    elif _is_rank_dedup_combine_mode(combine_mode):
        ep_ws._ensure_partial_combine_buf()
        rank_dedup_combine_triton(
            ep_ws.y_symm,
            ep_ws.s_rev_symm,
            dst_rank_flat,
            scores_global,
            peer_present_mask,
            ep_ws.partial_combine_buf,
            out,
            K=K,
            T_local=T_local,
            group=ep_ws.ep_group,
            partial_combine_hdl=ep_ws.partial_combine_hdl,
            partial_combine_peer_bufs=ep_ws.partial_combine_peer_bufs,
            my_rank=ep_ws.my_rank,
        )
        return out
    elif _is_rs_combine_mode(combine_mode):
        ep_ws._ensure_partial_combine_buf()
        rs_combine_triton(
            ep_ws.y_symm,
            ep_ws.s_rev_symm,
            dst_rank_flat,
            scores_global,
            ep_ws.partial_combine_buf,
            out,
            K,
            T_local,
            group=ep_ws.ep_group,
            partial_combine_hdl=ep_ws.partial_combine_hdl,
            partial_combine_peer_bufs=ep_ws.partial_combine_peer_bufs,
            my_rank=ep_ws.my_rank,
        )
        return out
    else:
        raise NotImplementedError()


class _MoeEPFunction(torch.autograd.Function):
    """Merged EP forward + backward for one MoE layer.

    Forward: dispatch x → up-proj GEMM (gemm_gated) → down-proj GEMM →
    NVLink-gather combine → y_local.

    Backward: dispatch do (configured mode) → reuse saved x_compute →
    all-gather scores → gemm_dgated (→ dh, ds, a_prime) → dW2 →
    reduce-scatter ds → up-proj-backward (writes dx_expanded into
    y_symm) → cross-rank gather to dx_local → dW1.

    Caches across the autograd boundary:
      - h: (total_m, H_act), where ``total_m = max_rows_per_rank_runtime``
        (= synced ``valid_rows`` under ``CPU_sync_on_runtime``,
        ``MAX_ROWS_PER_RANK_STATIC`` otherwise). gemm_gated / gemm_dgated
        strict-check ``preact.shape[0] == total_m``.
      - x_compute: (W·T_local, H) for AG / RANK_DEDUP and (total_m, H)
        for A2A.
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
        dispatch_mode = cfg.dispatch_mode
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
        x_idx_expanded_remap_for_rank_dedup = meta.get("x_idx_expanded_remap_for_rank_dedup")

        H_act = 2 * I if is_glu_act else I

        if is_inference_mode_enabled:
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
        if is_inference_mode_enabled:
            # x_compute isn't saved-for-backward in inference mode, so
            # reuse the workspace recv buffer where available.
            if _is_ag_dispatch_mode(dispatch_mode):
                ws_buf = ep_ws.ag_compute
            elif _is_a2a_dispatch_mode(dispatch_mode):
                ws_buf = ep_ws.a2a_recv
            else:
                ws_buf = torch.empty(ep_ws.world_size * T_local, H, dtype=x_dtype, device=device)
            x_compute = _do_dispatch(
                ep_ws.x_symm,
                ws_buf,
                dispatch_mode,
                dst_rank_flat=dst_rank_flat,
                recv_pos=recv_pos,
                K=K,
                group=ep_ws.ep_group,
                H=H,
                src_peer_bufs=ep_ws.x_peer_bufs,
                my_rank=ep_ws.my_rank,
                pair_present_mask=pair_present_mask,
                rank_dedup_recv_pos=rank_dedup_recv_pos,
            )
            # A2A's recv is consumed by gemm_gated WITHOUT an A_idx,
            # so the kernel reads ``total_m = A.shape[0]`` directly —
            # trim x_compute to the runtime row count for the strict-
            # check ``A.shape[0] == h.shape[0]`` to pass. AG and
            # RANK_DEDUP both feed the GEMM via A_idx (so total_m comes
            # from A_idx.shape[0], independent of x_compute's shape) and
            # need no trim.
            if _is_a2a_dispatch_mode(dispatch_mode):
                x_compute = x_compute[:max_rows_per_rank_runtime]
        else:
            # Per-call recv buffer for the cache path. The fresh
            # ``torch.empty`` decouples the saved x_compute from any
            # workspace, so backward step 1's dout dispatch can't
            # clobber it. Mode-aware sizing:
            #   AG_DISPATCH_TRITON   — per-token recv at W·T_local rows.
            #   RANK_DEDUP  — packed-by-source recv at W·T_local rows
            #                 (each home rank contributes ≤ T_local
            #                 distinct tokens; the GEMM gathers via
            #                 A_idx and ignores any tail).
            #   A2A_DISPATCH_TRITON  — expert-grouped recv at
            #                 ``max_rows_per_rank_runtime`` (synced
            #                 ``valid_rows`` under
            #                 ``CPU_sync_on_runtime``, otherwise the
            #                 structural ceiling
            #                 T_local·W·min(K, E_local)).
            if _is_ag_dispatch_mode(dispatch_mode) or _is_rank_dedup_dispatch_mode(dispatch_mode):
                fresh = torch.empty(ep_ws.world_size * T_local, H, dtype=x_dtype, device=device)
            else:
                fresh = torch.empty(max_rows_per_rank_runtime, H, dtype=x_dtype, device=device)
            x_compute = _do_dispatch(
                ep_ws.x_symm,
                fresh,
                dispatch_mode,
                dst_rank_flat=dst_rank_flat,
                recv_pos=recv_pos,
                K=K,
                group=ep_ws.ep_group,
                H=H,
                src_peer_bufs=ep_ws.x_peer_bufs,
                my_rank=ep_ws.my_rank,
                pair_present_mask=pair_present_mask,
                rank_dedup_recv_pos=rank_dedup_recv_pos,
            )

        # ====================================================================
        # 2. Up-proj GEMM with fused gated activation: x_compute @ w1 → (h, a)
        # ====================================================================
        a = torch.empty(max_rows_per_rank_runtime, I, dtype=x_dtype, device=device)
        h = torch.empty(max_rows_per_rank_runtime, H_act, dtype=x_dtype, device=device)

        # Three-way A_idx selection for the up-proj GEMM:
        #   A2A_DISPATCH_TRITON      — recv already expert-grouped, A_idx=None.
        #   RANK_DEDUP_DISPATCH_TRITON — recv packed-by-source, A_idx maps each
        #                     expert-grouped row → its packed row.
        #   AG_DISPATCH_TRITON       — recv is per-token; A_idx=x_gather_idx maps
        #                     expert-grouped row → token row.
        if _is_a2a_dispatch_mode(dispatch_mode):
            a_idx_for_up = None
        elif _is_rank_dedup_dispatch_mode(dispatch_mode):
            a_idx_for_up = x_idx_expanded_remap_for_rank_dedup[:max_rows_per_rank_runtime]
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
        if cfg.combine_mode in (CombineMode.RS_COMBINE_TRITON, CombineMode.RANK_DEDUP_COMBINE_TRITON):
            scores_global = _all_gather_topk_scores(topk_scores_local, ep_ws.ep_group, ep_ws.world_size, T_local, K)
        o_local = _do_combine(
            ep_ws,
            my_dst_rank=my_dst_rank,
            dst_rank_flat=dst_rank_flat,
            topk_scores_local=topk_scores_local,
            scores_global=scores_global,
            K=K,
            T_local=T_local,
            H=H,
            out_dtype=x_dtype,
            combine_mode=cfg.combine_mode,
            peer_present_mask=peer_present_mask,
        )

        # ====================================================================
        # 5. Save state for backward (training only)
        # ====================================================================
        if not is_inference_mode_enabled:
            # h, a are alloc'd fresh at the runtime row count (step 2);
            # x_compute is alloc'd fresh in step 1.
            ctx.save_for_backward(
                x_compute,
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
        # 13 forward inputs → 13 grads (only the first 6 are tensor inputs).
        (
            x_compute,
            w1,
            b1,
            w2,
            b2,
            h,
            topk_scores_local,
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
        cpu_synced = ctx.CPU_sync_on_runtime
        # Metadata tensors (no autograd flow).
        expert_frequency_offset = meta["expert_frequency_offset"]
        x_gather_idx = meta["x_gather_idx"]
        s_scatter_idx = meta["s_scatter_idx"]
        my_dst_rank = meta["my_dst_rank"]
        recv_pos = meta["recv_pos"]
        dst_rank_flat = meta["dst_rank_flat"]
        pair_present_mask = meta.get("pair_present_mask")
        rank_dedup_recv_pos = meta.get("rank_dedup_recv_pos")
        peer_present_mask = meta.get("peer_present_mask")
        x_idx_expanded_remap_for_rank_dedup = meta.get("x_idx_expanded_remap_for_rank_dedup")

        max_rows_per_rank_runtime = ctx.max_rows_per_rank_runtime
        ep_ws = ctx.ep_ws
        dispatch_mode = cfg.dispatch_mode

        H = ep_ws.d
        I = w2.shape[1]
        device, dtype = dout_local.device, dout_local.dtype

        # ====================================================================
        # 1. Dispatch do_local → dout_dispatched
        # ====================================================================
        # The forward x dispatch is the last consumer of x_symm, so the
        # backward freely reuses it to publish dout for the dO dispatch.
        do_buf, do_hdl, do_peer_bufs = ep_ws.x_symm, ep_ws.x_hdl, ep_ws.x_peer_bufs
        do_buf.copy_(dout_local)
        do_hdl.barrier()

        if _is_rank_dedup_dispatch_mode(dispatch_mode):
            do_recv_buf = torch.empty(ep_ws.world_size * T_local, H, dtype=dtype, device=device)
        elif _is_ag_dispatch_mode(dispatch_mode):
            do_recv_buf = ep_ws.ag_compute
        else:
            do_recv_buf = ep_ws.a2a_recv
        dout_dispatched = _do_dispatch(
            do_buf,
            do_recv_buf,
            dispatch_mode,
            dst_rank_flat=dst_rank_flat,
            recv_pos=recv_pos,
            K=K,
            group=ep_ws.ep_group,
            H=H,
            src_peer_bufs=do_peer_bufs,
            my_rank=ep_ws.my_rank,
            pair_present_mask=pair_present_mask,
            rank_dedup_recv_pos=rank_dedup_recv_pos,
        )

        # ====================================================================
        # 2. All-gather topk scores (or reuse the forward's cached AG)
        # ====================================================================
        # Forward step 4's RS- or RANK_DEDUP-combine path already AGs
        # scores; if so it cached the result on ``ctx.scores_global``.
        # Reuse to avoid a redundant NCCL collective. A2A_COMBINE_TRITON
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
        # 3. Down-proj backward act (gemm_dgated): dh, ds, a_prime
        # ====================================================================
        dh = torch.empty_like(h)
        ds = torch.zeros(TK_global, dtype=topk_scores_global.dtype, device=device)
        a_prime = torch.empty(max_rows_per_rank_runtime, I, dtype=h.dtype, device=device)
        db2 = None if b2 is None else torch.empty_like(b2)

        # Three-way A_idx for the down-proj backward / dW2 GEMM:
        #   A2A_DISPATCH_TRITON      — expert-grouped; A_idx=None, slice dout to total_m.
        #   RANK_DEDUP_DISPATCH_TRITON — packed-by-source; A_idx maps expert-grouped row
        #                     → packed row (same x_idx_expanded_remap_for_rank_dedup the up-proj used
        #                     in forward; identical metadata for both passes
        #                     since dst_rank_flat / s_reverse_local are layer-
        #                     and-pass invariant). No slice — all packed rows
        #                     are valid.
        #   AG_DISPATCH_TRITON       — A_idx=x_gather_idx[:total_m], no slice.
        if _is_a2a_dispatch_mode(dispatch_mode):
            a_idx_for_dout = None
            dout_for_kernel = dout_dispatched[:max_rows_per_rank_runtime]
        elif _is_rank_dedup_dispatch_mode(dispatch_mode):
            a_idx_for_dout = x_idx_expanded_remap_for_rank_dedup[:max_rows_per_rank_runtime]
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
        # 4. dW2 GEMM
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
        # 5. Reduce-scatter ds → ds_local
        # ====================================================================
        ds_local = torch.empty(T_local * K, dtype=ds.dtype, device=ds.device)
        dist.reduce_scatter_tensor(
            ds_local,
            ds,
            op=dist.ReduceOp.SUM,
            group=ep_ws.ep_group,
        )
        ds_local = ds_local.view(T_local, K)

        # ====================================================================
        # 6. Up-proj backward act: dh → dx_expanded (in y_symm), db1
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
        # 7. Cross-rank combine of dx_expanded → dx_local
        # ====================================================================
        dx_local = _do_combine(
            ep_ws,
            my_dst_rank=my_dst_rank,
            dst_rank_flat=dst_rank_flat,
            topk_scores_local=None,
            scores_global=None,
            K=K,
            T_local=T_local,
            H=H,
            out_dtype=dtype,
            combine_mode=cfg.combine_mode,
            peer_present_mask=peer_present_mask,
        )

        # ====================================================================
        # 8. dW1 GEMM
        # ====================================================================
        # K dim is the slot count, bounded to total_m by cu_seqlens_k.
        # Three-way A_idx selection (analogous to up-proj):
        #   A2A_DISPATCH_TRITON       — expert-grouped layout; A_idx=None.
        #   RANK_DEDUP_DISPATCH_TRITON — packed-by-source layout; A_idx=x_idx_expanded_remap_for_rank_dedup.
        #   AG_DISPATCH_TRITON        — per-token layout; A_idx=x_gather_idx.
        if _is_a2a_dispatch_mode(dispatch_mode):
            a_idx_for_dw1 = None
        elif _is_rank_dedup_dispatch_mode(dispatch_mode):
            a_idx_for_dw1 = x_idx_expanded_remap_for_rank_dedup[:max_rows_per_rank_runtime]
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
            *([None] * 7),
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
        None,  # num_activated_expert_per_token_offset
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
    cfg: RuntimeEPConfig,
    activation_type: ActivationType,
    is_inference_mode_enabled: bool,
    concat_layout: bool,
    CPU_sync_on_runtime: bool,
) -> torch.Tensor:
    W, my_rank = ep_ws.world_size, ep_ws.my_rank
    T_local, K, E_local = ep_ws.T_local, ep_ws.K, ep_ws.E_local
    TK_global = W * T_local * K
    dispatch_mode = cfg.dispatch_mode
    combine_mode = cfg.combine_mode

    meta = compute_dispatch_metadata(topk_idx_global, my_rank=my_rank, E_local=E_local)
    dst_rank_flat = meta["dst_rank_flat"]
    my_dst_rank = meta["my_dst_rank"]
    expert_local_padded = meta["expert_local_padded"]
    is_grouped_dispatch = _is_a2a_dispatch_mode(dispatch_mode) or _is_rank_dedup_dispatch_mode(dispatch_mode)
    a2a_token_indices = meta["a2a_token_indices"] if is_grouped_dispatch else None

    if _is_ag_dispatch_mode(dispatch_mode):
        token_indices = ep_ws.t_global_pattern
    else:
        token_indices = a2a_token_indices

    metadata = _build_consumer_metadata(
        expert_indices=expert_local_padded,
        token_indices=token_indices,
        TK=TK_global,
        E_local=E_local,
        s_reverse_idx_symm=ep_ws.s_rev_symm,
    )

    # A2A and RANK_DEDUP_DISPATCH_TRITON both need the expert-grouped recv_pos
    # (s_reverse_local). RANK_DEDUP_DISPATCH_TRITON additionally consumes pair_present_mask
    # / rank_dedup_recv_pos for the canonical-only peer pull.
    recv_pos = metadata["s_reverse_local"] if is_grouped_dispatch else None
    pair_present_mask = meta.get("pair_present_mask") if _is_rank_dedup_dispatch_mode(dispatch_mode) else None
    rank_dedup_recv_pos = meta.get("rank_dedup_recv_pos") if _is_rank_dedup_dispatch_mode(dispatch_mode) else None

    # peer_present_mask is only consumed by the rank-dedup combine
    # gather kernel; pass None for the other combine modes to keep the
    # bundle small.
    needs_dedup_combine = _is_rank_dedup_combine_mode(combine_mode)
    peer_present_mask = meta.get("peer_present_mask") if needs_dedup_combine else None

    # RANK_DEDUP_DISPATCH_TRITON: build the up-proj A_idx that gathers expert-grouped rows from the packed dispatch buffer.
    x_idx_expanded_remap_for_rank_dedup: Optional[torch.Tensor] = None
    if _is_rank_dedup_dispatch_mode(dispatch_mode):
        x_idx_expanded_remap_for_rank_dedup = build_rank_dedup_a_idx(
            dst_rank_flat=dst_rank_flat,
            s_reverse_local=metadata["s_reverse_local"],
            rank_dedup_recv_pos=meta["rank_dedup_recv_pos"],
            my_rank=my_rank,
            out=ep_ws.x_idx_expanded_remap_for_rank_dedup_buf,
        )

    meta_bundle = {
        "expert_frequency_offset": metadata["expert_frequency_offset"],
        "x_gather_idx": metadata["x_gather_idx"],
        "s_scatter_idx": metadata["s_scatter_idx"],
        "my_dst_rank": my_dst_rank,
        "recv_pos": recv_pos,
        "dst_rank_flat": dst_rank_flat,
        # RANK_DEDUP_DISPATCH_TRITON-only — None elsewhere.
        "pair_present_mask": pair_present_mask,
        "rank_dedup_recv_pos": rank_dedup_recv_pos,
        "x_idx_expanded_remap_for_rank_dedup": x_idx_expanded_remap_for_rank_dedup,
        # RANK_DEDUP_COMBINE_TRITON-only — None elsewhere.
        "peer_present_mask": peer_present_mask,
    }
    return _MoeEPFunction.apply(
        x_local,
        w1,
        b1,
        w2,
        b2,
        topk_scores_local,
        cfg,
        meta_bundle,
        activation_type,
        is_inference_mode_enabled,
        concat_layout,
        CPU_sync_on_runtime,
        ep_ws,
    )


def _default_ep_config(W: int, K: int) -> RuntimeEPConfig:
    """Pick a (dispatch, combine) pair without profiling.

    Heuristic (checked in order):
      * ``W <= 2`` or ``W <= 0.8 * K`` → **AG dispatch + RS combine**.
        Very small world size, or K large enough relative to W that AG's
        K-independent ``W·T_local·d`` traffic beats A2A's K-fanout.
      * ``K >= 1.25 * W`` → **A2A dispatch + A2A combine**.
        Algebraically equivalent to ``W <= 0.8*K``, so the AG/RS branch
        already swallows this region; the branch is kept for parity with
        the documented rule.
      * otherwise → **rank-dedup dispatch + rank-dedup combine**.

    Override by passing ``ep_config=`` from :meth:`NetworkProfiler.profile`
    or by constructing a :class:`RuntimeEPConfig` directly.
    """
    if W <= 2 or W <= 0.8 * K:
        return RuntimeEPConfig(
            dispatch_mode=DispatchMode.AG_DISPATCH_TRITON,
            W=W,
            K=K,
            combine_mode=CombineMode.RS_COMBINE_TRITON,
        )
    if K >= 1.25 * W:
        return RuntimeEPConfig(
            dispatch_mode=DispatchMode.A2A_DISPATCH_TRITON,
            W=W,
            K=K,
            combine_mode=CombineMode.A2A_COMBINE_TRITON,
        )
    return RuntimeEPConfig(
        dispatch_mode=DispatchMode.RANK_DEDUP_DISPATCH_TRITON,
        W=W,
        K=K,
        combine_mode=CombineMode.RANK_DEDUP_COMBINE_TRITON,
    )


def _validate_runtime_ep_config(cfg: RuntimeEPConfig, W: int, K: int) -> None:
    if cfg.W != W:
        raise ValueError(
            f"ep_config.W={cfg.W} does not match the current EP world size ({W}); "
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
    *,
    group: Optional[dist.ProcessGroup] = None,
    activation_type: ActivationType = ActivationType.SWIGLU,
    is_inference_mode_enabled: bool = False,
    is_softmax_over_topk: bool = True,
    norm_topk_probs: bool = False,
    concat_layout: bool = False,
    ep_config: Optional[RuntimeEPConfig] = None,
    CPU_sync_on_runtime: bool = False,
) -> torch.Tensor:
    """EP forward with TC softmax-topk router.

    Uses :class:`EP_Router_Replicated_Across_Ranks` for the router projection so the backward
    automatically all-reduces ``drouter_w`` across the EP group — each
    rank's ``router_w`` accumulates the global-batch gradient.

    The symm-mem workspace is cached per ``(group, x.device)`` inside
    the module — see :func:`clear_ep_cache` for explicit cleanup. Pass
    ``group=`` to target an EP group other than ``dist.group.WORLD``.

    ``ep_config`` selects the dispatch + combine primitives. When
    ``None`` (default), one is built via :func:`_default_ep_config` from
    the EP world size and ``K`` (see that function for the heuristic).
    Pass the result of :meth:`NetworkProfiler.profile` for a profiled
    decision, or a hand-built :class:`RuntimeEPConfig` to force a
    specific pair.
    """
    mgr = SymmMemManager(group, x.device)
    W = mgr.world_size
    if E % W != 0:
        raise ValueError(f"E ({E}) must be divisible by EP world size ({W}).")
    E_local = E // W
    if ep_config is None:
        ep_config = _default_ep_config(W, K)
    _validate_runtime_ep_config(ep_config, W, K)
    activation_type = _normalize_activation(activation_type)
    T_local, d = x.shape

    # Build the call-local cfg by promoting the user's (dispatch, combine,
    # W, K) selection with the layer-static fields that
    # ``_moe_ep_forward_inner`` / ``_MoeEPFunction`` consume. A fresh
    # instance avoids mutating the caller's ``ep_config`` (which is
    # frozen, and may be reused across multiple EP layers with different
    # I / E_local / T_local).
    _I = w1.shape[0]
    _is_glu_act = is_glu(activation_type)
    if _is_glu_act:
        _I //= 2
    cfg = RuntimeEPConfig(
        dispatch_mode=ep_config.dispatch_mode,
        W=W,
        K=K,
        combine_mode=ep_config.combine_mode,
        I=_I,
        is_glu_act=_is_glu_act,
        E_local=E_local,
        MAX_ROWS_PER_RANK_STATIC=T_local * W * min(K, E_local),
    )

    ws = mgr._get_or_alloc(T_local, d, K, E_local, x.dtype, cfg.dispatch_mode)

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
        x_local=x,
        topk_idx_global=topk_idx_g,
        topk_scores_local=topk_scores_l,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        ep_ws=ws,
        cfg=cfg,
        activation_type=activation_type,
        is_inference_mode_enabled=is_inference_mode_enabled,
        concat_layout=concat_layout,
        CPU_sync_on_runtime=CPU_sync_on_runtime,
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
    *,
    group: Optional[dist.ProcessGroup] = None,
    activation_type: ActivationType = ActivationType.SWIGLU,
    is_inference_mode_enabled: bool = False,
    concat_layout: bool = False,
    ep_config: Optional[RuntimeEPConfig] = None,
    CPU_sync_on_runtime: bool = False,
) -> torch.Tensor:
    """EP forward with caller-supplied top-K routing.

    Same workspace/cache semantics as :func:`moe_ep_TC_softmax_topk_forward`.
    When ``ep_config`` is ``None`` a default is built via
    :func:`_default_ep_config` from the EP world size and the K inferred
    from ``topk_indices.shape[1]``.
    """
    K = topk_indices.shape[1]
    mgr = SymmMemManager(group, x.device)
    W = mgr.world_size
    if E % W != 0:
        raise ValueError(f"E ({E}) must be divisible by EP world size ({W}).")
    E_local = E // W
    if ep_config is None:
        ep_config = _default_ep_config(W, K)
    _validate_runtime_ep_config(ep_config, W, K)
    activation_type = _normalize_activation(activation_type)
    T_local, d = x.shape

    # Build the call-local cfg with the layer-static fields filled in;
    # see ``moe_ep_TC_softmax_topk_forward`` for the rationale.
    _I = w1.shape[0]
    _is_glu_act = is_glu(activation_type)
    if _is_glu_act:
        _I //= 2
    cfg = RuntimeEPConfig(
        dispatch_mode=ep_config.dispatch_mode,
        W=W,
        K=K,
        combine_mode=ep_config.combine_mode,
        I=_I,
        is_glu_act=_is_glu_act,
        E_local=E_local,
        MAX_ROWS_PER_RANK_STATIC=T_local * W * min(K, E_local),
    )

    ep_ws = mgr._get_or_alloc(T_local, d, K, E_local, x.dtype, cfg.dispatch_mode)
    ep_ws.x_symm.copy_(x)

    topk_idx_g = _ag_routing_decision(ep_ws, topk_indices.to(torch.int32))
    ep_ws.x_hdl.barrier()

    return _moe_ep_forward_inner(
        x_local=x,
        topk_idx_global=topk_idx_g,
        topk_scores_local=topk_scores,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        ep_ws=ep_ws,
        cfg=cfg,
        activation_type=activation_type,
        is_inference_mode_enabled=is_inference_mode_enabled,
        concat_layout=concat_layout,
        CPU_sync_on_runtime=CPU_sync_on_runtime,
    )
