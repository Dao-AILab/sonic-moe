# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
# Expert Parallelism (EP) for SonicMoE. RANK_DEDUP dispatch/combine Pareto-dominates AG/A2A on inbound
# NVLink bytes (Dedup_d <= min(AG_d, A2A_d)); NetworkProfiler measures all modes on HW to pick winners.
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
    _is_hier_node_dedup_combine_gin_mode,
    _is_hier_node_dedup_dispatch_gin_mode,
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
    all_gather_triton,
    build_rank_dedup_a_idx,
    compute_dispatch_metadata,
    local_combine,
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
    """F.linear(x, router_w) but backward all-reduces drouter_w across ranks (router_w is replicated, not sharded)."""

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


def _is_grouped_recv_layout(mode: DispatchMode) -> bool:
    """True for dispatch modes whose recv buffer is rank-dedup's packed-by-source layout (HIER writes
    recv_packed identically, so it shares the GEMM/A_idx path). Kept separate from _is_rank_dedup_dispatch_mode so the workspace allocator's rank-dedup branch isn't mis-triggered."""
    return _is_rank_dedup_dispatch_mode(mode) or _is_hier_node_dedup_dispatch_gin_mode(mode)


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
    # HIER_NODE_DEDUP_DISPATCH_GIN-only: GIN workspace/backend, hierarchical metadata, and fwd/bwd buffer selector.
    ep_ws: "Optional[_EPWorkspace]" = None,
    meta: Optional[dict] = None,
    is_forward: bool = True,
) -> torch.Tensor:
    if _is_ag_dispatch_mode(dispatch_mode):
        return all_gather_triton(src_symm, group, out=out_buf, peer_bufs=src_peer_bufs)
    elif _is_hier_node_dedup_dispatch_gin_mode(dispatch_mode):
        # x is already staged in ep_ws.x_gin_* by the caller (not src_symm); bound fn keeps ep.py GIN-import-free.
        # out_buf.dtype MUST equal the x/window dtype.
        ep_ws.gin_dispatch_fn(
            ep_ws.gin_backend,
            ep_ws.x_gin_fwd if is_forward else ep_ws.x_gin_bwd,
            ep_ws.dst_node_buf_fwd if is_forward else ep_ws.dst_node_buf_bwd,
            ep_ws.gin_least_fwd if is_forward else ep_ws.gin_least_bwd,
            ep_ws.x_lsa_fwd if is_forward else ep_ws.x_lsa_bwd,
            ep_ws.dst_node_buf_lsa_fwd if is_forward else ep_ws.dst_node_buf_lsa_bwd,
            out_buf, meta, rank=my_rank, world_size=ep_ws.world_size, node_size=ep_ws.node_size,
            T_local=ep_ws.T_local, K=K, H=H, group=group, node_hdl=ep_ws.x_hdl,
            staging_win=ep_ws.staging_fwd if is_forward else ep_ws.staging_bwd)
        return out_buf.view(-1, H)
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
        # out_buf may be (W, TK_local, H) or flat (rows, H); collapse to (rows, H) regardless.
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
    # Persist-producer metadata (host-sync-free); routes to the tight persist+BLOCK_SLOT kernel (~5 TB/s)
    # instead of the all-K fallback. None -> all-K fallback.
    mine_slot_idx: Optional[torch.Tensor] = None,
    mine_count: Optional[torch.Tensor] = None,
    combine_contrib_C: Optional[int] = None,
    combine_work_list: Optional[torch.Tensor] = None,
    combine_work_count: Optional[torch.Tensor] = None,
    # RANK_DEDUP gather reads single-contributor peers' y_symm directly (their
    # pre-reduce is a no-op), via a precomputed source row (single_row).
    combine_single_k: Optional[torch.Tensor] = None,
    single_row: Optional[torch.Tensor] = None,
    # HIER_NODE_DEDUP_COMBINE_GIN-only: hierarchical metadata dict + fwd/bwd combine window selector.
    meta: Optional[dict] = None,
    is_forward: bool = True,
) -> torch.Tensor:
    out = torch.empty(T_local, H, dtype=out_dtype, device=ep_ws.y_symm.device)
    if _is_a2a_combine_mode(combine_mode):
        ep_ws.o_hdl.barrier()
        ep_ws._ensure_a2a_peer_base()
        a2a_combine_triton(
            ep_ws.y_symm,
            ep_ws.s_rev_symm,
            my_dst_rank,
            topk_scores_local,
            out,
            K=K,
            group=ep_ws.ep_group,
            peer_y_base=ep_ws.a2a_peer_y_base,
            peer_s_base=ep_ws.a2a_peer_s_base,
            my_rank=ep_ws.my_rank,
        )
        return out
    elif _is_rank_dedup_combine_mode(combine_mode):
        ep_ws._ensure_partial_combine_buf()
        rank_dedup_combine_triton(
            ep_ws.y_symm,
            ep_ws.s_rev_symm,
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
            mine_slot_idx=mine_slot_idx,
            mine_count=mine_count,
            combine_contrib_C=combine_contrib_C,
            combine_work_list=combine_work_list,
            combine_work_count=combine_work_count,
            combine_single_k=combine_single_k,
            y_peer_bufs=ep_ws.y_peer_bufs,
            s_reverse_peer_bufs=ep_ws.s_rev_peer_bufs,
            single_row=single_row,
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
    elif _is_hier_node_dedup_combine_gin_mode(combine_mode):
        # Inter-node combine = reverse-mirror of HIER dispatch; a REDUCTION (no RDMA atomics). fwd/bwd MUST
        # use separate combine windows (invariant #7). HW-validated path (jobs 3596675/3596701).
        local_combine(
            ep_ws.y_symm, ep_ws.s_rev_symm, dst_rank_flat, scores_global,
            ep_ws.partial_combine_buf, K, T_local, ep_ws.ep_group, skip_empty=False)
        send_win = ep_ws.combine_send_fwd if is_forward else ep_ws.combine_send_bwd
        recv_win = ep_ws.combine_recv_fwd if is_forward else ep_ws.combine_recv_bwd
        ep_ws.gin_combine_fn(
            ep_ws.gin_backend, send_win, recv_win, ep_ws.combine_least,
            ep_ws.partial_combine_peer_bufs, recv_win.tensor.view(-1, H), out, meta,
            rank=ep_ws.my_rank, world_size=ep_ws.world_size, node_size=ep_ws.node_size,
            num_nodes=ep_ws.num_nodes, T_local=T_local, d=H, group=ep_ws.ep_group,
            node_hdl=ep_ws.partial_combine_hdl)
        return out
    else:
        raise NotImplementedError()


class _MoeEPFunction(torch.autograd.Function):
    """Merged EP forward+backward for one MoE layer (steps numbered inline below).
    x_compute is (W*T_local, H) for AG/RANK_DEDUP dispatch, (total_m, H) for A2A."""

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

        I = cfg.I
        is_glu_act = cfg.is_glu_act
        E_local = cfg.E_local
        MAX_ROWS_PER_RANK_STATIC = cfg.MAX_ROWS_PER_RANK_STATIC
        dispatch_mode = cfg.dispatch_mode
        K = cfg.K
        T_local = ep_ws.T_local

        expert_frequency_offset = meta["expert_frequency_offset"]
        x_gather_idx = meta["x_gather_idx"]
        my_dst_rank = meta["my_dst_rank"]
        recv_pos = meta["recv_pos"]
        dst_rank_flat = meta["dst_rank_flat"]
        pair_present_mask = meta.get("pair_present_mask")
        rank_dedup_recv_pos = meta.get("rank_dedup_recv_pos")
        peer_present_mask = meta.get("peer_present_mask")
        x_idx_expanded_remap_for_rank_dedup = meta.get("x_idx_expanded_remap_for_rank_dedup")
        # persist-producer combine metadata (host-sync-free; None → all-K fallback)
        combine_mine_slot_idx = meta.get("mine_slot_idx")
        combine_mine_count = meta.get("mine_count")
        combine_contrib_C = meta.get("combine_contrib_C")

        H_act = 2 * I if is_glu_act else I

        if is_inference_mode_enabled:
            CPU_sync_on_runtime = False

        # Row count for all this forward's allocations (h, a, dispatch recv, y_symm slice). Under
        # CPU_sync_on_runtime it's a synced D2H .item() from expert_frequency_offset[E_local]; else the structural ceiling.
        max_rows_per_rank_runtime = MAX_ROWS_PER_RANK_STATIC
        if CPU_sync_on_runtime:
            max_rows_per_rank_runtime = expert_frequency_offset[E_local].item()

        # --- 1. Dispatch x -> x_compute ---
        # HIER reads tokens from the GIN window (NOT x_symm); stage them here so the node-dedup put sees
        # them (_do_dispatch's HIER branch then ignores x_symm/peer_bufs). Flat modes leave x_gin_fwd None.
        if _is_hier_node_dedup_dispatch_gin_mode(dispatch_mode):
            ep_ws.x_gin_fwd.tensor.view(T_local, H).copy_(x_local)
        if is_inference_mode_enabled:
            # Inference mode doesn't save x_compute for backward, so reuse the workspace recv buffer.
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
                ep_ws=ep_ws,
                meta=meta,
                is_forward=True,
            )
            # A2A feeds gemm_gated with no A_idx, so total_m=A.shape[0] directly -> trim to pass the
            # A.shape[0]==h.shape[0] check. AG/RANK_DEDUP use A_idx (total_m independent of x_compute) -> no trim.
            if _is_a2a_dispatch_mode(dispatch_mode):
                x_compute = x_compute[:max_rows_per_rank_runtime]
        else:
            # Fresh torch.empty (not a workspace buffer) so backward step 1's dout dispatch can't clobber
            # the saved x_compute. AG/RANK_DEDUP size W*T_local rows; A2A sizes max_rows_per_rank_runtime.
            if _is_ag_dispatch_mode(dispatch_mode) or _is_grouped_recv_layout(dispatch_mode):
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
                ep_ws=ep_ws,
                meta=meta,
                is_forward=True,
            )

        # --- 2. Up-proj GEMM with fused gated activation: x_compute @ w1 -> (h, a) ---
        a = torch.empty(max_rows_per_rank_runtime, I, dtype=x_dtype, device=device)
        h = torch.empty(max_rows_per_rank_runtime, H_act, dtype=x_dtype, device=device)

        # A_idx by dispatch mode: A2A recv is already expert-grouped (None); RANK_DEDUP recv is
        # packed-by-source (remap); AG recv is per-token (x_gather_idx maps expert-grouped row -> token row).
        if _is_a2a_dispatch_mode(dispatch_mode):
            a_idx_for_up = None
        elif _is_grouped_recv_layout(dispatch_mode):
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

        # --- 3. Down-proj GEMM: a @ w2 -> y_symm ---
        # y_symm is (MAX_ROWS_PER_RANK_STATIC, H) symm-mem; kernel writes [0, offset[E_local]) and peers
        # gather from those positions via s_rev_symm.
        gemm(
            a,
            w2,
            out=ep_ws.y_symm[:max_rows_per_rank_runtime],
            cu_seqlens_m=expert_frequency_offset,
            bias=b2,
            dynamic_scheduler=False,
        )
        del a

        # --- 4. NVLink combine -> o_local ---
        # Mode-dispatched by ``_do_combine``; barrier placement is internal to each branch there.
        scores_global: Optional[torch.Tensor] = None
        if cfg.combine_mode in (CombineMode.RS_COMBINE_TRITON, CombineMode.RANK_DEDUP_COMBINE_TRITON,
                                CombineMode.HIER_NODE_DEDUP_COMBINE_GIN):
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
            mine_slot_idx=combine_mine_slot_idx,
            mine_count=combine_mine_count,
            combine_contrib_C=combine_contrib_C,
            # RANK_DEDUP producer pre-reduces only multi-contributor pairs (singles read y_symm directly).
            # meta.get returns None for non-dedup modes, whose branches ignore these kwargs.
            combine_work_list=meta.get("combine_work_list_multi"),
            combine_work_count=meta.get("combine_work_count_multi"),
            combine_single_k=meta.get("combine_single_k"),
            single_row=meta.get("single_row"),
            meta=meta,
            is_forward=True,
        )

        # --- 5. Save state for backward (training only) ---
        if not is_inference_mode_enabled:
            # h, a, x_compute were all alloc'd fresh (steps 1-2), so no workspace-aliasing risk here.
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
            # Cache the RS/RANK_DEDUP-combine forward's topk_scores AG so backward step 3 can reuse it
            # (avoids a duplicate all-gather).
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
        # persist-producer combine metadata (host-sync-free; None → all-K fallback)
        combine_mine_slot_idx = meta.get("mine_slot_idx")
        combine_mine_count = meta.get("mine_count")
        combine_contrib_C = meta.get("combine_contrib_C")

        max_rows_per_rank_runtime = ctx.max_rows_per_rank_runtime
        ep_ws = ctx.ep_ws
        dispatch_mode = cfg.dispatch_mode

        H = ep_ws.d
        I = w2.shape[1]
        device, dtype = dout_local.device, dout_local.dtype

        # --- 1. Dispatch do_local -> dout_dispatched ---
        # x_symm's last reader was the forward's x dispatch, so backward safely reuses it to publish dout.
        do_buf, do_hdl, do_peer_bufs = ep_ws.x_symm, ep_ws.x_hdl, ep_ws.x_peer_bufs
        if _is_hier_node_dedup_dispatch_gin_mode(dispatch_mode):
            # HIER reads dO from the GIN window (not x_symm); stage it here. HIER's dispatch barriers
            # internally, so the x_symm publish/barrier dance below is skipped (do_buf/do_peer_bufs ignored).
            ep_ws.x_gin_bwd.tensor.view(T_local, H).copy_(dout_local)
        else:
            do_buf.copy_(dout_local)
            do_hdl.barrier()

        if _is_grouped_recv_layout(dispatch_mode):
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
            ep_ws=ep_ws,
            meta=meta,
            is_forward=False,
        )

        # --- 2. All-gather topk scores (or reuse the forward's cached AG) ---
        # Forward's RS/RANK_DEDUP-combine path already AG'd scores into ctx.scores_global; reuse it here
        # to avoid a redundant NCCL collective. A2A_COMBINE forward leaves it None -> fresh AG below.
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

        # --- 3. Down-proj backward act (gemm_dgated): dh, ds, a_prime ---
        dh = torch.empty_like(h)
        ds = torch.zeros(TK_global, dtype=topk_scores_global.dtype, device=device)
        a_prime = torch.empty(max_rows_per_rank_runtime, I, dtype=h.dtype, device=device)
        db2 = None if b2 is None else torch.empty_like(b2)

        # Same A_idx-by-mode split as the up-proj forward, but A2A also needs dout sliced to total_m
        # (others don't). dst_rank_flat/s_reverse_local are pass-invariant, so the up-proj's A_idx metadata is reused.
        if _is_a2a_dispatch_mode(dispatch_mode):
            a_idx_for_dout = None
            dout_for_kernel = dout_dispatched[:max_rows_per_rank_runtime]
        elif _is_grouped_recv_layout(dispatch_mode):
            a_idx_for_dout = x_idx_expanded_remap_for_rank_dedup[:max_rows_per_rank_runtime]
            dout_for_kernel = dout_dispatched
        else:
            a_idx_for_dout = x_gather_idx[:max_rows_per_rank_runtime]
            dout_for_kernel = dout_dispatched
        s_scatter_idx_local = s_scatter_idx[:max_rows_per_rank_runtime]
        # dst_rank_flat gates the ds-scatter to skip sentinel slots; only needed when max_rows_per_rank_runtime
        # can exceed real_total. cpu_synced already trims to real_total, so pass None there to skip the check.
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

        # --- 4. dW2 GEMM ---
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

        # --- 5. Reduce-scatter ds -> ds_local ---
        ds_local = torch.empty(T_local * K, dtype=ds.dtype, device=ds.device)
        dist.reduce_scatter_tensor(
            ds_local,
            ds,
            op=dist.ReduceOp.SUM,
            group=ep_ws.ep_group,
        )
        ds_local = ds_local.view(T_local, K)

        # --- 6. Up-proj backward act: dh -> dx_expanded (in y_symm), db1 ---
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

        # --- 7. Cross-rank combine of dx_expanded -> dx_local ---
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
            mine_slot_idx=combine_mine_slot_idx,
            mine_count=combine_mine_count,
            combine_contrib_C=combine_contrib_C,
            # dx combine is score-less (scores=None); singles read y_symm directly, multis use the work-list.
            combine_work_list=meta.get("combine_work_list_multi"),
            combine_work_count=meta.get("combine_work_count_multi"),
            combine_single_k=meta.get("combine_single_k"),
            single_row=meta.get("single_row"),
            meta=meta,
            is_forward=False,
        )

        # --- 8. dW1 GEMM ---
        # K dim is the slot count, bounded to total_m via cu_seqlens_k. Same A_idx-by-mode split as
        # the up-proj forward (A2A=None, RANK_DEDUP=remap, AG=x_gather_idx).
        if _is_a2a_dispatch_mode(dispatch_mode):
            a_idx_for_dw1 = None
        elif _is_grouped_recv_layout(dispatch_mode):
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
    """Per-expert metadata. Sentinels (peer-routed slots) bucket into expert id E_local; sliced to
    the first E_local+1 offset entries so downstream kernels iterate only real experts."""
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


def _build_single_row(
    topk_idx_global: torch.Tensor,
    combine_single_k: torch.Tensor,
    my_rank: int,
    E_local: int,
) -> torch.Tensor:
    """single_row[q,t] = peer q's y_symm row for token t's single contributor, else 0. Host-sync-free;
    verified bit-exact vs real s_reverse (verify_single_row.py) — replaces the gather's pointer-chase with one dense load."""
    device = topk_idx_global.device
    W, T_local, K = topk_idx_global.shape
    TK_local = T_local * K
    TK_global = W * TK_local
    E = W * E_local
    topk_flat = topk_idx_global.reshape(-1).to(torch.int32)
    tok = torch.arange(TK_global, dtype=torch.int32, device=device)  # feeds x_gather only
    buf = torch.empty(TK_global, dtype=torch.int32, device=device)
    cm = _build_consumer_metadata(topk_flat, tok, TK_global, E, buf)
    s_rev_global = cm["s_reverse_local"].to(torch.int64)
    global_offset = cm["expert_frequency_offset"].to(torch.int64)  # (E+1,)

    sk = combine_single_k.to(torch.int64)  # (W, T_local), -1 or k
    t_idx = torch.arange(T_local, device=device, dtype=torch.int64)
    pos = my_rank * TK_local + t_idx[None, :] * K + sk.clamp(min=0)  # (W, T_local)
    q_base = torch.arange(W, device=device, dtype=torch.int64)[:, None] * E_local  # (W,1)
    single_row = torch.where(sk >= 0, s_rev_global[pos] - global_offset[q_base], torch.zeros_like(pos))
    return single_row.to(torch.int32)


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
    # use_gin triggers emit of the node-axis dispatch tensors + combine present/contrib/expected-count
    # tensors (superset that also includes the rank-dedup tensors HIER reuses).
    use_gin = _is_hier_node_dedup_dispatch_gin_mode(dispatch_mode)

    # Combine-producer metadata (mine_slot_idx + work-list) is rank-dedup-only; built once here and
    # shared with the backward dx-combine via ctx.meta.
    meta = compute_dispatch_metadata(
        topk_idx_global,
        my_rank=my_rank,
        E_local=E_local,
        emit_combine=_is_rank_dedup_combine_mode(combine_mode),
        emit_hier=use_gin,
        node_size=(cfg.node_size if use_gin else 1),
    )
    # single_row is precomputed ONCE here and shared with the backward dx-combine via ctx.meta.
    if _is_rank_dedup_combine_mode(combine_mode):
        meta["single_row"] = _build_single_row(topk_idx_global, meta["combine_single_k"], my_rank, E_local)
    dst_rank_flat = meta["dst_rank_flat"]
    my_dst_rank = meta["my_dst_rank"]
    expert_local_padded = meta["expert_local_padded"]
    is_grouped_dispatch = _is_a2a_dispatch_mode(dispatch_mode) or _is_grouped_recv_layout(dispatch_mode)
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

    # A2A and RANK_DEDUP both need the expert-grouped recv_pos (s_reverse_local); RANK_DEDUP additionally
    # needs pair_present_mask/rank_dedup_recv_pos for the canonical-only peer pull.
    recv_pos = metadata["s_reverse_local"] if is_grouped_dispatch else None
    pair_present_mask = meta.get("pair_present_mask") if _is_grouped_recv_layout(dispatch_mode) else None
    rank_dedup_recv_pos = meta.get("rank_dedup_recv_pos") if _is_grouped_recv_layout(dispatch_mode) else None

    # peer_present_mask is rank-dedup-combine-only; None for other modes keeps the meta bundle small.
    needs_dedup_combine = _is_rank_dedup_combine_mode(combine_mode)
    peer_present_mask = meta.get("peer_present_mask") if needs_dedup_combine else None

    # RANK_DEDUP_DISPATCH_TRITON: build the up-proj A_idx that gathers expert-grouped rows from the packed dispatch buffer.
    x_idx_expanded_remap_for_rank_dedup: Optional[torch.Tensor] = None
    if _is_grouped_recv_layout(dispatch_mode):
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
    # RANK_DEDUP combine's producer/gather metadata (from emit_combine=True) + single_row; carried
    # through (None-safe) to the backward dx-combine via the meta bundle.
    if needs_dedup_combine:
        for _k in (
            "mine_slot_idx",
            "mine_count",
            "combine_contrib_C",
            "combine_work_list_multi",
            "combine_work_count_multi",
            "combine_single_k",
            "single_row",
        ):
            meta_bundle[_k] = meta.get(_k)
    # HIER adds the node-axis dispatch tensors (node_present/dst_node/dst_slot/dst_recv_count/is_local_slot)
    # + combine present/contrib/expected-count tensors (rank-dedup tensors already in the bundle via gating).
    if use_gin:
        for _k in (
            "node_present_mask",
            "dst_node_flat",
            "dst_slot",
            "dst_recv_count",
            "stripe_base",        # coalesced dispatch: per-(rank,node) stripe base offset
            "node_token_count",   # coalesced dispatch: per-(rank,node) stripe token count
            "is_local_slot",
            "combine_peer_present_all",
            "contrib_node_mask",
            "expected_count_combine",
        ):
            meta_bundle[_k] = meta.get(_k)
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
    """Heuristic (dispatch, combine) pick without profiling: W<=2 or W<=0.8*K -> AG+RS; K>=1.25*W -> A2A+A2A
    (same inequality as the first branch, so actually unreachable — kept for rule-parity); else rank-dedup."""
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
    """EP forward with TC softmax-topk routing; router backward all-reduces drouter_w across the EP group.
    Workspace caches per (group, x.device) — see clear_ep_cache(); ep_config picks dispatch/combine (default heuristic, or pass NetworkProfiler.profile())."""
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

    # Build a fresh cfg (not mutate ep_config, which is frozen and may be reused across layers with
    # different I/E_local/T_local) with the layer-static fields _moe_ep_forward_inner/_MoeEPFunction need.
    _I = w1.shape[0]
    _is_glu_act = is_glu(activation_type)
    if _is_glu_act:
        _I //= 2
    # Workspace built BEFORE cfg so the GIN path can self-derive node_size/num_nodes from the NVLink/LSA
    # domain (ws.node_size/ws.num_nodes); flat path defaults to 1 (byte-identical to the old behavior).
    ws = mgr._get_or_alloc(T_local, d, K, E_local, x.dtype, ep_config.dispatch_mode)
    _node_size = ws.node_size if ws.node_size else 1
    cfg = RuntimeEPConfig(
        dispatch_mode=ep_config.dispatch_mode,
        W=W,
        K=K,
        combine_mode=ep_config.combine_mode,
        I=_I,
        is_glu_act=_is_glu_act,
        E_local=E_local,
        MAX_ROWS_PER_RANK_STATIC=T_local * W * min(K, E_local),
        num_nodes=ws.num_nodes,
        node_size=_node_size,
        node_id=mgr.rank // _node_size,
        local_id=mgr.rank % _node_size,
        use_gin=ep_config.use_gin,
    )

    # Router projection with EP-aware drouter_w all-reduce.
    router_logits = EP_Router_Replicated_Across_Ranks.apply(x, router_w, ws.ep_group)
    topk_scores_l, topk_idx_l = TC_Softmax_Topk_Router_Function.apply(
        router_logits, W * E_local, K, is_softmax_over_topk, norm_topk_probs
    )

    # Publish x to peers (the forward dispatch inside _moe_ep_forward_inner reads peer x_symm) before AG-ing topk_idx.
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
    """EP forward with caller-supplied top-K routing; same workspace/cache semantics as
    moe_ep_TC_softmax_topk_forward. ep_config defaults via _default_ep_config using K=topk_indices.shape[1]."""
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

    # Build the call-local cfg (see moe_ep_TC_softmax_topk_forward for the fresh-instance rationale).
    _I = w1.shape[0]
    _is_glu_act = is_glu(activation_type)
    if _is_glu_act:
        _I //= 2
    # Workspace built BEFORE cfg so the GIN path can self-derive node_size/num_nodes from the NVLink/LSA
    # domain (ep_ws.node_size/ep_ws.num_nodes); flat path defaults to 1 (byte-identical to old behavior).
    ep_ws = mgr._get_or_alloc(T_local, d, K, E_local, x.dtype, ep_config.dispatch_mode)
    _node_size = ep_ws.node_size if ep_ws.node_size else 1
    cfg = RuntimeEPConfig(
        dispatch_mode=ep_config.dispatch_mode,
        W=W,
        K=K,
        combine_mode=ep_config.combine_mode,
        I=_I,
        is_glu_act=_is_glu_act,
        E_local=E_local,
        MAX_ROWS_PER_RANK_STATIC=T_local * W * min(K, E_local),
        num_nodes=ep_ws.num_nodes,
        node_size=_node_size,
        node_id=mgr.rank // _node_size,
        local_id=mgr.rank % _node_size,
        use_gin=ep_config.use_gin,
    )
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
