# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
#
# Expert Parallelism (EP) — AG dispatch + gather GEMM, or pull-based A2A
# dispatch with either gather or nogather GEMM. AG-pattern NVLink combine.
#
# Public surface:
#
#   SymmMemManager(ep_group, device)
#   moe_ep_TC_softmax_topk_forward(...)
#   moe_ep_general_routing_forward(...)
#
# ----------------------------------------------------------------------------
# Threading-safe symm-mem (deadlock fix — Round 4)
# ----------------------------------------------------------------------------
# PyTorch's autograd engine runs Function.backward() on a worker thread by
# default, and `_symm_mem.rendezvous(buf, group)` has been observed to
# deadlock when invoked from that thread — even on cache hits — because
# the lookup goes through Python+C++ machinery whose thread-coupling is
# fragile in combination with NCCL group state.
#
# Round-1 fix (eager allocation): rendezvous all symm-mem buffers from
# the main thread at workspace setup, instead of lazy-allocating on
# first backward.
#
# Round-2 fix (cached handles): cache the rendezvous handle on the
# workspace and use `hdl.barrier()` directly, so explicit barriers in
# backward never re-enter rendezvous.
#
# Round-3 fix (THIS round): also pass the cached handle into every
# `triton_comm` collective wrapper (`all_gather_triton`,
# `reduce_scatter_triton`, `gather_aggregation_triton`,
# `a2a_dispatch_pull_triton`) called from backward, so those wrappers
# also skip their internal rendezvous. Combined effect: ZERO calls to
# `_symm_mem.rendezvous` from the autograd backward thread.
#
# This requires the `hdl=` (and `hdl_y=`/`hdl_s=` for gather_aggregation)
# parameter on the triton_comm wrappers. See triton_comm_hdl_patch.py.
#
# ----------------------------------------------------------------------------
# Naming
# ----------------------------------------------------------------------------
#   T_local     tokens per rank
#   K           top-K experts per token
#   TK_local    T_local * K
#   W           EP world size
#   TK_global   W * TK_local
#   E_local     experts per rank (E // W)
#
# ----------------------------------------------------------------------------
# Forward dispatch + GEMM modes
# ----------------------------------------------------------------------------
# "ag"  — All-gather x to all peers (W*T_local rows). Always paired with
#         the gather GEMM (concat_layout=False).
# "a2a" — Pull-based A2A dispatch.
#         flat   (concat_layout=False): recv_pos = a2a_token_indices.
#         sorted (concat_layout=True):  recv_pos = s_reverse_local.
#
# Mode is fixed for a workspace's lifetime.
#
# ----------------------------------------------------------------------------
# Function ownership
# ----------------------------------------------------------------------------
# `_UpProjectionEP`   owns:  dispatch-X       → up-proj GEMM   → (a, h)
#                     bwd:    up-proj-act-grad → dW₁ → cross-rank reverse
#
# `_DownProjectionEP` owns:  down-proj GEMM   → forward combine → o_local
#                     bwd:    dispatch-dO + AG-scores → down-proj-act-grad
#                             → dW₂ → reduce-scatter ds
#
# Together they implement the full forward+backward chain. `dh` flows from
# _DownProjectionEP.backward into _UpProjectionEP.backward via autograd.
#
# ----------------------------------------------------------------------------
# Sentinels-at-end metadata layout
# ----------------------------------------------------------------------------
# Sentinel slots — those whose expert lives on a peer rank — are assigned
# expert id `E_local`. After the routing kernel sorts by expert id (with
# `E_total = E_local + 1` bins), sentinels land in a single trailing bin.
# `_build_consumer_metadata` slices `expert_frequency_offset[:E_local + 1]`
# before returning, so every grouped GEMM and db1/db2/ds kernel iterates
# only [0, real_total). No post-hoc masking required.
#
# ----------------------------------------------------------------------------
# Buffer reuse
# ----------------------------------------------------------------------------
# y_symm is idle between forward combine and backward up-proj-act-grad,
# so `_UpProjectionEP.bwd` writes dx_expanded into y_symm in place.
#
# `_bwd_stage_symm` is shared between `_DownProjectionEP.bwd` (stages
# dout, dispatched first) and `_UpProjectionEP.bwd` (stages x_local for
# re-dispatch, second). They run sequentially; one buffer instead of two.
#
# ----------------------------------------------------------------------------
# x_compute lifetime — `redispatch_x_in_backward` flag
# ----------------------------------------------------------------------------
#   False (default, cache):  forward dispatches into a fresh tensor saved
#                            on ctx. +(TK_global, d) memory.
#   True (re-dispatch):      forward uses workspace buffer; backward redoes
#                            the dispatch from saved x_local using
#                            _bwd_stage_symm. +(T_local, d) symm-mem.
#
# ----------------------------------------------------------------------------
# Caller contract
# ----------------------------------------------------------------------------
# Inputs are assumed aligned and well-shaped; forward does not re-validate.
# ********************************************************************************

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from quack.gemm_interface import gemm, gemm_gated
from torch.distributed import _symmetric_memory as _symm_mem

from sonicmoe.functional import TC_Softmax_Topk_Router_Function
from sonicmoe.functional.backward import _down_projection_backward_act, _up_projection_backward_act
from sonicmoe.functional.triton_kernels import general_routing_router_metadata_triton

from .enums import ActivationType, is_glu
from .functional.ep.triton_comm import (
    a2a_dispatch_pull_triton,
    all_gather_copy_engine_async,
    all_gather_triton,
    compute_dispatch_metadata,
    gather_aggregation_triton,
)


__all__ = [
    "SymmMemManager",
    "moe_ep_TC_softmax_topk_forward",
    "moe_ep_general_routing_forward",
]


# ============================================================================
# Mode selection
# ============================================================================

_VALID_MODES = ("ag", "a2a")


def _select_dispatch_mode(W: int, K: int) -> str:
    if W <= 1:
        return "ag"
    cap_major = torch.cuda.get_device_capability()[0]
    if cap_major >= 10:
        return "a2a" if K < W else "ag"
    if cap_major == 9:
        return "a2a" if 5 * K < 4 * W else "ag"
    return "ag"


def _resolve_mode(mode: str, W: int, K: int) -> str:
    if mode == "auto":
        return _select_dispatch_mode(W, K)
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be 'ag', 'a2a', or 'auto'; got {mode!r}")
    return mode


def _normalize_activation(activation_type) -> ActivationType:
    if isinstance(activation_type, str):
        return ActivationType[activation_type.upper()]
    return activation_type


# ============================================================================
# Shared dispatch helper — accepts the SOURCE buffer's cached handle.
# ============================================================================


def _do_dispatch(
    src_symm: torch.Tensor,
    out_buf: torch.Tensor,
    mode: str,
    *,
    dst_rank_flat: Optional[torch.Tensor],
    recv_pos: Optional[torch.Tensor],
    K: int,
    group: dist.ProcessGroup,
    TK_global: int,
    H: int,
    src_hdl: Any,  # cached handle for src_symm
) -> torch.Tensor:
    """Single dispatch step (AG or A2A pull) writing into out_buf.

    `src_hdl` is the pre-fetched rendezvous handle for `src_symm`. Always
    passed through, so the wrapper can skip its internal rendezvous.
    """
    if mode == "ag":
        return all_gather_triton(src_symm, group, out=out_buf, hdl=src_hdl)
    a2a_dispatch_pull_triton(
        x_symm=src_symm,
        dst_rank_flat=dst_rank_flat,
        recv_pos=recv_pos,
        recv=out_buf,
        K=K,
        group=group,
        hdl=src_hdl,
    )
    return out_buf.view(TK_global, H)


# ============================================================================
# _UpProjectionEP — owns dispatch-X (forward) + reverse-dispatch (backward)
# ============================================================================


class _UpProjectionEP(torch.autograd.Function):
    """EP up-projection. Forward returns (a, h). Backward returns
    dx_local + dW₁ + db1 via score-less cross-rank gather of peer
    dx_expanded buffers."""

    @staticmethod
    def forward(
        ctx,
        x_local: torch.Tensor,
        w1: torch.Tensor,
        b1: Optional[torch.Tensor],
        expert_frequency_offset: torch.Tensor,
        x_gather_idx: torch.Tensor,
        my_dst_rank: torch.Tensor,
        recv_pos: Optional[torch.Tensor],
        dst_rank_flat: Optional[torch.Tensor],
        TK_global: int,
        T_local: int,
        K: int,
        activation_type: ActivationType,
        is_inference_mode_enabled: bool,
        concat_layout: bool,
        redispatch_x_in_backward: bool,
        ep_ws,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        H = x_local.shape[1]
        mode = ep_ws.mode
        device, x_dtype = x_local.device, x_local.dtype

        # --- Forward dispatch — uses cached x_hdl.
        if is_inference_mode_enabled or redispatch_x_in_backward:
            ws_buf = ep_ws.ag_compute if mode == "ag" else ep_ws.a2a_recv
            x_compute = _do_dispatch(
                ep_ws.x_symm,
                ws_buf,
                mode,
                dst_rank_flat=dst_rank_flat,
                recv_pos=recv_pos,
                K=K,
                group=ep_ws.ep_group,
                TK_global=TK_global,
                H=H,
                src_hdl=ep_ws.x_hdl,
            )
        else:
            if mode == "ag":
                fresh = torch.empty(ep_ws.W * T_local, H, dtype=x_dtype, device=device)
            else:
                fresh = torch.empty(ep_ws.W, T_local * K, H, dtype=x_dtype, device=device)
            x_compute = _do_dispatch(
                ep_ws.x_symm,
                fresh,
                mode,
                dst_rank_flat=dst_rank_flat,
                recv_pos=recv_pos,
                K=K,
                group=ep_ws.ep_group,
                TK_global=TK_global,
                H=H,
                src_hdl=ep_ws.x_hdl,
            )

        # --- Up-proj fused GEMM + gated activation (matches non-EP _UpProjection).
        I = w1.shape[0]
        is_glu_act = is_glu(activation_type)
        if is_glu_act:
            I //= 2

        a = torch.empty(TK_global, I, dtype=x_dtype, device=device)
        h = (
            torch.empty(TK_global, (2 * I if is_glu_act else I), dtype=x_dtype, device=device)
            if not is_inference_mode_enabled
            else None
        )

        # Gather vs non-gather GEMM, paired with the dispatch:
        #   ag                            → ag_compute is (W*T_local, d), NOT
        #                                   expert-sorted → gather GEMM (A_idx).
        #   a2a + concat_layout=True      → a2a recv_pos=s_reverse_local writes
        #     (sorted recv layout)          rows in expert-sorted order →
        #                                   non-gather GEMM (no A_idx), like
        #                                   the down-proj.
        #   a2a + concat_layout=False     → recv_pos=a2a_token_indices writes
        #     (flat recv layout)            in per-rank-slot order, NOT expert
        #                                   sorted → gather GEMM (A_idx).
        x_compute_is_grouped = mode == "a2a" and concat_layout
        a_idx_for_up = None if x_compute_is_grouped else x_gather_idx

        assert activation_type.value in (
            "swiglu",
            "geglu",
        ), f"QuACK gemm_gated only supports glu activations, got {activation_type.value}"
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

        if not is_inference_mode_enabled:
            if redispatch_x_in_backward:
                ctx.save_for_backward(
                    x_local,
                    w1,
                    b1,
                    expert_frequency_offset,
                    x_gather_idx,
                    my_dst_rank,
                    recv_pos,
                    dst_rank_flat,
                )
            else:
                ctx.save_for_backward(
                    x_compute,
                    w1,
                    b1,
                    expert_frequency_offset,
                    x_gather_idx,
                    my_dst_rank,
                    None,
                    None,
                )
            ctx.is_glu_act = is_glu_act
            ctx.concat_layout = concat_layout
            ctx.redispatch_x_in_backward = redispatch_x_in_backward
            ctx.K = K
            ctx.T_local = T_local
            ctx.TK_global = TK_global
            ctx.ep_ws = ep_ws
            ctx.mark_non_differentiable(a)
            ctx.set_materialize_grads(False)

        return a, h

    @staticmethod
    def backward(ctx, _: None, dh: Optional[torch.Tensor]):
        if dh is None:
            return (None,) * 16

        (
            x_or_xlocal,
            w1,
            b1,
            expert_frequency_offset,
            x_gather_idx,
            my_dst_rank,
            recv_pos,
            dst_rank_flat,
        ) = ctx.saved_tensors
        is_glu_act = ctx.is_glu_act
        concat_layout = ctx.concat_layout
        redispatch = ctx.redispatch_x_in_backward
        K = ctx.K
        T_local = ctx.T_local
        TK_global = ctx.TK_global
        ep_ws = ctx.ep_ws
        mode = ep_ws.mode

        # ────────────────────────────────────────────────────────────────
        # Backward schedule (with `redispatch_x_in_backward=True`):
        #
        #   t0   issue AG-of-x via copy engines (async — runs on CE
        #        streams in parallel with the main torch stream)
        #   t1   _up_projection_backward_act: dh@w1 → dx_expanded, db1
        #        (only needs dh + w1 + b1; x_compute not yet needed)
        #   t2   cross-rank gather of dx_expanded → dx_local
        #        (only reads peer.y_symm + peer.s_rev_symm; CE-engine
        #         AG can keep running in the background)
        #   t3   wait on CE handle → x_compute materialized
        #   t4   dW₁ = x_compute.T @ dh  ← syncs here, last step
        #
        # The CE-async AG keeps NVLink busy on the copy engines while
        # the SMs work on Step 1 + Step 3, and dW₁ (the heaviest
        # compute) lands at the end with x_compute already in HBM.
        # ────────────────────────────────────────────────────────────────

        # --- t0: issue redispatch X.
        # Always all-gather + copy-engine for the redispatch path.
        # The forward saved x_local (T_local, H) instead of x_compute
        # (W*T_local, H) to save (W-1)/W of the activation memory; we
        # re-dispatch here to recover x_compute. Forcing AG (rather
        # than honoring the forward's mode) keeps this path simple and
        # serves only the gather-GEMM dW₁ contract that wants
        # (W*T_local, H) with x_gather_idx values pointing into it.
        ce_handle = None
        if redispatch:
            x_local = x_or_xlocal
            H = x_local.shape[1]
            stage = ep_ws._bwd_stage_symm
            stage.copy_(x_local)
            ep_ws._bwd_stage_hdl.barrier()  # fence stage before peer reads
            assert mode == "ag", (
                "redispatch_x_in_backward currently requires mode='ag' "
                "so the AG-copy-engine redispatch matches the saved "
                f"x_gather_idx layout (got mode={mode!r})."
            )
            ce_handle = all_gather_copy_engine_async(
                stage,
                ep_ws.ep_group,
                out=ep_ws.ag_compute,
            )
            x_compute = ce_handle.out  # not yet ready — read at t3
        else:
            x_compute = x_or_xlocal
            H = x_compute.shape[1]

        device, dtype = dh.device, dh.dtype
        dw1 = torch.empty_like(w1)
        db1 = None if b1 is None else torch.empty_like(b1)

        # --- t1: dx_expanded + db1 into y_symm (reused buffer).
        dx_expanded_symm = ep_ws.y_symm
        _up_projection_backward_act(
            w1=w1,
            dx_expanded=dx_expanded_symm,
            dh=dh,
            db1=db1,
            expert_frequency_offset=expert_frequency_offset,
            is_glu_activation=is_glu_act,
            concat_layout=concat_layout,
        )

        # --- t2: cross-rank gather → dx_local. Score-less mode.
        # Same per-buffer fence pattern as the forward gather: the gather
        # reads peer.dx_expanded_symm (= peer.y_symm) AND peer.s_rev_symm,
        # so both buffers' handles must be barriered. y_hdl covers the
        # _up_projection_backward_act write above; s_rev_hdl covers the
        # forward-time metadata write that's still resident in s_rev_symm.
        ep_ws.s_rev_hdl.barrier()
        ep_ws.y_hdl.barrier()  # y_symm is reused as dx_expanded_symm

        dx_local = torch.empty(T_local, H, dtype=dtype, device=device)
        gather_aggregation_triton(
            dx_expanded_symm,
            ep_ws.s_rev_symm,
            my_dst_rank,
            None,  # ← score-less
            dx_local,
            K=K,
            group=ep_ws.ep_group,
            hdl_y=ep_ws.y_hdl,
            hdl_s=ep_ws.s_rev_hdl,
        )

        # --- t3: sync the CE-async AG so x_compute is ready for dW₁.
        if ce_handle is not None:
            ce_handle.wait()

        # --- t4: dW₁. Gather/non-gather pairing matches forward up-proj
        # (see comment there): ag and a2a-flat use gather (A_idx); a2a-sorted
        # is already expert-grouped → non-gather.
        x_compute_is_grouped = mode == "a2a" and concat_layout
        a_idx_for_dw1 = None if x_compute_is_grouped else x_gather_idx
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

        return (
            dx_local,
            dw1,
            db1,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# ============================================================================
# _DownProjectionEP — owns down-proj GEMM + forward combine + full backward
# ============================================================================


class _DownProjectionEP(torch.autograd.Function):
    """EP down-projection + forward combine.

    Forward output: y_local (T_local, H).
    Backward output: (None for a, dh, dw2, db2, ds_local, *None×12).
    """

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        h: Optional[torch.Tensor],
        w2: torch.Tensor,
        b2: Optional[torch.Tensor],
        topk_scores_local: torch.Tensor,
        expert_frequency_offset: torch.Tensor,
        x_gather_idx: torch.Tensor,
        s_scatter_idx: torch.Tensor,
        dst_rank_flat: torch.Tensor,
        my_dst_rank: torch.Tensor,
        recv_pos: Optional[torch.Tensor],
        T_local: int,
        K: int,
        TK_global: int,
        activation_type: ActivationType,
        is_inference_mode_enabled: bool,
        concat_layout: bool,
        ep_ws,
    ) -> torch.Tensor:
        H = ep_ws.d
        device, dtype = a.device, a.dtype

        # --- Down-proj GEMM into y_symm. w2 already in (E_local, I, H).
        gemm(
            a,
            w2,
            out=ep_ws.y_symm,
            cu_seqlens_m=expert_frequency_offset,
            bias=b2,
            dynamic_scheduler=False,
        )

        # --- B3: per-buffer barriers covering BOTH peer reads in the gather.
        # gather_aggregation_triton reads from peer.y_symm AND peer.s_rev_symm.
        # y_symm was just written above; s_rev_symm was written much earlier
        # by general_routing_router_metadata_triton in _build_consumer_metadata
        # and has NOT been barriered since. Without s_rev_hdl.barrier() here,
        # faster ranks read peer.s_rev_symm via NVLink before peers' metadata
        # kernel has flushed, getting stale row indices that dereference into
        # the wrong rows of peer.y_symm. Symptom at large T: max(o_diff) ≈
        # max(o_ref) — output for affected tokens is the value of an
        # unrelated row, not bf16-precision noise. Both barriers are needed.
        ep_ws.s_rev_hdl.barrier()
        ep_ws.y_hdl.barrier()

        # --- Forward combine (score-weighted gather of peer.y_symm).
        y_local = torch.empty(T_local, H, dtype=dtype, device=device)
        gather_aggregation_triton(
            ep_ws.y_symm,
            ep_ws.s_rev_symm,
            my_dst_rank,
            topk_scores_local,
            y_local,
            K=K,
            group=ep_ws.ep_group,
            hdl_y=ep_ws.y_hdl,
            hdl_s=ep_ws.s_rev_hdl,
        )

        if not is_inference_mode_enabled:
            # The backward now AGs `topk_scores_local` via NCCL on a
            # regular HBM tensor — no symm-mem `scores_symm` buffer
            # involved, so no forward-time staging is needed here.
            ctx.save_for_backward(
                h,
                w2,
                b2,
                topk_scores_local,
                expert_frequency_offset,
                x_gather_idx,
                s_scatter_idx,
                dst_rank_flat,
                my_dst_rank,
                recv_pos,
            )
            ctx.activation_type = activation_type
            ctx.K = K
            ctx.T_local = T_local
            ctx.TK_global = TK_global
            ctx.concat_layout = concat_layout
            ctx.ep_ws = ep_ws
            ctx.set_materialize_grads(False)

        return y_local

    @staticmethod
    def backward(ctx, dout: Optional[torch.Tensor]):
        if dout is None:
            return (None,) * 18

        (
            h,
            w2,
            b2,
            topk_scores_local,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            dst_rank_flat,
            my_dst_rank,
            recv_pos,
        ) = ctx.saved_tensors
        activation_type = ctx.activation_type
        K = ctx.K
        T_local = ctx.T_local
        TK_global = ctx.TK_global
        ep_ws = ctx.ep_ws
        mode = ep_ws.mode
        H = dout.shape[1]
        I = w2.shape[1]
        device, dtype = dout.device, dout.dtype

        # --- Step 1: stage dout into _bwd_stage_symm.
        stage = ep_ws._bwd_stage_symm
        stage.copy_(dout)
        ep_ws._bwd_stage_hdl.barrier()  # B_bwd_d (fences _bwd_stage_symm)

        # --- Step 2: dispatch dout. Uses cached _bwd_stage_hdl.
        ws_buf = ep_ws.ag_compute if mode == "ag" else ep_ws.a2a_recv
        dout_dispatched = _do_dispatch(
            stage,
            ws_buf,
            mode,
            dst_rank_flat=dst_rank_flat,
            recv_pos=recv_pos,
            K=K,
            group=ep_ws.ep_group,
            TK_global=TK_global,
            H=H,
            src_hdl=ep_ws._bwd_stage_hdl,
        )

        # --- Step 3: AG scores → flat (TK_global,) via NCCL on the
        # regular HBM `topk_scores_local` tensor. Same justification as
        # the routing-decision AG in `_ag_routing_decision` — see
        # `benchmarks/ep/bench-meta-allgather.py`: NCCL on regular HBM
        # is faster than copy-to-symm + triton AG, and doesn't need
        # the symm-mem `scores_symm` buffer.
        W_topk = dist.get_world_size(ep_ws.ep_group)
        topk_scores_global = torch.empty(
            W_topk * T_local * K,
            dtype=topk_scores_local.dtype,
            device=topk_scores_local.device,
        )
        dist.all_gather_into_tensor(
            topk_scores_global,
            topk_scores_local.contiguous().view(-1),
            group=ep_ws.ep_group,
        )

        # --- Step 4: dh, ds, db2, a_prime via the standard kernel.
        dh = torch.empty_like(h)
        ds = torch.zeros(TK_global, dtype=topk_scores_global.dtype, device=device)
        a_prime = torch.empty(TK_global, I, dtype=h.dtype, device=device)
        db2 = None if b2 is None else torch.empty_like(b2)

        # Gather/non-gather pairing matches the forward up-proj:
        #   ag / a2a-flat  → dout_dispatched is NOT expert-sorted → gather GEMM
        #   a2a + concat=T → dout_dispatched lands expert-sorted (recv_pos =
        #                    s_reverse_local) → non-gather GEMM, like the
        #                    forward down-proj.
        # `dst_rank_flat` + `my_rank` go to `_down_projection_backward_act`
        # so its internal scatter into `ds` skips sentinel slot positions
        # in one Triton kernel — replaces the previous broken-tail scatter
        # + post-hoc `ds.masked_fill_` torch sequence.
        x_compute_is_grouped = mode == "a2a" and ctx.concat_layout
        a_idx_for_dout = None if x_compute_is_grouped else x_gather_idx
        my_rank_id = dist.get_rank(ep_ws.ep_group)
        _down_projection_backward_act(
            dout=dout_dispatched,
            h=h,
            w2=w2.permute(2, 1, 0),  # (E_local, I, H) → (H, I, E_local) view
            # for kernel's internal permute.
            dh=dh,
            ds=ds,
            b2=b2,
            db2=db2,
            a_prime=a_prime,
            topk_scores=topk_scores_global,
            expert_frequency_offset=expert_frequency_offset,
            x_gather_idx=a_idx_for_dout,
            s_scatter_idx=s_scatter_idx,
            activation_type=activation_type.value,
            dst_rank_flat=dst_rank_flat,
            my_rank=my_rank_id,
        )

        # --- Step 5: dW₂. dw2 has shape (E_local, I, H).
        # Same gather/non-gather pairing as the gemm_dgated above.
        dw2 = torch.empty_like(w2)
        gemm(
            dout_dispatched.T,
            a_prime,
            out=dw2.permute(0, 2, 1),
            cu_seqlens_k=expert_frequency_offset,
            A_idx=a_idx_for_dout,
            batch_idx_permute=None,
            dynamic_scheduler=False,
        )

        # --- Step 6: reduce-scatter ds via NCCL on the regular HBM tensor.
        #
        # `ds` is a regular HBM tensor produced by `_scatter_ds_kernel`.
        # Earlier the path was: copy ds into `ds_global_symm` (symm-mem)
        # → fence → `reduce_scatter_triton` (symm-mem-based).
        #
        # `benchmarks/ep/bench-ds-allreduce.py` shows that NCCL's
        # `reduce_scatter_tensor` on the regular tensor is ~2× faster
        # at production scale (W=4, T=131072, K=8 → 4 MiB):
        #   triton RS path (with symm copy + barrier):  ~58 µs
        #   NCCL reduce_scatter_tensor (no copy):       ~27 µs
        # Smaller scales (≤2 MiB) also favour NCCL once the copy
        # overhead is included. multimem_all_reduce_ on symm-mem only
        # wins below ~1 MiB AND only if the producer scatter is
        # refactored to write directly into ds_global_symm.
        ds_local = torch.empty(
            T_local * K,
            dtype=ds.dtype,
            device=ds.device,
        )
        dist.reduce_scatter_tensor(
            ds_local,
            ds,
            op=dist.ReduceOp.SUM,
            group=ep_ws.ep_group,
        )
        # autograd expects this gradient to match `topk_scores_local`
        # which is shape (T_local, K).
        ds_local = ds_local.view(T_local, K)

        return (
            None,  # a
            dh,  # h
            dw2,  # w2
            db2,  # b2
            ds_local,  # topk_scores_local
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# ============================================================================
# Internal workspace — holds buffers + cached handles. All handles are
# obtained via `_symm_mem.rendezvous` on the main thread inside
# `SymmMemManager._alloc_workspace`. They are passed to every triton_comm
# wrapper called from forward AND backward, so no rendezvous (cached or
# otherwise) ever runs from the autograd backward thread.
# ============================================================================


@dataclass
class _EPWorkspace:
    """Symm-mem buffers + cached rendezvous handles + static patterns.

    Routing-decision and routing-scores AGs no longer go through
    symm-mem buffers (`idx_symm` / `scores_symm` were deleted) — they
    use NCCL `all_gather_into_tensor` on the regular HBM tensors
    directly. See `_ag_routing_decision` and the down-proj backward
    Step 3 comment for the bench numbers driving that choice.
    """

    # Forward symm-mem buffers + their rendezvous handles.
    x_symm: torch.Tensor
    x_hdl: Any
    y_symm: torch.Tensor  # also dx_expanded_symm in bwd
    y_hdl: Any  # used for B3 + B_bwd_dx
    s_rev_symm: torch.Tensor
    s_rev_hdl: Any

    ep_group: dist.ProcessGroup
    E_local: int
    _T_local: int
    _K: int
    mode: str = "ag"

    # Dispatch destinations (regular tensors, not symm-mem).
    a2a_recv: Optional[torch.Tensor] = None
    ag_compute: Optional[torch.Tensor] = None

    # Static patterns.
    # `pos_2d_pattern` (= arange(TK_local) + my_rank * TK_local, reshaped to
    # (T_local, K)) was previously cached here for `gather_aggregation_triton`.
    # The kernel now computes that index inline from `pid_t`, `K`, and
    # `my_rank * TK_local` — no per-call tensor allocation, no HBM round-trip.
    t_global_pattern: Optional[torch.Tensor] = None

    # Backward symm-mem buffers + their rendezvous handles.
    _bwd_stage_symm: Optional[torch.Tensor] = None
    _bwd_stage_hdl: Optional[Any] = None

    @property
    def W(self) -> int:
        return dist.get_world_size(self.ep_group)

    @property
    def T_local(self) -> int:
        return self._T_local

    @property
    def K(self) -> int:
        return self._K

    @property
    def d(self) -> int:
        return self.y_symm.shape[1]


# ============================================================================
# SymmMemManager
# ============================================================================


class SymmMemManager:
    def __init__(self, ep_group: dist.ProcessGroup, device: torch.device):
        self.ep_group = ep_group
        self.device = torch.device(device)
        gname = getattr(ep_group, "group_name", None)
        if gname is None:
            gname = "0" if ep_group is dist.group.WORLD else None
        if gname is None:
            raise RuntimeError("Cannot resolve symm-mem group name for ep_group.")
        self._group_name = gname
        self._cache: dict = {}

    @property
    def world_size(self) -> int:
        return dist.get_world_size(self.ep_group)

    @property
    def rank(self) -> int:
        return dist.get_rank(self.ep_group)

    def clear(self) -> None:
        self._cache.clear()

    def prewarm(
        self,
        T_locals: Sequence[int],
        *,
        d: int,
        K: int,
        E_local: int,
        dtype: torch.dtype,
        mode: str,
    ) -> None:
        resolved = _resolve_mode(mode, self.world_size, K)
        for T in T_locals:
            self._get_or_alloc(T, d, K, E_local, dtype, resolved)

    def _alloc_symm(self, shape: Tuple[int, ...], dtype: torch.dtype) -> Tuple[torch.Tensor, Any]:
        """Allocate a symm-mem tensor and rendezvous it. Returns (buf, hdl).

        Always called from the main thread during workspace setup. The
        returned hdl is cached on `_EPWorkspace` and passed into every
        triton_comm wrapper used by forward AND backward, so no further
        rendezvous calls ever happen anywhere — in particular, none from
        the autograd backward thread."""
        buf = _symm_mem.empty(*shape, dtype=dtype, device=self.device)
        hdl = _symm_mem.rendezvous(buf, group=self._group_name)
        return buf, hdl

    def _alloc_workspace(
        self,
        T_local: int,
        d: int,
        K: int,
        E_local: int,
        dtype: torch.dtype,
        mode: str,
    ) -> _EPWorkspace:
        W = self.world_size
        my_rank = self.rank
        TK_local = T_local * K
        TK_global = W * TK_local
        dev = self.device

        # Forward symm-mem (handles cached for hdl.barrier() use AND for
        # passing to triton_comm wrappers).
        # Note: routing-decision and routing-scores AGs no longer use
        # symm-mem — `_ag_routing_decision` and the down-proj backward
        # both call NCCL `all_gather_into_tensor` on the regular HBM
        # tensors directly, which is faster than the prior
        # `idx_symm.copy_() + barrier + all_gather_triton` path.
        x_symm, x_hdl = self._alloc_symm((T_local, d), dtype)
        y_symm, y_hdl = self._alloc_symm((TK_global, d), dtype)
        s_rev_symm, s_rev_hdl = self._alloc_symm((TK_global,), torch.int32)

        # Backward symm-mem (also eagerly allocated + handles cached).
        # `_bwd_stage_symm` stages dout (and, in the redispatch_x path,
        # x_local) before the cross-rank dispatch. `ds` itself is now
        # reduced via NCCL `reduce_scatter_tensor` on the regular HBM
        # tensor — no symm-mem ds_global buffer needed (see Step 6 in
        # `_DownProjectionEP.backward`).
        bwd_stage_symm, bwd_stage_hdl = self._alloc_symm((T_local, d), dtype)

        a2a_recv = None
        ag_compute = None
        t_global_pattern = None

        if mode == "a2a":
            a2a_recv = torch.empty((W, TK_local, d), dtype=dtype, device=dev)
        else:
            ag_compute = torch.empty((W * T_local, d), dtype=dtype, device=dev)
            t_global_pattern = torch.arange(TK_global, device=dev, dtype=torch.int32) // K

        return _EPWorkspace(
            x_symm=x_symm,
            x_hdl=x_hdl,
            y_symm=y_symm,
            y_hdl=y_hdl,
            s_rev_symm=s_rev_symm,
            s_rev_hdl=s_rev_hdl,
            ep_group=self.ep_group,
            E_local=E_local,
            _T_local=T_local,
            _K=K,
            mode=mode,
            a2a_recv=a2a_recv,
            ag_compute=ag_compute,
            t_global_pattern=t_global_pattern,
            _bwd_stage_symm=bwd_stage_symm,
            _bwd_stage_hdl=bwd_stage_hdl,
        )

    def _get_or_alloc(
        self,
        T_local: int,
        d: int,
        K: int,
        E_local: int,
        dtype: torch.dtype,
        mode: str,
    ) -> _EPWorkspace:
        key = (T_local, d, K, E_local, str(dtype), mode)
        ws = self._cache.get(key)
        if ws is None:
            ws = self._alloc_workspace(T_local, d, K, E_local, dtype, mode)
            self._cache[key] = ws
        return ws


# ============================================================================
# Consumer-side metadata
# ============================================================================


def _build_consumer_metadata(
    expert_indices: torch.Tensor,
    token_indices: torch.Tensor,
    TK: int,
    E_local: int,
    s_reverse_idx_symm: torch.Tensor,
):
    """Build consumer-side per-expert metadata.

    SENTINELS-AT-END layout: phase 1 of `compute_dispatch_metadata` writes
    `expert_local_padded` with sentinel id == E_local, so the routing
    kernel — invoked here with `E_total = E_local + 1` bins — sorts
    sentinels into a single trailing bin. We slice
    `expert_frequency_offset[:E_local + 1]` before returning, so every
    downstream grouped GEMM and db1/db2/ds kernel iterates only
    [0, real_total). No post-hoc masking required.
    """
    device = expert_indices.device
    E_total = E_local + 1  # +1 sentinel bin

    expert_frequency_offset = torch.empty(E_total + 1, dtype=torch.int32, device=device)
    s_reverse_local = s_reverse_idx_symm[:TK]
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)

    expert_frequency = torch.empty(E_total, dtype=torch.int32, device=device)

    # `num_activated_expert_per_token_offset` (4th output of
    # `general_routing_router_metadata_triton`) is consumed only by the
    # non-EP `_token_broadcast_backward`. EP aggregates dx via cross-rank
    # gather and never reads it — pass None to skip the (TK+1)-int32
    # allocation and the parallel-searchsorted kernel launch.
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
        None,
    )

    expert_frequency_offset = expert_frequency_offset[: E_local + 1]

    return {
        "expert_frequency_offset": expert_frequency_offset,
        "s_reverse_local": s_reverse_local,
        "x_gather_idx": x_gather_idx,
        "s_scatter_idx": s_scatter_idx,
    }


# ============================================================================
# Core forward
# ============================================================================


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
) -> torch.Tensor:
    W, my_rank = ep_ws.W, dist.get_rank(ep_ws.ep_group)
    T_local, d, K, E_local = ep_ws.T_local, ep_ws.d, ep_ws.K, ep_ws.E_local
    TK_global = W * T_local * K
    mode = ep_ws.mode

    meta = compute_dispatch_metadata(topk_idx_global, my_rank=my_rank, E_local=E_local)
    dst_rank_flat = meta["dst_rank_flat"]
    my_dst_rank = meta["my_dst_rank"]
    expert_local_padded = meta["expert_local_padded"]
    a2a_token_indices = meta["a2a_token_indices"] if mode == "a2a" else None

    if mode == "ag":
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

    if mode == "a2a":
        recv_pos = metadata["s_reverse_local"] if concat_layout else a2a_token_indices
    else:
        recv_pos = None

    a, h = _UpProjectionEP.apply(
        x_local,
        w1,
        b1,
        metadata["expert_frequency_offset"],
        metadata["x_gather_idx"],
        my_dst_rank,
        recv_pos,
        dst_rank_flat,
        TK_global,
        T_local,
        K,
        activation_type,
        is_inference_mode_enabled,
        concat_layout,
        redispatch_x_in_backward,
        ep_ws,
    )

    y_local = _DownProjectionEP.apply(
        a,
        h,
        w2,
        b2,
        topk_scores_local,
        metadata["expert_frequency_offset"],
        metadata["x_gather_idx"],
        metadata["s_scatter_idx"],
        dst_rank_flat,
        my_dst_rank,
        recv_pos,
        T_local,
        K,
        TK_global,
        activation_type,
        is_inference_mode_enabled,
        concat_layout,
        ep_ws,
    )

    return y_local


# ============================================================================
# Public entry-point helpers
# ============================================================================


def _validate_and_resolve(
    mgr: "SymmMemManager",
    mode: str,
    E: int,
    K: int,
) -> Tuple[int, int, str]:
    W = mgr.world_size
    resolved = _resolve_mode(mode, W, K)
    if E % W != 0:
        raise ValueError(f"E ({E}) must be divisible by EP world size ({W}).")
    return W, E // W, resolved


def _ag_routing_decision(
    ep_ws: _EPWorkspace,
    topk_idx_l: torch.Tensor,
) -> torch.Tensor:
    """Routing-decision AG via NCCL on the regular HBM tensor.

    `benchmarks/ep/bench-meta-allgather.py` shows that NCCL
    `all_gather_into_tensor` on a regular HBM int32 tensor is ~2-3×
    faster than the prior `idx_symm.copy_() + barrier + all_gather_triton`
    path at every T we benched (W=4, K=8, T ∈ {8K, 16K, 32K, 64K, 128K}).
    multimem_all_gather_out is marginally faster (~1µs) but requires
    keeping `idx_symm` in symm-mem; the NCCL path needs no symm-mem
    buffer at all.
    """
    W = dist.get_world_size(ep_ws.ep_group)
    T_local, K = topk_idx_l.shape
    out = torch.empty(
        W * T_local,
        K,
        dtype=topk_idx_l.dtype,
        device=topk_idx_l.device,
    )
    dist.all_gather_into_tensor(out, topk_idx_l.contiguous(), group=ep_ws.ep_group)
    return out.view(W, T_local, K)


# ============================================================================
# Public entry point #1 — TC softmax top-K routing
# ============================================================================


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
    redispatch_x_in_backward: bool = False,
) -> torch.Tensor:
    W, E_local, mode = _validate_and_resolve(mgr, mode, E, K)
    activation_type = _normalize_activation(activation_type)
    T_local, d = x.shape

    ws = mgr._get_or_alloc(T_local, d, K, E_local, x.dtype, mode)

    router_logits = F.linear(x, router_w)
    topk_scores_l, topk_idx_l = TC_Softmax_Topk_Router_Function.apply(
        router_logits, W * E_local, K, is_softmax_over_topk, norm_topk_probs
    )

    # x_symm needs its own barrier — barriers are per-buffer (each
    # handle's signal pad is independent), so the idx_hdl.barrier() inside
    # _ag_routing_decision below does NOT fence x_symm. Without this,
    # _UpProjectionEP.forward later reads peer.x_symm via NVLink dispatch
    # before peers have flushed their .copy_() write, gets stale/in-flight
    # bytes for some rows, and the up-proj GEMM produces wrong outputs.
    # Symptom at large T: max(o_diff) ≈ max(o_ref) — affected tokens get
    # the value of an unrelated row, not bf16 precision noise.
    ws.x_symm.copy_(x)
    ws.x_hdl.barrier()
    topk_idx_g = _ag_routing_decision(ws, topk_idx_l)

    return _moe_ep_forward_inner(
        x_local=x,
        topk_idx_global=topk_idx_g,
        topk_scores_local=topk_scores_l,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        ep_ws=ws,
        activation_type=activation_type,
        is_inference_mode_enabled=is_inference_mode_enabled,
        concat_layout=concat_layout,
        redispatch_x_in_backward=redispatch_x_in_backward,
    )


# ============================================================================
# Public entry point #2 — caller-supplied routing decision
# ============================================================================


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
    redispatch_x_in_backward: bool = False,
) -> torch.Tensor:
    K = topk_indices.shape[1]
    W, E_local, mode = _validate_and_resolve(mgr, mode, E, K)
    activation_type = _normalize_activation(activation_type)
    T_local, d = x.shape

    ep_ws = mgr._get_or_alloc(T_local, d, K, E_local, x.dtype, mode)

    topk_idx_l = topk_indices.to(torch.int32)

    # See moe_ep_TC_softmax_topk_forward for why x_symm needs its own
    # barrier — barriers are per-buffer; idx_hdl can't fence x_hdl.
    ep_ws.x_symm.copy_(x)
    ep_ws.x_hdl.barrier()
    topk_idx_g = _ag_routing_decision(ep_ws, topk_idx_l)

    return _moe_ep_forward_inner(
        x_local=x,
        topk_idx_global=topk_idx_g,
        topk_scores_local=topk_scores,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        ep_ws=ep_ws,
        activation_type=activation_type,
        is_inference_mode_enabled=is_inference_mode_enabled,
        concat_layout=concat_layout,
        redispatch_x_in_backward=redispatch_x_in_backward,
    )
