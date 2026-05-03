# ********************************************************************************
# Copyright (c) 2025 Sonic-MoE contributors
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
# Dispatch + GEMM modes
# ----------------------------------------------------------------------------
#   "ag"  — All-gather x to all peers (W*T_local rows). Always paired with
#           the gather GEMM (concat_layout=False). Metadata is fed
#           token_indices = arange(TK_global) // K.
#
#   "a2a" — Pull-based A2A dispatch. Two layout strategies:
#           concat_layout=False (interleaved weights, gather GEMM):
#             Pass `meta["a2a_token_indices"]` as both metadata
#             token_indices and pull recv_pos. Legacy per-rank-slot layout.
#           concat_layout=True (concat weights, nogather GEMM):
#             Pass `metadata["s_reverse_local"]` as recv_pos for the pull.
#             Expert-sorted layout.
#
# In all three the up-proj forward and dw1 backward GEMMs are called the
# same way the non-EP `_UpProjection` calls them: A_idx=x_gather_idx,
# concat_layout threaded through to QuACK.
#
# ----------------------------------------------------------------------------
# `_UpProjectionEP` owns dispatch
# ----------------------------------------------------------------------------
# x_local is the autograd-tracked input to the Function. The Function's
# forward owns: dispatch (AG or A2A pull) → up-proj GEMM. Its backward
# owns: up-proj backward → local K-fold reduction → reverse-dispatch back
# to (T_local, H) dx_local. So `dx_local` returned to autograd is uniform
# in shape across all three layout modes — autograd-clean: backward output
# shape matches forward input shape.
#
# Why dispatch lives inside the Function rather than outside:
#
#   * dx genuinely has to be reduced. With dispatch outside, the Function
#     would have to return dx_compute matching x_compute's shape (autograd
#     contract), which means dx_compute = (TK_global, H) for A2A and the
#     K-fold reduction has to happen *somewhere else* — typically a
#     parent layer doing reverse-A2A. Splitting the up-proj backward
#     between Function (per-recv-row dx) and parent (K-fold + cross-rank
#     reverse) is brittle; the parent has to know the Function's internal
#     layout convention. Pulling dispatch in lets the Function control
#     the entire forward-input → backward-input loop.
#
#   * x_compute is a workspace buffer reused by the next forward call.
#     Saving it for backward via `ctx.save_for_backward(x_compute)` is
#     unsafe — the next iteration overwrites it before backward consumes
#     it. With dispatch inside, we save the autograd-owned `x_local` and
#     redo dispatch in backward. Costs one extra dispatch per training
#     step; pipelines naturally with backward compute.
#
# Backward shape contract (uniform across modes):
#
#   AG  : up-proj bwd → dx_expanded (TK_global, H) → token_broadcast_local
#         (W*T_local, H) → reduce_scatter → dx_local (T_local, H).
#   A2A : up-proj bwd → dx_expanded (TK_global, H) → permute or identity
#         (TK_global, H) → reverse-A2A pull with K-fold accumulation per
#         source token → dx_local (T_local, H).
#
# A2A reverse pull kernel does not exist yet in triton_comm.py. The
# Function structure is ready for it; today's A2A backward raises
# NotImplementedError. AG backward is fully implemented via the existing
# `reduce_scatter_triton`. Backward is dormant in the current EP forward
# (down-proj is bare gemm, produces no `dh`); none of these paths fire
# until a future `_DownProjectionEP` consumes `h`.
#
# ----------------------------------------------------------------------------
# Mode selection (mode="auto", default)
# ----------------------------------------------------------------------------
# Mode is a pure function of (W, K) and is fixed for the lifetime of a
# workspace. _resolve_mode runs once at workspace allocation; the result
# is cached on _EPWorkspace.mode.
#
# Heuristic (bench-comm.py on B300/W=4 and H100/W=8). A2A pull moves
# ≈ (K/W) × the bytes AG moves; A2A achieves rho × BW_AG of measured
# NVLink bandwidth where rho is hardware-dependent:
#
#   sm_100+ (B300): rho ~ 1.0       -> A2A iff K < W
#   sm_90   (H100): rho ~ 0.7-0.85  -> A2A iff K/W < 0.8
#   else            (conservative)  -> AG
#
# ----------------------------------------------------------------------------
# Combine mode
# ----------------------------------------------------------------------------
# `gather_aggregation` wins every benchmarked config (1.2-6× over rs+RS),
# so there is no combine mode selector — it is hardcoded.
#
# ----------------------------------------------------------------------------
# Barrier accounting
# ----------------------------------------------------------------------------
# Symm-mem barrier(buf, grp) is a group-level sync. Forward path:
#
#   B1   post-write fence on idx_symm AND x_symm before peer reads either.
#        (Issued by _ag_routing_decision; this is also the fence x_symm
#        needs before AG / A2A pull reads it.)
#   B3   post-write fence on y_symm before fused_gemm_combine reads it.
#   B4   pre-overwrite fence on y_symm AND x_symm before next iteration
#        rewrites them.
#
# Backward path (when wired):
#   B_bwd1   post-write fence on dx_compute_symm before reduce_scatter
#            reads it (AG only).
#
# ----------------------------------------------------------------------------
# CPU/GPU sync accounting
# ----------------------------------------------------------------------------
# Forward issues zero host-blocking syncs. Up-proj/down-proj/combine run at
# worst-case shape TK_global = W * T_local * K (a Python int from static
# workspace shape). Invalid lanes carry a sentinel local-expert id and
# produce garbage rows in y_symm that combine never reads.
#
# Patterns to avoid (each one re-introduces a host sync):
#   * BAD:  x[bool_mask]
#   * BAD:  tensor.item() / int(tensor) / bool(tensor)
#   * BAD:  x[(arange // step) == my_rank]
#
# ----------------------------------------------------------------------------
# Caller contract
# ----------------------------------------------------------------------------
# We assume callers pass aligned, well-shaped inputs:
#   - x is 2D (T_local, d) on the manager's device, T_local > 0.
#   - topk_indices / topk_scores (when provided) are 2D (T_local, K) and
#     match each other.
#   - Weights are sharded along the leading E axis with E_local = E // W.
# The forward does not re-validate any of these.
# ********************************************************************************

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
from quack.gemm_interface import gemm, gemm_gated
from torch.distributed import _symmetric_memory as _symm_mem

from sonicmoe.functional import TC_Softmax_Topk_Router_Function
from sonicmoe.functional.backward import _token_broadcast_backward, _up_projection_backward_act
from sonicmoe.functional.triton_kernels import general_routing_router_metadata_triton

from .enums import ActivationType, is_glu
from .functional.ep.triton_comm import (
    a2a_dispatch_pull_triton,
    all_gather_copy_engine_async,
    all_gather_triton,
    barrier,
    compute_dispatch_metadata,
    gather_aggregation_triton,
    reduce_scatter_triton,
)


__all__ = [
    "SymmMemManager",
    "moe_ep_TC_softmax_topk_forward",
    "moe_ep_general_routing_forward",
]


# ============================================================================
# Mode selection policy
# ============================================================================

_VALID_MODES = ("ag", "a2a")


def _select_dispatch_mode(W: int, K: int) -> str:
    """Heuristic pick between 'ag' and 'a2a' dispatch."""
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
# _UpProjectionEP — owns dispatch (forward) and reverse-dispatch (backward)
# ============================================================================


class _UpProjectionEP(torch.autograd.Function):
    """EP-specialized up-projection. Forward: stage → dispatch → gemm_gated.
    Backward: up-proj-bwd → local reduction → reverse-dispatch → dx_local.

    Inputs differentiable: x_local, w1, b1.
    Output: a (TK_global, I).

    Dispatch mode + concat_layout combination is encoded in `mode` (str).
    Backward returns dx_local of shape (T_local, H), uniform across modes.

    AG backward uses `reduce_scatter_triton` for the cross-rank reverse.
    A2A backward needs a reverse-A2A-pull triton kernel that doesn't exist
    yet — currently raises NotImplementedError. The path is dormant
    regardless (today's down-proj is bare gemm, produces no `dh`).
    """

    @staticmethod
    def forward(
        ctx,
        x_local: torch.Tensor,  # (T_local, d) — autograd input
        w1: torch.Tensor,  # (2I, H, E_local)
        b1: Optional[torch.Tensor],
        # Routing metadata (per-call; built by _build_consumer_metadata).
        expert_frequency_offset: torch.Tensor,
        x_gather_idx: torch.Tensor,
        s_reverse_local: torch.Tensor,
        dst_rank_flat: torch.Tensor,
        a2a_token_indices: Optional[torch.Tensor],  # only used in A2A
        # Constants.
        TK_global: int,
        T_local: int,
        K: int,
        activation_type: ActivationType,
        is_inference_mode_enabled: bool,
        concat_layout: bool,
        mode: str,  # "ag" | "a2a"
        # Workspace handle. Holds:
        #   ep_ws.x_symm          — pre-staged, pre-fenced by B1.
        #   ep_ws.ag_compute      — AG dispatch dest (or None).
        #   ep_ws.a2a_recv        — A2A dispatch dest (or None).
        # Plus lazily-allocated backward symm-mem buffers (see backward).
        # ep_ws is attached to ctx but NOT in saved_tensors (autograd only
        # accepts tensors there; ep_ws is a dataclass instance).
        ep_ws,
    ) -> torch.Tensor:
        # x_symm is pre-staged by the parent and pre-fenced by B1
        # (issued in _ag_routing_decision). The Function does NOT re-stage:
        # restaging here would happen *after* B1, breaking the peer-read
        # invariant. We dispatch directly from ep_ws.x_symm.

        # --- Forward dispatch.
        if mode == "ag":
            x_compute = all_gather_triton(ep_ws.x_symm, ep_ws.ep_group, out=ep_ws.ag_compute)
        else:
            recv_pos = s_reverse_local if concat_layout else a2a_token_indices
            a2a_dispatch_pull_triton(
                x_symm=ep_ws.x_symm,
                dst_rank_flat=dst_rank_flat,
                recv_pos=recv_pos,
                recv=ep_ws.a2a_recv,
                K=K,
                group=ep_ws.ep_group,
            )
            x_compute = ep_ws.a2a_recv.view(TK_global, x_local.shape[1])

        # --- Up-proj GEMM (same call shape as non-EP _UpProjection).
        I = w1.shape[0]
        is_glu_act = is_glu(activation_type)
        if is_glu_act:
            I //= 2

        a = torch.empty(TK_global, I, dtype=x_compute.dtype, device=x_compute.device)
        h = (
            torch.empty(
                TK_global,
                (2 * I if is_glu_act else I),
                dtype=x_compute.dtype,
                device=x_compute.device,
            )
            if not is_inference_mode_enabled
            else None
        )
        gemm_gated(
            x_compute,
            w1.permute(2, 1, 0),
            activation=activation_type.value,
            cu_seqlens_m=expert_frequency_offset,
            A_idx=x_gather_idx,
            preact_out=h,
            postact_out=a,
            store_preact=(not is_inference_mode_enabled),
            bias=b1,
            concat_layout=(("B", "bias") if b1 is not None else ("B",)) if concat_layout else None,
        )

        if not is_inference_mode_enabled:
            # Save autograd-owned x_local (NOT the workspace buffers — those
            # are reused next iteration; backward redoes dispatch from x_local).
            ctx.save_for_backward(
                x_local,
                w1,
                b1,
                expert_frequency_offset,
                x_gather_idx,
                s_reverse_local,
                dst_rank_flat,
                (
                    a2a_token_indices
                    if a2a_token_indices is not None
                    else torch.empty(0, dtype=torch.int32, device=x_local.device)
                ),
            )
            ctx.is_glu_act = is_glu_act
            ctx.concat_layout = concat_layout
            ctx.mode = mode
            ctx.K = K
            ctx.T_local = T_local
            ctx.TK_global = TK_global
            ctx.has_a2a_token_indices = a2a_token_indices is not None
            ctx.ep_ws = ep_ws  # Python attr, not in saved_tensors
            ctx.mark_non_differentiable(a)
            ctx.set_materialize_grads(False)

        return a

    @staticmethod
    def backward(ctx, dh: Optional[torch.Tensor]):
        # Dormant path: today's down-proj produces no dh, so this is None.
        if dh is None:
            return (None,) * 16

        (
            x_local,
            w1,
            b1,
            expert_frequency_offset,
            x_gather_idx,
            s_reverse_local,
            dst_rank_flat,
            a2a_token_indices_or_empty,
        ) = ctx.saved_tensors
        is_glu_act = ctx.is_glu_act
        concat_layout = ctx.concat_layout
        mode = ctx.mode
        K = ctx.K
        T_local = ctx.T_local
        TK_global = ctx.TK_global
        ep_ws = ctx.ep_ws
        device, dtype = dh.device, dh.dtype
        H = x_local.shape[1]

        a2a_token_indices = a2a_token_indices_or_empty if ctx.has_a2a_token_indices else None

        # --- Re-dispatch to rebuild x_compute. Forward saved x_local rather
        # than x_compute because workspace buffers are reused next iteration.
        # Allocate fresh non-workspace buffers in backward — the perf cost of
        # one extra dispatch per training step is the price of correctness;
        # workspace-side bwd buffers can be added later as an optimization.
        # Note: the symm-mem staging buffer for backward's redo-dispatch
        # also has to be fresh (workspace x_symm is reused). For this we
        # use ep_ws.x_symm_bwd, lazily allocated on first backward call.
        x_symm_bwd = ep_ws._lazy_alloc_x_symm_bwd()
        x_symm_bwd.copy_(x_local.contiguous())
        # Fence x_symm_bwd before peer reads it.
        barrier(x_symm_bwd, ep_ws.ep_group)

        if mode == "ag":
            ag_compute_bwd = torch.empty(
                ep_ws.W * T_local,
                H,
                dtype=dtype,
                device=device,
            )
            x_compute = all_gather_triton(x_symm_bwd, ep_ws.ep_group, out=ag_compute_bwd)
        else:
            a2a_recv_bwd = torch.empty(
                ep_ws.W,
                T_local * K,
                H,
                dtype=dtype,
                device=device,
            )
            recv_pos = s_reverse_local if concat_layout else a2a_token_indices
            a2a_dispatch_pull_triton(
                x_symm=x_symm_bwd,
                dst_rank_flat=dst_rank_flat,
                recv_pos=recv_pos,
                recv=a2a_recv_bwd,
                K=K,
                group=ep_ws.ep_group,
            )
            x_compute = a2a_recv_bwd.view(TK_global, H)

        # --- Up-proj backward step 1: dx_expanded + db1 (same call as non-EP).
        dx_expanded = torch.empty(TK_global, H, dtype=dtype, device=device)
        dw1 = torch.empty_like(w1)
        db1 = None if b1 is None else torch.empty_like(b1)

        _up_projection_backward_act(
            w1=w1,
            dx_expanded=dx_expanded,
            dh=dh,
            db1=db1,
            expert_frequency_offset=expert_frequency_offset,
            is_glu_activation=is_glu_act,
            concat_layout=concat_layout,
        )

        # --- Up-proj backward step 2: dw1 (same call as non-EP).
        gemm(
            x_compute.T,
            dh,
            out=dw1.permute(2, 1, 0),
            cu_seqlens_k=expert_frequency_offset,
            A_idx=x_gather_idx,
            batch_idx_permute=None,
            dynamic_scheduler=False,
            concat_layout=(("out",) if concat_layout else None),
        )

        # --- Local K-fold reduction + reverse-dispatch → dx_local.
        if mode == "ag":
            # AG: dual of router_forward. K-fold collapse via s_reverse_local
            # into a (W*T_local, H) buffer, then reduce_scatter across W ranks
            # to land each rank's T_local rows in dx_local.
            dx_compute_symm = ep_ws._lazy_alloc_dx_compute_symm(H, dtype)
            _token_broadcast_backward(
                dx_reduced=dx_compute_symm,
                dx_expanded=dx_expanded,
                s_reverse_scatter_idx=s_reverse_local,
                num_activated_expert_per_token_offset=None,
                varlen_K_max=K,
                H=H,
                is_varlen_K=False,
            )
            barrier(dx_compute_symm, ep_ws.ep_group)
            dx_local = reduce_scatter_triton(dx_compute_symm, ep_ws.ep_group)
        else:
            # A2A reverse-pull is not yet implemented in triton_comm.py.
            # Sketch of what's needed:
            #
            #   For each (token t, expert k) on this rank, find the K
            #   recv-buffer rows on peer ranks that hold copies of t's
            #   x_local row (one per expert). Read peer.dx_recv at those
            #   positions, accumulate into dx_local[t]. Symmetric inverse
            #   of a2a_dispatch_pull_triton.
            #
            # Workspace would need:
            #   - dx_recv_symm: symm-mem (TK_global, d), peers read this.
            #
            # The local reduction step (dx_expanded → per-recv-row gradient
            # in dx_recv) IS:
            #   concat_layout=False (A2A flat): permutation via x_gather_idx.
            #   concat_layout=True  (A2A sorted): identity (dx_expanded IS
            #                                      dx_recv).
            raise NotImplementedError(
                "A2A backward requires a reverse-A2A-pull triton kernel that "
                "is not yet implemented in triton_comm.py. The Function is "
                "structured for it; see the comment in _UpProjectionEP.backward "
                "for the kernel contract."
            )

        # 16 forward inputs. Differentiable: x_local, w1, b1.
        return (
            dx_local,
            dw1,
            db1,
            None,  # expert_frequency_offset
            None,  # x_gather_idx
            None,  # s_reverse_local
            None,  # dst_rank_flat
            None,  # a2a_token_indices
            None,  # TK_global
            None,  # T_local
            None,  # K
            None,  # activation_type
            None,  # is_inference_mode_enabled
            None,  # concat_layout
            None,  # mode
            None,  # ep_ws
        )


# ============================================================================
# Internal workspace
# ============================================================================


@dataclass
class _EPWorkspace:
    """Symm-mem buffers and cached static patterns for one
    (T_local, d, K, E_local, dtype, mode) shape."""

    idx_symm: torch.Tensor
    scores_symm: torch.Tensor
    x_symm: torch.Tensor
    y_symm: torch.Tensor
    s_rev_symm: torch.Tensor
    ep_group: dist.ProcessGroup
    E_local: int
    mode: str = "ag"

    a2a_recv: Optional[torch.Tensor] = None  # A2A only, (W, TK_local, d)
    ag_compute: Optional[torch.Tensor] = None  # AG only,  (W * T_local, d)
    pos_2d_pattern: Optional[torch.Tensor] = None  # (T_local, K) int32
    invalid_lane_expert: Optional[torch.Tensor] = None  # (TK_global,) int32
    t_global_pattern: Optional[torch.Tensor] = None  # AG only

    # Backward-side symm-mem buffers, lazily allocated on first backward
    # call. Allocation is a collective (rendezvous) — all ranks must hit
    # the lazy alloc together. Autograd backward fires synchronously on
    # all ranks for the same Function, so this is safe.
    _x_symm_bwd: Optional[torch.Tensor] = None  # (T_local, d), symm-mem
    _dx_compute_symm: Optional[torch.Tensor] = None  # AG only, (W*T_local, H), symm-mem

    @property
    def W(self) -> int:
        return dist.get_world_size(self.ep_group)

    @property
    def T_local(self) -> int:
        return self.idx_symm.shape[0]

    @property
    def K(self) -> int:
        return self.idx_symm.shape[1]

    @property
    def d(self) -> int:
        return self.y_symm.shape[1]

    def _lazy_alloc_x_symm_bwd(self) -> torch.Tensor:
        if self._x_symm_bwd is None:
            self._x_symm_bwd = _symm_mem.empty(
                self.T_local,
                self.d,
                dtype=self.x_symm.dtype,
                device=self.x_symm.device,
            )
            _symm_mem.rendezvous(
                self._x_symm_bwd,
                group=getattr(self.ep_group, "group_name", "0"),
            )
        return self._x_symm_bwd

    def _lazy_alloc_dx_compute_symm(self, H: int, dtype: torch.dtype) -> torch.Tensor:
        if self._dx_compute_symm is None:
            self._dx_compute_symm = _symm_mem.empty(
                self.W * self.T_local,
                H,
                dtype=dtype,
                device=self.x_symm.device,
            )
            _symm_mem.rendezvous(
                self._dx_compute_symm,
                group=getattr(self.ep_group, "group_name", "0"),
            )
        return self._dx_compute_symm


# ============================================================================
# SymmMemManager — owns allocation, rendezvous, and cache
# ============================================================================


class SymmMemManager:
    """Owns symm-mem buffer allocation and the workspace cache for one
    (ep_group, device) pair."""

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

    def _alloc_symm(self, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        buf = _symm_mem.empty(*shape, dtype=dtype, device=self.device)
        _symm_mem.rendezvous(buf, group=self._group_name)
        return buf

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

        idx_symm = self._alloc_symm((T_local, K), torch.int32)
        scores_symm = self._alloc_symm((T_local, K), torch.float32)
        x_symm = self._alloc_symm((T_local, d), dtype)
        y_symm = self._alloc_symm((TK_global, d), dtype)
        s_rev_symm = self._alloc_symm((TK_global,), torch.int32)

        pos_2d_pattern = (torch.arange(TK_local, device=dev, dtype=torch.int32) + my_rank * TK_local).view(T_local, K)
        invalid_lane_expert = torch.arange(TK_global, device=dev, dtype=torch.int32) % E_local

        a2a_recv = None
        ag_compute = None
        t_global_pattern = None

        if mode == "a2a":
            a2a_recv = torch.empty((W, TK_local, d), dtype=dtype, device=dev)
        else:  # ag
            ag_compute = torch.empty((W * T_local, d), dtype=dtype, device=dev)
            t_global_pattern = torch.arange(TK_global, device=dev, dtype=torch.int32) // K

        return _EPWorkspace(
            idx_symm=idx_symm,
            scores_symm=scores_symm,
            x_symm=x_symm,
            y_symm=y_symm,
            s_rev_symm=s_rev_symm,
            ep_group=self.ep_group,
            E_local=E_local,
            mode=mode,
            a2a_recv=a2a_recv,
            ag_compute=ag_compute,
            pos_2d_pattern=pos_2d_pattern,
            invalid_lane_expert=invalid_lane_expert,
            t_global_pattern=t_global_pattern,
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
# Consumer-side metadata (worst-case sized)
# ============================================================================


def _build_consumer_metadata(
    expert_indices: torch.Tensor,
    token_indices: torch.Tensor,
    TK: int,
    E_local: int,
    s_reverse_idx_symm: torch.Tensor,
):
    """Build consumer-side per-expert metadata. See module docstring for
    which kernel outputs are live and which are scratch."""
    device = expert_indices.device

    expert_frequency_offset = torch.empty(E_local + 1, dtype=torch.int32, device=device)
    s_reverse_local = s_reverse_idx_symm[:TK]
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)

    # Scratch — written by the kernel, never read by ep.py.
    expert_frequency = torch.empty(E_local, dtype=torch.int32, device=device)
    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    num_offset = torch.empty(TK + 1, dtype=torch.int32, device=device)

    general_routing_router_metadata_triton(
        token_indices,
        expert_indices,
        TK,
        E_local,
        expert_frequency,
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_local,
        num_offset,
    )

    return {
        "expert_frequency_offset": expert_frequency_offset,
        "s_reverse_local": s_reverse_local,
        "x_gather_idx": x_gather_idx,
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
) -> torch.Tensor:
    W, my_rank = ep_ws.W, dist.get_rank(ep_ws.ep_group)
    T_local, d, K, E_local = ep_ws.T_local, ep_ws.d, ep_ws.K, ep_ws.E_local
    TK_local = T_local * K
    TK_global = W * TK_local
    dev = x_local.device
    grp = ep_ws.ep_group
    mode = ep_ws.mode

    # Routing-side metadata — feeds dispatch (dst_rank_flat, a2a_token_indices)
    # and combine (my_dst_rank, my_pos_per_rank).
    meta = compute_dispatch_metadata(topk_idx_global, my_rank=my_rank, E_local=E_local)
    dst_rank_flat = meta["dst_rank_flat"]
    rank_2d = meta["my_dst_rank"]
    expert_local_padded = meta["expert_local_padded"]
    a2a_token_indices = meta["a2a_token_indices"] if mode == "a2a" else None

    # Consumer-side metadata — feeds the up-proj GEMM and the combine.
    if mode == "ag":
        token_indices = ep_ws.t_global_pattern  # arange(TK_global) // K
    else:
        token_indices = a2a_token_indices  # legacy per-rank-slot layout

    metadata = _build_consumer_metadata(
        expert_indices=expert_local_padded,
        token_indices=token_indices,
        TK=TK_global,
        E_local=E_local,
        s_reverse_idx_symm=ep_ws.s_rev_symm,
    )

    # _UpProjectionEP owns dispatch + up-proj GEMM. x_local is the autograd
    # input; backward returns dx_local of shape (T_local, H). The Function
    # reaches into ep_ws for the workspace buffers (x_symm, ag_compute,
    # a2a_recv) directly, so we don't pass them as positional args.
    a = _UpProjectionEP.apply(
        x_local,
        w1,
        b1,
        metadata["expert_frequency_offset"],
        metadata["x_gather_idx"],
        metadata["s_reverse_local"],
        dst_rank_flat,
        a2a_token_indices,
        TK_global,
        T_local,
        K,
        activation_type,
        is_inference_mode_enabled,
        concat_layout,
        mode,
        ep_ws,
    )

    gemm(
        a,
        w2,
        out=ep_ws.y_symm,
        cu_seqlens_m=metadata["expert_frequency_offset"],
        bias=b2,
    )

    barrier(ep_ws.y_symm, grp)

    y_local = torch.empty(T_local, d, dtype=x_local.dtype, device=dev)
    gather_aggregation_triton(
        ep_ws.y_symm, ep_ws.s_rev_symm, rank_2d, ep_ws.pos_2d_pattern, topk_scores_local, y_local, K=K, group=grp
    )

    return y_local


# ============================================================================
# Helpers shared by the public entry points
# ============================================================================


def _validate_and_resolve(
    mgr: "SymmMemManager",
    mode: str,
    E: int,
    K: int,
) -> Tuple[int, int, str]:
    """Resolve auto mode, validate divisibility, return (W, E_local, mode)."""
    W = mgr.world_size
    resolved = _resolve_mode(mode, W, K)
    if E % W != 0:
        raise ValueError(f"E ({E}) must be divisible by EP world size ({W}).")
    return W, E // W, resolved


def _ag_routing_decision(
    idx_symm: torch.Tensor,
    topk_idx_l: torch.Tensor,
    grp: dist.ProcessGroup,
    *,
    scores_symm: Optional[torch.Tensor] = None,
    topk_scores_l: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Routing-decision AG. Issues B1 — fences idx_symm AND x_symm before
    peer reads. _UpProjectionEP relies on this fence; it does NOT issue
    its own barrier on x_symm before dispatch."""
    idx_symm.copy_(topk_idx_l)
    stage_scores = scores_symm is not None
    if stage_scores:
        assert topk_scores_l is not None, "_ag_routing_decision: scores_symm given but topk_scores_l is None"
        scores_symm.copy_(topk_scores_l)

    barrier(idx_symm, grp)  # B1: fences idx_symm + scores_symm + x_symm

    W = dist.get_world_size(grp)
    T_local, K = idx_symm.shape
    topk_idx_g = all_gather_triton(idx_symm, grp).view(W, T_local, K)

    topk_scores_g = None
    if stage_scores:
        topk_scores_g = all_gather_triton(scores_symm, grp)

    return topk_idx_g, topk_scores_g


# ============================================================================
# Public entry point #1 — TC softmax top-K routing (router computed inside)
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
) -> torch.Tensor:
    """EP MoE forward with TC softmax top-K routing computed internally.

    `mode='auto'` (default) selects 'ag' or 'a2a' from W, K, and the device's
    compute capability — see module docstring."""
    W, E_local, mode = _validate_and_resolve(mgr, mode, E, K)
    activation_type = _normalize_activation(activation_type)
    T_local, d = x.shape

    ws = mgr._get_or_alloc(T_local, d, K, E_local, x.dtype, mode)

    # NOTE: x_local staging into x_symm now happens INSIDE _UpProjectionEP.
    # We pass x as the autograd-tracked input; the Function copies it into
    # x_symm. We still need the routing-decision AG (which issues B1 — the
    # barrier that also fences x_symm).
    router_logits = F.linear(x, router_w)
    topk_scores_l, topk_idx_l = TC_Softmax_Topk_Router_Function.apply(
        router_logits, W * E_local, K, is_softmax_over_topk, norm_topk_probs
    )

    # The Function expects x_symm to be pre-fenced when it dispatches.
    # That fence is B1, issued inside _ag_routing_decision. So we have to
    # stage x_symm BEFORE calling _ag_routing_decision so B1 covers it.
    ws.x_symm.copy_(x.contiguous())
    topk_idx_g, _ = _ag_routing_decision(ws.idx_symm, topk_idx_l, ws.ep_group)

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
) -> torch.Tensor:
    """EP MoE forward with caller-supplied routing decision (uniform K)."""
    K = topk_indices.shape[1]
    W, E_local, mode = _validate_and_resolve(mgr, mode, E, K)
    activation_type = _normalize_activation(activation_type)
    T_local, d = x.shape

    ep_ws = mgr._get_or_alloc(T_local, d, K, E_local, x.dtype, mode)

    topk_idx_l = topk_indices.to(torch.int32)

    # Pre-stage x_symm so B1 in _ag_routing_decision also fences it.
    ep_ws.x_symm.copy_(x.contiguous())
    topk_idx_g, _ = _ag_routing_decision(ep_ws.idx_symm, topk_idx_l, ep_ws.ep_group)

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
    )
