# ********************************************************************************
# Copyright (c) 2025 Sonic-MoE contributors
#
# Expert Parallelism (EP) — AG dispatch or pull-based A2A dispatch, AG-pattern
# NVLink combine.
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
#   TK_local    T_local * K   (this rank's max slot count along the source axis)
#   W           EP world size
#   TK_global   W * TK_local  (worst-case consumer-stream length)
#   E_local     experts per rank (E // W)
#
# ----------------------------------------------------------------------------
# Dispatch mode selection (mode="auto", default)
# ----------------------------------------------------------------------------
# Derived from `benchmarks/ep/bench-comm.py` on B300 (W=4) and H100 (W=8).
# A2A pull moves ≈ (K/W) × the bytes AG moves; A2A achieves rho × BW_AG of
# measured NVLink bandwidth where rho is hardware-dependent:
#
#   sm_100+ (B300): rho ~ 1.0       -> A2A iff K < W
#   sm_90   (H100): rho ~ 0.7-0.85  -> A2A iff K/W < 0.8
#   else            (conservative)  -> AG
#
# Validated against all 15 dispatch rows in the bench-comm output (B300 W=4
# and H100 W=8): correct or tied in every config.
#
# ----------------------------------------------------------------------------
# Combine mode
# ----------------------------------------------------------------------------
# `gather_aggregation` wins every benchmarked config (1.2-6× over rs+RS),
# so there is no combine mode selector — it is hardcoded.
#
# ----------------------------------------------------------------------------
# Barrier accounting (3 barriers per forward in BOTH modes)
# ----------------------------------------------------------------------------
# Symm-mem barrier(buf, grp) is a group-level sync. Minimal set kept:
#
#     B1   post-write fence on idx_symm AND x_symm before peer reads either.
#          (idx_symm is read by the AG of the routing decision; x_symm is
#          read by the AG kernel (AG mode) or by the pull kernel (A2A mode).)
#     B3   post-write fence on y_symm before fused_gemm_combine reads it.
#     B4   pre-overwrite fence on y_symm AND x_symm before next iteration
#          rewrites them.
#
# A2A dispatch is done by the pull kernel, which reads peer.x_symm directly.
# There is no a2a_send buffer and no separate post-A2A barrier: B1 already
# fences x_symm, and B4 of the previous iteration fences peer reads of
# x_symm before the next stage_input.
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
#   * BAD:  x[bool_mask]  — implicit .nonzero(); host blocks for output shape.
#   * BAD:  tensor.item() / int(tensor) / bool(tensor)
#   * BAD:  x[(arange // step) == my_rank]  — same as above; use static slice.
#
# ----------------------------------------------------------------------------
# Caller contract
# ----------------------------------------------------------------------------
# We assume callers pass aligned, well-shaped inputs:
#   - x is 2D (T_local, d) on the manager's device, T_local > 0.
#   - topk_indices / topk_scores (when provided) are 2D (T_local, K) and
#     match each other.
#   - Weights are sharded along the leading E axis with E_local = E // W.
# The forward does not re-validate any of these. Caller errors typically
# surface as kernel shape errors deeper in the stack.
# ********************************************************************************

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton

# Down projection used to be wrapped by sonicmoe.functional.forward._down_projection_forward,
# which was a one-line `gemm(a, w2.permute(2, 1, 0), out=y, cu_seqlens_m=...)` shim.
# Recent sonic-moe main dropped that shim, so we call QuACK's gemm directly.
from quack.gemm_interface import gemm
from torch.distributed import _symmetric_memory as _symm_mem

from sonicmoe.functional import TC_Softmax_Topk_Router_Function, _UpProjection
from sonicmoe.functional.triton_kernels import general_routing_router_metadata_triton

from .enums import ActivationType
from .functional.ep.triton_comm import (
    a2a_dispatch_pull,
    all_gather,
    barrier,
    compute_dispatch_metadata,
    gather_aggregation,
    rendezvous,
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
    """Heuristic pick between 'ag' and 'a2a' dispatch.

    See module-level docstring for derivation. Single-rank trivially uses
    AG (no off-rank traffic either way; the AG path skips the pull kernel).
    """
    if W <= 1:
        return "ag"
    cap_major = torch.cuda.get_device_capability()[0]
    if cap_major >= 10:  # B300+
        return "a2a" if K < W else "ag"
    if cap_major == 9:  # H100
        return "a2a" if 5 * K < 4 * W else "ag"
    return "ag"  # conservative default


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
# Internal workspace
# ============================================================================


@dataclass
class _EPWorkspace:
    """Symm-mem buffers and cached static patterns for one
    (T_local, d, K, E_local, dtype, mode) shape."""

    idx_symm: torch.Tensor
    x_symm: torch.Tensor  # always allocated (used by both modes)
    y_symm: torch.Tensor
    s_rev_symm: torch.Tensor
    ep_group: dist.ProcessGroup
    E_local: int

    # Mode-specific reusable compute buffers. Cached here (not torch.empty'd
    # per call) so the AG and A2A paths are symmetric in their allocation
    # pattern and so the steady-state forward issues no torch.empty calls
    # for the dispatch output.
    a2a_recv: Optional[torch.Tensor] = None  # A2A only, (W, TK_local, d)
    ag_compute: Optional[torch.Tensor] = None  # AG only,  (W * T_local, d)

    # Static patterns cached at allocation time.
    pos_2d_pattern: Optional[torch.Tensor] = None  # (T_local, K) int32
    invalid_lane_expert: Optional[torch.Tensor] = None  # (TK_global,) int32
    t_global_pattern: Optional[torch.Tensor] = None  # (TK_global,) int32, AG only
    src_rank_pattern: Optional[torch.Tensor] = None  # (TK_global,) int64, A2A only

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
        """Pre-allocate workspaces for the given shapes. `mode` is resolved
        if 'auto' is passed."""
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
        x_symm = self._alloc_symm((T_local, d), dtype)
        y_symm = self._alloc_symm((TK_global, d), dtype)
        s_rev_symm = self._alloc_symm((TK_global,), torch.int32)

        pos_2d_pattern = (torch.arange(TK_local, device=dev, dtype=torch.int32) + my_rank * TK_local).view(T_local, K)
        invalid_lane_expert = torch.arange(TK_global, device=dev, dtype=torch.int32) % E_local

        a2a_recv = None
        ag_compute = None
        t_global_pattern = None
        src_rank_pattern = None

        if mode == "a2a":
            # Pull kernel writes here. (W, TK_local, d) layout: each source
            # rank's slots are contiguous along axis 1.
            a2a_recv = torch.empty((W, TK_local, d), dtype=dtype, device=dev)
            src_rank_pattern = (torch.arange(TK_global, device=dev) // TK_local).to(torch.int64)
        else:  # ag
            # AG kernel writes here.
            ag_compute = torch.empty((W * T_local, d), dtype=dtype, device=dev)
            t_global_pattern = torch.arange(TK_global, device=dev, dtype=torch.int32) // K

        return _EPWorkspace(
            idx_symm=idx_symm,
            x_symm=x_symm,
            y_symm=y_symm,
            s_rev_symm=s_rev_symm,
            ep_group=self.ep_group,
            E_local=E_local,
            a2a_recv=a2a_recv,
            ag_compute=ag_compute,
            pos_2d_pattern=pos_2d_pattern,
            invalid_lane_expert=invalid_lane_expert,
            t_global_pattern=t_global_pattern,
            src_rank_pattern=src_rank_pattern,
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
    """Build per-expert metadata for the up-projection."""
    device = expert_indices.device
    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    expert_frequency = torch.empty(E_local, dtype=torch.int32, device=device)
    expert_frequency_offset = torch.empty(E_local + 1, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)
    num_offset = torch.empty(TK + 1, dtype=torch.int32, device=device)
    s_reverse_local = s_reverse_idx_symm[:TK]

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
        "x_gather_idx": x_gather_idx,
        "s_scatter_idx": s_scatter_idx,
        "s_reverse_local": s_reverse_local,
        "num_offset": num_offset,
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
    mode: str,
) -> torch.Tensor:
    W, my_rank = ep_ws.W, dist.get_rank(ep_ws.ep_group)
    T_local, d, K, E_local = ep_ws.T_local, ep_ws.d, ep_ws.K, ep_ws.E_local
    TK_local = T_local * K
    TK_global = W * TK_local
    dev = x_local.device
    grp = ep_ws.ep_group

    meta = compute_dispatch_metadata(topk_idx_global, my_rank=my_rank, E_local=E_local)
    dst_rank_flat = meta["dst_rank_flat"]
    rank_2d = meta["my_dst_rank"]
    expert_local_padded = meta["expert_local_padded"]

    if mode == "ag":
        x_compute = all_gather(x_local, grp, out=ep_ws.ag_compute)
        token_indices_padded = ep_ws.t_global_pattern
    else:
        slot_flat_per_rank = meta["slot_flat_per_rank"]
        a2a_dispatch_pull(
            x_symm=x_local,
            dst_rank_flat=dst_rank_flat,
            slot_flat_per_rank=slot_flat_per_rank,
            recv=ep_ws.a2a_recv,
            K=K,
            group=grp,
        )
        x_compute = ep_ws.a2a_recv.view(W * TK_local, d)
        token_indices_padded = meta["a2a_token_indices"]

    metadata = _build_consumer_metadata(
        expert_indices=expert_local_padded,
        token_indices=token_indices_padded,
        TK=TK_global,
        E_local=E_local,
        s_reverse_idx_symm=ep_ws.s_rev_symm,
    )
    a, _h = _UpProjection.apply(
        x_compute,
        w1,
        b1,
        metadata["expert_frequency_offset"],
        TK_global,
        None,
        metadata["x_gather_idx"],
        metadata["s_scatter_idx"],
        metadata["s_reverse_local"],
        metadata["num_offset"],
        True,
        activation_type,
        is_inference_mode_enabled,
        concat_layout,
    )

    gemm(
        a,
        w2.permute(2, 1, 0),
        out=ep_ws.y_symm,
        cu_seqlens_m=metadata["expert_frequency_offset"],
        bias=b2,
    )

    barrier(ep_ws.y_symm, grp)

    y_local = torch.empty(T_local, d, dtype=x_local.dtype, device=dev)
    gather_aggregation(
        ep_ws.y_symm, ep_ws.s_rev_symm, rank_2d, ep_ws.pos_2d_pattern, topk_scores_local, y_local, K=K, group=grp
    )

    barrier(ep_ws.y_symm, grp)
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


def _ag_routing_indices(idx_symm: torch.Tensor, topk_idx_l: torch.Tensor, grp: dist.ProcessGroup) -> torch.Tensor:
    idx_symm.copy_(topk_idx_l.contiguous())
    barrier(idx_symm, grp)  # B1: fences both
    # idx_symm and x_symm
    g = all_gather(idx_symm, grp)
    return g.view(dist.get_world_size(grp), idx_symm.shape[0], idx_symm.shape[1])


def _stage_input(ws: _EPWorkspace, x: torch.Tensor) -> torch.Tensor:
    """Stage x into the workspace's symm-mem x buffer. Always uses ws.x_symm
    regardless of dispatch mode."""
    ws.x_symm.copy_(x.contiguous())
    return ws.x_symm


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
    x_in = _stage_input(ws, x)

    router_logits = F.linear(x_in, router_w)
    topk_scores_l, topk_idx_l = TC_Softmax_Topk_Router_Function.apply(
        router_logits, W * E_local, K, is_softmax_over_topk, norm_topk_probs
    )
    topk_idx_g = _ag_routing_indices(ws.idx_symm, topk_idx_l, ws.ep_group)

    return _moe_ep_forward_inner(
        x_local=x_in,
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
        mode=mode,
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
    """EP MoE forward with caller-supplied routing decision (uniform K).

    Caller contract: x is (T_local, d); topk_indices and topk_scores are
    both (T_local, K). The forward does not check shapes — feed it aligned
    inputs.

    `mode='auto'` (default) selects 'ag' or 'a2a' from W, K, and the device's
    compute capability — see module docstring."""
    K = topk_indices.shape[1]
    W, E_local, mode = _validate_and_resolve(mgr, mode, E, K)
    activation_type = _normalize_activation(activation_type)
    T_local, d = x.shape

    ep_ws = mgr._get_or_alloc(T_local, d, K, E_local, x.dtype, mode)
    x_in = _stage_input(ep_ws, x)

    topk_idx_l = topk_indices.to(torch.int32)
    topk_idx_g = _ag_routing_indices(ep_ws.idx_symm, topk_idx_l, ep_ws.ep_group)

    return _moe_ep_forward_inner(
        x_local=x_in,
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
        mode=mode,
    )
