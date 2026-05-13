# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
#
# Layer-static EP config + per-rank symm-mem workspace + the
# manager that owns/caches workspaces.
# ********************************************************************************

from __future__ import annotations

import atexit
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed import _symmetric_memory as _symm_mem


__all__ = [
    "CombineMode",
    "DispatchMode",
    "NetworkProfiler",
    "RuntimeEPConfig",
    "SymmMemManager",
    "_EPWorkspace",
    "_is_ag_dispatch_mode",
    "_is_a2a_dispatch_mode",
    "_is_rank_dedup_dispatch_mode",
    "_is_a2a_combine_mode",
    "_is_rs_combine_mode",
    "_is_rank_dedup_combine_mode",
    "clear_ep_cache",
]


# ============================================================================
# Mode resolution
# ============================================================================


class DispatchMode(str, Enum):
    """Dispatch strategy for the EP forward / backward."""

    AG_DISPATCH_TRITON = "AG_DISPATCH_TRITON"
    A2A_DISPATCH_TRITON = "A2A_DISPATCH_TRITON"
    RANK_DEDUP_DISPATCH_TRITON = "RANK_DEDUP_DISPATCH_TRITON"


class CombineMode(str, Enum):
    """Combine strategy for the forward combine + backward dx combine."""

    A2A_COMBINE_TRITON = "A2A_COMBINE_TRITON"
    RS_COMBINE_TRITON = "RS_COMBINE_TRITON"
    RANK_DEDUP_COMBINE_TRITON = "RANK_DEDUP_COMBINE_TRITON"


def _is_ag_dispatch_mode(mode: DispatchMode) -> bool:
    return mode == DispatchMode.AG_DISPATCH_TRITON


def _is_a2a_dispatch_mode(mode: DispatchMode) -> bool:
    return mode == DispatchMode.A2A_DISPATCH_TRITON


def _is_rank_dedup_dispatch_mode(mode: DispatchMode) -> bool:
    return mode == DispatchMode.RANK_DEDUP_DISPATCH_TRITON


def _is_a2a_combine_mode(mode: CombineMode) -> bool:
    return mode == CombineMode.A2A_COMBINE_TRITON


def _is_rs_combine_mode(mode: CombineMode) -> bool:
    return mode == CombineMode.RS_COMBINE_TRITON


def _is_rank_dedup_combine_mode(mode: CombineMode) -> bool:
    return mode == CombineMode.RANK_DEDUP_COMBINE_TRITON


# ============================================================================
# Per-workspace symm-mem state
# ============================================================================


@dataclass
class _EPWorkspace:
    # x_symm — (T_local, d) NVLink P2P staging. Forward writes x;
    # the backward X redispatch reads peers' published x_local from
    # it. With redispatch on, step 1's do publish goes through
    # ``do_symm`` instead so x_symm is preserved across the whole
    # forward→backward sequence; without redispatch, step 1 freely
    # overwrites x_symm with dout (x_symm has no further consumer).
    x_symm: Optional[torch.Tensor]
    x_hdl: Any
    x_peer_bufs: Tuple[torch.Tensor, ...]
    # y_symm — (MAX_ROWS_PER_RANK, d) gemm output in forward; reused
    # as dx_expanded in backward. Sized to the structural ceiling on
    # local-routed slots (rather than TK_global) since neither the
    # forward down-proj nor the backward up-proj-act ever writes
    # past it, and peers only gather within that range too.
    y_symm: Optional[torch.Tensor]
    o_hdl: Any
    y_peer_bufs: Tuple[torch.Tensor, ...]
    # s_rev_symm — (TK_global,) reverse dispatch index for the gather.
    s_rev_symm: Optional[torch.Tensor]
    s_rev_hdl: Any
    s_rev_peer_bufs: Tuple[torch.Tensor, ...]

    ep_group: dist.ProcessGroup
    world_size: int
    my_rank: int
    E_local: int
    _T_local: int
    _K: int
    _d: int
    dispatch_mode: DispatchMode = DispatchMode.AG_DISPATCH_TRITON

    a2a_recv: Optional[torch.Tensor] = None
    ag_compute: Optional[torch.Tensor] = None
    t_global_pattern: Optional[torch.Tensor] = None

    # do_symm — (T_local, d) NVLink P2P staging for the backward do
    # dispatch. Allocated *during the backward pass* (runtime) by
    # :meth:`_ensure_do_symm` on the first call where
    # ``redispatch_x_in_backward=True``; without redispatch the do
    # dispatch reuses ``x_symm`` and do_symm stays None. x_symm and
    # do_symm never need to coexist in the no-redispatch path, so we
    # save one (T_local, d) symm allocation per workspace for the
    # common case.
    do_symm: Optional[torch.Tensor] = None
    do_hdl: Any = None
    do_peer_bufs: Tuple[torch.Tensor, ...] = ()

    # Persistent (W*T_local, d) buffer used as the AG-CE destination
    # for the backward X redispatch. Lazy-allocated on first redispatch
    # call. Distinct from ``ag_compute`` to avoid aliasing with the dO
    # dispatch in AG mode (step 1 writes dO into ag_compute, the CE in
    # step 2 must not clobber it before down-proj-act reads it).
    _ag_redispatch_buf: Optional[torch.Tensor] = None

    # partial_combine_buf — (W*T_local, d) bf16 symm-mem staging for the
    # per-(home_rank, home_t) partial-combine accumulator. Shared by both
    # combine paths that use a local pre-sum:
    #   * ``RS_COMBINE_TRITON``: ``local_combine`` writes here locally,
    #     then ``reduce_scatter_triton`` reads peer partial_combine_buf to
    #     combine across ranks.
    #   * ``RANK_DEDUP_COMBINE_TRITON``: ``local_combine`` writes
    #     non-self stripes here, then the per-token sparse gather kernel
    #     peer-reads them via NVLink.
    # Lazy-allocated only when one of these agg paths is actually used.
    # Storage is the model dtype (bf16) — both producer and consumer
    # kernels accumulate in fp32 registers and cast at store time, so
    # NVLink moves half the bytes a fp32 partial_combine_buf would. End-to-end this
    # adds one extra bf16 rounding (partial_combine_buf store) on top of the
    # unavoidable y_local cast — total error per element ≈ 2·ε_bf16,
    # independent of K and W.
    partial_combine_buf: Optional[torch.Tensor] = None
    partial_combine_hdl: Any = None
    partial_combine_peer_bufs: Tuple[torch.Tensor, ...] = ()

    # x_idx_expanded_remap_for_rank_dedup_buf — (MAX_ROWS_PER_RANK_STATIC,) int32 plain HBM,
    # the up-proj GEMM's A_idx for the dedup mode. Built every forward
    # by ``build_rank_dedup_a_idx`` and consumed by gemm_gated /
    # gemm_dgated / dW2 / dW1 GEMMs. Allocated only when the workspace
    # is configured for RANK_DEDUP_DISPATCH_TRITON
    # (``dispatch_mode == DispatchMode.RANK_DEDUP_DISPATCH_TRITON``);
    # None otherwise.
    x_idx_expanded_remap_for_rank_dedup_buf: Optional[torch.Tensor] = None

    @property
    def T_local(self) -> int:
        return self._T_local

    @property
    def K(self) -> int:
        return self._K

    @property
    def d(self) -> int:
        return self._d

    def _ensure_do_symm(self) -> None:
        """Lazy-allocate do_symm on the first backward call that needs it.

        Called from backward step 1 when ``redispatch_x_in_backward`` is
        set. The allocation is collective (``_symm_mem.rendezvous``)
        but symmetric — all ranks enter this backward at the same
        logical point. First call pays the rendezvous cost; later
        backwards reuse the buffer.
        """
        if self.do_symm is not None:
            return
        shape = (self._T_local, self._d)
        dtype = self.x_symm.dtype
        device = self.x_symm.device
        buf = _symm_mem.empty(*shape, dtype=dtype, device=device)
        hdl = _symm_mem.rendezvous(buf, group=self.ep_group)
        peer_bufs = tuple(hdl.get_buffer(r, shape, dtype) for r in range(self.world_size))
        self.do_symm = buf
        self.do_hdl = hdl
        self.do_peer_bufs = peer_bufs

    def _ensure_partial_combine_buf(self) -> None:
        """Lazy-allocate partial_combine_buf for the RS_COMBINE_TRITON
        and RANK_DEDUP_COMBINE_TRITON combine paths.

        Called from forward/backward step 4 when ``cfg.combine_mode`` is
        ``RS_COMBINE_TRITON`` or ``RANK_DEDUP_COMBINE_TRITON``.
        ``(W·T_local, d)`` symm-mem buffer in the model dtype (bf16) —
        register accumulation stays fp32 in both producer and consumer
        kernels; only the storage is bf16, which halves NVLink bytes
        for the cross-rank read. The allocation is collective
        (``_symm_mem.rendezvous``) but symmetric — all ranks enter
        this combine at the same logical point. First call pays the
        rendezvous cost; later calls reuse.
        """
        if self.partial_combine_buf is not None:
            return
        W = self.world_size
        shape = (W * self._T_local, self._d)
        device = self.x_symm.device
        dtype = self.x_symm.dtype
        buf = _symm_mem.empty(*shape, dtype=dtype, device=device)
        hdl = _symm_mem.rendezvous(buf, group=self.ep_group)
        peer_bufs = tuple(hdl.get_buffer(r, shape, dtype) for r in range(W))
        self.partial_combine_buf = buf
        self.partial_combine_hdl = hdl
        self.partial_combine_peer_bufs = peer_bufs

    def release(self) -> None:
        """Drop ALL tensor refs (use when evicting / disposing).

        After this returns no field pins GPU memory; calling forward on
        a released workspace is undefined.

        Destruction order matters: peer_bufs are per-rank torch.Tensor
        aliases of the local symm allocation, so they must drop BEFORE
        the symm tensor itself. Otherwise the SymmetricMemory backing
        the symm tensor can be torn down while peer aliases still
        hold an AllocationRef into it; their later destructor calls
        cuMemUnmap on freed pages and ``~AllocationRef`` throws
        ``CUDA driver error: invalid argument`` (which is noexcept-
        false ⇒ ``std::terminate`` ⇒ SIGABRT).
        """
        self.x_peer_bufs = ()
        self.do_peer_bufs = ()
        self.y_peer_bufs = ()
        self.s_rev_peer_bufs = ()
        self.partial_combine_peer_bufs = ()
        # Each *_hdl is the SymmetricMemory wrapper for its buffer
        # If it survives past the symm tensor below it would keep an internal AllocationRef
        # alive and we'd hit the same cuMemUnmap("invalid argument") crash one rendezvous later.
        self.x_hdl = None
        self.do_hdl = None
        self.o_hdl = None
        self.s_rev_hdl = None
        self.partial_combine_hdl = None
        # ↓ now the symm-mem buffers themselves.
        self.x_symm = None
        self.do_symm = None
        self.y_symm = None
        self.s_rev_symm = None
        self.partial_combine_buf = None
        # Plain HBM (not symm-mem) — order doesn't matter.
        self.a2a_recv = None
        self.ag_compute = None
        self.t_global_pattern = None
        self._ag_redispatch_buf = None
        self.x_idx_expanded_remap_for_rank_dedup_buf = None


# ============================================================================
# Workspace cache / allocator
# ============================================================================


class SymmMemManager:
    # Process-wide intern table keyed by ``(id(group), str(device))`` —
    # one manager per ``(group, device)`` pair so workspaces and the
    # symm-mem rendezvous they hold are reused across forward calls.
    # ``__new__`` returns the cached instance on a hit; ``__init__`` is
    # guarded by ``_initialized`` so the body runs exactly once per
    # intern. CPython's GIL makes the get/setitem pair atomic for our
    # single-main-thread usage; concurrent rendezvous would already be
    # broken at the collectives layer so we don't add a lock.
    _instances: "dict[Tuple[int, str], SymmMemManager]" = {}

    def __new__(
        cls,
        ep_group: Optional[dist.ProcessGroup] = None,
        device: Optional[Union[torch.device, int, str]] = None,
    ) -> "SymmMemManager":
        if ep_group is None:
            ep_group = dist.group.WORLD
        if device is None:
            device = torch.cuda.current_device()
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        key = (id(ep_group), str(dev))
        inst = cls._instances.get(key)
        if inst is None:
            inst = super().__new__(cls)
            inst._initialized = False
            cls._instances[key] = inst
        return inst

    def __init__(
        self,
        ep_group: Optional[dist.ProcessGroup] = None,
        device: Optional[Union[torch.device, int, str]] = None,
    ) -> None:
        if getattr(self, "_initialized", False):
            return
        if ep_group is None:
            ep_group = dist.group.WORLD
        if device is None:
            device = torch.cuda.current_device()
        self.ep_group = ep_group
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.world_size = dist.get_world_size(ep_group)
        self.rank = dist.get_rank(ep_group)
        self._cache: dict = dict()
        self._initialized = True

    def clear(self) -> None:
        # Release each workspace's tensor refs explicitly. Dropping
        # the dict alone is normally sufficient (the workspace is
        # then unreferenced and Python collects it), but autograd
        # ctx may still pin a workspace via ctx.ep_ws if a backward
        # is pending. Explicit release lets the caching allocator
        # reclaim symm-mem and ws_buf in those cases too.
        for ws in self._cache.values():
            ws.release()
        self._cache.clear()

    def prewarm(
        self,
        T_locals: Sequence[int],
        *,
        d: int,
        K: int,
        E_local: int,
        dtype: torch.dtype,
        mode: DispatchMode,
    ) -> None:
        for T in T_locals:
            self._get_or_alloc(T, d, K, E_local, dtype, mode)

    def _alloc_symm(
        self, shape: Tuple[int, ...], dtype: torch.dtype
    ) -> Tuple[torch.Tensor, Any, Tuple[torch.Tensor, ...]]:
        buf = _symm_mem.empty(*shape, dtype=dtype, device=self.device)
        hdl = _symm_mem.rendezvous(buf, group=self.ep_group)
        peer_bufs = tuple(hdl.get_buffer(r, shape, dtype) for r in range(self.world_size))
        return buf, hdl, peer_bufs

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
        TK_local = T_local * K
        TK_global = W * TK_local
        # Per-rank ceiling on (token, expert) slots routed to this
        # rank — only this many rows of y_symm ever get written, so we
        # size the symm buffer at this bound instead of TK_global.
        MAX_ROWS_PER_RANK = T_local * W * min(K, E_local)
        dev = self.device

        x_symm, x_hdl, x_peer_bufs = self._alloc_symm((T_local, d), dtype)
        # do_symm is lazy — only the redispatch path needs it (so
        # x_symm can stay live for the X redispatch in step 2).
        y_symm, o_hdl, y_peer_bufs = self._alloc_symm((MAX_ROWS_PER_RANK, d), dtype)
        s_rev_symm, s_rev_hdl, s_rev_peer_bufs = self._alloc_symm((TK_global,), torch.int32)

        a2a_recv = None
        ag_compute = None
        t_global_pattern = None
        x_idx_expanded_remap_for_rank_dedup_buf = None

        # A2A_DISPATCH_TRITON consumes an expert-grouped recv buffer; allocate ``a2a_recv``.
        # RANK_DEDUP_DISPATCH_TRITON consumes the dedup packed buffer DIRECTLY via an
        # A_idx-driven GEMM gather — no separate expert-grouped buffer
        # needed. The A_idx tensor itself lives in ``x_idx_expanded_remap_for_rank_dedup_buf``.
        # AG_DISPATCH_TRITON uses the (W*T_local, d) gather buffer instead.
        if _is_a2a_dispatch_mode(mode):
            a2a_recv = torch.empty((W, TK_local, d), dtype=dtype, device=dev)
        elif _is_rank_dedup_dispatch_mode(mode):
            x_idx_expanded_remap_for_rank_dedup_buf = torch.empty(MAX_ROWS_PER_RANK, dtype=torch.int32, device=dev)
        else:
            ag_compute = torch.empty((W * T_local, d), dtype=dtype, device=dev)
            t_global_pattern = torch.arange(TK_global, device=dev, dtype=torch.int32) // K

        return _EPWorkspace(
            x_symm=x_symm,
            x_hdl=x_hdl,
            x_peer_bufs=x_peer_bufs,
            y_symm=y_symm,
            o_hdl=o_hdl,
            y_peer_bufs=y_peer_bufs,
            s_rev_symm=s_rev_symm,
            s_rev_hdl=s_rev_hdl,
            s_rev_peer_bufs=s_rev_peer_bufs,
            ep_group=self.ep_group,
            world_size=W,
            my_rank=self.rank,
            E_local=E_local,
            _T_local=T_local,
            _K=K,
            _d=d,
            dispatch_mode=mode,
            a2a_recv=a2a_recv,
            ag_compute=ag_compute,
            t_global_pattern=t_global_pattern,
            x_idx_expanded_remap_for_rank_dedup_buf=x_idx_expanded_remap_for_rank_dedup_buf,
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
# Process-wide EP workspace cache
# ----------------------------------------------------------------------------
# ``SymmMemManager`` interns one instance per ``(group, device)`` pair on
# its own class — see ``SymmMemManager._instances``. The cache state
# lives there; this section just wires the lifecycle hooks.
#
# Why hooks at all: a manager holds symm-mem allocations + peer-buffer
# handles that need to be released BEFORE the CUDA context tears down —
# otherwise ``~CUDASymmetricMemory`` calls ``cuMemUnmap`` on a dead
# context and throws from a destructor → ``std::terminate``. The
# ``atexit`` hook runs ahead of PyTorch's own CUDA-teardown atexit
# (reverse-registration order, and we import after torch.cuda is set up)
# so symm-mem is freed while the driver is still live. The fork hook
# drops cached entries in the child so inherited mappings — which point
# at the parent's CUDA context — don't get freed in-child.
# ============================================================================


def clear_ep_cache() -> None:
    """Release every cached EP symm-mem workspace.

    Registered as an ``atexit`` and fork-after-in-child hook at import
    time, so normally there's no need to call this explicitly. Useful
    if you destroy and re-create a process group mid-process, or want
    to free GPU memory between independent runs in the same script.
    """
    instances = list(SymmMemManager._instances.values())
    SymmMemManager._instances.clear()
    for mgr in instances:
        try:
            mgr.clear()
        except Exception:
            # We're often in the middle of interpreter / driver teardown;
            # swallow so the rest of the cache still gets a chance to drop.
            pass


atexit.register(clear_ep_cache)
os.register_at_fork(after_in_child=SymmMemManager._instances.clear)


# ============================================================================
# Runtime config + network profiler
# ----------------------------------------------------------------------------
# ``RuntimeEPConfig`` is the user-facing handle for the dispatch
# decision. ``NetworkProfiler.profile()`` benchmarks the three
# dispatch primitives on the actual hardware and returns one. The
# public ``moe_ep_*_forward`` entry points accept it as an optional
# ``ep_config=`` argument.
# ============================================================================


@dataclass(frozen=True)
class RuntimeEPConfig:
    """End-to-end EP runtime config: workload-level decisions + layer-static dims.
    * ``dispatch_mode`` — dispatch mode (AG_DISPATCH_TRITON / A2A_DISPATCH_TRITON / RANK_DEDUP_DISPATCH_TRITON),
      used for the forward x dispatch and the backward dO dispatch.
    * ``W`` — EP world size (validated against the current process group at call time).
    * ``K`` — top-K experts per token (validated against the call's K).
    * ``combine_mode`` — combine mode (A2A_COMBINE_TRITON / RS_COMBINE_TRITON /
      RANK_DEDUP_COMBINE_TRITON), used for the forward combine and the
      backward dx combine.
    * ``MAX_ROWS_PER_RANK_STATIC`` — worst-case (token, expert) slots
      routed to this rank, given by ``T_local · W · min(K, E_local)``.
      Top-K's distinct-pick guarantee plus this rank holding ``E_local``
      of ``E`` experts means at most ``min(K, E_local)`` of any token's
      picks land here; the bound is achievable (degenerate router) and
      so cannot be tightened without routing assumptions. Equals
      ``TK_global`` in the paper-typical ``E_local ≥ K`` regime;
      tightens to ``T_local · E`` only when ``E_local < K``.
    """

    W: int
    K: int
    dispatch_mode: DispatchMode = DispatchMode.A2A_DISPATCH_TRITON
    combine_mode: CombineMode = CombineMode.A2A_COMBINE_TRITON
    I: Optional[int] = None
    is_glu_act: Optional[bool] = None
    E_local: Optional[int] = None
    # Per-expert GEMM output bound (h, a, y_symm, A_idx domain). Always
    # ``T_local · W · min(K, E_local)``: each (token, expert) slot routed
    # here gets its own row, regardless of dispatch mode. RANK_DEDUP's
    # dedup-packed input is gathered via A_idx into K-fanout outputs.
    MAX_ROWS_PER_RANK_STATIC: Optional[int] = None


class NetworkProfiler:
    """Benchmark AG_DISPATCH_TRITON / A2A_DISPATCH_TRITON / RANK_DEDUP_DISPATCH_TRITON and pick a winner.

    Time each dispatch primitive on the local hardware for the given
    ``(T_local, d, K)`` shape, with synthetic uniform routing for the
    A2A measurement (``E = K · W`` so ``E_local = K``, balanced).
    Returns a :class:`RuntimeEPConfig` whose ``dispatch_mode`` and
    ``combine_mode`` fields are the fastest of each kind.

    Recommended usage: call once per ``(T_local, d, K, dtype)``
    after ``dist.init_process_group`` but before the training loop;
    pass the result as ``ep_config=`` to every subsequent
    ``moe_ep_TC_softmax_topk_forward`` / ``moe_ep_general_routing_forward``
    call.

    Args:
        T_local: Tokens per rank.
        H: Hidden dim (the "d" axis of x).
        K: Top-K.
        dtype: x dtype (``torch.bfloat16`` / ``torch.float16`` etc).
        group: EP process group. Defaults to ``dist.group.WORLD``.
        device: CUDA device for the symm-mem buffers. Defaults to the
            current CUDA device.
        warmup: Warm-up iterations before timing each kernel.
        repeat: Timed iterations per kernel.
    """

    def __init__(
        self,
        T_local: int,
        H: int,
        K: int,
        dtype: torch.dtype,
        *,
        group: Optional[dist.ProcessGroup] = None,
        device: Optional[Union[torch.device, int, str]] = None,
        warmup: int = 10,
        repeat: int = 50,
    ) -> None:
        if device is None:
            device = torch.cuda.current_device()
        self.mgr = SymmMemManager(group, device)
        self.T_local = T_local
        self.H = H
        self.K = K
        self.dtype = dtype
        self.warmup = warmup
        self.repeat = repeat

    def profile(self) -> RuntimeEPConfig:
        # Local imports to avoid a top-level dependency on the Triton
        # comm primitives (keeps distributed_utils.py importable without
        # a CUDA context for tests / type stubs).
        from .functional.distributed import (
            a2a_combine_triton,
            a2a_dispatch_triton,
            all_gather_triton,
            compute_dispatch_metadata,
            rank_dedup_combine_triton,
            rank_dedup_dispatch_triton,
            rs_combine_triton,
        )
        from .functional.metadata import general_routing_router_metadata_triton

        mgr = self.mgr
        W = mgr.world_size
        rank = mgr.rank
        device = mgr.device
        T_local = self.T_local
        H = self.H
        K = self.K
        dtype = self.dtype

        # ── Setup symm-mem buffer for the AG primitives ──────────────
        # NOTE: this is an extra rendezvous outside the SymmMemManager
        # cache — that's fine because the buffer + handle + peer_bufs
        # are released in the right order at the bottom of this method.
        x_symm = _symm_mem.empty((T_local, H), dtype=dtype, device=device)
        x_symm.normal_()
        x_hdl = _symm_mem.rendezvous(x_symm, group=mgr.ep_group)
        x_peer_bufs = tuple(x_hdl.get_buffer(r, (T_local, H), dtype) for r in range(W))
        x_hdl.barrier()  # publish x_symm to peers before the AG reads

        ag_out = torch.empty(W * T_local, H, dtype=dtype, device=device)

        def ag_triton_call() -> None:
            all_gather_triton(x_symm, mgr.ep_group, out=ag_out, peer_bufs=x_peer_bufs)

        # ── Setup A2A: synthetic uniform routing with E = K · W ──────
        # E = K·W ⇒ E_local = K, perfectly balanced (each rank pulls
        # exactly TK_local rows from peers on average).
        E = K * W
        E_local = K
        TK_local = T_local * K
        TK_global = W * TK_local

        # Same-on-all-ranks routing: every rank seeds with 0 so the
        # gathered (W, T_local, K) tensor is consistent.
        rng = torch.Generator(device=device).manual_seed(0)
        topk_idx_local = torch.randint(0, E, (T_local, K), generator=rng, device=device, dtype=torch.int32)
        topk_idx_global = torch.empty(W * T_local * K, dtype=torch.int32, device=device)
        dist.all_gather_into_tensor(
            topk_idx_global,
            topk_idx_local.view(-1).contiguous(),
            group=mgr.ep_group,
        )
        topk_idx_global = topk_idx_global.view(W, T_local, K)

        meta = compute_dispatch_metadata(topk_idx_global, my_rank=rank, E_local=E_local)
        dst_rank_flat = meta["dst_rank_flat"]
        expert_local_padded = meta["expert_local_padded"]
        a2a_token_indices = meta["a2a_token_indices"]
        my_dst_rank = meta["my_dst_rank"]  # (T_local, K) — for A2A_combine

        # Build s_reverse_local (the A2A recv_pos) via the metadata
        # kernel — same as ``_build_consumer_metadata`` in distributed.py.
        E_total = E_local + 1
        a2a_s_rev = torch.empty(TK_global, dtype=torch.int32, device=device)
        _x_gather_idx = torch.empty(TK_global, dtype=torch.int32, device=device)
        _s_scatter_idx = torch.empty(TK_global, dtype=torch.int32, device=device)
        _expert_freq = torch.empty(E_total, dtype=torch.int32, device=device)
        _expert_freq_off = torch.empty(E_total + 1, dtype=torch.int32, device=device)
        general_routing_router_metadata_triton(
            a2a_token_indices,
            expert_local_padded,
            TK_global,
            E_total,
            _expert_freq,
            _expert_freq_off,
            _x_gather_idx,
            _s_scatter_idx,
            a2a_s_rev,
            None,
        )

        a2a_recv = torch.empty(W, TK_local, H, dtype=dtype, device=device)

        def a2a_call() -> None:
            a2a_dispatch_triton(
                x_symm=x_symm,
                dst_rank_flat=dst_rank_flat,
                recv_pos=a2a_s_rev,
                recv=a2a_recv,
                K=K,
                group=mgr.ep_group,
                peer_bufs=x_peer_bufs,
                my_rank=rank,
            )

        # RANK_DEDUP dispatch: same metadata, plus pair_present_mask /
        # rank_dedup_recv_pos. Packed buffer sized to MAX_PAIR_COUNT =
        # W·T_local. Local-write only (peers don't read from the recv
        # buffer), so plain ``torch.empty`` — no symm-mem rendezvous
        # needed.
        MAX_PAIR_COUNT = W * T_local
        rank_dedup_packed_local = torch.empty((MAX_PAIR_COUNT, H), dtype=dtype, device=device)

        def rank_dedup_dispatch_call() -> None:
            rank_dedup_dispatch_triton(
                x_symm,
                dst_rank_flat,
                meta["pair_present_mask"],
                meta["rank_dedup_recv_pos"],
                rank_dedup_packed_local,
                K=K,
                group=mgr.ep_group,
                peer_bufs=x_peer_bufs,
                my_rank=rank,
            )

        # ── Time each dispatch primitive ─────────────────────────────
        t_ag_triton = self._bench(ag_triton_call)
        t_a2a = self._bench(a2a_call)
        t_rank_dedup_dispatch = self._bench(rank_dedup_dispatch_call)

        timings = {
            DispatchMode.AG_DISPATCH_TRITON: t_ag_triton,
            DispatchMode.A2A_DISPATCH_TRITON: t_a2a,
            DispatchMode.RANK_DEDUP_DISPATCH_TRITON: t_rank_dedup_dispatch,
        }
        winner = min(timings, key=timings.get)

        # ── Setup symm-mem buffers for the combine primitives ────
        # a2a_combine_triton reads peers' y_symm + s_rev_symm in
        # a single fused kernel. local_combine writes a bf16
        # partial_combine_buf locally; reduce_scatter_triton and the
        # rank_dedup sparse gather both consume that buffer across
        # ranks. All three reuse the routing meta we already computed
        # above.
        y_symm = _symm_mem.empty((TK_global, H), dtype=dtype, device=device)
        y_symm.normal_()
        o_hdl = _symm_mem.rendezvous(y_symm, group=mgr.ep_group)
        y_peer_bufs = tuple(o_hdl.get_buffer(r, (TK_global, H), dtype) for r in range(W))

        s_rev_symm = _symm_mem.empty((TK_global,), dtype=torch.int32, device=device)
        s_rev_symm.copy_(a2a_s_rev)
        s_rev_hdl = _symm_mem.rendezvous(s_rev_symm, group=mgr.ep_group)
        s_rev_peer_bufs = tuple(s_rev_hdl.get_buffer(r, (TK_global,), torch.int32) for r in range(W))

        # One barrier after the writes; covers both y_symm and s_rev_symm
        # (cached symm-mem handles fence the rank's whole stream).
        o_hdl.barrier()

        # a2a_combine_triton inputs: per-rank scores (T_local, K).
        topk_scores_local = torch.softmax(
            torch.randn(T_local, K, device=device, dtype=torch.float32),
            dim=-1,
        )
        gather_out = torch.empty(T_local, H, dtype=dtype, device=device)

        def gather_call() -> None:
            a2a_combine_triton(
                y_symm,
                s_rev_symm,
                my_dst_rank,
                topk_scores_local,
                gather_out,
                K=K,
                group=mgr.ep_group,
                y_peer_bufs=y_peer_bufs,
                s_peer_bufs=s_rev_peer_bufs,
                my_rank=rank,
            )

        scores_global = torch.empty(W * T_local, K, device=device, dtype=torch.float32)
        dist.all_gather_into_tensor(scores_global, topk_scores_local.contiguous(), group=mgr.ep_group)
        scores_flat = scores_global.view(-1)

        partial_combine_buf = _symm_mem.empty((W * T_local, H), dtype=dtype, device=device)
        partial_combine_hdl = _symm_mem.rendezvous(partial_combine_buf, group=mgr.ep_group)
        partial_combine_peer_bufs = tuple(partial_combine_hdl.get_buffer(r, (W * T_local, H), dtype) for r in range(W))
        rs_out = torch.empty(T_local, H, dtype=dtype, device=device)

        def rs_call() -> None:
            rs_combine_triton(
                y_symm,
                s_rev_symm,
                dst_rank_flat,
                scores_flat,
                partial_combine_buf,
                rs_out,
                K,
                T_local,
                group=mgr.ep_group,
                partial_combine_hdl=partial_combine_hdl,
                partial_combine_peer_bufs=partial_combine_peer_bufs,
                my_rank=rank,
            )

        out_dedup = torch.empty(T_local, H, dtype=dtype, device=device)

        def dedup_call() -> None:
            rank_dedup_combine_triton(
                y_symm,
                s_rev_symm,
                dst_rank_flat,
                scores_flat,
                meta["peer_present_mask"],
                partial_combine_buf,
                out_dedup,
                K=K,
                T_local=T_local,
                group=mgr.ep_group,
                partial_combine_hdl=partial_combine_hdl,
                partial_combine_peer_bufs=partial_combine_peer_bufs,
                my_rank=rank,
            )

        # ── Time combine primitives ──────────────────────────────
        t_gather = self._bench(gather_call)
        t_rs = self._bench(rs_call)
        t_dedup = self._bench(dedup_call)
        combine_timings = {
            CombineMode.A2A_COMBINE_TRITON: t_gather,
            CombineMode.RS_COMBINE_TRITON: t_rs,
            CombineMode.RANK_DEDUP_COMBINE_TRITON: t_dedup,
        }
        combine_winner = min(combine_timings, key=combine_timings.get)

        # ── Dispatch-dedup byte volume from the actual routing ───────
        # Compute the per-rank cross-rank receive count from the
        # measured (W, W) pair_count, then average across ranks so it
        # matches the per-rank mean time used in the GB/s reporting
        # below. The previous analytical fallback
        # ``(W-1) · T_local · (1 - (1-1/W)^K) · H · itemsize`` is only
        # exact under balanced uniform routing.
        itemsize = torch.tensor([], dtype=dtype).element_size()
        pc = meta["pair_count"]
        dedup_rows_local = int(pc[:, rank].sum().item() - pc[rank, rank].item())
        _dedup_rows_t = torch.tensor([dedup_rows_local], dtype=torch.int64, device=device)
        dist.all_reduce(_dedup_rows_t, op=dist.ReduceOp.SUM, group=mgr.ep_group)
        dedup_bytes = (_dedup_rows_t.item() / W) * H * itemsize

        # Expose per-rank-mean timings (ms) so callers can report the
        # measured cost of the chosen dispatch/combine pair without
        # re-running the bench.
        self.dispatch_timings_ms = {mode: t / W * 1e3 for mode, t in timings.items()}
        self.combine_timings_ms = {mode: t / W * 1e3 for mode, t in combine_timings.items()}

        # ── Cleanup symm-mem in safe destruction order ───────────────
        # peer aliases first → handle next → buffer last (mirrors
        # _EPWorkspace.release()'s rationale).
        del x_peer_bufs, y_peer_bufs, s_rev_peer_bufs, partial_combine_peer_bufs
        del x_hdl, o_hdl, s_rev_hdl, partial_combine_hdl
        del x_symm, y_symm, s_rev_symm, partial_combine_buf, rank_dedup_packed_local
        del a2a_recv, a2a_s_rev, out_dedup
        del gather_out, rs_out
        del topk_scores_local, scores_global, scores_flat
        del _x_gather_idx, _s_scatter_idx, _expert_freq, _expert_freq_off
        del topk_idx_global, topk_idx_local
        del meta, dst_rank_flat, expert_local_padded, a2a_token_indices, my_dst_rank
        torch.cuda.empty_cache()

        if rank == 0:
            # `_bench` returns the SUM across W ranks (chosen so that
            # A2A's per-rank variance under random routing doesn't get
            # picked up by MAX; SUM == W·MEAN preserves ordering).
            # Display the per-rank mean = sum/W, and compute per-rank
            # GB/s against that mean.
            #
            # Bytes traversing NVLink per rank per call:
            #   AG: (W-1) · T_local · H · itemsize_dtype     (received)
            #   A2A: (W-1)/W · TK_local · H · itemsize_dtype (received)
            #   A2A_COMBINE_TRITON: same as A2A_DISPATCH_TRITON's volume (peer y_symm reads).
            #   RS pipeline: dominated by reduce_scatter, which moves
            #     (W-1) · T_local · H · itemsize_dtype bytes per rank
            #     (partial_combine_buf is now stored in the model dtype; reducer
            #     register acc stays fp32, store casts at the boundary).
            ag_bytes = (W - 1) * T_local * H * itemsize
            a2a_bytes = ((W - 1) / W) * TK_local * H * itemsize
            gather_bytes = a2a_bytes
            # ``dedup_bytes`` was already computed above from the actual
            # ``meta["pair_count"]`` (cross-rank averaged).
            t_ag_triton_mean = t_ag_triton / W
            t_a2a_mean = t_a2a / W
            t_rank_dedup_dispatch_mean = t_rank_dedup_dispatch / W
            t_gather_mean = t_gather / W
            t_rs_mean = t_rs / W
            t_dedup_mean = t_dedup / W
            ag_triton_gbs = ag_bytes / t_ag_triton_mean / 1e9
            a2a_gbs = a2a_bytes / t_a2a_mean / 1e9
            rank_dedup_dispatch_gbs = dedup_bytes / t_rank_dedup_dispatch_mean / 1e9
            gather_gbs = gather_bytes / t_gather_mean / 1e9
            # RS / RANK_DEDUP_COMBINE_TRITON: a single NVLink GB/s figure is
            # misleading — both paths spend significant time in the
            # HBM-bound ``local_combine`` producer step, so a "bytes /
            # mean_time" ratio mixes HBM and NVLink throughputs. Drop
            # the GB/s annotation; ``benchmarks/distributed/bench-ep-comm.py``
            # reports the two tiers separately when wanted.
            head = f"[NetworkProfiler] T_local={T_local} H={H} K={K} W={W}"
            print(
                f"{head}\n"
                f"  Dispatch:    "
                f"AG_DISPATCH_TRITON={t_ag_triton_mean * 1e3:.2f}ms ({ag_triton_gbs:.1f} GB/s)  "
                f"A2A_DISPATCH_TRITON={t_a2a_mean * 1e3:.2f}ms ({a2a_gbs:.1f} GB/s)  "
                f"RANK_DEDUP_DISPATCH_TRITON={t_rank_dedup_dispatch_mean * 1e3:.2f}ms ({rank_dedup_dispatch_gbs:.1f} GB/s)  "
                f"→  winner={winner.value}\n"
                f"  Combine: "
                f"A2A_COMBINE_TRITON={t_gather_mean * 1e3:.2f}ms ({gather_gbs:.1f} GB/s)  "
                f"RS_COMBINE_TRITON={t_rs_mean * 1e3:.2f}ms  "
                f"RANK_DEDUP_COMBINE_TRITON={t_dedup_mean * 1e3:.2f}ms  "
                f"→  winner={combine_winner.value}",
                flush=True,
            )

        return RuntimeEPConfig(dispatch_mode=winner, W=W, K=K, combine_mode=combine_winner)

    def _bench(self, fn) -> float:
        """Time ``fn`` over ``warmup + repeat`` iterations.

        Returns mean ms-per-iter, sum-reduced across ranks
        """
        for _ in range(self.warmup):
            fn()
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(self.repeat):
            fn()
        end.record()
        torch.cuda.synchronize()
        local_ms = start.elapsed_time(end) / self.repeat
        if dist.is_initialized():
            t = torch.tensor([local_ms], device=self.mgr.device, dtype=torch.float64)
            # choose sum here because A2A will be severely disfavored by max-reduction for random routing decisions.
            # sum-reduction slightly balances it.
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            return t.item() * 1e-3  # ms → s
        return local_ms * 1e-3
