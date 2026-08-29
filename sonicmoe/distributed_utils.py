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
    "_is_hier_node_dedup_dispatch_gin_mode",
    "_is_a2a_combine_mode",
    "_is_rs_combine_mode",
    "_is_rank_dedup_combine_mode",
    "_is_hier_node_dedup_combine_gin_mode",
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
    # AG via NVLink multicast (multimem.st) — same semantics as AG_DISPATCH_TRITON, rides the
    # multicast fabric instead of per-peer unicast reads. Requires NVLink SHARP / MNNVL.
    AG_DISPATCH_MULTIMEM_TRITON = "AG_DISPATCH_MULTIMEM_TRITON"
    # HIER inter-node dispatch over NCCL-GIN: node-dedup RDMA put + NVLink expand into the SAME
    # recv_packed[rank_dedup_recv_pos] layout as RANK_DEDUP. Degrades to RANK_DEDUP_DISPATCH_TRITON at num_nodes==1.
    HIER_NODE_DEDUP_DISPATCH_GIN = "HIER_NODE_DEDUP_DISPATCH_GIN"


class CombineMode(str, Enum):
    """Combine strategy for the forward combine + backward dx combine."""

    A2A_COMBINE_TRITON = "A2A_COMBINE_TRITON"
    RS_COMBINE_TRITON = "RS_COMBINE_TRITON"
    RANK_DEDUP_COMBINE_TRITON = "RANK_DEDUP_COMBINE_TRITON"
    # local_combine producer + reduce-scatter via NVLink multicast (multimem.ld_reduce) — same
    # semantics as RS_COMBINE_TRITON, rides the multicast fabric. Requires multicast support.
    RS_COMBINE_MULTIMEM_TRITON = "RS_COMBINE_MULTIMEM_TRITON"
    # HIER inter-node combine over NCCL-GIN: local node-reduce, ONE rack-reduced partial per remote
    # node via GIN (no RDMA float atomics), origin does the final reduce. Degrades at num_nodes==1.
    HIER_NODE_DEDUP_COMBINE_GIN = "HIER_NODE_DEDUP_COMBINE_GIN"


def _is_ag_dispatch_mode(mode: DispatchMode) -> bool:
    # Both AG variants share the same dispatch metadata + gather-GEMM path;
    # they differ only in which all-gather primitive moves the bytes.
    return mode in (DispatchMode.AG_DISPATCH_TRITON, DispatchMode.AG_DISPATCH_MULTIMEM_TRITON)


def _is_ag_dispatch_multimem_mode(mode: DispatchMode) -> bool:
    return mode == DispatchMode.AG_DISPATCH_MULTIMEM_TRITON


def _is_a2a_dispatch_mode(mode: DispatchMode) -> bool:
    return mode == DispatchMode.A2A_DISPATCH_TRITON


def _is_rank_dedup_dispatch_mode(mode: DispatchMode) -> bool:
    return mode == DispatchMode.RANK_DEDUP_DISPATCH_TRITON


def _is_hier_node_dedup_dispatch_gin_mode(mode: DispatchMode) -> bool:
    # Shares the RANK_DEDUP recv_packed/A_idx GEMM layout (so GEMM call sites treat it the same);
    # only the TRANSPORT differs (node-dedup GIN put + NVLink expand), gated on this predicate.
    return mode == DispatchMode.HIER_NODE_DEDUP_DISPATCH_GIN


def _is_a2a_combine_mode(mode: CombineMode) -> bool:
    return mode == CombineMode.A2A_COMBINE_TRITON


def _is_rs_combine_mode(mode: CombineMode) -> bool:
    # Both RS variants share the local_combine producer + reduce-scatter
    # structure; they differ only in the reduce-scatter primitive.
    return mode in (CombineMode.RS_COMBINE_TRITON, CombineMode.RS_COMBINE_MULTIMEM_TRITON)


def _is_rs_combine_multimem_mode(mode: CombineMode) -> bool:
    return mode == CombineMode.RS_COMBINE_MULTIMEM_TRITON


def _is_rank_dedup_combine_mode(mode: CombineMode) -> bool:
    return mode == CombineMode.RANK_DEDUP_COMBINE_TRITON


def _is_hier_node_dedup_combine_gin_mode(mode: CombineMode) -> bool:
    # Reuses the RANK_DEDUP local-combine producer + sparse gather within the node;
    # the cross-node reduce rides GIN (gated here).
    return mode == CombineMode.HIER_NODE_DEDUP_COMBINE_GIN


# ============================================================================
# Per-workspace symm-mem state
# ============================================================================


@dataclass
class _EPWorkspace:
    # x_symm — (T_local, d) NVLink P2P staging. Forward dispatch is its last consumer, so backward
    # safely overwrites it with dout to publish the dO dispatch.
    x_symm: Optional[torch.Tensor]
    x_hdl: Any
    x_peer_bufs: Tuple[torch.Tensor, ...]
    # y_symm — (MAX_ROWS_PER_RANK, d) gemm output (fwd) / dx_expanded (bwd). Sized to the local-routed
    # ceiling, not TK_global — neither GEMM writes past it, and peers only gather within that range.
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

    # partial_combine_buf — (W*T_local, d) bf16 pre-sum staging shared by RS/RANK_DEDUP combine (lazy-
    # alloc'd on first use). bf16 storage halves NVLink bytes; total error is ~2*eps_bf16, independent of K/W.
    partial_combine_buf: Optional[torch.Tensor] = None
    partial_combine_hdl: Any = None
    partial_combine_peer_bufs: Tuple[torch.Tensor, ...] = ()

    # a2a_peer_{y,s}_base — int64[W] peer base addresses for a2a_combine_triton's runtime peer
    # addressing. Built once (data_ptrs are constant for the allocation's lifetime) to stay host-sync-free.
    a2a_peer_y_base: Optional[torch.Tensor] = None
    a2a_peer_s_base: Optional[torch.Tensor] = None

    # x_idx_expanded_remap_for_rank_dedup_buf — up-proj GEMM's A_idx for dedup mode, rebuilt every
    # forward. Allocated only for RANK_DEDUP_DISPATCH_TRITON; None otherwise.
    x_idx_expanded_remap_for_rank_dedup_buf: Optional[torch.Tensor] = None

    # HIER NCCL-GIN state: populated only when use_gin selects a HIER mode; None on the flat path.
    # Forward/backward buffers are SEPARATE (invariant #7 — bwd dO dispatch must not clobber saved fwd recv_packed).
    gin_backend: Any = None
    num_nodes: int = 1
    node_size: int = 1
    x_gin_fwd: Any = None           # GinWindow (T_local*d) — fwd x: put source + same-node read source
    x_gin_bwd: Any = None           # GinWindow (T_local*d) — bwd dO
    dst_node_buf_fwd: Any = None    # GinWindow (DST_NODE_BUF_ROWS*d) — fwd landing buffer
    dst_node_buf_bwd: Any = None    # GinWindow (DST_NODE_BUF_ROWS*d) — bwd landing buffer
    staging_fwd: Any = None         # GinWindow (num_nodes*T_local*d) — fwd coalesced-put source (compact per-node)
    staging_bwd: Any = None         # GinWindow (num_nodes*T_local*d) — bwd coalesced-put source
    gin_least_fwd: Any = None       # GinWindow int64[1] — fwd signal epoch
    gin_least_bwd: Any = None       # GinWindow int64[1] — bwd signal epoch
    x_lsa_fwd: Optional[torch.Tensor] = None             # int64[node_size] LSA base addrs of x_gin_fwd
    x_lsa_bwd: Optional[torch.Tensor] = None
    dst_node_buf_lsa_fwd: Optional[torch.Tensor] = None  # int64[node_size] of dst_node_buf_fwd
    dst_node_buf_lsa_bwd: Optional[torch.Tensor] = None
    # nccl_gin.dispatch.hier_dispatch_forward, bound at HIER alloc so ep.py drives the GIN dispatch
    # without importing the optional GIN stack (it is reached only via this workspace handle).
    gin_dispatch_fn: Any = None

    # HIER combine GIN state: reverse-mirror of dispatch (a REDUCTION, no RDMA atomics). SEPARATE
    # signal slot from dispatch's — one shared slot would conflate receiver puts with origin puts.
    combine_send_fwd: Any = None     # GinWindow ((num_nodes-1)*T_local*d) — fwd gateway reduce staging
    combine_send_bwd: Any = None
    combine_recv_fwd: Any = None     # GinWindow ((num_nodes-1)*T_local*d) — fwd origin recv landing
    combine_recv_bwd: Any = None
    combine_least: Any = None        # GinWindow int64[1] — combine signal epoch (slot 1)
    # nccl_gin.combine.hier_combine_forward, bound at HIER alloc (same optional-stack isolation as dispatch).
    gin_combine_fn: Any = None

    @property
    def T_local(self) -> int:
        return self._T_local

    @property
    def K(self) -> int:
        return self._K

    @property
    def d(self) -> int:
        return self._d

    def _ensure_partial_combine_buf(self) -> None:
        """Lazy-allocates partial_combine_buf for RS/RANK_DEDUP combine. Collective (rendezvous) but
        symmetric — all ranks must call this at the same logical point; first call pays rendezvous cost, later calls reuse."""
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

    def _ensure_a2a_peer_base(self) -> None:
        """Lazy-builds int64[W] peer base addresses for a2a_combine_triton. One-time host op (data_ptrs
        are constant for the allocation's lifetime) kept off the hot path / out of CUDA-graph capture."""
        if self.a2a_peer_y_base is not None:
            return
        device = self.y_symm.device
        self.a2a_peer_y_base = torch.tensor(
            [b.data_ptr() for b in self.y_peer_bufs], dtype=torch.int64, device=device
        )
        self.a2a_peer_s_base = torch.tensor(
            [b.data_ptr() for b in self.s_rev_peer_bufs], dtype=torch.int64, device=device
        )

    def release(self) -> None:
        """Drops ALL tensor refs (workspace is unusable after). Destruction order matters: peer_bufs must
        drop BEFORE the symm tensor, else ~AllocationRef hits cuMemUnmap on freed pages -> SIGABRT."""
        self.x_peer_bufs = ()
        self.y_peer_bufs = ()
        self.s_rev_peer_bufs = ()
        self.partial_combine_peer_bufs = ()
        self.a2a_peer_y_base = None
        self.a2a_peer_s_base = None
        # Each *_hdl is the SymmetricMemory wrapper; if it outlives the symm tensor below it keeps an
        # AllocationRef alive -> same cuMemUnmap crash one rendezvous later.
        self.x_hdl = None
        self.o_hdl = None
        self.s_rev_hdl = None
        self.partial_combine_hdl = None
        # ↓ now the symm-mem buffers themselves.
        self.x_symm = None
        self.y_symm = None
        self.s_rev_symm = None
        self.partial_combine_buf = None
        # Plain HBM (not symm-mem) — order doesn't matter.
        self.a2a_recv = None
        self.ag_compute = None
        self.t_global_pattern = None
        self.x_idx_expanded_remap_for_rank_dedup_buf = None
        # HIER GIN teardown uses NON-collective ncclCommAbort (not collective close()) — safe during
        # eviction, where a collective call could hang waiting for peers that already left.
        self.x_lsa_fwd = self.x_lsa_bwd = None
        self.dst_node_buf_lsa_fwd = self.dst_node_buf_lsa_bwd = None
        self.gin_dispatch_fn = None
        self.x_gin_fwd = self.x_gin_bwd = None
        self.dst_node_buf_fwd = self.dst_node_buf_bwd = None
        self.staging_fwd = self.staging_bwd = None
        self.gin_least_fwd = self.gin_least_bwd = None
        self.gin_combine_fn = None
        self.combine_send_fwd = self.combine_send_bwd = None
        self.combine_recv_fwd = self.combine_recv_bwd = None
        self.combine_least = None
        if self.gin_backend is not None:
            self.gin_backend.abort()  # closes ALL windows (dispatch + combine) + aborts the comm
            self.gin_backend = None


# ============================================================================
# Workspace cache / allocator
# ============================================================================


class SymmMemManager:
    # Process-wide intern table keyed by (group, device); __new__ returns the cached instance,
    # __init__ runs once (_initialized guard). No lock: CPython's GIL makes get/setitem atomic here.
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
        # Explicit release (not just dict.clear()) because a pending backward's autograd ctx may
        # still pin a workspace via ctx.ep_ws; this lets the caching allocator reclaim it too.
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
        self, shape: Tuple[int, ...], dtype: torch.dtype, group: Optional[dist.ProcessGroup] = None
    ) -> Tuple[torch.Tensor, Any, Tuple[torch.Tensor, ...]]:
        # HIER passes the per-node subgroup so rendezvous stays within the NVLink/LSA domain — a
        # WORLD rendezvous across nodes hangs/errors (invariant #11). peer_bufs become node-local.
        grp = group if group is not None else self.ep_group
        n = dist.get_world_size(grp)
        buf = _symm_mem.empty(*shape, dtype=dtype, device=self.device)
        hdl = _symm_mem.rendezvous(buf, group=grp)
        peer_bufs = tuple(hdl.get_buffer(r, shape, dtype) for r in range(n))
        return buf, hdl, peer_bufs

    def _node_subgroup(self, num_nodes: int, node_size: int) -> dist.ProcessGroup:
        """Per-node symm-mem rendezvous subgroup (the NVLink/LSA domain), built deterministically on
        ALL ranks in increasing node order and cached. ``new_group`` is collective over WORLD."""
        cached = getattr(self, "_node_grp", None)
        if cached is not None and cached[0] == (num_nodes, node_size):
            return cached[1]

        my_node = self.rank // node_size
        grp = None
        for n in range(num_nodes):
            g = dist.new_group(ranks=list(range(n * node_size, (n + 1) * node_size)))
            if n == my_node:
                grp = g

        self._node_grp = ((num_nodes, node_size), grp)
        return grp

    def _alloc_hier_workspace(
        self, T_local: int, d: int, K: int, E_local: int, dtype: torch.dtype, mode: str,
    ) -> _EPWorkspace:
        """Inter-node (NCCL-GIN) workspace with a SEPARATE GIN comm (needs NCCL_GIN_TYPE=3). SELF-CONFIG:
        node_size/num_nodes are derived from the GIN backend's lsa_size, not passed in (like DeepEP's num_nvl_ranks)."""
        import nccl.core as nccl  # optional GIN stack — imported only on the use_gin path
        from sonicmoe.functional.distributed.nccl_gin import NCCLGin
        from sonicmoe.functional.distributed.nccl_gin import dispatch as gin_dispatch
        from sonicmoe.functional.distributed.nccl_gin import combine as gin_combine

        W = self.world_size
        dev = self.device
        dev_idx = dev.index if dev.index is not None else torch.cuda.current_device()

        # Separate NCCL GIN comm (single-use unique_id broadcast over the EP group).
        uid = nccl.get_unique_id() if self.rank == 0 else None
        obj = [uid]
        dist.broadcast_object_list(obj, src=0, group=self.ep_group)
        be = NCCLGin(self.rank, W, obj[0], device=dev_idx)
        # Two size-independent epoch windows first (slot 0 = dispatch, slot 1 = combine), so >=1 window is
        # registered before make_dev_comm (the established window-register -> create_dev_comm order).
        gin_least = be.alloc_window(1, torch.int64)
        cmb_least = be.alloc_window(1, torch.int64)
        be.make_dev_comm(signal_count=2)  # slot 0 = dispatch, slot 1 = combine

        # Reconcile lsa_size to ONE value via MAX across ranks (a rank outside an LSA team reports 0)
        # so node subgroups + cross-rank collectives agree.
        lsa_t = torch.tensor([be.lsa_size or 0], device=dev, dtype=torch.int64)
        dist.all_reduce(lsa_t, op=dist.ReduceOp.MAX, group=self.ep_group)
        node_size = int(lsa_t.item())
        assert node_size > 0 and W % node_size == 0, \
            f"self-config: bad reconciled lsa_size {node_size} for world_size {W}"
        num_nodes = W // node_size

        TK_local = T_local * K
        TK_global = W * TK_local
        MAX_ROWS_PER_RANK = T_local * W * min(K, E_local)
        DST_NODE_BUF_ROWS = max(T_local * (num_nodes - 1), 1)
        COMBINE_RECV_ROWS = max(T_local * (num_nodes - 1), 1)  # remote-node stripes per rank (combine)
        STAGING_ROWS = num_nodes * T_local  # COMPACT per-node coalesced-put staging: node n at [n*T_local, +cnt_n)

        # symm-mem rendezvous over the NODE subgroup, never WORLD. partial_combine_buf is node-group so
        # gateway/origin reduce reads node-local NVLink peers (flat _ensure_partial_combine_buf then early-returns).
        node_grp = self._node_subgroup(num_nodes, node_size)
        x_symm, x_hdl, x_peer_bufs = self._alloc_symm((T_local, d), dtype, group=node_grp)
        y_symm, o_hdl, y_peer_bufs = self._alloc_symm((MAX_ROWS_PER_RANK, d), dtype, group=node_grp)
        s_rev_symm, s_rev_hdl, s_rev_peer_bufs = self._alloc_symm((TK_global,), torch.int32, group=node_grp)
        pc_buf, pc_hdl, pc_peer_bufs = self._alloc_symm((W * T_local, d), dtype, group=node_grp)
        x_idx_buf = torch.empty(MAX_ROWS_PER_RANK, dtype=torch.int32, device=dev)  # HIER reuses rank-dedup A_idx

        # GIN data windows (sized by the derived num_nodes). alloc_window registers into the comm, so
        # registering after make_dev_comm is fine — the devcomm reaches them by handle at kernel time.
        x_gin_fwd = be.alloc_window(T_local * d, dtype)
        x_gin_bwd = be.alloc_window(T_local * d, dtype)
        dst_fwd = be.alloc_window(DST_NODE_BUF_ROWS * d, dtype)
        dst_bwd = be.alloc_window(DST_NODE_BUF_ROWS * d, dtype)
        stg_fwd = be.alloc_window(STAGING_ROWS * d, dtype)
        stg_bwd = be.alloc_window(STAGING_ROWS * d, dtype)
        # combine windows: gateway send staging + origin recv landing (fwd+bwd)
        cmb_send_fwd = be.alloc_window(COMBINE_RECV_ROWS * d, dtype)
        cmb_send_bwd = be.alloc_window(COMBINE_RECV_ROWS * d, dtype)
        cmb_recv_fwd = be.alloc_window(COMBINE_RECV_ROWS * d, dtype)
        cmb_recv_bwd = be.alloc_window(COMBINE_RECV_ROWS * d, dtype)

        be.bind_signal(gin_least, 0)
        be.reset_epoch(0)
        cmb_least.tensor.fill_(0)  # combine epoch (slot 1) base — launch_combine_put advances it device-side
        x_lsa_fwd = gin_dispatch.build_lsa_base(be, x_gin_fwd, node_size)
        x_lsa_bwd = gin_dispatch.build_lsa_base(be, x_gin_bwd, node_size)
        dst_lsa_fwd = gin_dispatch.build_lsa_base(be, dst_fwd, node_size)
        dst_lsa_bwd = gin_dispatch.build_lsa_base(be, dst_bwd, node_size)
        torch.cuda.synchronize()

        return _EPWorkspace(
            x_symm=x_symm, x_hdl=x_hdl, x_peer_bufs=x_peer_bufs,
            y_symm=y_symm, o_hdl=o_hdl, y_peer_bufs=y_peer_bufs,
            s_rev_symm=s_rev_symm, s_rev_hdl=s_rev_hdl, s_rev_peer_bufs=s_rev_peer_bufs,
            partial_combine_buf=pc_buf, partial_combine_hdl=pc_hdl, partial_combine_peer_bufs=pc_peer_bufs,
            ep_group=self.ep_group, world_size=W, my_rank=self.rank, E_local=E_local,
            _T_local=T_local, _K=K, _d=d, dispatch_mode=mode,
            x_idx_expanded_remap_for_rank_dedup_buf=x_idx_buf,
            gin_backend=be, num_nodes=num_nodes, node_size=node_size,
            x_gin_fwd=x_gin_fwd, x_gin_bwd=x_gin_bwd, dst_node_buf_fwd=dst_fwd, dst_node_buf_bwd=dst_bwd,
            staging_fwd=stg_fwd, staging_bwd=stg_bwd,
            gin_least_fwd=gin_least, gin_least_bwd=gin_least,  # one signal slot/epoch (fwd+bwd accumulate)
            x_lsa_fwd=x_lsa_fwd, x_lsa_bwd=x_lsa_bwd,
            dst_node_buf_lsa_fwd=dst_lsa_fwd, dst_node_buf_lsa_bwd=dst_lsa_bwd,
            gin_dispatch_fn=gin_dispatch.hier_dispatch_forward,
            combine_send_fwd=cmb_send_fwd, combine_send_bwd=cmb_send_bwd,
            combine_recv_fwd=cmb_recv_fwd, combine_recv_bwd=cmb_recv_bwd,
            combine_least=cmb_least, gin_combine_fn=gin_combine.hier_combine_forward,
        )

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
        # Per-rank ceiling on (token, expert) slots routed here — only this many rows of y_symm ever
        # get written, so we size the buffer at this bound instead of TK_global.
        MAX_ROWS_PER_RANK = T_local * W * min(K, E_local)
        dev = self.device

        x_symm, x_hdl, x_peer_bufs = self._alloc_symm((T_local, d), dtype)
        y_symm, o_hdl, y_peer_bufs = self._alloc_symm((MAX_ROWS_PER_RANK, d), dtype)
        s_rev_symm, s_rev_hdl, s_rev_peer_bufs = self._alloc_symm((TK_global,), torch.int32)

        a2a_recv = None
        ag_compute = None
        t_global_pattern = None
        x_idx_expanded_remap_for_rank_dedup_buf = None

        # A2A allocates an expert-grouped a2a_recv buffer. RANK_DEDUP skips it — its A_idx-driven GEMM
        # gathers the dedup packed buffer directly. AG uses the (W*T_local, d) gather buffer instead.
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
        # use_gin selects a HIER GIN dispatch mode; node_size/num_nodes derive from the backend's
        # lsa_size — a single NVLink domain (lsa_size==world) yields num_nodes==1, degrading HIER to flat.
        use_gin = _is_hier_node_dedup_dispatch_gin_mode(mode)
        key = (T_local, d, K, E_local, str(dtype), mode)
        ws = self._cache.get(key)
        if ws is None:
            if use_gin:
                ws = self._alloc_hier_workspace(T_local, d, K, E_local, dtype, mode)
            else:
                ws = self._alloc_workspace(T_local, d, K, E_local, dtype, mode)
            self._cache[key] = ws
        return ws


# atexit hook (below) frees symm-mem before CUDA-context teardown — otherwise ~CUDASymmetricMemory's
# cuMemUnmap on a dead context -> std::terminate. Fork hook drops entries so inherited mappings aren't double-freed.


def clear_ep_cache() -> None:
    """Releases every cached EP symm-mem workspace. Auto-registered as an atexit/fork hook, so normally
    no need to call directly — useful when destroying/re-creating a process group mid-process."""
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


# RuntimeEPConfig is the user-facing dispatch/combine choice; NetworkProfiler.profile() benchmarks
# hardware and returns one. Public moe_ep_*_forward entry points accept it as ep_config=.


@dataclass(frozen=True)
class RuntimeEPConfig:
    """End-to-end EP runtime config. MAX_ROWS_PER_RANK_STATIC = T_local*W*min(K,E_local): the worst-case
    (top-K distinct-pick + E_local of E experts) bound, tight only when E_local>=K; else T_local*E is tighter."""

    W: int
    K: int
    dispatch_mode: DispatchMode = DispatchMode.A2A_DISPATCH_TRITON
    combine_mode: CombineMode = CombineMode.A2A_COMBINE_TRITON
    I: Optional[int] = None
    is_glu_act: Optional[bool] = None
    E_local: Optional[int] = None
    # Per-expert GEMM output bound (h, a, y_symm, A_idx domain) — same T_local*W*min(K,E_local) as above.
    # Applies regardless of dispatch mode; RANK_DEDUP gathers its packed input via A_idx into this domain.
    MAX_ROWS_PER_RANK_STATIC: Optional[int] = None
    # HIER partition: W = num_nodes*node_size, node_of(r)=r//node_size. use_gin must equal
    # (num_nodes>1); at num_nodes==1 HIER degrades to flat rank-dedup and no GIN comm is created.
    num_nodes: int = 1
    node_size: Optional[int] = None
    node_id: Optional[int] = None
    local_id: Optional[int] = None
    use_gin: bool = False


class NetworkProfiler:
    """Benchmarks AG/A2A/RANK_DEDUP dispatch (synthetic BALANCED routing, E=K*W) and returns the fastest
    as a RuntimeEPConfig. Call once per (T_local,d,K,dtype) before the training loop; reuse the result as ep_config=."""

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
        # Local imports avoid a top-level dependency on the Triton comm primitives, keeping this
        # module importable without a CUDA context (tests / type stubs).
        from .functional.distributed import (
            a2a_combine_triton,
            a2a_dispatch_triton,
            all_gather_triton,
            build_a2a_peer_base,
            compute_dispatch_metadata,
            rank_dedup_combine_triton,
            rank_dedup_dispatch_triton,
            rs_combine_triton,
        )
        from .functional.ep import _build_single_row
        from .functional.metadata import general_routing_router_metadata_triton

        mgr = self.mgr
        W = mgr.world_size
        rank = mgr.rank
        device = mgr.device
        T_local = self.T_local
        H = self.H
        K = self.K
        dtype = self.dtype

        # Extra rendezvous outside the SymmMemManager cache — fine since the buffer/handle/peer_bufs
        # are released in the right order at the bottom of this method.
        x_symm = _symm_mem.empty((T_local, H), dtype=dtype, device=device)
        x_symm.normal_()
        x_hdl = _symm_mem.rendezvous(x_symm, group=mgr.ep_group)
        x_peer_bufs = tuple(x_hdl.get_buffer(r, (T_local, H), dtype) for r in range(W))
        x_hdl.barrier()  # publish x_symm to peers before the AG reads

        ag_out = torch.empty(W * T_local, H, dtype=dtype, device=device)

        def ag_triton_call() -> None:
            all_gather_triton(x_symm, mgr.ep_group, out=ag_out, peer_bufs=x_peer_bufs)

        # Synthetic uniform routing: E = K*W => E_local = K, perfectly balanced (each rank pulls
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

        meta = compute_dispatch_metadata(topk_idx_global, my_rank=rank, E_local=E_local, emit_combine=True)
        dst_rank_flat = meta["dst_rank_flat"]
        expert_local_padded = meta["expert_local_padded"]
        a2a_token_indices = meta["a2a_token_indices"]
        my_dst_rank = meta["my_dst_rank"]  # (T_local, K) — for A2A_combine
        # RANK_DEDUP combine gather needs the precomputed single_row.
        meta["single_row"] = _build_single_row(topk_idx_global, meta["combine_single_k"], rank, E_local)

        # Build s_reverse_local (the A2A recv_pos) via the metadata
        # kernel — same as ``_build_consumer_metadata`` in ep.py.
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

        # RANK_DEDUP: packed buffer sized to MAX_PAIR_COUNT=W*T_local. Local-write only (peers never
        # read the recv buffer) so plain torch.empty suffices — no symm-mem rendezvous needed.
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

        t_ag_triton = self._bench(ag_triton_call)
        t_a2a = self._bench(a2a_call)
        t_rank_dedup_dispatch = self._bench(rank_dedup_dispatch_call)

        timings = {
            DispatchMode.AG_DISPATCH_TRITON: t_ag_triton,
            DispatchMode.A2A_DISPATCH_TRITON: t_a2a,
            DispatchMode.RANK_DEDUP_DISPATCH_TRITON: t_rank_dedup_dispatch,
        }
        winner = min(timings, key=timings.get)

        # a2a_combine_triton fuses peer y_symm+s_rev_symm reads; local_combine writes bf16
        # partial_combine_buf, consumed across ranks by reduce_scatter_triton / rank_dedup gather.
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
        a2a_peer_y_base, a2a_peer_s_base, _ = build_a2a_peer_base(y_symm, s_rev_symm, mgr.ep_group)

        def gather_call() -> None:
            a2a_combine_triton(
                y_symm,
                s_rev_symm,
                my_dst_rank,
                topk_scores_local,
                gather_out,
                K=K,
                group=mgr.ep_group,
                peer_y_base=a2a_peer_y_base,
                peer_s_base=a2a_peer_s_base,
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
                mine_slot_idx=meta["mine_slot_idx"],
                mine_count=meta["mine_count"],
                combine_contrib_C=meta["combine_contrib_C"],
                combine_work_list=meta["combine_work_list_multi"],
                combine_work_count=meta["combine_work_count_multi"],
                combine_single_k=meta["combine_single_k"],
                y_peer_bufs=y_peer_bufs,
                s_reverse_peer_bufs=s_rev_peer_bufs,
                single_row=meta["single_row"],
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

        # Uses the MEASURED (W,W) pair_count, not the analytical (W-1)*T_local*(1-(1-1/W)^K)*H*itemsize
        # fallback — that formula is only exact under balanced uniform routing.
        itemsize = torch.tensor([], dtype=dtype).element_size()
        pc = meta["pair_count"]
        dedup_rows_local = int(pc[:, rank].sum().item() - pc[rank, rank].item())
        _dedup_rows_t = torch.tensor([dedup_rows_local], dtype=torch.int64, device=device)
        dist.all_reduce(_dedup_rows_t, op=dist.ReduceOp.SUM, group=mgr.ep_group)
        dedup_bytes = (_dedup_rows_t.item() / W) * H * itemsize

        # Expose per-rank-mean timings (ms) so callers can report the measured dispatch/combine cost
        # without re-running the bench.
        self.dispatch_timings_ms = {mode: t / W * 1e3 for mode, t in timings.items()}
        self.combine_timings_ms = {mode: t / W * 1e3 for mode, t in combine_timings.items()}

        # Cleanup order: peer aliases -> handle -> buffer (mirrors _EPWorkspace.release()'s
        # cuMemUnmap-crash-avoidance rationale).
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
            # _bench returns the SUM across ranks (not MAX) so A2A's per-rank variance under random
            # routing doesn't skew the pick; SUM == W*MEAN preserves ordering. Displayed as per-rank mean.
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
            # RS/RANK_DEDUP_COMBINE GB/s is intentionally omitted — both spend real time in the HBM-bound
            # local_combine producer, so bytes/time mixes HBM and NVLink throughput. See bench-ep-nvlink.py for the split.
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
        """Times fn over warmup+repeat iterations. Returns mean ms-per-iter, SUM-reduced across ranks."""
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
