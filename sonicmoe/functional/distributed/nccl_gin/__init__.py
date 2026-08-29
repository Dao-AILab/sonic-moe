# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
"""Production CuTeDSL NCCL-GIN backend — transport primitives only (no dispatch/combine).

Hard-imports CuTeDSL/nccl4py (see interrack-gin-ep skill); signal proof: docs/interrack_gin_signal_ordering.md.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import cutlass
import cutlass.cute as cute
from cutlass.cute.arch.nvvm_wrappers import WARP_SIZE
import cuda.bindings.driver as cuda_driver  # CUstream
import nccl.core as nccl
import nccl.core.device.cute as nccl_cute  # noqa: F401 — triggers BitCode(device_bitcode_path())
import nccl.core.interop.torch as nccl_torch


# Module-level so @cute.jit string-annotation eval resolves `cutlass`. Weak-attached SignalInc put
# iff inc>0; receiver advances device-resident epoch in-graph (le=least[0]+inc) and waits on le.
@cute.kernel
def _put_wait_kernel(dev_comm, send_win, recv_win, least_win,
                     n: cutlass.Int32, src: cutlass.Int32, dst: cutlass.Int32,
                     sig: cutlass.Int32, inc: cutlass.Int64):
    dev_comm = nccl_cute.DevComm(dev_comm)
    send_win = nccl_cute.Window(send_win)
    recv_win = nccl_cute.Window(recv_win)
    least_win = nccl_cute.Window(least_win)
    team = dev_comm.team_world
    gin = dev_comm.gin(nccl_cute.GinBackendMask.ALL, 0)
    coop = nccl_cute.cta()
    tidx, _, _ = cute.arch.thread_idx()

    if team.rank == src:
        if inc > 0:  # zero-count: no inbound rows expected -> no put
            send = send_win.tensor(cutlass.BFloat16, cute.make_layout(n))
            recv = recv_win.tensor(cutlass.BFloat16, cute.make_layout(n))
            # weak, indexed: one attached SignalInc per put (signal_op=0 -> ADD 1).
            gin.put(team, dst, recv_win, recv, send_win, send, coop,
                    is_signal=True, signal_id=sig, signal_op=0, signal_op_arg=1)

    if team.rank == dst:
        # device-resident least: read accumulated epoch, advance by inc, persist, wait on it.
        least_t = least_win.tensor(cutlass.Int64, cute.make_layout(1))
        le = least_t[0] + inc
        if tidx == 0:
            least_t[0] = le  # in-graph epoch advance (captured); persists for the next replay

        gin.wait_signal(coop, signal=sig, least=le)


@cute.jit
def _put_wait_launch(dev_comm: cutlass.Int64, send_win: cutlass.Int64, recv_win: cutlass.Int64,
                     least_win: cutlass.Int64, n: cutlass.Int32, src: cutlass.Int32,
                     dst: cutlass.Int32, sig: cutlass.Int32, inc: cutlass.Int64,
                     stream: cuda_driver.CUstream):
    _put_wait_kernel(dev_comm, send_win, recv_win, least_win, n, src, dst, sig, inc).launch(
        grid=[1, 1, 1], block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
        cooperative=True, stream=stream)


def compile_kernels():
    """Offline-compile the put/wait launch (no live comm) — CI smoke + pre-capture warmup."""
    return cute.compile(_put_wait_launch, 0, 0, 0, 0, cutlass.Int32(8), cutlass.Int32(0),
                        cutlass.Int32(1), cutlass.Int32(0), cutlass.Int64(1), cuda_driver.CUstream(0))


class GinLaunchHelper:
    """Threads torch's current CUDA stream into every cute launch; forbids stream=None (creates its
    own stream + host-syncs), which would break CUDA-graph capture."""

    @staticmethod
    def current_cu_stream():
        return cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)

    @classmethod
    def launch(cls, jit_fn, *args, stream=None):
        st = stream if stream is not None else cls.current_cu_stream()
        if st is None:  # belt-and-suspenders: current_cu_stream() is never None
            raise RuntimeError("stream=None forbidden in production GIN launch (breaks graph capture)")

        jit_fn(*args, st)  # the jit launch fn takes the CUstream as its LAST positional param


@dataclass
class GinWindow:
    """A registered NCCL GIN window: ``(torch tensor, RegisteredWindowHandle)`` with a double-close guard.
    Buffer is C-contiguous (nccl_torch.empty), matching grouped-GEMM's row-major recv_packed contract."""

    tensor: object  # torch.Tensor
    _handle: object  # nccl RegisteredWindowHandle
    _closed: bool = False

    @property
    def handle(self) -> int:
        if self._closed:
            raise RuntimeError("GinWindow is closed")

        return self._handle.handle

    @property
    def is_valid(self) -> bool:
        return (not self._closed) and self._handle is not None and self._handle.is_valid

    @property
    def data_ptr(self) -> int:
        return self.tensor.data_ptr()

    def close(self) -> None:
        if not self._closed and self._handle is not None:
            self._handle.close()

        self._closed = True


class GinSignalState:
    """One indexed signal slot: device-resident ``least`` epoch, advanced INSIDE the kernel so a graph
    replay never stale-passes (proof: docs/interrack_gin_signal_ordering.md); count==0 ⇒ no-op wait."""

    def __init__(self, signal_id: int, least_window: "GinWindow"):
        self.signal_id = int(signal_id)
        self._least = least_window
        self._epoch = 0  # host mirror (the device window is authoritative; this is for asserts/logs)

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def least_handle(self) -> int:
        return self._least.handle

    def reset(self, base: int = 0) -> None:
        """Set the device-resident epoch to ``base`` (host fill, on the current stream) at setup."""
        self._epoch = int(base)
        self._least.tensor.fill_(int(base))

    def note(self, inc: int) -> None:
        """Mirror the device-side advance host-side (bookkeeping only)."""
        if inc > 0:
            self._epoch += int(inc)


class _GINWorkspace:
    """Two-group model: ``ep_world_group`` (routing AG + GIN comm) + LSA subgroup (symm-mem rendezvous).
    Contract: ``use_lsa_only == (W > L)`` — when True, rendezvous MUST use ``lsa_group`` only, never WORLD."""

    def __init__(self, ep_world_group, lsa_group, lsa_size, node_id, lsa_rank, world_size, rank):
        self.ep_world_group = ep_world_group
        self.lsa_group = lsa_group
        self.lsa_size = lsa_size
        self.node_id = node_id
        self.lsa_rank = lsa_rank
        self.world_size = world_size
        self.rank = rank
        self.use_lsa_only = world_size > lsa_size

    @classmethod
    def build(cls, rank, world_size, lsa_size, ep_world_group=None):
        """Deterministically creates per-node LSA subgroups on EVERY rank (same order), keeps local.
        ``lsa_size`` MUST already be reconciled to one value across ranks (see ``build_workspace``)."""
        assert lsa_size > 0 and world_size % lsa_size == 0, \
            f"bad lsa_size {lsa_size} for world_size {world_size}"
        n_nodes = world_size // lsa_size
        node_id = rank // lsa_size
        lsa_rank = rank % lsa_size
        lsa_group = None
        for nid in range(n_nodes):
            ranks = list(range(nid * lsa_size, (nid + 1) * lsa_size))
            grp = dist.new_group(ranks=ranks)  # collective over WORLD; identical order on all ranks
            if nid == node_id:
                lsa_group = grp

        dist.barrier()
        return cls(ep_world_group, lsa_group, lsa_size, node_id, lsa_rank, world_size, rank)


class NCCLGin:
    """Owns a SEPARATE NCCL GIN comm+devcomm+windows+signal. On setup failure use ``abort()`` (non-
    collective), NOT ``close()`` (its collective ``comm.destroy()`` hangs other ranks); call either BEFORE destroy_process_group()."""

    def __init__(self, rank: int, world_size: int, unique_id, device: int = 0, *,
                 require_gdaki: bool = True):
        # Set teardown-relevant fields FIRST so __del__ is safe even if init raises later.
        self._closed = False
        self._comm = None
        self._windows: list = []
        self._dev = None            # DevCommResource
        self._signal: Optional[GinSignalState] = None
        self.workspace: Optional[_GINWorkspace] = None
        self.lsa_size: Optional[int] = None
        self.rank, self.world_size, self.device = rank, world_size, device

        # Preflight BEFORE allocating any NCCL resource: GDAKI (NCCL_GIN_TYPE=3) is required for graph
        # capture + RDMA — without it nccl silently falls back to a CPU proxy path (see check_cutedsl_gin_env.py).
        if require_gdaki and os.environ.get("NCCL_GIN_TYPE") != "3":
            raise RuntimeError(
                f"NCCL_GIN_TYPE={os.environ.get('NCCL_GIN_TYPE')!r}: the GIN backend needs GDAKI "
                f"(type 3) for graph capture + RDMA. Export NCCL_GIN_TYPE=3 (see interrack-gin-ep).")

        # Inherits NCCL's process-global IB device list — the LAUNCHER must set NCCL_IB_HCA to the
        # fast IB NICs or NCCL may grab slow/management NICs (QP timeouts / a fraction of cross-node BW).

        # 2. Separate NCCL communicator (NOT the torch ProcessGroup).
        self._comm = nccl.Communicator.init(nranks=world_size, rank=rank, unique_id=unique_id)

    # ── window registration (init-time only; NEVER in a hot path / graph capture) ──
    def alloc_window(self, n_elems: int, dtype) -> GinWindow:
        buf = nccl_torch.empty(n_elems, dtype=dtype, device=self.device)
        buf.zero_()
        torch.cuda.synchronize()
        h = self._comm.register_window(buf)
        if h is None or not h.is_valid:  # validate BEFORE wrapping/appending (no invalid handle leaks)
            raise RuntimeError("register_window failed or unsupported on this platform")

        w = GinWindow(buf, h)
        self._windows.append(w)
        return w

    def make_dev_comm(self, signal_count: int = 1, connection: str = "FULL"):
        """Creates the GIN devcomm AFTER all windows are registered. ``connection``: "FULL" (default, peers
        addressed by GLOBAL rank, byte-identical to existing callers) or "RAIL" (perf-only NIC-affinity switch)."""
        assert signal_count >= 1, "need >=1 indexed signal slot"
        conn = (nccl.NcclGinConnectionType.RAIL if str(connection).upper() == "RAIL"
                else nccl.NcclGinConnectionType.FULL)
        reqs = nccl.NCCLDevCommRequirements(
            gin_connection_type=conn,
            gin_signal_count=signal_count,
        )
        # Weak/indexed contract must NOT request strong/VA signals — the CuTeDSL put C-ABI hardcodes
        # WEAK/INDEXED regardless; warn only if either default (None) is truthy (see signal_ordering.md).
        if self.rank == 0 and (reqs.gin_strong_signals_required or reqs.gin_va_signals_required):
            print(f"[cutedsl-gin] WARNING: strong/VA signals requested "
                  f"(strong={reqs.gin_strong_signals_required!r}, va={reqs.gin_va_signals_required!r}) — "
                  f"contradicts the weak/indexed signal contract; re-review the signal-ordering proof.",
                  flush=True)

        dev = self._comm.create_dev_comm(requirements=reqs)
        assert dev.is_valid and dev.ptr != 0, "create_dev_comm failed"
        self._dev = dev
        self.lsa_size = int(dev.dev_comm.lsa_size)  # plain int; 0 if this rank is not in an LSA team

        if self.rank == 0:  # log once, on global rank 0 (avoid N-rank spam)
            print(f"[cutedsl-gin] devcomm: signal_mode=WEAK/INDEXED (signal_op=0 Inc, op_arg=1) "
                  f"connection={str(connection).upper()} gin_signal_count={signal_count} backend=GDAKI(type3) "
                  f"lsa_size={self.lsa_size}  [1 attached weak signal per put]", flush=True)

        return dev

    @property
    def railed_gin_type(self):
        """RAIL availability probe: the fabric's railed GIN type. A NONE/0 value ⇒ RAIL is NOT
        available on this fabric (do not request connection="RAIL"); a real GIN type ⇒ RAIL is usable."""
        return self._comm.railed_gin_type

    @property
    def dev_ptr(self) -> int:
        """The devcomm pointer (first arg to every GIN cute launch). Valid after make_dev_comm()."""
        assert self._dev is not None, "call make_dev_comm() first"
        return self._dev.ptr

    def bind_signal(self, least_window: GinWindow, signal_id: int = 0) -> GinSignalState:
        """Binds the device-resident ``least`` window to a signal slot. dtype MUST be int64 (the wait
        kernel reads Int64/64 bits) — an int32 window would silently truncate the epoch."""
        assert least_window.tensor.dtype == torch.int64, \
            f"least_window dtype must be torch.int64, got {least_window.tensor.dtype}"
        self._signal = GinSignalState(signal_id, least_window)
        return self._signal

    def build_workspace(self, lsa_size: Optional[int] = None) -> _GINWorkspace:
        """Builds the real LSA subgroup. ``lsa_size`` is RECONCILED across all ranks first (per-rank
        ``DevComm.lsa_size`` can be 0/divergent) — an unreconciled divergent value would deadlock ``new_group``."""
        L = lsa_size if lsa_size is not None else (self.lsa_size or 0)

        # Consensus: every rank must build the SAME subgroups. all_gather_object is backend-agnostic
        # (gloo/nccl). Take the max nonzero L; fall back to single-domain (world_size).
        if dist.is_available() and dist.is_initialized() and self.world_size > 1:
            gathered = [None] * self.world_size
            dist.all_gather_object(gathered, int(L))
            vals = [int(x) for x in gathered if x is not None and int(x) > 0]
            L = max(vals) if vals else 0

        if L <= 0:
            L = self.world_size  # single LSA domain (W <= L); rendezvous == WORLD is acceptable

        assert self.world_size % L == 0, \
            f"reconciled lsa_size {L} must divide world_size {self.world_size}"
        self.workspace = _GINWorkspace.build(self.rank, self.world_size, L)
        return self.workspace

    # ── launches (always explicit-stream via GinLaunchHelper; graph-capturable) ──
    def reset_epoch(self, base: int = 0) -> None:
        """Zero/set the device-resident epoch once at setup (call before the first launch / capture)."""
        assert self._signal is not None, "call bind_signal first"
        self._signal.reset(base)

    def launch_put_wait(self, send_win: GinWindow, recv_win: GinWindow, n: int, src: int, dst: int,
                        *, inc: int = 1, stream=None):
        """Always-launched put/wait. ``inc``=expected_count for ``dst``; ``inc==0`` returns immediately
        (no deadlock). Graph-safe — the epoch advance is captured inside the kernel."""
        assert self._signal is not None and self._dev is not None, \
            "call make_dev_comm + bind_signal first"
        GinLaunchHelper.launch(
            _put_wait_launch, self._dev.ptr, send_win.handle, recv_win.handle,
            self._signal.least_handle, cutlass.Int32(n), cutlass.Int32(src),
            cutlass.Int32(dst), cutlass.Int32(self._signal.signal_id), cutlass.Int64(inc),
            stream=stream)
        self._signal.note(inc)

    def put_wait(self, send_win: GinWindow, recv_win: GinWindow, n: int, src: int, dst: int,
                 *, expected_count: int = 1, stream=None) -> None:
        """Eager convenience = ``launch_put_wait(inc=expected_count)``. Always launches (graph-safe);
        ``expected_count == 0`` is a no-op wait (no put, no deadlock).
        """
        self.launch_put_wait(send_win, recv_win, n, src, dst, inc=expected_count, stream=stream)

    # ── teardown ──
    def close(self) -> None:
        """COLLECTIVE clean path (sync → devcomm → windows → ``comm.destroy()``), idempotent. ALL ranks
        must call in lockstep (use ``abort()`` for a single-rank failure) — BEFORE destroy_process_group()."""
        if self._closed:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()  # ensure in-flight GIN kernels / GEMM consumers completed

        try:
            if self._dev is not None:
                self._dev.close()

            for w in self._windows:
                w.close()
        finally:
            # ``comm.destroy()`` is COLLECTIVE — it MUST run on every rank (even if a local
            # devcomm/window close raised) or peers deadlock; so it lives in finally with the reset.
            self._dev = None
            self._windows.clear()
            self._signal = None
            self.workspace = None
            self._closed = True
            comm, self._comm = self._comm, None
            if comm is not None:
                comm.destroy()

    def abort(self) -> None:
        """NON-COLLECTIVE failure path: local window close + ``ncclCommAbort``. Use when only this rank
        failed (``close()``'s collective ``comm.destroy()`` would hang waiting for peers). Idempotent.
        """
        if self._closed:
            return

        try:
            for w in self._windows:
                w.close()  # window deregister is LOCAL (non-collective)
        finally:
            self._dev = None  # devcomm destroy is collective; skip it on the abort path
            self._windows.clear()
            self._signal = None
            self.workspace = None
            self._closed = True
            comm, self._comm = self._comm, None
            if comm is not None:
                comm.abort()  # ncclCommAbort (non-collective)

    def __del__(self):
        # NO collectives/NCCL calls in GC — best-effort warn only if the caller forgot close()/abort().
        # __init__ sets _closed/_comm/rank first, so these reads are safe even if init raised later.
        if self.rank == 0 and not self._closed and self._comm is not None:
            print(f"[cutedsl-gin] WARNING: NCCLGin(rank={self.rank}) garbage-collected "
                  f"without close()/abort() — comm/windows may leak. Call close() (all ranks) or "
                  f"abort() (failure path) explicitly.", flush=True)
