# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
"""Hierarchical inter-node dispatch GIN kernels (built on nccl_gin's transport primitive): each rank
puts ONE RDMA row per (token, remote dst_node), addressed BY NODE; details: docs/interrack_gin_signal_ordering.md.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import cutlass
import cutlass.cute as cute
from cutlass.cute.arch.nvvm_wrappers import WARP_SIZE
import cuda.bindings.driver as cuda_driver  # CUstream
import nccl.core.device.cute as nccl_cute  # noqa: F401 — triggers BitCode(device_bitcode_path())

from . import GinLaunchHelper
from ..ep_dispatch import hier_gather_rt_triton, hier_stage_coalesced_triton  # Triton: NVLink gather half + coalesced-put staging


# Coalesced dispatch (one big put per remote node) is the default; pass ``coalesce=False`` to
# hier_dispatch_forward for the validated per-token put fallback (diagnostic / safety).
_COALESCE_DEFAULT = True


# Node-dedup dispatch kernel: (1) each rank puts one weak-signalled row per (token, remote dst_node)
# into dst_node_buffer[dst_slot], then (2) waits for dst_recv_count[my_rank] inbound; inc==0 ⇒ no-op.
@cute.kernel
def _gin_dispatch_kernel(
    dev_comm, x_win, dst_node_buf_win, least_win,
    node_present_ptr, dst_node_ptr, dst_slot_ptr, dst_recv_count_ptr,
    my_rank: cutlass.Int32, my_base: cutlass.Int32, my_local: cutlass.Int32,
    node_size: cutlass.Int32, TK_local: cutlass.Int32, K: cutlass.Int32,
    d: cutlass.Int32, n_meta: cutlass.Int32, sig: cutlass.Int32):
    dev_comm = nccl_cute.DevComm(dev_comm)
    x_win = nccl_cute.Window(x_win)
    dst_node_buf_win = nccl_cute.Window(dst_node_buf_win)
    least_win = nccl_cute.Window(least_win)
    team = dev_comm.team_world
    gin = dev_comm.gin(nccl_cute.GinBackendMask.ALL, 0)
    coop = nccl_cute.cta()
    tidx, _, _ = cute.arch.thread_idx()

    # metadata views over plain HBM device tensors (not GIN windows)
    np_t = cute.make_tensor(
        cute.make_ptr(cutlass.Int8, node_present_ptr, cute.AddressSpace.gmem), cute.make_layout(n_meta))
    dn_t = cute.make_tensor(
        cute.make_ptr(cutlass.Int32, dst_node_ptr, cute.AddressSpace.gmem), cute.make_layout(n_meta))
    ds_t = cute.make_tensor(
        cute.make_ptr(cutlass.Int32, dst_slot_ptr, cute.AddressSpace.gmem), cute.make_layout(n_meta))

    # 1) post my node-dedup puts (one per canonical remote (token, dst_node) slot)
    for i in cutlass.range(TK_local, unroll=1):
        orig = my_base + i
        present = np_t[orig]
        if present != 0:
            dn = dn_t[orig]
            ds = ds_t[orig]
            t = i // K
            peer = dn * node_size + my_local
            src = x_win.tensor(cutlass.BFloat16, cute.make_layout(d),
                               offset=cutlass.Int64(t) * cutlass.Int64(d) * 2)
            dst = dst_node_buf_win.tensor(cutlass.BFloat16, cute.make_layout(d),
                                    offset=cutlass.Int64(ds) * cutlass.Int64(d) * 2)
            gin.put(team, peer, dst_node_buf_win, dst, x_win, src, coop,
                    is_signal=True, signal_id=sig, signal_op=0, signal_op_arg=1)

    # 2) wait for my inbound puts (dst_recv_count[my_rank]); device-resident in-graph epoch advance
    cnt_t = cute.make_tensor(
        cute.make_ptr(cutlass.Int32, dst_recv_count_ptr, cute.AddressSpace.gmem), cute.make_layout(n_meta))
    inc = cutlass.Int64(cnt_t[my_rank])
    least_t = least_win.tensor(cutlass.Int64, cute.make_layout(1))
    le = least_t[0] + inc
    if tidx == 0:
        least_t[0] = le

    gin.wait_signal(coop, signal=sig, least=le)


@cute.jit
def _gin_dispatch_launch(
    dev_comm: cutlass.Int64, x_win: cutlass.Int64, dst_node_buf_win: cutlass.Int64, least_win: cutlass.Int64,
    node_present_ptr: cutlass.Int64, dst_node_ptr: cutlass.Int64, dst_slot_ptr: cutlass.Int64,
    dst_recv_count_ptr: cutlass.Int64, my_rank: cutlass.Int32, my_base: cutlass.Int32,
    my_local: cutlass.Int32, node_size: cutlass.Int32, TK_local: cutlass.Int32, K: cutlass.Int32,
    d: cutlass.Int32, n_meta: cutlass.Int32, sig: cutlass.Int32,
    stream: cuda_driver.CUstream):
    _gin_dispatch_kernel(
        dev_comm, x_win, dst_node_buf_win, least_win, node_present_ptr, dst_node_ptr, dst_slot_ptr,
        dst_recv_count_ptr, my_rank, my_base, my_local, node_size, TK_local, K, d, n_meta,
        sig).launch(
        grid=[1, 1, 1], block=[cute.size(WARP_SIZE, mode=[0]), 1, 1], cooperative=True, stream=stream)


# RAIL variant of _gin_dispatch_kernel: same node-dedup put/wait but via team_rail with peer=dst_node
# (rail-relative, lands on the same receiving GPU as FULL). Correctness-equivalent; perf-only; needs connection="RAIL".
@cute.kernel
def _gin_dispatch_rail_kernel(
    dev_comm, x_win, dst_node_buf_win, least_win,
    node_present_ptr, dst_node_ptr, dst_slot_ptr, dst_recv_count_ptr,
    my_rank: cutlass.Int32, my_base: cutlass.Int32, TK_local: cutlass.Int32, K: cutlass.Int32,
    d: cutlass.Int32, n_meta: cutlass.Int32, sig: cutlass.Int32):
    # RAIL peer = dn (rail-relative node index), so this variant needs neither node_size nor my_local.
    dev_comm = nccl_cute.DevComm(dev_comm)
    x_win = nccl_cute.Window(x_win)
    dst_node_buf_win = nccl_cute.Window(dst_node_buf_win)
    least_win = nccl_cute.Window(least_win)
    team = dev_comm.team_rail        # RAIL (vs team_world)
    gin = dev_comm.gin(nccl_cute.GinBackendMask.ALL, 0)
    coop = nccl_cute.cta()
    tidx, _, _ = cute.arch.thread_idx()

    np_t = cute.make_tensor(
        cute.make_ptr(cutlass.Int8, node_present_ptr, cute.AddressSpace.gmem), cute.make_layout(n_meta))
    dn_t = cute.make_tensor(
        cute.make_ptr(cutlass.Int32, dst_node_ptr, cute.AddressSpace.gmem), cute.make_layout(n_meta))
    ds_t = cute.make_tensor(
        cute.make_ptr(cutlass.Int32, dst_slot_ptr, cute.AddressSpace.gmem), cute.make_layout(n_meta))

    for i in cutlass.range(TK_local, unroll=1):
        orig = my_base + i
        present = np_t[orig]
        if present != 0:
            dn = dn_t[orig]
            ds = ds_t[orig]
            t = i // K
            peer = dn                # RAIL: rail-relative node index (vs dn*node_size+my_local)
            src = x_win.tensor(cutlass.BFloat16, cute.make_layout(d),
                               offset=cutlass.Int64(t) * cutlass.Int64(d) * 2)
            dst = dst_node_buf_win.tensor(cutlass.BFloat16, cute.make_layout(d),
                                    offset=cutlass.Int64(ds) * cutlass.Int64(d) * 2)
            gin.put(team, peer, dst_node_buf_win, dst, x_win, src, coop,
                    is_signal=True, signal_id=sig, signal_op=0, signal_op_arg=1)

    cnt_t = cute.make_tensor(
        cute.make_ptr(cutlass.Int32, dst_recv_count_ptr, cute.AddressSpace.gmem), cute.make_layout(n_meta))
    inc = cutlass.Int64(cnt_t[my_rank])
    least_t = least_win.tensor(cutlass.Int64, cute.make_layout(1))
    le = least_t[0] + inc
    if tidx == 0:
        least_t[0] = le

    gin.wait_signal(coop, signal=sig, least=le)


@cute.jit
def _gin_dispatch_rail_launch(
    dev_comm: cutlass.Int64, x_win: cutlass.Int64, dst_node_buf_win: cutlass.Int64, least_win: cutlass.Int64,
    node_present_ptr: cutlass.Int64, dst_node_ptr: cutlass.Int64, dst_slot_ptr: cutlass.Int64,
    dst_recv_count_ptr: cutlass.Int64, my_rank: cutlass.Int32, my_base: cutlass.Int32,
    TK_local: cutlass.Int32, K: cutlass.Int32,
    d: cutlass.Int32, n_meta: cutlass.Int32, sig: cutlass.Int32,
    stream: cuda_driver.CUstream):
    _gin_dispatch_rail_kernel(
        dev_comm, x_win, dst_node_buf_win, least_win, node_present_ptr, dst_node_ptr, dst_slot_ptr,
        dst_recv_count_ptr, my_rank, my_base, TK_local, K, d, n_meta,
        sig).launch(
        grid=[1, 1, 1], block=[cute.size(WARP_SIZE, mode=[0]), 1, 1], cooperative=True, stream=stream)


# COALESCED: one big put per remote node from a pre-staged compact block; src(n*T_local) != dst
# (stripe_base[r,n]) since reuse would alias for num_nodes>2. signal_op_arg=cnt (not 1) keeps the receiver's wait-sum correct.
@cute.kernel
def _gin_dispatch_coalesced_kernel(
    dev_comm, staging_win, dst_node_buf_win, least_win,
    stripe_base_ptr, node_count_ptr, dst_recv_count_ptr,
    my_rank: cutlass.Int32, my_node: cutlass.Int32, my_local: cutlass.Int32,
    node_size: cutlass.Int32, num_nodes: cutlass.Int32, T_local: cutlass.Int32, d: cutlass.Int32,
    n_meta: cutlass.Int32, sig: cutlass.Int32, n_cta: cutlass.Constexpr, rail: cutlass.Constexpr):
    dev_comm = nccl_cute.DevComm(dev_comm)
    staging_win = nccl_cute.Window(staging_win)
    dst_node_buf_win = nccl_cute.Window(dst_node_buf_win)
    least_win = nccl_cute.Window(least_win)
    team = dev_comm.team_rail if rail else dev_comm.team_world
    gin = dev_comm.gin(nccl_cute.GinBackendMask.ALL, 0)
    coop = nccl_cute.cta()
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # stripe_base[my_rank, :] and node_token_count[my_rank, :] — the caller passes ROW pointers.
    sb_t = cute.make_tensor(
        cute.make_ptr(cutlass.Int32, stripe_base_ptr, cute.AddressSpace.gmem), cute.make_layout(num_nodes))
    cnt_t = cute.make_tensor(
        cute.make_ptr(cutlass.Int32, node_count_ptr, cute.AddressSpace.gmem), cute.make_layout(num_nodes))

    # one coalesced put per remote node, strided across the n_cta CTAs (CTA bidx owns node bidx)
    for n in cutlass.range(num_nodes, unroll=1):
        if n != my_node:
            if n % n_cta == bidx:
                cnt = cnt_t[n]
                if cnt > 0:
                    base = sb_t[n]
                    if cutlass.const_expr(rail):
                        peer = n
                    else:
                        peer = n * node_size + my_local
                    nelem = cnt * d
                    src = staging_win.tensor(cutlass.BFloat16, cute.make_layout(nelem),  # COMPACT per-node src block
                                             offset=cutlass.Int64(n) * cutlass.Int64(T_local) * cutlass.Int64(d) * 2)
                    dst = dst_node_buf_win.tensor(cutlass.BFloat16, cute.make_layout(nelem),  # r's stripe in receiver n
                                                  offset=cutlass.Int64(base) * cutlass.Int64(d) * 2)
                    gin.put(team, peer, dst_node_buf_win, dst, staging_win, src, coop,
                            is_signal=True, signal_id=sig, signal_op=1,  # ncclGinSignalAdd (op 0 Inc
                            signal_op_arg=cutlass.Int64(cnt))            # forces arg→1; Add respects +cnt)

    if bidx == 0:
        cnt_recv = cute.make_tensor(
            cute.make_ptr(cutlass.Int32, dst_recv_count_ptr, cute.AddressSpace.gmem), cute.make_layout(n_meta))
        inc = cutlass.Int64(cnt_recv[my_rank])
        least_t = least_win.tensor(cutlass.Int64, cute.make_layout(1))
        le = least_t[0] + inc
        if tidx == 0:
            least_t[0] = le

        gin.wait_signal(coop, signal=sig, least=le)


@cute.jit
def _gin_dispatch_coalesced_launch(
    dev_comm: cutlass.Int64, staging_win: cutlass.Int64, dst_node_buf_win: cutlass.Int64,
    least_win: cutlass.Int64, stripe_base_ptr: cutlass.Int64, node_count_ptr: cutlass.Int64,
    dst_recv_count_ptr: cutlass.Int64, my_rank: cutlass.Int32, my_node: cutlass.Int32,
    my_local: cutlass.Int32, node_size: cutlass.Int32, num_nodes: cutlass.Int32, T_local: cutlass.Int32,
    d: cutlass.Int32, n_meta: cutlass.Int32, sig: cutlass.Int32, n_cta: cutlass.Constexpr,
    rail: cutlass.Constexpr, stream: cuda_driver.CUstream):
    _gin_dispatch_coalesced_kernel(
        dev_comm, staging_win, dst_node_buf_win, least_win, stripe_base_ptr, node_count_ptr,
        dst_recv_count_ptr, my_rank, my_node, my_local, node_size, num_nodes, T_local, d, n_meta, sig,
        n_cta, rail).launch(
        grid=[n_cta, 1, 1], block=[cute.size(WARP_SIZE, mode=[0]), 1, 1], cooperative=True, stream=stream)


def launch_dispatch_coalesced(backend, staging_win, dst_node_buf_win, least_win, meta, *,
                              rank: int, node_size: int, num_nodes: int, T_local: int, d: int, sig: int = 0,
                              rail: bool = False, n_cta: int = 0, stream=None):
    """Host run-path for the COALESCED GIN dispatch. PRECONDITION: ``hier_stage_coalesced_triton`` already
    scattered rows into ``staging_win``. ``rail=True`` needs comm created with connection="RAIL"."""
    nc = int(n_cta) if n_cta and n_cta > 0 else int(num_nodes)
    my_node = rank // node_size
    my_local = rank % node_size
    W = backend.world_size
    GinLaunchHelper.launch(
        _gin_dispatch_coalesced_launch, backend.dev_ptr, staging_win.handle, dst_node_buf_win.handle,
        least_win.handle, meta["stripe_base"][rank].data_ptr(), meta["node_token_count"][rank].data_ptr(),
        meta["dst_recv_count"].data_ptr(), cutlass.Int32(rank), cutlass.Int32(my_node),
        cutlass.Int32(my_local), cutlass.Int32(node_size), cutlass.Int32(num_nodes),
        cutlass.Int32(T_local), cutlass.Int32(d), cutlass.Int32(W), cutlass.Int32(sig), nc, bool(rail),
        stream=stream)


# Extracts node_size LSA peer base addresses into an int64 tensor, ONCE at setup (peer VAs are stable
# for the window's lifetime) — runtime int->ptr addressing like a2a_combine; no torch symm-mem, no copies.
@cute.kernel
def _lsa_base_extract_kernel(win, out_ptr, node_size: cutlass.Int32):
    win = nccl_cute.Window(win)
    tidx, _, _ = cute.arch.thread_idx()
    out_t = cute.make_tensor(
        cute.make_ptr(cutlass.Int64, out_ptr, cute.AddressSpace.gmem), cute.make_layout(node_size))
    if tidx == 0:
        for p in cutlass.range(node_size, unroll=1):
            out_t[p] = cutlass.Int64(cute.make_ptr(cutlass.Int8, win.lsa_pointer(0, p)).toint())


@cute.jit
def _lsa_base_extract_launch(win: cutlass.Int64, out_ptr: cutlass.Int64, node_size: cutlass.Int32,
                             stream: cuda_driver.CUstream):
    _lsa_base_extract_kernel(win, out_ptr, node_size).launch(
        grid=[1, 1, 1], block=[cute.size(WARP_SIZE, mode=[0]), 1, 1], cooperative=True, stream=stream)


def build_lsa_base(backend, win, node_size: int, *, stream=None) -> torch.Tensor:
    """Extracts a window's ``node_size`` LSA peer base addresses. Call ONCE per window (VAs are stable),
    AFTER make_dev_comm; assumes ``node_size == lsa_size`` (asserted in ``build_workspace``)."""
    out = torch.empty(node_size, dtype=torch.int64, device=backend.device)
    GinLaunchHelper.launch(_lsa_base_extract_launch, win.handle, out.data_ptr(),
                           cutlass.Int32(node_size), stream=stream)
    return out


def compile_dispatch_kernels():
    """Offline-compile the dispatch + LSA-base-extract launches (no live comm) — login compile-check."""
    z32 = cutlass.Int32(0)
    h_dispatch = cute.compile(
        _gin_dispatch_launch, 0, 0, 0, 0, 0, 0, 0, 0,
        z32, z32, z32, cutlass.Int32(1), cutlass.Int32(8), cutlass.Int32(2),
        cutlass.Int32(16), cutlass.Int32(64), z32, cuda_driver.CUstream(0))
    h_rail = cute.compile(
        _gin_dispatch_rail_launch, 0, 0, 0, 0, 0, 0, 0, 0,
        z32, z32, cutlass.Int32(8), cutlass.Int32(2),
        cutlass.Int32(16), cutlass.Int32(64), z32, cuda_driver.CUstream(0))
    h_coalesced = cute.compile(
        _gin_dispatch_coalesced_launch, 0, 0, 0, 0, 0, 0, 0,
        z32, z32, z32, cutlass.Int32(4), cutlass.Int32(2), cutlass.Int32(8), cutlass.Int32(16),
        cutlass.Int32(64), z32, 2, False, cuda_driver.CUstream(0))
    h_extract = cute.compile(_lsa_base_extract_launch, 0, 0, cutlass.Int32(4), cuda_driver.CUstream(0))
    return h_dispatch, h_rail, h_coalesced, h_extract


def launch_dispatch(backend, x_win, dst_node_buf_win, least_win, meta, *,
                    rank: int, node_size: int, T_local: int, K: int, d: int, sig: int = 0,
                    rail: bool = False, stream=None):
    """Host run-path for the node-dedup GIN dispatch. Always launched (graph-capturable); ``dst_recv_count
    [rank]==0`` ⇒ no-op wait. ``rail=True`` needs comm created with connection="RAIL"."""
    TK_local = T_local * K
    TK_global = backend.world_size * TK_local
    my_base = rank * TK_local
    args = [backend.dev_ptr, x_win.handle, dst_node_buf_win.handle, least_win.handle,
            meta["node_present_mask"].data_ptr(), meta["dst_node_flat"].data_ptr(),
            meta["dst_slot"].data_ptr(), meta["dst_recv_count"].data_ptr(),
            cutlass.Int32(rank), cutlass.Int32(my_base)]
    if rail:
        launch_fn = _gin_dispatch_rail_launch
    else:  # FULL peer = dn*node_size + my_local; RAIL peer = dn, so it omits both
        launch_fn = _gin_dispatch_launch
        args += [cutlass.Int32(rank % node_size), cutlass.Int32(node_size)]
    args += [cutlass.Int32(TK_local), cutlass.Int32(K), cutlass.Int32(d), cutlass.Int32(TK_global),
             cutlass.Int32(sig)]
    GinLaunchHelper.launch(launch_fn, *args, stream=stream)


def hier_dispatch_forward(backend, x_gin_win, dst_node_buf_win, least_win, x_lsa_base,
                          dst_node_buf_lsa_base, recv_packed, meta, *, rank, world_size, node_size,
                          T_local, K, H, group, node_hdl=None, rail=False,
                          coalesce=_COALESCE_DEFAULT, staging_win=None, sig=0, stream=None):
    """Full inter-node dispatch (no copies/symm-mem). PRECONDITION: rows already staged in ``x_gin_win``.
    Under graph capture, host sync is illegal — requires ``node_hdl`` for a capturable barrier."""
    capturing = torch.cuda.is_current_stream_capturing()

    def _node_barrier():
        # intra-node NVLink visibility (same-node peers read x_gin / dst_node_buffer in the gather);
        # cross-node ordering is the GIN signal/wait. Capturable symm-mem barrier under graph capture.
        if capturing:
            assert node_hdl is not None, "hier_dispatch_forward needs node_hdl (node-group symm-mem) under CUDA-graph capture"
            node_hdl.barrier()
        else:
            torch.cuda.synchronize()
            dist.barrier(group=group)

    _node_barrier()  # x_gin filled (by caller) before any cross-node put / same-node read

    if coalesce:
        assert staging_win is not None, "hier_dispatch_forward(coalesce=True) needs staging_win"
        # COALESCED: local scatter x_gin[token] → staging[dst_slot], then ONE big contiguous put per
        # remote node (replaces ~512 tiny per-token puts; the per-token path stayed ~1 GB/s — tiny RDMA).
        hier_stage_coalesced_triton(x_gin_win.tensor, meta["node_present_mask"], meta["dst_slot"],
                                    meta["dst_node_flat"], meta["stripe_base"][rank], staging_win.tensor,
                                    rank=rank, T_local=T_local, K=K, d=H)
        launch_dispatch_coalesced(backend, staging_win, dst_node_buf_win, least_win, meta,
                                  rank=rank, node_size=node_size, num_nodes=world_size // node_size,
                                  T_local=T_local, d=H, sig=sig, rail=rail, stream=stream)
    else:
        launch_dispatch(backend, x_gin_win, dst_node_buf_win, least_win, meta,
                        rank=rank, node_size=node_size, T_local=T_local, K=K, d=H, sig=sig, rail=rail,
                        stream=stream)
    _node_barrier()  # dst_node_buffers filled (GIN-wait) + x_gin NVLink-visible to same-node peers

    hier_gather_rt_triton(
        x_lsa_base, dst_node_buf_lsa_base, meta["pair_present_mask"], meta["is_local_slot"],
        meta["dst_rank_flat"], meta["dst_slot"], meta["rank_dedup_recv_pos"],
        recv_packed, K, rank, world_size, node_size)
    return recv_packed
