# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
"""Hierarchical inter-node COMBINE = reduction, not expand (NO RDMA atomics): gateway NVLink-reduce ->
GIN stripe-put -> origin NVLink-reduce. Uses a SEPARATE signal slot from dispatch — sharing one would let combine's wait pass too early.
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
from ..ep_combine import (  # always-available (Triton); the NVLink reduce halves
    hier_combine_gateway_reduce_triton, hier_combine_origin_reduce_triton,
)


# Combine put kernel: (1) gateway posts one weak-signalled stripe-put per served remote origin-node,
# (2) waits on its own COMBINE epoch (expected_count_combine[my_rank]); inc==0 ⇒ returns immediately.
@cute.kernel
def _gin_combine_put_kernel(
    dev_comm, send_buf_win, recv_buf_win, least_win,
    contrib_ptr, expected_count_ptr,
    my_rank: cutlass.Int32, my_local: cutlass.Int32, node_size: cutlass.Int32,
    num_nodes: cutlass.Int32, T_local: cutlass.Int32, d: cutlass.Int32,
    W: cutlass.Int32, sig: cutlass.Int32):
    dev_comm = nccl_cute.DevComm(dev_comm)
    send_buf_win = nccl_cute.Window(send_buf_win)
    recv_buf_win = nccl_cute.Window(recv_buf_win)
    least_win = nccl_cute.Window(least_win)
    team = dev_comm.team_world
    gin = dev_comm.gin(nccl_cute.GinBackendMask.ALL, 0)
    coop = nccl_cute.cta()
    tidx, _, _ = cute.arch.thread_idx()

    g = my_rank // node_size          # my node
    block = T_local * d               # bf16 elements per dense stripe

    # contrib_node_mask (W, num_nodes) int8 over plain HBM (not a GIN window)
    contrib_t = cute.make_tensor(
        cute.make_ptr(cutlass.Int8, contrib_ptr, cute.AddressSpace.gmem),
        cute.make_layout(W * num_nodes))

    # 1) post gateway stripe-puts: R=rank_of(n,my_local) is served iff contrib_node_mask[R][g]!=0
    # (combine mirrors dispatch's node set). s_send/s_recv are diagonal-compacted stripe indices.
    for n in cutlass.range(num_nodes, unroll=1):
        if n != g:
            R = n * node_size + my_local
            if contrib_t[R * num_nodes + g] != 0:
                s_send = n if n < g else n - 1     # stripe in MY send_buf for node n
                s_recv = g if g < n else g - 1     # stripe in R's recv_buf for MY node g
                src = send_buf_win.tensor(cutlass.BFloat16, cute.make_layout(block),
                                          offset=cutlass.Int64(s_send) * cutlass.Int64(block) * 2)
                dst = recv_buf_win.tensor(cutlass.BFloat16, cute.make_layout(block),
                                          offset=cutlass.Int64(s_recv) * cutlass.Int64(block) * 2)
                gin.put(team, R, recv_buf_win, dst, send_buf_win, src, coop,
                        is_signal=True, signal_id=sig, signal_op=0, signal_op_arg=1)

    # 2) wait for my inbound combine puts (expected_count_combine[my_rank]); device-resident in-graph epoch
    cnt_t = cute.make_tensor(
        cute.make_ptr(cutlass.Int32, expected_count_ptr, cute.AddressSpace.gmem), cute.make_layout(W))
    inc = cutlass.Int64(cnt_t[my_rank])
    least_t = least_win.tensor(cutlass.Int64, cute.make_layout(1))
    le = least_t[0] + inc
    if tidx == 0:
        least_t[0] = le

    gin.wait_signal(coop, signal=sig, least=le)


@cute.jit
def _gin_combine_put_launch(
    dev_comm: cutlass.Int64, send_buf_win: cutlass.Int64, recv_buf_win: cutlass.Int64,
    least_win: cutlass.Int64, contrib_ptr: cutlass.Int64, expected_count_ptr: cutlass.Int64,
    my_rank: cutlass.Int32, my_local: cutlass.Int32, node_size: cutlass.Int32,
    num_nodes: cutlass.Int32, T_local: cutlass.Int32, d: cutlass.Int32, W: cutlass.Int32,
    sig: cutlass.Int32, stream: cuda_driver.CUstream):
    _gin_combine_put_kernel(
        dev_comm, send_buf_win, recv_buf_win, least_win, contrib_ptr, expected_count_ptr,
        my_rank, my_local, node_size, num_nodes, T_local, d, W, sig).launch(
        grid=[1, 1, 1], block=[cute.size(WARP_SIZE, mode=[0]), 1, 1], cooperative=True, stream=stream)


def compile_combine_kernels():
    """Offline-compile the combine put launch (no live comm) — login-node compile-check (no GPU)."""
    z32 = cutlass.Int32(0)
    return cute.compile(
        _gin_combine_put_launch, 0, 0, 0, 0, 0, 0,
        z32, z32, cutlass.Int32(4), cutlass.Int32(2), cutlass.Int32(8), cutlass.Int32(64),
        cutlass.Int32(8), cutlass.Int32(1), cuda_driver.CUstream(0))


def launch_combine_put(backend, send_buf_win, recv_buf_win, combine_least_win, meta, *,
                       rank: int, node_size: int, num_nodes: int, T_local: int, d: int,
                       sig: int = 1, stream=None):
    """Host run-path for the node-dedup GIN combine put. ``sig`` defaults to 1 (distinct from dispatch's 0).
    Always launched (graph-capturable); ``expected_count_combine[rank]==0`` ⇒ no-op wait."""
    W = backend.world_size
    my_local = rank % node_size
    GinLaunchHelper.launch(
        _gin_combine_put_launch, backend.dev_ptr, send_buf_win.handle, recv_buf_win.handle,
        combine_least_win.handle, meta["contrib_node_mask"].data_ptr(),
        meta["expected_count_combine"].data_ptr(),
        cutlass.Int32(rank), cutlass.Int32(my_local), cutlass.Int32(node_size),
        cutlass.Int32(num_nodes), cutlass.Int32(T_local), cutlass.Int32(d), cutlass.Int32(W),
        cutlass.Int32(sig), stream=stream)


def hier_combine_forward(backend, send_buf_win, recv_buf_win, combine_least_win,
                         partial_combine_peer_bufs, recv_buf_tensor, out, meta, *,
                         rank, world_size, node_size, num_nodes, T_local, d, group,
                         node_hdl=None, sig=1, stream=None):
    """Full inter-node combine (no copies/symm-mem/RDMA atomics). PRECONDITION: ``local_combine`` already
    filled every rank's ``partial_combine_buf``. Under graph capture, ``node_hdl`` MUST be ``partial_combine_hdl`` (the same symm-mem the partials live in)."""
    present_all = meta["combine_peer_present_all"]
    contrib_node_mask = meta["contrib_node_mask"]
    send_buf_tensor = send_buf_win.tensor.view(-1, d)
    capturing = torch.cuda.is_current_stream_capturing()

    def _node_barrier():
        if capturing:
            assert node_hdl is not None, "hier_combine_forward needs node_hdl (node-group symm-mem) under CUDA-graph capture"
            node_hdl.barrier()
        else:
            torch.cuda.synchronize()
            dist.barrier(group=group)

    # 0) PRODUCER BARRIER: gateway/origin reduces below read node-local PEERS' partial_combine_buf over
    #    NVLink — without this, the gateway reads stale peer partials (error grows with #contributors).
    _node_barrier()

    # 1) gateway reduce: node-local peers' partials -> send_buf (one dense (T_local,d) stripe per remote node)
    hier_combine_gateway_reduce_triton(
        partial_combine_peer_bufs, present_all, send_buf_tensor,
        T_local=T_local, node_size=node_size, num_nodes=num_nodes, W=world_size, my_rank=rank, d=d)
    _node_barrier()   # all gateways' send_bufs ready before any cross-node put

    # 2) GIN combine put+wait: gateway stripes -> origins' recv_buf
    launch_combine_put(backend, send_buf_win, recv_buf_win, combine_least_win, meta,
                       rank=rank, node_size=node_size, num_nodes=num_nodes, T_local=T_local, d=d,
                       sig=sig, stream=stream)
    _node_barrier()   # all recv_bufs filled (GIN-wait) + partials NVLink-visible to same-node peers

    # 3) origin reduce: node-local peers (NVLink, present-gated) + remote stripes (recv_buf, contrib-gated) -> out
    hier_combine_origin_reduce_triton(
        partial_combine_peer_bufs, present_all, contrib_node_mask, recv_buf_tensor, out,
        T_local=T_local, node_size=node_size, num_nodes=num_nodes, W=world_size, my_rank=rank, d=d)
    return out
