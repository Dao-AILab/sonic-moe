# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
"""Cluster validation: hierarchical GIN DISPATCH transport (put/wait -> unified gather), multi-node;
mirror of hier_ep_combine_gin_test. srun on >=2 nodes (set NODE_SIZE=GPUs/node)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent))  # repo root for the sonicmoe package


def _world():
    return (int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0))),
            int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1))),
            int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0))))


def _make_global_routing(W, num_nodes, node_size, T_local, K, E_local, seed):
    """Deterministic global routing (identical on every rank), biased remote so cross-node GIN fires."""
    g = torch.Generator().manual_seed(seed)
    E = W * E_local
    out = torch.empty(W, T_local, K, dtype=torch.int64)
    for r in range(W):
        my_node = r // node_size
        remote_ranks = [rk for rk in range(W) if rk // node_size != my_node]
        for t in range(T_local):
            # ~half the picks forced remote (exercise GIN), rest free — distinct experts per token
            picks = set()
            n_remote = K // 2 if num_nodes > 1 else 0
            while len(picks) < n_remote and remote_ranks:
                rk = remote_ranks[int(torch.randint(0, len(remote_ranks), (1,), generator=g))]
                picks.add(int(torch.randint(rk * E_local, (rk + 1) * E_local, (1,), generator=g)))
            while len(picks) < K:
                picks.add(int(torch.randint(0, E, (1,), generator=g)))
            out[r, t] = torch.tensor(sorted(picks)[:K], dtype=torch.int64)
    return out


def test_dispatch_kernel_compiles_singleproc():
    """Offline smoke: the node-dedup GIN dispatch kernel compiles with no live comm."""
    import pytest
    pytest.importorskip("sonicmoe.functional.distributed.nccl_gin",
                        reason="CuTeDSL/nccl4py stack or libnccl_device.bc unavailable")
    from sonicmoe.functional.distributed.nccl_gin import dispatch
    dispatch.compile_dispatch_kernels()


def _run_e2e():
    import torch.distributed as dist
    from torch.distributed import _symmetric_memory as _symm_mem
    import nccl.core as nccl
    from sonicmoe.functional.distributed.nccl_gin import NCCLGin
    from sonicmoe.functional.distributed.nccl_gin import dispatch as gin_dispatch
    from sonicmoe.functional.distributed.metadata import compute_dispatch_metadata
    import sonicmoe.functional.distributed.ep_dispatch as ep
    from hier_ep_reference import compute_hier_dispatch_reference, recv_gpu

    rank, world, local = _world()
    if world < 2:
        print(f"[rank {rank}] need >=2 ranks", flush=True)
        return 1

    if "MASTER_ADDR" not in os.environ:
        import subprocess
        nl = os.environ.get("SLURM_NODELIST", "127.0.0.1")
        os.environ["MASTER_ADDR"] = (subprocess.check_output(
            ["scontrol", "show", "hostnames", nl]).decode().split()[0]
            if ("[" in nl or "," in nl) else nl)

    os.environ.setdefault("MASTER_PORT", "29563")
    if local >= torch.cuda.device_count():  # SLURM gpu-bind: each task sees only its 1 GPU as device 0
        local = 0
    torch.cuda.set_device(local)
    # NCCL backend (+ gloo for objects) so the node-group symm-mem barrier used by the CUDA-graph capture
    # path can rendezvous; WORLD control ops still go over gloo, GIN handles the cross-node data.
    dist.init_process_group(backend="cuda:nccl,cpu:gloo", rank=rank, world_size=world)

    node_size = int(os.environ.get("NODE_SIZE", os.environ.get("SLURM_NTASKS_PER_NODE", 1)))
    assert world % node_size == 0, f"world {world} not divisible by node_size {node_size}"
    num_nodes = world // node_size
    my_node = rank // node_size

    T_local = int(os.environ.get("T_LOCAL", "8"))
    K = int(os.environ.get("K", "4"))
    E_local = int(os.environ.get("E_LOCAL", "2"))
    d = int(os.environ.get("D", "128"))
    assert world * T_local <= 256, "token id must stay bf16-exact (<256)"
    SIG = 0
    # CONNECTION=RAIL routes via dev_comm.team_rail (node-relative peer) instead of team_world
    # (global rank) — correctness-equivalent to FULL, only the wire/connection type differs.
    CONNECTION = os.environ.get("CONNECTION", "FULL").upper()
    rail = CONNECTION == "RAIL"
    errors = 0

    def _verdict(s):
        print(f"[rank {rank}] {s}", flush=True)

    topk = _make_global_routing(world, num_nodes, node_size, T_local, K, E_local, seed=1234)
    topk_dev = topk.to(torch.int32).to(local)
    meta = compute_dispatch_metadata(topk_dev, rank, E_local, emit_combine=False,
                                     emit_hier=True, node_size=node_size)
    ref = compute_hier_dispatch_reference(topk, num_nodes, node_size, E_local)
    # (Metadata parity vs the oracle is covered by metadata_test::HierDispatchMetadataTest.)

    # ── liveness + correctness: node-dedup GIN dispatch put/wait, then the unified gather ──
    DST_NODE_BUF_ROWS = max(T_local * (num_nodes - 1), 1)

    # node-group symm-mem for the capturable barrier (node_hdl.barrier()); rendezvous BEFORE the GIN comm
    # (coexistence recipe). Intra-node only (NVLink) — no cross-node IB involved.
    node_group = None
    for nid in range(num_nodes):
        grp = dist.new_group(ranks=list(range(nid * node_size, (nid + 1) * node_size)))
        if nid == my_node:
            node_group = grp
    dist.barrier()
    _bar_buf = _symm_mem.empty((max(node_size, 1),), dtype=torch.int32, device=local)
    node_hdl = _symm_mem.rendezvous(_bar_buf, group=node_group)

    def fresh_uid():
        u = nccl.get_unique_id() if rank == 0 else None
        o = [u]
        dist.broadcast_object_list(o, src=0)
        return o[0]

    be = NCCLGin(rank, world, fresh_uid(), device=local)
    x_win = be.alloc_window(T_local * d, torch.bfloat16)
    dst_node_buf_win = be.alloc_window(DST_NODE_BUF_ROWS * d, torch.bfloat16)
    staging_win = be.alloc_window(num_nodes * T_local * d, torch.bfloat16)  # compact per-node coalesced-put staging
    least_win = be.alloc_window(1, torch.int64)
    be.make_dev_comm(signal_count=1, connection=CONNECTION)
    be.bind_signal(least_win, SIG)
    be.reset_epoch(0)
    if rank == 0:
        _verdict(f"[probe] connection={CONNECTION} railed_gin_type={be.railed_gin_type} (rail={rail})")

    # expected dst_node_buffer for MY rank: which (src, t) lands at each of my dst_slot rows
    np_m = ref.node_present_mask.view(world, T_local, K)
    dn_m = ref.dst_node_flat.view(world, T_local, K)
    ds_m = ref.dst_slot.view(world, T_local, K)
    expected = {}  # ds -> token id
    for src in range(world):
        for t in range(T_local):
            for k in range(K):
                if int(np_m[src, t, k]) and recv_gpu(src, int(dn_m[src, t, k]), node_size) == rank:
                    expected[int(ds_m[src, t, k])] = src * T_local + t

    # build LSA-base addrs, then gather same-node x + remote dst_node_buffer (both via lsa_pointer) into
    # recv_packed, bit-exact vs the oracle golden.
    assert be.lsa_size == node_size, \
        f"unified gather assumes node==LSA domain: lsa_size {be.lsa_size} != node_size {node_size}"
    x_lsa = gin_dispatch.build_lsa_base(be, x_win, node_size)               # int64[node_size]
    dnb_lsa = gin_dispatch.build_lsa_base(be, dst_node_buf_win, node_size)  # int64[node_size]
    torch.cuda.synchronize()
    pp_m = ref.pair_present_mask.view(world, T_local, K)
    dr_m = ref.dst_rank_flat.view(world, T_local, K)
    pos_m = ref.rank_dedup_recv_pos.view(world, T_local, K)
    recv_rows = int(ref.recv_rows_per_rank[rank])
    golden_recv = {}  # recv_pos -> token id (canonical slots routed to my rank)
    for src in range(world):
        for t in range(T_local):
            for k in range(K):
                if int(pp_m[src, t, k]) and int(dr_m[src, t, k]) == rank:
                    golden_recv[int(pos_m[src, t, k])] = src * T_local + t

    for rnd in range(2):  # 2 rounds: exercise the device-resident epoch across dispatches
        xt = x_win.tensor.view(T_local, d)
        for t in range(T_local):
            xt[t].fill_(float(rank * T_local + t))

        dst_node_buf_win.tensor.fill_(-1.0)
        torch.cuda.synchronize()
        dist.barrier()

        # rail=False -> FULL kernel (team_world, global peer); rail=True -> RAIL kernel (team_rail, node
        # peer). Same landing either way, so the dst_node_buffer + gather checks below are connection-agnostic.
        gin_dispatch.launch_dispatch(
            be, x_win, dst_node_buf_win, least_win, meta,
            rank=rank, node_size=node_size, T_local=T_local, K=K, d=d, sig=SIG, rail=rail)
        # the signal epoch advances device-side INSIDE the kernel (least[0] += dst_recv_count), so it
        # carries across rounds with no host arming; round 2 waits on 2*dst_recv_count.
        torch.cuda.synchronize()  # reaching here => the wait completed => no deadlock

        db = dst_node_buf_win.tensor.view(DST_NODE_BUF_ROWS, d).to("cpu")
        rnd_err = 0
        for ds, tid in expected.items():
            if not bool((db[ds] == float(tid)).all()):
                rnd_err += 1
        # unused rows must remain at the -1 init (no stray writes)
        used = set(expected.keys())
        for ds in range(DST_NODE_BUF_ROWS):
            if ds not in used and not bool((db[ds] == -1.0).all()):
                rnd_err += 1

        # unified gather: same-node x + remote dst_node_buffer (via lsa) -> recv_packed.
        dist.barrier()  # all ranks' put/wait done -> peers' dst_node_buffers + x NVLink-visible
        # recv_packed MUST match the GIN window dtype (bf16) — the gather reads peer windows AS
        # recv_packed.dtype; a float32 recv would read the bf16 windows as f32 => all rows garbage.
        recv = torch.full((max(recv_rows, 1), d), -2.0, device=local, dtype=torch.bfloat16)
        ep.hier_gather_rt_triton(
            x_lsa, dnb_lsa, meta["pair_present_mask"], meta["is_local_slot"], meta["dst_rank_flat"],
            meta["dst_slot"], meta["rank_dedup_recv_pos"], recv, K, rank, world, node_size)
        torch.cuda.synchronize()
        rv = recv.to("cpu")
        gather_err = 0
        for p, tid in golden_recv.items():
            if not bool((rv[p] == float(tid)).all()):
                gather_err += 1
        if recv_rows > 0 and not bool((rv[:recv_rows] >= 0).all()):
            gather_err += 1  # an unfilled recv_packed row
        rnd_err += gather_err

        errors += rnd_err
        if rank == 0:
            _verdict(f"[ok] round {rnd}: dst_node_buffer + recv_packed ({recv_rows} rows) bit-exact, "
                     f"no deadlock" if rnd_err == 0
                     else f"[FAIL] round {rnd}: {rnd_err} mismatches (gather={gather_err})")
        dist.barrier()

    # Under capture: node_hdl.barrier() replaces host sync, GIN epoch is device-resident in-graph, and
    # the node-group symm-mem barrier must also fence the non-symm-mem GIN window writes the gather reads.
    cap_err = 0
    try:
        xt = x_win.tensor.view(T_local, d)

        def _stage_x():
            for t in range(T_local):
                xt[t].fill_(float(rank * T_local + t))

        _stage_x()
        recv_cap = torch.full((max(recv_rows, 1), d), -2.0, device=local, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        dist.barrier()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):  # warmup (eager path) to compile before capture
            gin_dispatch.hier_dispatch_forward(
                be, x_win, dst_node_buf_win, least_win, x_lsa, dnb_lsa, recv_cap, meta,
                rank=rank, world_size=world, node_size=node_size, T_local=T_local, K=K, H=d,
                group=dist.group.WORLD, node_hdl=node_hdl, rail=rail, sig=SIG, staging_win=staging_win)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        dist.barrier()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            gin_dispatch.hier_dispatch_forward(
                be, x_win, dst_node_buf_win, least_win, x_lsa, dnb_lsa, recv_cap, meta,
                rank=rank, world_size=world, node_size=node_size, T_local=T_local, K=K, H=d,
                group=dist.group.WORLD, node_hdl=node_hdl, rail=rail, sig=SIG, staging_win=staging_win)
        torch.cuda.synchronize()
        dist.barrier()
        for rep in range(2):
            recv_cap.fill_(-2.0)
            _stage_x()
            torch.cuda.synchronize()
            dist.barrier()
            g.replay()
            torch.cuda.synchronize()
            rv = recv_cap.to("cpu")
            for p, tid in golden_recv.items():
                if not bool((rv[p] == float(tid)).all()):
                    cap_err += 1
            if recv_rows > 0 and not bool((rv[:recv_rows] >= 0).all()):
                cap_err += 1
        errors += cap_err
        if rank == 0:
            _verdict("[ok] dispatch CUDA-graph capture+replay (2 replays) recv_packed bit-exact"
                     if cap_err == 0 else f"[FAIL] dispatch cudagraph: {cap_err} mismatches")
    except Exception as e:
        errors += 1
        if rank == 0:
            _verdict(f"[FAIL] dispatch cudagraph raised: {type(e).__name__}: {str(e)[:180]}")
    dist.barrier()

    # teardown: drop node-group symm-mem refs (handle before its buffer) before closing the GIN comm / PG
    node_hdl = None      # noqa: F841
    _bar_buf = None      # noqa: F841
    import gc
    gc.collect()
    torch.cuda.synchronize()
    be.close()
    dist.barrier()

    all_err = [None] * world
    dist.all_gather_object(all_err, errors)
    total = sum(int(e) for e in all_err if e is not None)
    if rank == 0:
        _verdict("[SUCCESS] full dispatch: GIN put/wait + lsa_pointer gather -> recv_packed bit-exact "
                 "(ONE GIN window, no copies/symm-mem)"
                 if total == 0 else f"[FAIL] dispatch e2e: {total} errors across ranks")
    dist.barrier()
    dist.destroy_process_group()
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_e2e())
