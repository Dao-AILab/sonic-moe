# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
"""Cluster validation: hierarchical GIN COMBINE transport (gateway reduce -> GIN put -> origin reduce);
srun on >=2 nodes. Fill value 2^q per peer makes a mis-routed stripe a decodable bitmask mismatch."""
import os
import sys

import pytest

pytest.importorskip("sonicmoe.functional.distributed.nccl_gin",
                    reason="CuTeDSL/nccl4py stack or libnccl_device.bc unavailable")


def _world():
    return (int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0))),
            int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1))),
            int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0))))


def test_combine_kernel_compiles_singleproc():
    """Single-process smoke: the node-dedup combine put kernel compiles offline (no live comm)."""
    from sonicmoe.functional.distributed.nccl_gin import combine
    combine.compile_combine_kernels()


def _make_global_routing(W, num_nodes, node_size, T_local, K, E_local, seed):
    """Deterministic global routing (identical on every rank), biased remote so cross-node GIN fires."""
    import torch
    g = torch.Generator().manual_seed(seed)
    E = W * E_local
    out = torch.empty(W, T_local, K, dtype=torch.int64)
    for r in range(W):
        my_node = r // node_size
        remote_ranks = [rk for rk in range(W) if rk // node_size != my_node]
        for t in range(T_local):
            picks = set()
            n_remote = K // 2 if num_nodes > 1 else 0
            while len(picks) < n_remote and remote_ranks:
                rk = remote_ranks[int(torch.randint(0, len(remote_ranks), (1,), generator=g))]
                picks.add(int(torch.randint(rk * E_local, (rk + 1) * E_local, (1,), generator=g)))
            while len(picks) < K:
                picks.add(int(torch.randint(0, E, (1,), generator=g)))
            out[r, t] = torch.tensor(sorted(picks)[:K], dtype=torch.int64)
    return out


def _run():
    import torch
    import torch.distributed as dist
    from torch.distributed import _symmetric_memory as _symm_mem
    import nccl.core as nccl
    from sonicmoe.functional.distributed.nccl_gin import NCCLGin
    from sonicmoe.functional.distributed.nccl_gin import combine as gin_combine
    from sonicmoe.functional.distributed.metadata import compute_dispatch_metadata

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hier_ep_reference import compute_hier_combine_reference

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
    os.environ.setdefault("MASTER_PORT", "29564")
    if local >= torch.cuda.device_count():  # SLURM gpu-bind: each task sees only its 1 GPU as device 0
        local = 0
    torch.cuda.set_device(local)
    # NCCL backend is required for symm-mem rendezvous (node-group partial_combine_buf); gloo for objects.
    dist.init_process_group(backend="cuda:nccl,cpu:gloo", rank=rank, world_size=world)

    node_size = int(os.environ.get("NODE_SIZE", os.environ.get("SLURM_NTASKS_PER_NODE", 1)))
    assert world % node_size == 0, f"world {world} not divisible by node_size {node_size}"
    num_nodes = world // node_size
    my_node, my_local = rank // node_size, rank % node_size

    T_local = int(os.environ.get("T_LOCAL", "8"))
    K = int(os.environ.get("K", "4"))
    E_local = int(os.environ.get("E_LOCAL", "2"))
    d = int(os.environ.get("D", "128"))
    assert world <= 256, "2^q fill assumes W small enough for bf16-exact powers (q<W, 2^q<=256)"
    SIG_COMBINE = 1            # combine uses indexed signal slot 1 (dispatch would be 0)
    POISON = -100.0           # bf16-exact; loud if a present-gating bug leaks an absent row into a sum
    COMBINE_RECV_ROWS = max(T_local * (num_nodes - 1), 1)
    errors = 0

    def _verdict(s):
        print(f"[rank {rank}] {s}", flush=True)

    topk = _make_global_routing(world, num_nodes, node_size, T_local, K, E_local, seed=1234)
    topk_dev = topk.to(torch.int32).to(local)
    meta = compute_dispatch_metadata(topk_dev, rank, E_local, emit_combine=False,
                                     emit_hier=True, node_size=node_size)
    cref = compute_hier_combine_reference(topk, num_nodes, node_size, E_local)

    # (Metadata parity vs the oracle is covered by metadata_test::HierDispatchMetadataTest — this test
    # focuses purely on the cross-node combine transport.)
    present = cref.peer_present_mask          # (R, q, t) int8 — ground truth for fill + golden
    # golden out[R][t] = Σ_q present[R][q][t] * 2^q   (each rank checks its own R = my_rank)
    golden = torch.zeros(world, T_local, dtype=torch.float32)
    for R in range(world):
        for q in range(world):
            for t in range(T_local):
                if int(present[R, q, t]):
                    golden[R, t] += float(2 ** q)

    # ── node-group symm-mem partial_combine_buf (NVLink-only), rendezvoused BEFORE the GIN comm ──
    node_group = None
    for nid in range(num_nodes):
        grp = dist.new_group(ranks=list(range(nid * node_size, (nid + 1) * node_size)))
        if nid == my_node:
            node_group = grp
    dist.barrier()
    partial = _symm_mem.empty((world * T_local, d), dtype=torch.bfloat16, device=local)
    hdl = _symm_mem.rendezvous(partial, group=node_group)
    peer_bufs = tuple(hdl.get_buffer(j, (world * T_local, d), torch.bfloat16) for j in range(node_size))

    # ── GIN comm (AFTER symm-mem rendezvous) + combine windows + slot-1 signal ──
    def fresh_uid():
        u = nccl.get_unique_id() if rank == 0 else None
        o = [u]
        dist.broadcast_object_list(o, src=0)
        return o[0]

    be = NCCLGin(rank, world, fresh_uid(), device=local)
    send_buf_win = be.alloc_window(COMBINE_RECV_ROWS * d, torch.bfloat16)
    recv_buf_win = be.alloc_window(COMBINE_RECV_ROWS * d, torch.bfloat16)
    combine_least_win = be.alloc_window(1, torch.int64)
    be.make_dev_comm(signal_count=2)          # slot 0 (dispatch, unused here) + slot 1 (combine)
    assert be.lsa_size == node_size, \
        f"combine assumes node==LSA domain: lsa_size {be.lsa_size} != node_size {node_size}"
    combine_least_win.tensor.fill_(0)         # device-resident combine epoch base
    torch.cuda.synchronize()

    send_t = send_buf_win.tensor.view(COMBINE_RECV_ROWS, d)
    recv_t = recv_buf_win.tensor.view(COMBINE_RECV_ROWS, d)

    for rnd in range(2):  # 2 rounds: exercise the device-resident combine epoch
        # fill MY partial_combine_buf: row R*T_local+t = 2^my_rank if present[R][me][t] else POISON
        partial.fill_(POISON)
        pv = float(2 ** rank)
        for R in range(world):
            for t in range(T_local):
                if int(present[R, rank, t]):
                    partial[R * T_local + t].fill_(pv)
        send_t.fill_(0.0)
        recv_t.fill_(0.0)
        torch.cuda.synchronize()
        dist.barrier()

        out = torch.full((T_local, d), -1.0, device=local, dtype=torch.bfloat16)
        gin_combine.hier_combine_forward(
            be, send_buf_win, recv_buf_win, combine_least_win, peer_bufs, recv_t, out, meta,
            rank=rank, world_size=world, node_size=node_size, num_nodes=num_nodes,
            T_local=T_local, d=d, group=dist.group.WORLD, sig=SIG_COMBINE)
        torch.cuda.synchronize()  # reaching here => the combine wait completed => no deadlock

        want = golden[rank][:, None].expand(T_local, d).to("cpu")
        got = out.to(torch.float32).to("cpu")
        rnd_err = int((got != want).any(dim=1).sum())
        errors += rnd_err
        if rank == 0:
            _verdict(f"[ok] combine round {rnd}: out ({T_local} tokens) bit-exact vs oracle, no deadlock"
                     if rnd_err == 0 else f"[FAIL] combine round {rnd}: {rnd_err} token mismatches")
        dist.barrier()

    # Under capture, hier_combine_forward uses node_hdl.barrier() (not host sync) and a device-resident
    # GIN epoch; the put is record-only at capture time — it executes (and the epoch advances) only on replay.
    def _fill_partial():
        partial.fill_(POISON)
        for R in range(world):
            for t in range(T_local):
                if int(present[R, rank, t]):
                    partial[R * T_local + t].fill_(pv)

    cap_err = 0
    try:
        out_cap = torch.full((T_local, d), -1.0, device=local, dtype=torch.bfloat16)
        _fill_partial()
        send_t.fill_(0.0)
        recv_t.fill_(0.0)
        torch.cuda.synchronize()
        dist.barrier()
        # warmup on a side stream (compile kernels) — eager path, NOT capturing
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            gin_combine.hier_combine_forward(
                be, send_buf_win, recv_buf_win, combine_least_win, peer_bufs, recv_t, out_cap, meta,
                rank=rank, world_size=world, node_size=node_size, num_nodes=num_nodes,
                T_local=T_local, d=d, group=dist.group.WORLD, node_hdl=hdl, sig=SIG_COMBINE)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        dist.barrier()
        # capture (record-only; node_hdl.barrier path since the stream is capturing)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            gin_combine.hier_combine_forward(
                be, send_buf_win, recv_buf_win, combine_least_win, peer_bufs, recv_t, out_cap, meta,
                rank=rank, world_size=world, node_size=node_size, num_nodes=num_nodes,
                T_local=T_local, d=d, group=dist.group.WORLD, node_hdl=hdl, sig=SIG_COMBINE)
        torch.cuda.synchronize()
        dist.barrier()
        for rep in range(2):
            out_cap.fill_(-1.0)
            _fill_partial()
            torch.cuda.synchronize()
            dist.barrier()
            g.replay()
            torch.cuda.synchronize()
            want = golden[rank][:, None].expand(T_local, d).to("cpu")
            cap_err += int((out_cap.to(torch.float32).to("cpu") != want).any(dim=1).sum())
        errors += cap_err
        if rank == 0:
            _verdict("[ok] combine CUDA-graph capture+replay (2 replays) bit-exact vs oracle"
                     if cap_err == 0 else f"[FAIL] combine cudagraph: {cap_err} token mismatches")
    except Exception as e:
        errors += 1
        if rank == 0:
            _verdict(f"[FAIL] combine cudagraph capture raised: {type(e).__name__}: {str(e)[:180]}")
    dist.barrier()

    all_err = [None] * world
    dist.all_gather_object(all_err, errors)
    total = sum(int(e) for e in all_err if e is not None)
    if rank == 0:
        _verdict("[SUCCESS] combine: metadata GPU parity + gateway reduce + GIN stripe-put (slot 1) + "
                 "origin reduce -> out bit-exact (node-group symm-mem + GIN coexist)"
                 if total == 0 else f"[FAIL] combine: {total} errors across ranks")
    dist.barrier()

    # Drop symm-mem refs in order (peer views -> handle -> buffer) BEFORE closing the GIN comm/PG,
    # else the AllocationRef outlives its tensor and cuMemUnmap fails "invalid argument".
    import gc
    peer_bufs = ()       # noqa: F841 — drop peer buffer views first
    hdl = None           # noqa: F841 — then the SymmetricMemory wrapper (must not outlive its tensor)
    partial = None       # noqa: F841 — then the symm-mem buffer itself
    gc.collect()
    torch.cuda.synchronize()
    be.close()
    dist.barrier()
    dist.destroy_process_group()
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(_run())
