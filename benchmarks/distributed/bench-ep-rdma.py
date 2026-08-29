# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
# Throughput-only benchmark (no correctness checks) for hierarchical inter-node EP dispatch+combine
# over NCCL-GIN. Run under srun on >=2 nodes; NCCL_IB_HCA must name the fast IB NICs or NCCL grabs the
# slow RoCE/mgmt ports -> QP timeouts. Example: T_LOCAL=512 K=8 E_LOCAL=8 D=2048
# NCCL_IB_HCA=<your fast IB NICs> NCCL_P2P_DISABLE=1 srun -N2 --ntasks-per-node=4 <python> bench-ep-rdma.py
# ********************************************************************************

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist


# ============================================================================
# MoE model-config sweep (mirrors bench-ep-nvlink.py _MODELS); replicated not imported since a
# hyphenated filename isn't importable as a module. MODEL_SWEEP=1 loops these, deriving E_local=E//world.
# ============================================================================
MODEL_CONFIGS = [
    ("mixtral", 6144, 8, 2),
    ("olmoe", 2048, 64, 8),
    ("d2880_e64k4", 2880, 64, 4),
    ("d2304_e256k8", 2304, 256, 8),
    ("e512_k10", 2048, 512, 10),
    ("d4096_e128k8", 4096, 128, 8),
    ("dsv3", 7168, 256, 8),
]


# ============================================================================
# SLURM / torch.distributed bootstrap (mirrors hier_ep_combine_gin_test._world())
# ============================================================================


def _world():
    return (
        int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0))),
        int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1))),
        int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0))),
    )


def _init_dist(rank: int, world: int, local: int) -> int:
    """Mirror the test's MASTER_ADDR/PORT bootstrap + cuda:nccl,cpu:gloo init."""
    if "MASTER_ADDR" not in os.environ:
        import subprocess

        nl = os.environ.get("SLURM_NODELIST", "127.0.0.1")
        os.environ["MASTER_ADDR"] = (
            subprocess.check_output(["scontrol", "show", "hostnames", nl]).decode().split()[0]
            if ("[" in nl or "," in nl)
            else nl
        )
    os.environ.setdefault("MASTER_PORT", "29565")
    if local >= torch.cuda.device_count():  # SLURM gpu-bind: each task sees only its 1 GPU as device 0
        local = 0
    torch.cuda.set_device(local)
    # NCCL backend for the node-group symm-mem rendezvous; gloo for object bcast (unique_id).
    dist.init_process_group(backend="cuda:nccl,cpu:gloo", rank=rank, world_size=world)
    return local


# ============================================================================
# Deterministic global routing (mirrors hier_ep_dispatch_gin_e2e_test._make_global_routing);
# biased remote (~half the picks) so the cross-node GIN path actually fires.
# ============================================================================


def _make_global_routing(W, num_nodes, node_size, T_local, K, E_local, seed):
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


# ============================================================================
# Timing primitive (mirrors bench-ep-nvlink.py bench_fn). No barrier_buf needed here — the HIER
# dispatch/combine forwards do their OWN internal barriers, so each round is already honest.
# ============================================================================


def bench_fn(fn, *, warmup: int = 10, repeat: int = 50, calls_per_iter: int = 3, cross_rank_avg: bool = True) -> float:
    """Time ``fn()`` and return mean per-iter milliseconds (cross-rank AVG)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    local_ms = 0.0
    for _ in range(calls_per_iter):
        start.record()
        for _ in range(repeat):
            fn()
        end.record()
        torch.cuda.synchronize()
        local_ms += start.elapsed_time(end) / repeat
    local_ms /= calls_per_iter
    if cross_rank_avg and dist.is_initialized():
        t = torch.tensor([local_ms], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t.item() / dist.get_world_size()
    return local_ms


def capture_graph(call):
    """Capture ``call`` into a replayable CUDA graph: warm up on a side stream (compiles kernels) so
    capture records only steady-state work; each replay is a self-synced round (no barrier_buf needed)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        call()  # eager warmup (compile) — NOT capturing
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()  # record-only; capturable node_hdl.barrier() path
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    return g


# ============================================================================
# Pretty-printing (mirrors bench-ep-nvlink.py _print_table)
# ============================================================================


def _print_table(rank: int, title: str, header, rows) -> None:
    if rank != 0:
        return
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    bar = "  ".join("-" * w for w in widths)
    print(f"\n=== {title} ===")
    print(fmt.format(*header))
    print(bar)
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))


def _gbps(bytes_moved: float, ms: float) -> float:
    """bytes / ms -> GB/s (1e9). ms==0 guard returns 0."""
    return bytes_moved / (ms / 1e3) / 1e9 if ms > 0 else 0.0


# ============================================================================
# Main
# ============================================================================


def _run() -> int:
    rank, world, local = _world()
    if world < 2:
        print(f"[rank {rank}] HIER inter-node bench needs >= 2 ranks (got world={world})", flush=True)
        return 1

    # Hard-import the optional GIN stack up front — failing here with a clear message beats a
    # deep ImportError surfacing later inside _alloc_hier_workspace.
    try:
        import nccl.core as _nccl  # noqa: F401
        from sonicmoe.functional.distributed import nccl_gin as _nccl_gin  # noqa: F401
    except Exception as e:  # pragma: no cover - environment gate
        print(
            f"[rank {rank}] CuTeDSL / nccl4py GIN stack unavailable ({type(e).__name__}: {e}); "
            f"this benchmark requires the inter-rack GIN env (see the interrack-gin-ep skill).",
            flush=True,
        )
        return 1

    from sonicmoe.distributed_utils import SymmMemManager, DispatchMode
    from sonicmoe.functional.distributed.metadata import compute_dispatch_metadata

    local = _init_dist(rank, world, local)

    # node_size (NVLink/LSA domain) + num_nodes SELF-DERIVE from the GIN backend's lsa_size inside
    # _bench_one_config (no NODE_SIZE knob); these are pre-derive placeholders only for the banner below.
    node_size = min(4, world)
    num_nodes = world // node_size

    # T_local is shared across modes (per-rank token count). bf16 model dtype.
    T_local = int(os.environ.get("T_LOCAL", "512"))
    dtype = torch.bfloat16

    mgr = SymmMemManager()

    # MODEL_SWEEP=1 loops MODEL_CONFIGS (deriving d/K/E_local per config), optionally filtered by
    # MODEL_FILTER; single-config mode (default) reads D/K/E_LOCAL from env instead.
    if os.environ.get("MODEL_SWEEP", "0") == "1":
        rc = _run_model_sweep(
            rank, world, local, num_nodes, node_size, T_local, dtype,
            mgr, DispatchMode, compute_dispatch_metadata,
        )
        dist.barrier()
        dist.destroy_process_group()
        return rc

    # ── single-config mode: realistic EP shape from env. ──
    K = int(os.environ.get("K", "8"))
    E_local = int(os.environ.get("E_LOCAL", "8"))
    d = int(os.environ.get("D", "2048"))

    if rank == 0:
        print(
            f"[bench-ep-rdma] W={world} num_nodes={num_nodes} node_size={node_size} "
            f"T_local={T_local} K={K} E_local={E_local} d={d} dtype={dtype}",
            flush=True,
        )

    res = _run_one_config(
        rank, world, local, num_nodes, node_size, T_local, d, K, E_local, dtype,
        mgr, DispatchMode, compute_dispatch_metadata, print_detail=True,
    )

    dispatch_ms, dispatch_gbs = res["dispatch_ms"], res["dispatch_gbs"]
    combine_ms, combine_gbs = res["combine_ms"], res["combine_gbs"]
    cross_node_puts_hier = res["cross_node_puts_hier"]
    cross_node_puts_perrank = res["cross_node_puts_perrank"]
    dedup_ratio = res["dedup_ratio"]
    if rank == 0:
        print(
            f"\n[bench-ep-rdma] dispatch={dispatch_ms:.3f}ms ({dispatch_gbs:.1f} GB/s)  "
            f"combine={combine_ms:.3f}ms ({combine_gbs:.1f} GB/s)  [cudagraph-replay per-iter cost]  "
            f"node-dedup savings={dedup_ratio:.2f}x "
            f"(hier {cross_node_puts_hier} vs per-rank {cross_node_puts_perrank} cross-node puts)",
            flush=True,
        )

    dist.barrier()
    dist.destroy_process_group()
    return 0


def _run_one_config(
    rank, world, local, num_nodes, node_size, T_local, d, K, E_local, dtype,
    mgr, DispatchMode, compute_dispatch_metadata, *, print_detail: bool,
) -> dict:
    """Build the HIER workspace + metadata for ONE (d, K, E_local) config, time cudagraph-captured
    dispatch + combine, and return a result dict; print_detail toggles the per-config detail tables."""
    # ── build the validated HIER workspace FIRST — it SELF-DERIVES node_size/num_nodes from the NVLink
    #    (LSA) domain (no NODE_SIZE knob); routing + metadata below use the derived partition. ──
    ws = mgr._get_or_alloc(
        T_local, d, K, E_local, dtype, DispatchMode.HIER_NODE_DEDUP_DISPATCH_GIN,
    )
    node_size = ws.node_size
    num_nodes = ws.num_nodes

    # ── routing + metadata (emit_hier so node_present_mask / dst_slot / dst_recv_count exist) ──
    topk = _make_global_routing(world, num_nodes, node_size, T_local, K, E_local, seed=1234)
    topk_dev = topk.to(torch.int32).to(local)
    meta = compute_dispatch_metadata(
        topk_dev, rank, E_local, emit_combine=False, emit_hier=True, node_size=node_size
    )

    TK_local = T_local * K
    my_base = rank * TK_local
    itemsize = torch.tensor([], dtype=dtype).element_size()  # bf16 -> 2

    # Byte accounting: cross_node_puts_hier = present (token,remote-node) slots in my stripe (one put
    # each); cross_node_puts_perrank = the per-rank-dedup baseline's equivalent (token,remote-rank) count.
    my = slice(my_base, my_base + TK_local)
    node_present_me = meta["node_present_mask"][my].to(torch.int64)
    pair_present_me = meta["pair_present_mask"][my].to(torch.int64)
    is_local_me = meta["is_local_slot"][my].to(torch.int64)

    cross_node_puts_hier = int(node_present_me.sum().item())
    # per-rank-dedup cross-node puts = canonical (token,rank) slots whose dst is on a remote node
    cross_node_puts_perrank = int((pair_present_me * (1 - is_local_me)).sum().item())
    dedup_ratio = (cross_node_puts_perrank / cross_node_puts_hier) if cross_node_puts_hier > 0 else 0.0

    dispatch_cross_node_bytes = cross_node_puts_hier * d * itemsize

    # Combine moves dense (T_local, d) stripes: ONE GIN stripe-put per remote node this gateway
    # serves (expected_count_combine[rank] inbound, symmetric send count); each is T_local*d*itemsize bytes.
    combine_puts = int(meta["expected_count_combine"][rank].item())
    combine_cross_node_bytes = combine_puts * T_local * d * itemsize

    # 1) DISPATCH timing, CUDA-graph-CAPTURED: the eager forward's 2x host sync + 2x barrier per call
    # would dominate the timing, so we capture (device-side node_hdl.barrier(), no host sync) and replay.
    ws.x_gin_fwd.tensor.view(T_local, d).normal_()

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "distributed"))
        from hier_ep_reference import compute_hier_dispatch_reference

        ref = compute_hier_dispatch_reference(topk, num_nodes, node_size, E_local)
        recv_rows = int(ref.recv_rows_per_rank[rank])
    except Exception:
        recv_rows = world * T_local
    recv_packed = torch.empty((max(recv_rows, 1), d), dtype=dtype, device=local)

    def _dispatch():
        ws.gin_dispatch_fn(
            ws.gin_backend,
            ws.x_gin_fwd,
            ws.dst_node_buf_fwd,
            ws.gin_least_fwd,
            ws.x_lsa_fwd,
            ws.dst_node_buf_lsa_fwd,
            recv_packed,
            meta,
            rank=rank,
            world_size=world,
            node_size=node_size,
            T_local=T_local,
            K=K,
            H=d,
            group=dist.group.WORLD,
            node_hdl=ws.x_hdl,  # capturable node-group symm-mem barrier under capture
            staging_win=ws.staging_fwd,  # coalesced-put source (coalesce=True default)
        )

    g_disp = capture_graph(_dispatch)
    dispatch_ms = bench_fn(lambda: g_disp.replay())

    # 2) COMBINE timing, same CUDA-graph-CAPTURED rationale as dispatch (dummy data, timed via replay).
    # node_hdl is the SAME partial_combine_hdl symm-mem the partials live in, so its barrier fences them.
    ws.partial_combine_buf.normal_()
    combine_recv_t = ws.combine_recv_fwd.tensor.view(-1, d)
    combine_out = torch.empty((T_local, d), dtype=dtype, device=local)

    def _combine():
        ws.gin_combine_fn(
            ws.gin_backend,
            ws.combine_send_fwd,
            ws.combine_recv_fwd,
            ws.combine_least,
            ws.partial_combine_peer_bufs,
            combine_recv_t,
            combine_out,
            meta,
            rank=rank,
            world_size=world,
            node_size=node_size,
            num_nodes=num_nodes,
            T_local=T_local,
            d=d,
            group=dist.group.WORLD,
            node_hdl=ws.partial_combine_hdl,  # capturable barrier under capture
        )

    g_comb = capture_graph(_combine)
    combine_ms = bench_fn(lambda: g_comb.replay())

    # 3) ISOLATED PUT timing: the cross-node RDMA wire alone (just the GIN put+wait kernel, no NVLink
    # gather/reduce or barriers) — wire GB/s >> end-to-end GB/s shows how much cost is transport vs compute.
    from sonicmoe.functional.distributed.nccl_gin import dispatch as _gd, combine as _gc

    ws.staging_fwd.tensor.normal_()  # dummy staged source (timing only) — coalesced put wire

    def _disp_put():
        _gd.launch_dispatch_coalesced(ws.gin_backend, ws.staging_fwd, ws.dst_node_buf_fwd,
                                      ws.gin_least_fwd, meta, rank=rank, node_size=node_size,
                                      num_nodes=num_nodes, T_local=T_local, d=d, sig=0)

    g_dp = capture_graph(_disp_put)
    dispatch_put_ms = bench_fn(lambda: g_dp.replay())

    ws.combine_send_fwd.tensor.normal_()  # dummy staged send_buf (timing only — content irrelevant)

    def _comb_put():
        _gc.launch_combine_put(ws.gin_backend, ws.combine_send_fwd, ws.combine_recv_fwd, ws.combine_least, meta,
                               rank=rank, node_size=node_size, num_nodes=num_nodes, T_local=T_local, d=d, sig=1)

    g_cp = capture_graph(_comb_put)
    combine_put_ms = bench_fn(lambda: g_cp.replay())

    # Optional (PROFILE_LOCAL_COMBINE=1): time the local_combine producer. s_rev is filled with valid
    # scattered indices (content irrelevant) so the byte volume + scatter pattern stay representative.
    local_combine_ms = 0.0
    routed_slots = 0
    if os.environ.get("PROFILE_LOCAL_COMBINE") == "1":
        from sonicmoe.functional.distributed.ep_combine import local_combine
        routed_slots = int((meta["dst_rank_flat"] == rank).sum().item())
        ws.s_rev_symm.random_(0, max(int(ws.y_symm.shape[0]), 1))
        ws.y_symm.normal_()
        def _local_combine():
            local_combine(ws.y_symm, ws.s_rev_symm, meta["dst_rank_flat"], None,
                          ws.partial_combine_buf, K, T_local, dist.group.WORLD, skip_empty=False)
        g_lc = capture_graph(_local_combine)
        local_combine_ms = bench_fn(lambda: g_lc.replay())
        if rank == 0:
            lc_bytes = (routed_slots + world * T_local) * d * 2  # reads routed rows + writes all partials
            print(f"[local_combine] {local_combine_ms:.4f} ms  ({_gbps(lc_bytes, local_combine_ms):.1f} GB/s "
                  f"on {lc_bytes:,} B = {routed_slots} routed + {world*T_local} written rows)", flush=True)

    # ========================================================================
    # Report (rank 0). Byte accounting (puts/ratio) is taken from rank 0's stripe;
    # the ms / GB/s are cross-rank AVG'd.
    # ========================================================================
    dispatch_gbs = _gbps(dispatch_cross_node_bytes, dispatch_ms)
    combine_gbs = _gbps(combine_cross_node_bytes, combine_ms)
    dispatch_put_gbs = _gbps(dispatch_cross_node_bytes, dispatch_put_ms)
    combine_put_gbs = _gbps(combine_cross_node_bytes, combine_put_ms)

    if print_detail:
        _print_table(
            rank,
            "HIER inter-node EP — geometry",
            ["W", "num_nodes", "node_size", "T_local", "K", "E_local", "d"],
            [[world, num_nodes, node_size, T_local, K, E_local, d]],
        )
        _print_table(
            rank,
            "HIER inter-node EP — throughput (CUDA-graph replay; cross-rank AVG; GB/s on cross-node RDMA bytes)",
            ["phase", "mean ms", "GB/s", "cross-node bytes/rank"],
            [
                ["dispatch e2e", f"{dispatch_ms:.3f}", f"{dispatch_gbs:.1f}", f"{dispatch_cross_node_bytes:,}"],
                ["dispatch wire", f"{dispatch_put_ms:.3f}", f"{dispatch_put_gbs:.1f}", f"{dispatch_cross_node_bytes:,}"],
                ["combine e2e", f"{combine_ms:.3f}", f"{combine_gbs:.1f}", f"{combine_cross_node_bytes:,}"],
                ["combine wire", f"{combine_put_ms:.3f}", f"{combine_put_gbs:.1f}", f"{combine_cross_node_bytes:,}"],
            ],
        )
        if rank == 0:
            print(
                "  note: mean ms is the CAPTURED (cudagraph-replay) per-iter cost — the eager forward's host "
                "syncs/barriers would otherwise dominate and hide RDMA throughput.",
                flush=True,
            )
        _print_table(
            rank,
            "HIER node-dedup traffic reduction (rank 0 stripe; puts = cross-node RDMA rows)",
            ["cross_node_puts_hier", "cross_node_puts_perrank", "reduction ratio (perrank/hier)"],
            [[cross_node_puts_hier, cross_node_puts_perrank, f"{dedup_ratio:.3f}x"]],
        )

    return {
        "d": d,
        "K": K,
        "E_local": E_local,
        "dispatch_ms": dispatch_ms,
        "dispatch_gbs": dispatch_gbs,
        "dispatch_put_ms": dispatch_put_ms,
        "dispatch_put_gbs": dispatch_put_gbs,
        "combine_ms": combine_ms,
        "combine_gbs": combine_gbs,
        "combine_put_ms": combine_put_ms,
        "combine_put_gbs": combine_put_gbs,
        "cross_node_puts_hier": cross_node_puts_hier,
        "cross_node_puts_perrank": cross_node_puts_perrank,
        "dedup_ratio": dedup_ratio,
    }


def _run_model_sweep(
    rank, world, local, num_nodes, node_size, T_local, dtype,
    mgr, DispatchMode, compute_dispatch_metadata,
) -> int:
    """Loop MODEL_CONFIGS (filtered by MODEL_FILTER), time each via _run_one_config, and release the
    workspace between configs (mgr.clear()) so the 7-config sweep doesn't accumulate symm-mem."""
    filt_raw = os.environ.get("MODEL_FILTER", "").strip()
    filt = {n.strip() for n in filt_raw.split(",") if n.strip()} if filt_raw else None

    if rank == 0:
        print(
            f"[bench-ep-rdma MODEL_SWEEP] W={world} num_nodes={num_nodes} node_size={node_size} "
            f"T_local={T_local} dtype={dtype}"
            + (f"  MODEL_FILTER={sorted(filt)}" if filt else "  (all configs)"),
            flush=True,
        )

    rows = []
    for name, d, E, K in MODEL_CONFIGS:
        if filt is not None and name not in filt:
            continue
        if E % world != 0:
            if rank == 0:
                print(f"[skip] {name}: E={E} not divisible by W={world}", flush=True)
            continue
        E_local = E // world
        if rank == 0:
            print(f"\n[model-sweep] {name}: d={d} E={E} K={K} E_local={E_local}", flush=True)
        try:
            res = _run_one_config(
                rank, world, local, num_nodes, node_size, T_local, d, K, E_local, dtype,
                mgr, DispatchMode, compute_dispatch_metadata, print_detail=False,
            )
        except Exception as e:  # one config failing must not abort the sweep
            if rank == 0:
                print(f"[model-sweep] {name}: EXC {type(e).__name__}: {str(e)[:200]}", flush=True)
            mgr.clear()
            torch.cuda.empty_cache()
            if dist.is_initialized():
                dist.barrier()
            continue
        rows.append([
            name, d, E, K, res["E_local"],
            f"{res['dispatch_ms']:.3f}", f"{res['dispatch_gbs']:.1f}", f"{res['dispatch_put_gbs']:.1f}",
            f"{res['combine_ms']:.3f}", f"{res['combine_gbs']:.1f}", f"{res['combine_put_gbs']:.1f}",
            f"{res['dedup_ratio']:.2f}x",
        ])
        # Release this config's workspace before the next one so the 7-config
        # sweep does not accumulate symm-mem (dsv3 d=7168 is large).
        mgr.clear()
        torch.cuda.empty_cache()

    _print_table(
        rank,
        "HIER inter-node EP — MoE model-config sweep "
        "(CUDA-graph replay; cross-rank AVG; GB/s on cross-node RDMA bytes)",
        ["model", "d", "E", "K", "E_local", "disp ms", "disp e2e", "disp wire", "comb ms", "comb e2e", "comb wire", "dedup"],
        rows,
    )
    return 0


if __name__ == "__main__":
    sys.exit(_run())
