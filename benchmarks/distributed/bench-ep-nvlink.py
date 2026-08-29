# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
# Benchmark suite for SonicMoE EP collectives and end-to-end forward (torchrun-launched).
# Run: torchrun --nproc_per_node=8 --standalone bench-ep-nvlink.py --phase <name...|all>; see _PHASES for names.
# ********************************************************************************

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch
import torch.distributed as dist
from torch.distributed import _symmetric_memory as _symm_mem

from sonicmoe.functional.distributed import (
    a2a_combine_triton,
    a2a_dispatch_triton,
    all_gather_multimem_triton,
    all_gather_triton,
    build_rank_dedup_a_idx,
    compute_dispatch_metadata,
    local_combine,
    rank_dedup_combine_triton,
    rank_dedup_dispatch_triton,
)

from sonicmoe.functional.metadata import general_routing_router_metadata_triton


# ============================================================================
# Timing primitive
# ============================================================================


def bench_fn(
    fn: Callable[[], None],
    *,
    warmup: int = 10,
    repeat: int = 50,
    cross_rank_avg: bool = True,
    calls_per_iter=3,
    barrier_buf: "torch.Tensor | None" = None,
) -> float:
    """Time `fn()`, return mean per-iter ms (cross-rank AVG). barrier_buf, if given, forces each iter to
    be a synced collective round — without it, pull-based dispatch/combine GB/s can exceed the NVLink ceiling."""
    for _ in range(warmup):
        if barrier_buf is not None:
            _barrier(barrier_buf)
        fn()
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    local_ms = 0
    for _ in range(calls_per_iter):
        start.record()
        for _ in range(repeat):
            if barrier_buf is not None:
                _barrier(barrier_buf)
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


# ============================================================================
# Pretty-printing
# ============================================================================


def _print_table(rank: int, title: str, header: List[str], rows: List[List[str]]) -> None:
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
    """bytes / ms -> GB/s. Used for NVLink-bound primitives."""
    return bytes_moved / ms / 1e6


def _tbps(bytes_moved: float, ms: float) -> float:
    """bytes / ms -> TB/s. Used for HBM-bound primitives (e.g. local_combine: per-rank read of y_symm
    + write of partial_combine_buf, no cross-rank traffic)."""
    return bytes_moved / ms / 1e9


# ============================================================================
# Symm-mem allocation helper: returns ONLY the buffer. Holding the rendezvous handle in a Python local
# would drop its refcount on rebind, firing cuMemUnmap MID-EXECUTION; the symm-mem cache keeps it alive instead.
# ============================================================================


def _alloc_symm(shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device):
    buf = _symm_mem.empty(*shape, dtype=dtype, device=device)
    _symm_mem.rendezvous(buf, group=dist.group.WORLD.group_name)
    return buf


def _barrier(buf: torch.Tensor) -> None:
    """GPU-side barrier on `buf`'s symm-mem group via the cached handle — fences pending writes before
    peers read via NVLink. Cache hit (cheap), avoiding the destruction-order trap in _alloc_symm."""
    _symm_mem.rendezvous(buf, group=dist.group.WORLD.group_name).barrier()


def _flush_async_errors() -> None:
    torch.cuda.synchronize()


def _post_bench_sync() -> None:
    if dist.is_initialized():
        dist.barrier()
    torch.cuda.synchronize()
    time.sleep(0.5)


def _iter_cleanup() -> None:
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    torch.cuda.empty_cache()


@dataclass
class _ModelCfg:
    name: str
    d: int
    E: int
    K: int


_MODELS = [
    _ModelCfg("mixtral", d=6144, E=8, K=2),
    _ModelCfg("olmoe", d=2048, E=64, K=8),
    _ModelCfg("d2880_e64k4", d=2880, E=64, K=4),
    _ModelCfg("d2304_e256k8", d=2304, E=256, K=8),
    _ModelCfg("e512_k10", d=2048, E=512, K=10),
    _ModelCfg("d4096_e128k8", d=4096, E=128, K=8),
    # DeepSeek-V3 MoE shape (matches DeepEP benchmark)
    _ModelCfg("dsv3", d=7168, E=256, K=8),
]


def _pick_t_local() -> int:
    """Pick a per-rank token count from the local GPU's SM capability: SM100+ (Blackwell) -> 32768,
    SM90 (Hopper) and everything else -> 16384 (conservative default)."""
    cap_major = torch.cuda.get_device_capability()[0]
    if cap_major >= 10:
        return 32768
    return 16384


_T_LOCAL = _pick_t_local()
_T_LOCALS = [_T_LOCAL]


def _t_tag(T: int) -> str:
    return f"{T // 1024}k"


def _dedupe_by_d(models):
    seen, out = set(), []
    for m in models:
        if m.d in seen:
            continue
        seen.add(m.d)
        out.append(m)
    return out


def _make_balanced_topk(
    T_local: int, K: int, E: int, my_rank: int, world_size: int, device: torch.device
) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(123 + my_rank)
    local = torch.randint(0, E, (T_local, K), generator=g, device=device, dtype=torch.int32)
    full = torch.empty((world_size, T_local, K), dtype=torch.int32, device=device)
    full[my_rank] = local
    dist.all_gather_into_tensor(full.view(-1), local.view(-1).contiguous(), group=dist.group.WORLD)
    full = full.view(world_size, T_local, K)
    return full


# ============================================================================
# Phase: dispatch metadata
# ============================================================================


@dataclass
class MetaCfg:
    name: str
    T_local: int
    K: int
    E: int


_META_CONFIGS = [
    MetaCfg(name=f"{m.name}_T{_t_tag(T)}", T_local=T, K=m.K, E=m.E) for m in _dedupe_by_d(_MODELS) for T in _T_LOCALS
]


def phase_metadata(rank: int, world_size: int, device: torch.device, args) -> None:
    rows = []
    for cfg in _META_CONFIGS:
        if cfg.E % world_size != 0:
            continue
        torch.cuda.empty_cache()
        E_local = cfg.E // world_size

        topk_idx_g = _make_balanced_topk(cfg.T_local, cfg.K, cfg.E, rank, world_size, device)

        def call():
            compute_dispatch_metadata(topk_idx_g, my_rank=rank, E_local=E_local)

        t = bench_fn(call, warmup=args.warmup, repeat=args.repeat, cross_rank_avg=False)
        _post_bench_sync()
        TK_global = world_size * cfg.T_local * cfg.K
        rows.append([cfg.name, f"{cfg.T_local}", f"{cfg.K}", f"{cfg.E}", f"{TK_global}", f"{t*1e3:.1f}"])

        del call, topk_idx_g
        _iter_cleanup()

    _print_table(
        rank,
        f"compute_dispatch_metadata (W={world_size})",
        ["name", "T_local", "K", "E", "TK_global", "µs"],
        rows,
    )


# ============================================================================
# Phase: AG (Triton vs NCCL)
# ============================================================================


@dataclass
class AGCfg:
    name: str
    T_local: int
    d: int
    dtype: torch.dtype = torch.bfloat16


_AG_CONFIGS = [
    AGCfg(name=f"{m.name}_T{_t_tag(T)}", T_local=T, d=m.d) for m in _dedupe_by_d(_MODELS) for T in _T_LOCALS
]


@dataclass
class A2ADispatchCfg:
    name: str
    T_local: int
    d: int
    K: int
    E: int
    dtype: torch.dtype = torch.bfloat16


_A2A_DISPATCH_CONFIGS = [
    A2ADispatchCfg(name=f"{m.name}_T{_t_tag(T)}", T_local=T, d=m.d, K=m.K, E=m.E) for m in _MODELS for T in _T_LOCALS
]


@dataclass
class CombineCfg:
    name: str
    T_local: int
    d: int
    K: int
    E: int
    dtype: torch.dtype = torch.bfloat16


_COMBINE_CONFIGS = [
    CombineCfg(name=f"{m.name}_T{_t_tag(T)}", T_local=T, d=m.d, K=m.K, E=m.E) for m in _MODELS for T in _T_LOCALS
]


# ============================================================================
# NCCL parity phases reuse the _AG_CONFIGS/_A2A_DISPATCH_CONFIGS/_COMBINE_CONFIGS lists above so the
# resulting tables are directly comparable with the rest of the suite.
# ============================================================================


def _make_a2a_split_sizes(peer_count_per_rank, my_rank):
    """Build (input_split_sizes, output_split_sizes) for an A2A NCCL call: in[my_rank->q] =
    peer_count_per_rank[my_rank,q], out[p->my_rank] = peer_count_per_rank[p,my_rank]."""
    pcpr = peer_count_per_rank.cpu()
    return (
        [int(x) for x in pcpr[my_rank, :].tolist()],
        [int(x) for x in pcpr[:, my_rank].tolist()],
    )


def _build_nogather_metadata(meta, expert_local_padded, TK_global, E_local, device):
    """Run general_routing_router_metadata_triton on the A2A token-id pattern to produce s_reverse_local
    + expert_freq_off (mirrors _build_consumer_metadata in ep.py)."""
    E_total = E_local + 1
    s_reverse_local = torch.empty(TK_global, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK_global, dtype=torch.int32, device=device)
    s_scatter_idx = torch.empty(TK_global, dtype=torch.int32, device=device)
    ef = torch.empty(E_total, dtype=torch.int32, device=device)
    efo = torch.empty(E_total + 1, dtype=torch.int32, device=device)
    general_routing_router_metadata_triton(
        meta["a2a_token_indices"],
        expert_local_padded,
        TK_global,
        E_total,
        ef,
        efo,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_local,
        None,
    )
    return s_reverse_local, x_gather_idx, efo


def phase_ag_dispatch(rank, world_size, device, args):
    """AG dispatch — Triton vs NCCL baseline (NVLink GB/s): each rank receives (W-1) chunks of (T_local,d)
    via NVLink peer reads; bytes/time is the effective throughput. Parity is covered by the test suite."""
    rows = []
    for cfg in _AG_CONFIGS:
        x = _alloc_symm((cfg.T_local, cfg.d), cfg.dtype, device)
        x.normal_()
        _barrier(x)
        out_ncl = torch.empty(world_size * cfg.T_local, cfg.d, dtype=cfg.dtype, device=device)
        # Pre-allocate the multimem output once (multicast-backed symm buffer)
        # so timing excludes allocation. None if the fabric lacks multicast.
        out_mm = _alloc_symm((world_size * cfg.T_local, cfg.d), cfg.dtype, device)
        has_mc = _symm_mem.rendezvous(x, group=dist.group.WORLD.group_name).multicast_ptr != 0
        _flush_async_errors()

        def tri_call():
            all_gather_triton(x, dist.group.WORLD)

        # fence=False: time the pure multicast transfer, matching unicast AG and both RS paths (no
        # per-call barrier); do_bench's trailing cuda sync still captures store completion.
        def mm_call():
            all_gather_multimem_triton(x, dist.group.WORLD, out=out_mm, fence=False)

        def ncl_call():
            dist.all_gather_into_tensor(out_ncl, x, group=dist.group.WORLD)

        t_tri = bench_fn(tri_call, warmup=args.warmup, repeat=args.repeat)
        _post_bench_sync()
        t_mm = bench_fn(mm_call, warmup=args.warmup, repeat=args.repeat) if has_mc else float("nan")
        _post_bench_sync()
        t_ncl = bench_fn(ncl_call, warmup=args.warmup, repeat=args.repeat)
        _post_bench_sync()

        elem = x.element_size()
        # Delivered bytes: (W-1) chunks/rank, shared numerator across all paths. Effective egress: 1
        # chunk — multimem's source only egresses its own chunk (the switch replicates), vs unicast's (W-1)x.
        nv_bytes = (world_size - 1) * cfg.T_local * cfg.d * elem
        egress_bytes = cfg.T_local * cfg.d * elem

        rows.append(
            [
                cfg.name,
                f"{cfg.T_local}",
                f"{cfg.d}",
                f"{t_tri*1e3:.1f}",
                f"{t_mm*1e3:.1f}",
                f"{t_ncl*1e3:.1f}",
                f"{_gbps(nv_bytes, t_tri):.0f}",
                f"{_gbps(nv_bytes, t_mm):.0f}",
                f"{_gbps(nv_bytes, t_ncl):.0f}",
                f"{_gbps(egress_bytes, t_mm):.0f}",
            ]
        )
        del x, out_ncl, out_mm, tri_call, mm_call, ncl_call
        _iter_cleanup()

    _print_table(
        rank,
        f"AG dispatch: cp.async vs multimem vs NCCL (W={world_size})",
        ["name", "T_local", "d", "cp.async µs", "multimem µs", "NCCL µs",
         "cp.async GB/s", "multimem GB/s", "NCCL GB/s", "multimem egress GB/s"],
        rows,
    )


def phase_a2a_dispatch(rank, world_size, device, args):
    rows = []
    for cfg in _A2A_DISPATCH_CONFIGS:
        if cfg.E % world_size != 0:
            continue
        E_local = cfg.E // world_size
        TK_local = cfg.T_local * cfg.K
        TK_global = world_size * TK_local

        x = _alloc_symm((cfg.T_local, cfg.d), cfg.dtype, device)
        x.normal_()
        _barrier(x)

        topk = _make_balanced_topk(cfg.T_local, cfg.K, cfg.E, rank, world_size, device)
        meta = compute_dispatch_metadata(topk, my_rank=rank, E_local=E_local)
        s_reverse_local, _, _ = _build_nogather_metadata(
            meta,
            meta["expert_local_padded"],
            TK_global,
            E_local,
            device,
        )

        recv = torch.empty((TK_global, cfg.d), dtype=cfg.dtype, device=device)

        in_splits, out_splits = _make_a2a_split_sizes(meta["peer_count_per_rank"], rank)
        send_ncl = torch.empty((sum(in_splits), cfg.d), dtype=cfg.dtype, device=device)
        recv_ncl = torch.empty((sum(out_splits), cfg.d), dtype=cfg.dtype, device=device)
        _flush_async_errors()

        def tri_call():
            a2a_dispatch_triton(
                x,
                meta["dst_rank_flat"],
                s_reverse_local,
                recv,
                K=cfg.K,
                group=dist.group.WORLD,
            )

        def ncl_call():
            dist.all_to_all_single(
                recv_ncl,
                send_ncl,
                output_split_sizes=out_splits,
                input_split_sizes=in_splits,
                group=dist.group.WORLD,
            )

        t_tri = bench_fn(tri_call, warmup=args.warmup, repeat=args.repeat)
        _post_bench_sync()
        t_ncl = bench_fn(ncl_call, warmup=args.warmup, repeat=args.repeat)
        _post_bench_sync()

        elem = x.element_size()
        pcpr = meta["peer_count_per_rank"]
        a2a_rows = int(pcpr[:, rank].sum().item() - pcpr[rank, rank].item())
        nv_bytes = a2a_rows * cfg.d * elem

        rows.append(
            [
                cfg.name,
                f"{cfg.T_local}",
                f"{cfg.d}",
                f"{cfg.K}",
                f"{cfg.E}",
                f"{t_tri*1e3:.1f}",
                f"{t_ncl*1e3:.1f}",
                f"{_gbps(nv_bytes, t_tri):.0f}",
                f"{_gbps(nv_bytes, t_ncl):.0f}",
            ]
        )
        del x, recv, send_ncl, recv_ncl, meta, s_reverse_local, tri_call, ncl_call
        _iter_cleanup()

    _print_table(
        rank,
        f"A2A dispatch: cp.async vs NCCL (W={world_size})",
        ["name", "T_local", "d", "K", "E", "cp.async µs", "NCCL µs", "cp.async NVLink GB/s", "NCCL NVLink GB/s"],
        rows,
    )


def phase_rank_dedup_dispatch(rank, world_size, device, args):
    """RANK_DEDUP_DISPATCH standalone (actual-bytes BW)."""
    rows = []
    for cfg in _A2A_DISPATCH_CONFIGS:
        if cfg.E % world_size != 0:
            continue
        E_local = cfg.E // world_size
        TK_local = cfg.T_local * cfg.K
        MAX_PAIR_COUNT = world_size * cfg.T_local

        x = _alloc_symm((cfg.T_local, cfg.d), cfg.dtype, device)
        x.normal_()
        _barrier(x)

        topk = _make_balanced_topk(cfg.T_local, cfg.K, cfg.E, rank, world_size, device)
        meta = compute_dispatch_metadata(topk, my_rank=rank, E_local=E_local)
        recv_packed = _alloc_symm((MAX_PAIR_COUNT, cfg.d), cfg.dtype, device)

        _flush_async_errors()

        def call():
            rank_dedup_dispatch_triton(
                x,
                meta["dst_rank_flat"],
                meta["pair_present_mask"],
                meta["rank_dedup_recv_pos"],
                recv_packed,
                K=cfg.K,
                group=dist.group.WORLD,
            )

        t = bench_fn(call, warmup=args.warmup, repeat=args.repeat)
        _post_bench_sync()

        elem = x.element_size()
        pc = meta["pair_count"]
        recv_rows = int(pc[:, rank].sum().item() - pc[rank, rank].item())
        nv_bytes = recv_rows * cfg.d * elem

        rows.append(
            [
                cfg.name,
                f"{cfg.T_local}",
                f"{cfg.d}",
                f"{cfg.K}",
                f"{cfg.E}",
                f"{t*1e3:.1f}",
                f"{_gbps(nv_bytes, t):.0f}",
            ]
        )
        del x, recv_packed, meta, call
        _iter_cleanup()

    _print_table(
        rank,
        f"RANK_DEDUP_DISPATCH standalone (W={world_size})",
        ["name", "T_local", "d", "K", "E", "µs", "NVLink GB/s"],
        rows,
    )


def phase_dispatch_compare(rank, world_size, device, args):
    """AG vs A2A vs RANK_DEDUP_DISPATCH head-to-head."""
    rows = []
    for cfg in _A2A_DISPATCH_CONFIGS:
        if cfg.E % world_size != 0:
            continue
        E_local = cfg.E // world_size
        TK_local = cfg.T_local * cfg.K
        TK_global = world_size * TK_local
        MAX_PAIR_COUNT = world_size * cfg.T_local

        x = _alloc_symm((cfg.T_local, cfg.d), cfg.dtype, device)
        x.normal_()
        _barrier(x)

        topk = _make_balanced_topk(cfg.T_local, cfg.K, cfg.E, rank, world_size, device)
        meta = compute_dispatch_metadata(topk, my_rank=rank, E_local=E_local)
        s_reverse_local, _, expert_freq_off = _build_nogather_metadata(
            meta,
            meta["expert_local_padded"],
            TK_global,
            E_local,
            device,
        )
        E_total = E_local + 1
        x_gather_idx_ag = torch.empty(TK_global, dtype=torch.int32, device=device)
        _s_scatter = torch.empty(TK_global, dtype=torch.int32, device=device)
        _ef = torch.empty(E_total, dtype=torch.int32, device=device)
        _efo = torch.empty(E_total + 1, dtype=torch.int32, device=device)
        _s_rev_throwaway = torch.empty(TK_global, dtype=torch.int32, device=device)
        t_global_pattern = torch.arange(TK_global, device=device, dtype=torch.int32) // cfg.K
        general_routing_router_metadata_triton(
            t_global_pattern,
            meta["expert_local_padded"],
            TK_global,
            E_total,
            _ef,
            _efo,
            x_gather_idx_ag,
            _s_scatter,
            _s_rev_throwaway,
            None,
        )

        ag_compute = torch.empty((world_size * cfg.T_local, cfg.d), dtype=cfg.dtype, device=device)
        a2a_recv = torch.empty((TK_global, cfg.d), dtype=cfg.dtype, device=device)
        dedup_packed = _alloc_symm((MAX_PAIR_COUNT, cfg.d), cfg.dtype, device)
        MAX_ROWS = cfg.T_local * world_size * min(cfg.K, E_local)
        a_idx_dedup = torch.empty(MAX_ROWS, dtype=torch.int32, device=device)
        build_rank_dedup_a_idx(
            dst_rank_flat=meta["dst_rank_flat"],
            s_reverse_local=s_reverse_local,
            rank_dedup_recv_pos=meta["rank_dedup_recv_pos"],
            my_rank=rank,
            out=a_idx_dedup,
        )

        _flush_async_errors()

        def ag_call():
            all_gather_triton(x, dist.group.WORLD, out=ag_compute)

        def a2a_call():
            a2a_dispatch_triton(
                x,
                meta["dst_rank_flat"],
                s_reverse_local,
                a2a_recv,
                K=cfg.K,
                group=dist.group.WORLD,
            )

        def dedup_call():
            rank_dedup_dispatch_triton(
                x,
                meta["dst_rank_flat"],
                meta["pair_present_mask"],
                meta["rank_dedup_recv_pos"],
                dedup_packed,
                K=cfg.K,
                group=dist.group.WORLD,
            )

        ag_call()
        a2a_call()
        dedup_call()

        t_ag = bench_fn(ag_call, warmup=args.warmup, repeat=args.repeat, barrier_buf=x)
        _post_bench_sync()
        t_a2a = bench_fn(a2a_call, warmup=args.warmup, repeat=args.repeat, barrier_buf=x)
        _post_bench_sync()
        t_dedup = bench_fn(dedup_call, warmup=args.warmup, repeat=args.repeat, barrier_buf=x)
        _post_bench_sync()

        elem = x.element_size()
        ag_bytes = (world_size - 1) * cfg.T_local * cfg.d * elem
        pcpr = meta["peer_count_per_rank"]
        pc = meta["pair_count"]
        a2a_rows = int(pcpr[:, rank].sum().item() - pcpr[rank, rank].item())
        a2a_bytes = a2a_rows * cfg.d * elem
        dedup_rows = int(pc[:, rank].sum().item() - pc[rank, rank].item())
        dedup_bytes = dedup_rows * cfg.d * elem

        rows.append(
            [
                cfg.name,
                f"{cfg.T_local}",
                f"{cfg.d}",
                f"{cfg.K}",
                f"{cfg.E}",
                f"{t_ag*1e3:.1f}",
                f"{t_a2a*1e3:.1f}",
                f"{t_dedup*1e3:.1f}",
                f"{_gbps(ag_bytes, t_ag):.0f}",
                f"{_gbps(a2a_bytes, t_a2a):.0f}",
                f"{_gbps(dedup_bytes, t_dedup):.0f}",
                f"{t_ag/t_dedup:.2f}x",
                f"{t_a2a/t_dedup:.2f}x",
                f"{dedup_bytes/ag_bytes:.2f}",
                f"{dedup_bytes/a2a_bytes:.2f}",
            ]
        )
        del x, ag_compute, a2a_recv, dedup_packed, a_idx_dedup, x_gather_idx_ag
        del _s_scatter, _ef, _efo, _s_rev_throwaway, t_global_pattern
        del meta, s_reverse_local, ag_call, a2a_call, dedup_call
        _iter_cleanup()

    _print_table(
        rank,
        f"Dispatch head-to-head: AG vs A2A vs RANK_DEDUP_DISPATCH (W={world_size})",
        [
            "name",
            "T_local",
            "d",
            "K",
            "E",
            "AG µs",
            "A2A µs",
            "DEDUP µs",
            "AG NVLink GB/s",
            "A2A NVLink GB/s",
            "DEDUP NVLink GB/s",
            "AG/DEDUP",
            "A2A/DEDUP",
            "DEDUP/AG bytes",
            "DEDUP/A2A bytes",
        ],
        rows,
    )


def phase_a2a_combine(rank, world_size, device, args):
    """a2a_combine_triton vs NCCL all_to_all_single (NVLink GB/s on cross-rank recv volume). Not apples-
    to-apples: Triton fuses the gather with the weighted top-K sum, NCCL only gathers. See phase_combine_compare for e2e."""
    rows = []
    for cfg in _COMBINE_CONFIGS:
        if cfg.E % world_size != 0:
            continue
        E_local = cfg.E // world_size
        TK_local = cfg.T_local * cfg.K
        TK_global = world_size * TK_local

        y = _alloc_symm((TK_global, cfg.d), cfg.dtype, device)
        y.normal_()
        sr = _alloc_symm((TK_global,), torch.int32, device)
        sr.copy_(torch.arange(TK_global, dtype=torch.int32, device=device))

        topk = _make_balanced_topk(cfg.T_local, cfg.K, cfg.E, rank, world_size, device)
        meta = compute_dispatch_metadata(topk, my_rank=rank, E_local=E_local)
        scores_local = torch.softmax(
            torch.randn(cfg.T_local, cfg.K, device=device, dtype=torch.float32),
            dim=-1,
        ).to(cfg.dtype)
        out_tri = torch.empty(cfg.T_local, cfg.d, dtype=cfg.dtype, device=device)
        _barrier(y)

        def tri_call():
            a2a_combine_triton(
                y,
                sr,
                meta["my_dst_rank"],
                scores_local,
                out_tri,
                K=cfg.K,
                group=dist.group.WORLD,
            )

        scores_global = torch.empty(TK_global, dtype=cfg.dtype, device=device)
        dist.all_gather_into_tensor(scores_global, scores_local.view(-1).contiguous(), group=dist.group.WORLD)
        scores_global = scores_global.view(world_size, cfg.T_local, cfg.K)

        in_splits, out_splits = _make_a2a_split_sizes(meta["peer_count_per_rank"], rank)
        dst_rank_flat = meta["dst_rank_flat"]
        my_dst = dst_rank_flat[rank * TK_local : (rank + 1) * TK_local]
        order = torch.argsort(my_dst, stable=True)
        global_slots = (rank * TK_local + order).long()
        s_rev_at = sr[global_slots].long()
        rows_at = y[s_rev_at].to(torch.float32)
        home_p = global_slots // TK_local
        home_t = (global_slots - home_p * TK_local) // cfg.K
        k_idx = global_slots - home_p * TK_local - home_t * cfg.K
        scr = scores_global[home_p, home_t, k_idx].to(torch.float32)
        send_buf = (rows_at * scr.unsqueeze(-1)).to(cfg.dtype)
        recv_ncl = torch.empty(sum(out_splits), cfg.d, dtype=cfg.dtype, device=device)

        def ncl_call():
            dist.all_to_all_single(
                recv_ncl,
                send_buf,
                output_split_sizes=out_splits,
                input_split_sizes=in_splits,
                group=dist.group.WORLD,
            )

        t_tri = bench_fn(tri_call, warmup=args.warmup, repeat=args.repeat)
        _post_bench_sync()
        t_ncl = bench_fn(ncl_call, warmup=args.warmup, repeat=args.repeat)
        _post_bench_sync()

        elem = y.element_size()
        # Per-rank received bytes across NVLink: rows from peers (excluding self) x d x elem. Mirrors
        # phase_a2a_dispatch — under balanced routing, combine recv volume == dispatch send volume.
        pcpr = meta["peer_count_per_rank"]
        a2a_rows = int(pcpr[:, rank].sum().item() - pcpr[rank, rank].item())
        nv_bytes = a2a_rows * cfg.d * elem

        rows.append(
            [
                cfg.name,
                f"{cfg.T_local}",
                f"{cfg.d}",
                f"{cfg.K}",
                f"{cfg.E}",
                f"{t_tri*1e3:.1f}",
                f"{t_ncl*1e3:.1f}",
                f"{_gbps(nv_bytes, t_tri):.0f}",
                f"{_gbps(nv_bytes, t_ncl):.0f}",
            ]
        )
        del y, sr, out_tri, send_buf, recv_ncl, scores_local, scores_global, meta, tri_call, ncl_call
        _iter_cleanup()

    _print_table(
        rank,
        f"A2A combine: cp.async vs NCCL (W={world_size})",
        ["name", "T_local", "d", "K", "E", "cp.async µs", "NCCL µs", "cp.async NVLink GB/s", "NCCL NVLink GB/s"],
        rows,
    )


def phase_combine_compare(rank, world_size, device, args):
    """A2A_combine (RT) vs RANK_DEDUP_COMBINE_TRITON head-to-head."""
    rows = []
    for cfg in _COMBINE_CONFIGS:
        if cfg.E % world_size != 0:
            continue
        E_local = cfg.E // world_size
        TK_local = cfg.T_local * cfg.K
        TK_global = world_size * TK_local

        y = _alloc_symm((TK_global, cfg.d), cfg.dtype, device)
        y.normal_()
        sr = _alloc_symm((TK_global,), torch.int32, device)
        sr.copy_(torch.arange(TK_global, dtype=torch.int32, device=device))

        topk = _make_balanced_topk(cfg.T_local, cfg.K, cfg.E, rank, world_size, device)
        meta = compute_dispatch_metadata(topk, my_rank=rank, E_local=E_local, emit_combine=True)
        # rank_dedup gather single-contributor source row (sr == arange here).
        _sk = meta["combine_single_k"].to(torch.int64)
        _tk = torch.arange(cfg.T_local, device=device, dtype=torch.int64)
        _pos = rank * TK_local + _tk[None, :] * cfg.K + _sk.clamp(min=0)
        single_row = torch.where(_sk >= 0, sr.to(torch.int64)[_pos], torch.zeros_like(_pos)).to(torch.int32)
        scores_local = torch.softmax(
            torch.randn(cfg.T_local, cfg.K, device=device, dtype=torch.float32),
            dim=-1,
        ).to(cfg.dtype)
        scores_global = torch.empty(world_size * cfg.T_local * cfg.K, dtype=cfg.dtype, device=device)
        dist.all_gather_into_tensor(scores_global, scores_local.view(-1).contiguous(), group=dist.group.WORLD)

        out_a2a = torch.empty(cfg.T_local, cfg.d, dtype=cfg.dtype, device=device)
        out_dedup = torch.empty(cfg.T_local, cfg.d, dtype=cfg.dtype, device=device)
        partial_combine_buf_dedup = _alloc_symm((world_size * cfg.T_local, cfg.d), cfg.dtype, device)
        partial_combine_buf_local = _alloc_symm((world_size * cfg.T_local, cfg.d), cfg.dtype, device)

        _barrier(y)
        _flush_async_errors()

        def a2a_call():
            a2a_combine_triton(
                y,
                sr,
                meta["my_dst_rank"],
                scores_local,
                out_a2a,
                K=cfg.K,
                group=dist.group.WORLD,
            )

        def dedup_call():
            rank_dedup_combine_triton(
                y,
                sr,
                scores_global,
                meta["peer_present_mask"],
                partial_combine_buf_dedup,
                out_dedup,
                K=cfg.K,
                T_local=cfg.T_local,
                group=dist.group.WORLD,
                mine_slot_idx=meta["mine_slot_idx"],
                mine_count=meta["mine_count"],
                combine_contrib_C=meta["combine_contrib_C"],
                combine_work_list=meta["combine_work_list_multi"],
                combine_work_count=meta["combine_work_count_multi"],
                combine_single_k=meta["combine_single_k"],
                single_row=single_row,
            )

        # Standalone local_combine producer step — the HBM-only leg of RANK_DEDUP combine
        # (rank_dedup_combine internally runs skip_empty=True; timed here next to its gather leg).
        def local_call_dedup():
            local_combine(
                y,
                sr,
                meta["dst_rank_flat"],
                scores_global,
                partial_combine_buf_local,
                cfg.K,
                cfg.T_local,
                group=dist.group.WORLD,
                skip_empty=True,
            )

        a2a_call()
        dedup_call()

        # barrier_buf=y aligns all ranks each timed iteration (same correction as dispatch_compare) —
        # a2a_combine reads peer y directly and is otherwise un-synced; local_dedup is pure-local HBM, no barrier needed.
        t_a2a = bench_fn(a2a_call, warmup=args.warmup, repeat=args.repeat, barrier_buf=y)
        _post_bench_sync()
        t_dedup = bench_fn(dedup_call, warmup=args.warmup, repeat=args.repeat, barrier_buf=y)
        _post_bench_sync()
        t_local_dedup = bench_fn(local_call_dedup, warmup=args.warmup, repeat=args.repeat)
        _post_bench_sync()

        elem = y.element_size()
        pcpr = meta["peer_count_per_rank"]
        pc = meta["pair_count"]
        a2a_rows = int(pcpr[:, rank].sum().item() - pcpr[rank, rank].item())
        a2a_bytes = a2a_rows * cfg.d * elem
        dedup_rows = int(pc[rank, :].sum().item() - pc[rank, rank].item())
        dedup_bytes = dedup_rows * cfg.d * elem
        # local_combine HBM bytes/rank, averaged across ranks to match bench_fn's cross-rank mean time:
        # reads = mine-slot count (is_mine predication kills the load otherwise); writes = rows with my_rank in their K slots.
        dst = meta["dst_rank_flat"].view(world_size, cfg.T_local, cfg.K)
        is_mine = dst == rank
        counts = torch.stack(
            [
                is_mine.sum().to(torch.int64),
                is_mine.any(dim=2).sum().to(torch.int64),
            ]
        )
        dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
        n_read_rows_avg = counts[0].item() / world_size
        n_write_dedup_avg = counts[1].item() / world_size
        hbm_bytes_dedup = int((n_read_rows_avg + n_write_dedup_avg) * cfg.d * elem)

        # A2A vs dedup pairwise allclose asserted inline; not printed. The HBM TB/s column reports the
        # dedup local_combine producer cost (skip_empty=True).
        rows.append(
            [
                cfg.name,
                f"{cfg.T_local}",
                f"{cfg.d}",
                f"{cfg.K}",
                f"{cfg.E}",
                f"{t_a2a*1e3:.1f}",
                f"{t_dedup*1e3:.1f}",
                f"{_gbps(a2a_bytes, t_a2a):.0f}",
                f"{_gbps(dedup_bytes, t_dedup):.0f}",
                f"{_tbps(hbm_bytes_dedup, t_local_dedup):.2f}",
                f"{t_a2a/t_dedup:.2f}x",
                f"{dedup_bytes/a2a_bytes:.2f}",
                f"{(t_dedup - t_local_dedup)*1e3:.1f}",  # DEDUP NVLink gather only (total - local reduce)
                f"{t_local_dedup*1e3:.1f}",              # DEDUP local_combine reduce (HBM-only) leg
            ]
        )
        del (
            y,
            sr,
            out_a2a,
            out_dedup,
            partial_combine_buf_dedup,
            partial_combine_buf_local,
        )
        del scores_local, scores_global, meta, a2a_call, dedup_call, local_call_dedup
        _iter_cleanup()

    _print_table(
        rank,
        f"Combine head-to-head: A2A(RT) vs RANK_DEDUP_COMBINE_TRITON (W={world_size})",
        [
            "name",
            "T_local",
            "d",
            "K",
            "E",
            "A2A µs",
            "DEDUP µs",
            "A2A NVLink GB/s",
            "DEDUP NVLink GB/s",
            "Dedup local HBM TB/s",
            "A2A/DEDUP",
            "DEDUP/A2A bytes",
            "DEDUP gather µs (NVLink only)",
            "DEDUP local µs",
        ],
        rows,
    )


# ============================================================================
# Phase dispatch
# ============================================================================

_PHASES = {
    # Metadata kernel timing.
    "metadata": phase_metadata,
    # Dispatch — Triton vs NCCL + RANK_DEDUP sweep + 3-way comparison.
    "ag_dispatch": phase_ag_dispatch,
    "a2a_dispatch": phase_a2a_dispatch,
    "rank_dedup_dispatch": phase_rank_dedup_dispatch,
    "dispatch_compare": phase_dispatch_compare,
    # Combine — A2A (RT) vs RANK_DEDUP head-to-head.
    "a2a_combine": phase_a2a_combine,
    "combine_compare": phase_combine_compare,
}


def run_phases(rank: int, world_size: int, device: torch.device, args) -> None:
    if rank == 0:
        print(f"\nSonicMoE EP benchmark suite (W={world_size}, " f"warmup={args.warmup}, repeat={args.repeat})")
    selected = list(_PHASES.keys()) if args.phase == ["all"] else args.phase
    for p in selected:
        if p not in _PHASES:
            if rank == 0:
                print(f"  [skip] unknown phase: {p}")
            continue
        if rank == 0:
            print(f"\n[phase] {p}")
        _PHASES[p](rank, world_size, device, args)
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        torch.cuda.empty_cache()


def _under_torchrun() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--phase", nargs="+", default=["all"], help="phases to run: " + ", ".join(_PHASES.keys()) + ", all"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument(
        "--t-local", type=int, default=None,
        help="override per-rank token count (default: auto from GPU cap, currently %d)" % _T_LOCAL,
    )
    parser.add_argument(
        "--filter-model", nargs="+", default=None,
        help="run only these model names (default: all). Available: " + ", ".join(m.name for m in _MODELS),
    )
    parser.add_argument(
        "--reps", type=int, default=1,
        help="re-run the selected phases this many times in ONE allocation "
        "(run-to-run spread on the same rack).",
    )
    args = parser.parse_args()

    # Rebuild config lists if --t-local or --filter-model is given.
    if args.t_local is not None or args.filter_model is not None:
        global _AG_CONFIGS, _A2A_DISPATCH_CONFIGS, _COMBINE_CONFIGS, _META_CONFIGS
        t_locals = [args.t_local] if args.t_local is not None else _T_LOCALS
        models = _MODELS
        if args.filter_model is not None:
            models = [m for m in _MODELS if m.name in args.filter_model]
            if not models:
                print(
                    f"ERROR: no models matched --filter-model {args.filter_model}. "
                    f"Available: {[m.name for m in _MODELS]}",
                    file=sys.stderr,
                )
                return 2
        _META_CONFIGS = [
            MetaCfg(name=f"{m.name}_T{_t_tag(T)}", T_local=T, K=m.K, E=m.E)
            for m in _dedupe_by_d(models) for T in t_locals
        ]
        _AG_CONFIGS = [
            AGCfg(name=f"{m.name}_T{_t_tag(T)}", T_local=T, d=m.d)
            for m in _dedupe_by_d(models) for T in t_locals
        ]
        _A2A_DISPATCH_CONFIGS = [
            A2ADispatchCfg(name=f"{m.name}_T{_t_tag(T)}", T_local=T, d=m.d, K=m.K, E=m.E)
            for m in models for T in t_locals
        ]
        _COMBINE_CONFIGS = [
            CombineCfg(name=f"{m.name}_T{_t_tag(T)}", T_local=T, d=m.d, K=m.K, E=m.E)
            for m in models for T in t_locals
        ]

    if not _under_torchrun():
        print(
            "ERROR: this benchmark must be launched with torchrun, e.g.:\n"
            "  torchrun --nproc_per_node=8 --standalone -m sonicmoe.benchmarks.ep_bench",
            file=sys.stderr,
        )
        return 2

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if world_size < 2:
        if rank == 0:
            print("EP benchmark requires world_size >= 2 GPUs", file=sys.stderr)
        return 1
    if local_rank >= torch.cuda.device_count():
        print(
            f"[r{rank}] ERROR: LOCAL_RANK={local_rank} but only " f"{torch.cuda.device_count()} CUDA devices visible",
            file=sys.stderr,
        )
        return 2

    torch.cuda.set_device(local_rank)
    torch.manual_seed(rank)
    # NCCL timeout bumped 10min->60min: Triton autotune over the combine/dispatch sweeps (51-78 configs)
    # can take >10min on a cold cache, tripping the watchdog on a straggler rank still compiling.
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{local_rank}"),
        timeout=datetime.timedelta(minutes=60),
    )
    device = torch.device(f"cuda:{local_rank}")

    try:
        for _rep in range(max(1, args.reps)):
            if args.reps > 1 and rank == 0:
                print(f"\n########## REP {_rep + 1}/{args.reps} ##########", flush=True)
            run_phases(rank, world_size, device, args)
    finally:
        try:
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            pass

    # ====================================================================
    # CRITICAL: hard-exit BEFORE returning — falling off the end triggers Python shutdown while symm-mem
    # state is still tracked, and ~CUDASymmetricMemory -> cuMemUnmap from a C++ destructor mid-teardown -> SIGABRT.
    # ====================================================================
    os._exit(0)


if __name__ == "__main__":
    # main() hard-exits via os._exit on the success path; only the pre-symm-mem early-skips (returns
    # 1/2 before any allocation) fall through here, where Python's normal exit is safe.
    sys.exit(main())
