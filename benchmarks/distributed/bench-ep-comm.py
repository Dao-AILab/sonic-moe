# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
#
# Benchmark suite for SonicMoE EP collectives and end-to-end forward.
#
# Run (torchrun-launched):
#
#   # Run every registered phase (default):
#   torchrun --nproc_per_node=4 --standalone benchmarks/distributed/bench-ep-comm.py
#
#   # Run one phase by name:
#   torchrun --nproc_per_node=8 --standalone benchmarks/distributed/bench-ep-comm.py \
#            --phase combine_compare
#
#   # Run several phases in one launch (space-separated):
#   torchrun --nproc_per_node=8 --standalone benchmarks/distributed/bench-ep-comm.py \
#            --phase ag_dispatch a2a_dispatch rank_dedup_dispatch
#
# Available phase names (registered in `_PHASES`, run individually
# via --phase <name> or together via --phase all):
#   metadata               — metadata kernel timing
#   ag_dispatch            — AG triton dispatch standalone sweep
#                            (NVLink GB/s)
#   a2a_dispatch           — A2A triton dispatch standalone sweep
#                            (NVLink GB/s)
#   rank_dedup_dispatch    — RANK_DEDUP_DISPATCH standalone sweep
#                            (NVLink GB/s)
#   dispatch_compare       — AG vs A2A vs RANK_DEDUP_DISPATCH head-to-head
#                            (NVLink GB/s for all three)
#   reduce_scatter         — RS triton standalone sweep (NVLink GB/s;
#                            excludes the local_combine producer step
#                            which is HBM-only and not a network primitive)
#   a2a_combine            — A2A_combine triton vs NCCL all_to_all_single
#                            (correctness only)
#   rank_dedup_combine     — RANK_DEDUP_COMBINE_TRITON on TWO profiles:
#                            local_combine producer (HBM TB/s) and
#                            cross-rank gather (NVLink GB/s)
#   combine_compare        — A2A_combine vs RS combine vs RANK_DEDUP_COMBINE_TRITON
#                            head-to-head; reports NVLink GB/s for all
#                            three plus the standalone local_combine
#                            HBM TB/s (the producer leg shared by RS
#                            and the dedup combine)
#
# torchrun sets RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT in
# each child; we just read them. --standalone picks a free master port; use
# --local-ranks-filter 0 to dedupe console output (rank 0 already does the
# printing, but Triton/NCCL warnings on other ranks can still be noisy).
#
# Symm-mem rendezvous handles are NOT held as Python locals across the
# bench loops — doing so reorders ~CUDASymmetricMemory to fire when the
# handle local is rebound on the next iteration, mid-execution, racing
# with in-flight CUDA work on the buffer's peer mappings and triggering
# `cuMemUnmap → CUDA_ERROR_INVALID_VALUE` from inside ~AllocationRef.
# Producer-→peer-read fences use the transient `_barrier(buf)` helper,
# which fetches the cached handle from PyTorch's symm-mem cache, calls
# `.barrier()`, and drops the local ref immediately — the cache keeps
# the actual handle alive bound to the buffer's lifetime. Coarse cross-
# rank syncs (in `bench_fn` between warmup and timing, in main()
# teardown) keep `dist.barrier()`.
# ********************************************************************************

from __future__ import annotations

import argparse
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
    all_gather_triton,
    build_rank_dedup_a_idx,
    compute_dispatch_metadata,
    local_combine,
    rank_dedup_combine_triton,
    rank_dedup_dispatch_triton,
    reduce_scatter_triton,
)

# Module-internal communication-only step — used by the bench's
# RANK_DEDUP_COMBINE_TRITON network-bandwidth phase to time the cross-rank
# gather kernel without the local_combine producer.
from sonicmoe.functional.distributed.ep_combine import _rank_dedup_combine_communication_triton
from sonicmoe.functional.metadata import general_routing_router_metadata_triton


# ============================================================================
# Timing primitive
# ============================================================================


def bench_fn(
    fn: Callable[[], None], *, warmup: int = 10, repeat: int = 50, cross_rank_avg: bool = True, calls_per_iter=3
) -> float:
    """Time `fn()` and return mean per-iter milliseconds.

    Cross-rank reduction defaults to AVG (more stable than MAX, which
    is dominated by stragglers on shared hardware).
    """
    for _ in range(warmup):
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
    """bytes / ms -> TB/s. Used for HBM-bound primitives (e.g.
    ``local_combine``: per-rank read of y_symm + write of partial_acc_buf, no
    cross-rank traffic)."""
    return bytes_moved / ms / 1e9


# ============================================================================
# Symm-mem allocation helper
# ----------------------------------------------------------------------------
# Returns ONLY the buffer. The rendezvous handle is intentionally NOT
# returned — holding it in a Python local across loop iterations causes
# the handle's refcount to drop when the local is rebound on the next
# iteration, firing ~CUDASymmetricMemory → ~AllocationRef → cuMemUnmap
# MID-EXECUTION while CUDA work on the buffer's peer mappings is still
# in flight, which crashes with CUDA_ERROR_INVALID_VALUE. PyTorch's
# symm-mem cache keeps the actual handle alive bound to the buffer's
# lifetime, so subsequent rendezvous(buf, ...) calls (inside wrappers
# or via `_barrier(buf)`) hit the cache without a round-trip.
# ============================================================================


def _alloc_symm(shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device):
    buf = _symm_mem.empty(*shape, dtype=dtype, device=device)
    _symm_mem.rendezvous(buf, group=dist.group.WORLD.group_name)
    return buf


def _barrier(buf: torch.Tensor) -> None:
    """GPU-side barrier on `buf`'s symm-mem group via the cached handle.

    Fences this rank's pending writes to `buf` before peers read it via
    NVLink. The local handle ref drops as soon as `.barrier()` returns;
    the cache keeps the underlying SymmetricMemory alive — so this is
    cheap (cache hit) and avoids the destruction-order trap described
    above `_alloc_symm`."""
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
]


def _pick_t_local() -> int:
    """Pick a per-rank token count based on the local GPU's SM capability.

    SM90 (Hopper: H100/H200) → 16384.
    SM100+ (Blackwell: B100/B200/SM110+) → 32768.
    Other / unknown → 16384 conservatively.

    Bench-comm uses one T_local for every config so the resulting
    table is directly comparable across phases; the value is chosen
    per-architecture so the reported numbers reflect a workload size
    that's representative for that class of GPU (Hopper has roughly
    half the per-chip resources of Blackwell, so a smaller T_local
    keeps total HBM footprint comparable).
    """
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
        rows.append(
            [
                cfg.name,
                f"{cfg.T_local}",
                f"{cfg.K}",
                f"{cfg.E}",
                f"{TK_global}",
                f"{t*1e3:.1f}",
            ]
        )

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
# NCCL parity phases reuse the _AG_CONFIGS / _A2A_DISPATCH_CONFIGS /
# _COMBINE_CONFIGS lists already defined above so the resulting tables
# are directly comparable with the rest of the suite.
# ============================================================================


def _make_a2a_split_sizes(peer_count_per_rank, my_rank):
    """Build (input_split_sizes, output_split_sizes) for an A2A NCCL call:
    in[my_rank → q] = peer_count_per_rank[my_rank, q]
    out[p → my_rank] = peer_count_per_rank[p, my_rank]
    """
    pcpr = peer_count_per_rank.cpu()
    return (
        [int(x) for x in pcpr[my_rank, :].tolist()],
        [int(x) for x in pcpr[:, my_rank].tolist()],
    )


def _build_nogather_metadata(meta, expert_local_padded, TK_global, E_local, device):
    """Run general_routing_router_metadata_triton on the A2A token-id pattern
    to produce s_reverse_local + expert_freq_off (mirrors what
    _build_consumer_metadata does in ep.py)."""
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
    """AG triton dispatch standalone bench (NVLink GB/s).

    Each rank receives (W-1) chunks of (T_local, d) via NVLink peer
    reads; the bytes / time ratio is the effective NVLink throughput.
    Parity vs NCCL all_gather is covered by the test suite.
    """
    rows = []
    for cfg in _AG_CONFIGS:
        x = _alloc_symm((cfg.T_local, cfg.d), cfg.dtype, device)
        x.normal_()
        _barrier(x)
        _flush_async_errors()

        def call():
            all_gather_triton(x, dist.group.WORLD)

        t = bench_fn(call, warmup=args.warmup, repeat=args.repeat)
        _post_bench_sync()

        elem = x.element_size()
        nv_bytes = (world_size - 1) * cfg.T_local * cfg.d * elem

        rows.append(
            [
                cfg.name,
                f"{cfg.T_local}",
                f"{cfg.d}",
                f"{t*1e3:.1f}",
                f"{_gbps(nv_bytes, t):.0f}",
            ]
        )
        del x, call
        _iter_cleanup()

    _print_table(
        rank,
        f"AG dispatch standalone (W={world_size})",
        ["name", "T_local", "d", "µs", "NVLink GB/s"],
        rows,
    )


def phase_a2a_dispatch(rank, world_size, device, args):
    """A2A dispatch triton standalone bench (NVLink GB/s, actual peer rows).

    Each rank pulls only the slots routed to its E_local experts via
    NVLink peer reads; bytes are counted from ``peer_count_per_rank``
    (excluding self) under the balanced synthetic routing used here.
    Parity vs NCCL all_to_all_single is covered by the test suite.
    """
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
        _flush_async_errors()

        def call():
            a2a_dispatch_triton(
                x,
                meta["dst_rank_flat"],
                s_reverse_local,
                recv,
                K=cfg.K,
                group=dist.group.WORLD,
            )

        t = bench_fn(call, warmup=args.warmup, repeat=args.repeat)
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
                f"{t*1e3:.1f}",
                f"{_gbps(nv_bytes, t):.0f}",
            ]
        )
        del x, recv, meta, s_reverse_local, call
        _iter_cleanup()

    _print_table(
        rank,
        f"A2A dispatch standalone (W={world_size})",
        ["name", "T_local", "d", "K", "E", "µs", "NVLink GB/s"],
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
        TK_global = world_size * TK_local
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
        n_routed = int(expert_freq_off[E_local].item())
        ag_view = ag_compute[x_gather_idx_ag[:n_routed].long()]
        a2a_view = a2a_recv[:n_routed]
        dedup_view = dedup_packed[a_idx_dedup[:n_routed].long()]
        ok = bool(torch.equal(ag_view, a2a_view) and torch.equal(ag_view, dedup_view))

        t_ag = bench_fn(ag_call, warmup=args.warmup, repeat=args.repeat)
        _post_bench_sync()
        t_a2a = bench_fn(a2a_call, warmup=args.warmup, repeat=args.repeat)
        _post_bench_sync()
        t_dedup = bench_fn(dedup_call, warmup=args.warmup, repeat=args.repeat)
        _post_bench_sync()

        elem = x.element_size()
        ag_bytes = (world_size - 1) * cfg.T_local * cfg.d * elem
        pcpr = meta["peer_count_per_rank"]
        pc = meta["pair_count"]
        a2a_rows = int(pcpr[:, rank].sum().item() - pcpr[rank, rank].item())
        a2a_bytes = a2a_rows * cfg.d * elem
        dedup_rows = int(pc[:, rank].sum().item() - pc[rank, rank].item())
        dedup_bytes = dedup_rows * cfg.d * elem

        # Three-way correctness asserted inline; not printed.
        assert ok, f"AG/A2A/DEDUP dispatch parity failed at {cfg.name}"
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


def phase_reduce_scatter(rank, world_size, device, args):
    """``reduce_scatter_triton`` standalone NVLink GB/s bench.

    Times the cross-rank reduce primitive only. The HBM-bound
    ``local_combine`` producer is excluded — for the full
    ``local_combine`` + ``reduce_scatter_triton`` wall-clock see
    ``phase_combine_compare``. Parity vs NCCL ``reduce_scatter_tensor``
    is covered by the test suite.
    """
    rows = []
    for cfg in _AG_CONFIGS:
        x = _alloc_symm((world_size * cfg.T_local, cfg.d), cfg.dtype, device)
        x.normal_()
        _barrier(x)

        out = torch.empty(cfg.T_local, cfg.d, dtype=cfg.dtype, device=device)
        _flush_async_errors()

        def call():
            reduce_scatter_triton(x, dist.group.WORLD, out=out)

        t = bench_fn(call, warmup=args.warmup, repeat=args.repeat)
        _post_bench_sync()

        elem = x.element_size()
        # Each rank pulls (W-1) chunks of (T_local, d) from peers and
        # reduces locally; receive bytes per rank = (W-1)·T_local·d·elem.
        nv_bytes = (world_size - 1) * cfg.T_local * cfg.d * elem

        rows.append(
            [
                cfg.name,
                f"{cfg.T_local}",
                f"{cfg.d}",
                f"{t*1e3:.1f}",
                f"{_gbps(nv_bytes, t):.0f}",
            ]
        )
        del x, out, call
        _iter_cleanup()

    _print_table(
        rank,
        f"RS standalone (W={world_size})",
        ["name", "T_local", "d", "µs", "NVLink GB/s"],
        rows,
    )


def phase_a2a_combine(rank, world_size, device, args):
    """``a2a_combine_triton`` vs NCCL ``all_to_all_single`` smoke test.

    Runs both paths to verify they complete without error on the
    target shapes. No bandwidth column — numerical parity to the NCCL
    recv would require a per-source reduction the bench does
    out-of-band, and timing the A2A combine head-to-head with the
    other combine paths lives in ``phase_combine_compare``.
    """
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

        # Smoke-test both paths complete without error; numerical parity
        # at the row layout level is non-trivial (the NCCL recv would
        # need a per-source reduction the bench does out-of-band).
        tri_call()
        ncl_call()

        rows.append([cfg.name, f"{cfg.T_local}", f"{cfg.d}", f"{cfg.K}", f"{cfg.E}"])
        del y, sr, out_tri, send_buf, recv_ncl, scores_local, scores_global, meta, tri_call, ncl_call
        _iter_cleanup()

    _print_table(
        rank,
        f"A2A combine triton vs NCCL all_to_all_single (smoke, W={world_size})",
        ["name", "T_local", "d", "K", "E"],
        rows,
    )


def phase_rank_dedup_combine(rank, world_size, device, args):
    """RANK_DEDUP_COMBINE_TRITON timing on TWO separate profiles:

      1. ``local_combine`` producer (HBM-bound): reads ``y_symm``
         (TK_global rows) + writes ``partial_acc_buf`` (W·T_local rows) — no
         cross-rank traffic. Reported as HBM TB/s.
      2. cross-rank sparse gather (NVLink-bound): the actual peer-pull
         kernel, with ``partial_acc_buf`` already populated. Reported as
         NVLink GB/s on actual peer bytes.

    The two are kept in separate timings on purpose — averaging across
    them would muddle the bandwidth figures (different memory tiers).
    For the combined ``local_combine + gather`` wall-clock head-to-head
    against A2A and RS, see ``phase_combine_compare``.
    """
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
        scores_global = (
            torch.softmax(
                torch.randn(world_size * cfg.T_local, cfg.K, device=device, dtype=torch.float32),
                dim=-1,
            )
            .to(cfg.dtype)
            .view(-1)
        )
        partial_acc_buf = _alloc_symm((world_size * cfg.T_local, cfg.d), cfg.dtype, device)
        out = torch.empty(cfg.T_local, cfg.d, dtype=cfg.dtype, device=device)

        # Pre-populate partial_acc_buf once so the gather call below has valid
        # input even before we time the local_combine producer below.
        local_combine(
            y,
            sr,
            meta["dst_rank_flat"],
            scores_global,
            partial_acc_buf,
            cfg.K,
            cfg.T_local,
            group=dist.group.WORLD,
            skip_empty=True,
        )
        _barrier(partial_acc_buf)
        _flush_async_errors()

        def local_call():
            local_combine(
                y,
                sr,
                meta["dst_rank_flat"],
                scores_global,
                partial_acc_buf,
                cfg.K,
                cfg.T_local,
                group=dist.group.WORLD,
                skip_empty=True,
            )

        def gather_call():
            _rank_dedup_combine_communication_triton(
                partial_acc_buf,
                meta["peer_present_mask"],
                out,
                T_local=cfg.T_local,
                group=dist.group.WORLD,
            )

        t_local = bench_fn(local_call, warmup=args.warmup, repeat=args.repeat)
        _post_bench_sync()
        t_gather = bench_fn(gather_call, warmup=args.warmup, repeat=args.repeat)
        _post_bench_sync()

        elem = y.element_size()
        pc = meta["pair_count"]
        gather_rows = int(pc[rank, :].sum().item() - pc[rank, rank].item())
        nv_bytes = gather_rows * cfg.d * elem
        # local_combine HBM bytes per rank.
        #   y_symm reads: only the K·is_mine masked loads move HBM
        #     (warps where is_mine==False are predicate-killed and
        #     don't fetch from DRAM). Under balanced routing the
        #     mine_slot count per rank is T_local·K → effective
        #     reads ≈ T_local·K rows · d · elem.
        #   partial_acc_buf writes: W·T_local rows · d · elem. With
        #     SKIP_EMPTY=True only ``any_mine`` rows are stored —
        #     under balanced routing that's
        #     ~W·T_local·(1 - (1 - 1/W)^K). For typical (K, W) the
        #     uncondensed-write count is within a few % of W·T_local
        #     and we ignore the small overcount.
        #   Metadata reads (dst_rank_flat, s_reverse, scores) are
        #     <1% of total and ignored.
        hbm_bytes = cfg.T_local * (cfg.K + world_size) * cfg.d * elem

        rows.append(
            [
                cfg.name,
                f"{cfg.T_local}",
                f"{cfg.d}",
                f"{cfg.K}",
                f"{cfg.E}",
                f"{t_local*1e3:.1f}",
                f"{t_gather*1e3:.1f}",
                f"{_tbps(hbm_bytes, t_local):.2f}",
                f"{_gbps(nv_bytes, t_gather):.0f}",
            ]
        )
        del y, sr, partial_acc_buf, out, scores_global, meta, local_call, gather_call
        _iter_cleanup()

    _print_table(
        rank,
        f"RANK_DEDUP_COMBINE_TRITON: local_combine (HBM) + gather (NVLink) — W={world_size}",
        ["name", "T_local", "d", "K", "E", "local µs", "gather µs", "local HBM TB/s", "gather NVLink GB/s"],
        rows,
    )


def phase_combine_compare(rank, world_size, device, args):
    """A2A_combine vs RS vs RANK_DEDUP_COMBINE_TRITON head-to-head."""
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
        scores_global = torch.empty(world_size * cfg.T_local * cfg.K, dtype=cfg.dtype, device=device)
        dist.all_gather_into_tensor(scores_global, scores_local.view(-1).contiguous(), group=dist.group.WORLD)

        out_a2a = torch.empty(cfg.T_local, cfg.d, dtype=cfg.dtype, device=device)
        out_rs = torch.empty(cfg.T_local, cfg.d, dtype=cfg.dtype, device=device)
        out_dedup = torch.empty(cfg.T_local, cfg.d, dtype=cfg.dtype, device=device)
        partial_acc_buf_rs = _alloc_symm((world_size * cfg.T_local, cfg.d), cfg.dtype, device)
        partial_acc_buf_dedup = _alloc_symm((world_size * cfg.T_local, cfg.d), cfg.dtype, device)
        partial_acc_buf_local = _alloc_symm((world_size * cfg.T_local, cfg.d), cfg.dtype, device)

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

        def rs_call():
            local_combine(
                y,
                sr,
                meta["dst_rank_flat"],
                scores_global,
                partial_acc_buf_rs,
                cfg.K,
                cfg.T_local,
                group=dist.group.WORLD,
            )
            _barrier(partial_acc_buf_rs)
            reduce_scatter_triton(partial_acc_buf_rs, dist.group.WORLD, out=out_rs)

        def dedup_call():
            rank_dedup_combine_triton(
                y,
                sr,
                meta["dst_rank_flat"],
                scores_global,
                meta["peer_present_mask"],
                partial_acc_buf_dedup,
                out_dedup,
                K=cfg.K,
                T_local=cfg.T_local,
                group=dist.group.WORLD,
            )

        # Standalone ``local_combine`` producer step (the HBM-only leg
        # shared by RS and RANK_DEDUP combines). Timed separately so
        # the per-tier bandwidth (HBM for this, NVLink for the others)
        # is reported on its native unit.
        def local_call():
            local_combine(
                y,
                sr,
                meta["dst_rank_flat"],
                scores_global,
                partial_acc_buf_local,
                cfg.K,
                cfg.T_local,
                group=dist.group.WORLD,
            )

        a2a_call()
        rs_call()
        dedup_call()
        atol = 1.5e-1
        rtol = 3e-2
        ok = bool(
            torch.allclose(out_a2a, out_rs, atol=atol, rtol=rtol)
            and torch.allclose(out_a2a, out_dedup, atol=atol, rtol=rtol)
        )

        t_a2a = bench_fn(a2a_call, warmup=args.warmup, repeat=args.repeat)
        _post_bench_sync()
        t_rs = bench_fn(rs_call, warmup=args.warmup, repeat=args.repeat)
        _post_bench_sync()
        t_dedup = bench_fn(dedup_call, warmup=args.warmup, repeat=args.repeat)
        _post_bench_sync()
        t_local = bench_fn(local_call, warmup=args.warmup, repeat=args.repeat)
        _post_bench_sync()

        elem = y.element_size()
        pcpr = meta["peer_count_per_rank"]
        pc = meta["pair_count"]
        a2a_rows = int(pcpr[:, rank].sum().item() - pcpr[rank, rank].item())
        a2a_bytes = a2a_rows * cfg.d * elem
        rs_bytes = (world_size - 1) * cfg.T_local * cfg.d * elem
        dedup_rows = int(pc[rank, :].sum().item() - pc[rank, rank].item())
        dedup_bytes = dedup_rows * cfg.d * elem
        # local_combine HBM bytes per rank.
        #   y_symm reads: only the K·is_mine masked loads move HBM —
        #     warps where is_mine==False are predicate-killed and
        #     don't fetch from DRAM. Under balanced routing the
        #     mine_slot count per rank is T_local·K, so effective
        #     reads ≈ T_local·K rows of size d·elem.
        #   partial_acc_buf writes: W·T_local rows · d · elem (SKIP_EMPTY=False
        #     in this path — every row written).
        #   dst_rank_flat / s_reverse / scores reads are <1% of the
        #     total and ignored.
        hbm_bytes = cfg.T_local * (cfg.K + world_size) * cfg.d * elem

        # Three-way pairwise allclose asserted inline; not printed.
        assert ok, f"A2A/RS/DEDUP combine parity failed at {cfg.name}"
        rows.append(
            [
                cfg.name,
                f"{cfg.T_local}",
                f"{cfg.d}",
                f"{cfg.K}",
                f"{cfg.E}",
                f"{t_a2a*1e3:.1f}",
                f"{t_rs*1e3:.1f}",
                f"{t_dedup*1e3:.1f}",
                f"{_gbps(a2a_bytes, t_a2a):.0f}",
                f"{_gbps(rs_bytes, t_rs):.0f}",
                # RS Local HBM TB/s and Dedup local HBM TB/s both report
                # the standalone ``local_combine`` HBM throughput (same
                # kernel both RS and the dedup combine run as their
                # producer). Reported alongside each cross-rank leg so
                # the producer cost is visible in context for both paths.
                f"{_tbps(hbm_bytes, t_local):.2f}",
                f"{_gbps(dedup_bytes, t_dedup):.0f}",
                f"{_tbps(hbm_bytes, t_local):.2f}",
                f"{t_a2a/t_dedup:.2f}x",
                f"{t_rs/t_dedup:.2f}x",
                f"{dedup_bytes/rs_bytes:.2f}",
                f"{dedup_bytes/a2a_bytes:.2f}",
            ]
        )
        del y, sr, out_a2a, out_rs, out_dedup, partial_acc_buf_rs, partial_acc_buf_dedup, partial_acc_buf_local
        del scores_local, scores_global, meta, a2a_call, rs_call, dedup_call, local_call
        _iter_cleanup()

    _print_table(
        rank,
        f"Combine head-to-head: A2A vs RS vs RANK_DEDUP_COMBINE_TRITON (W={world_size})",
        [
            "name",
            "T_local",
            "d",
            "K",
            "E",
            "A2A µs",
            "RS µs",
            "DEDUP µs",
            "A2A NVLink GB/s",
            "RS NVLink GB/s",
            "RS Local HBM TB/s",
            "DEDUP NVLink GB/s",
            "Dedup local HBM TB/s",
            "A2A/DEDUP",
            "RS/DEDUP",
            "DEDUP/RS bytes",
            "DEDUP/A2A bytes",
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
    # Combine — Triton vs NCCL + RANK_DEDUP sweep + 3-way comparison.
    # ``reduce_scatter`` benches the standalone RS primitive only — the
    # ``local_combine`` producer is HBM-only and is excluded from
    # network-bandwidth measurement (it still runs as one leg of the
    # full RS combine path inside ``combine_compare``).
    "reduce_scatter": phase_reduce_scatter,
    "a2a_combine": phase_a2a_combine,
    "rank_dedup_combine": phase_rank_dedup_combine,
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


# ============================================================================
# Driver — torchrun-driven. dist is initialized once in main(); everything
# else is plain rank-aware code.
# ============================================================================


def _under_torchrun() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--phase", nargs="+", default=["all"], help="phases to run: " + ", ".join(_PHASES.keys()) + ", all"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()

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
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    device = torch.device(f"cuda:{local_rank}")

    try:
        run_phases(rank, world_size, device, args)
    finally:
        try:
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            pass

    # ====================================================================
    # CRITICAL: Hard-exit BEFORE this function returns. Symm-mem state
    # allocated inside the phases (the bench's transient symm-mem
    # buffers plus internal `_symmetric_memory` module-level registries)
    # is still tracked at this point.
    # Falling off the end of main → sys.exit(0) → SystemExit → Python
    # interpreter shutdown → module globals cleared → ~CUDASymmetricMemory()
    # → cuMemUnmap() runs from a C++ destructor while the CUDA context
    # is being torn down concurrently → c10::Error → C++ destructors are
    # not allowed to throw → std::terminate() → SIGABRT.
    #
    # By calling os._exit() here, the CUDA driver releases the device
    # context atomically at process exit without going through any C++
    # destructors. atexit handlers don't fire either, but we don't have
    # any registered, and pytest-style result reporting isn't relevant
    # for a benchmark.
    # ====================================================================
    os._exit(0)


if __name__ == "__main__":
    # main() hard-exits via os._exit on the success path. The only paths
    # that fall through to here are the pre-symm-mem early-skips inside
    # main (returns 1 / 2 before any symm-mem allocation), where Python's
    # normal exit is safe.
    sys.exit(main())
