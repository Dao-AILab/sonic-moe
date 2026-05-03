# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
#
# Benchmark suite for SonicMoE EP collectives and end-to-end forward.
#
# Run (torchrun-launched):
#
#   torchrun --nproc_per_node=4 --standalone benchmarks/ep/bench-comm.py
#   torchrun --nproc_per_node=8 --standalone --local-ranks-filter 0 \
#            benchmarks/ep/bench-comm.py --phase ag a2a_pull
#
# torchrun sets RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT in
# each child; we just read them. --standalone picks a free master port; use
# --local-ranks-filter 0 to dedupe console output (rank 0 already does the
# printing, but Triton/NCCL warnings on other ranks can still be noisy).
# ********************************************************************************

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import _symmetric_memory as _symm_mem

from sonicmoe.enums import ActivationType
from sonicmoe.ep import SymmMemManager, moe_ep_TC_softmax_topk_forward
from sonicmoe.functional.ep import (
    a2a_dispatch_pull_triton,
    all_gather_copy_engine_async,
    all_gather_triton,
    compute_dispatch_metadata,
    gather_aggregation_triton,
    reduce_scatter_triton,
    rs_aggregation,
)
from sonicmoe.functional.triton_kernels import general_routing_router_metadata_triton


# ============================================================================
# Distributed setup
# ============================================================================


def _setup_dist(rank: int, world_size: int, master_port: str = "29555") -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=torch.device(f"cuda:{rank}"))
    _symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)


# ============================================================================
# Timing primitive
# ============================================================================


def bench_fn(
    fn: Callable[[], None], *, warmup: int = 10, repeat: int = 50, cross_rank_max: bool = True, calls_per_iter=3
) -> float:
    """Time `fn()` and return mean per-iter milliseconds."""
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
    if cross_rank_max and dist.is_initialized():
        t = torch.tensor([local_ms], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return t.item()
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
    """bytes / ms -> GB/s."""
    return bytes_moved / ms / 1e6


# ============================================================================
# Symm-mem allocation helper
# ============================================================================


def _alloc_symm(shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    buf = _symm_mem.empty(*shape, dtype=dtype, device=device)
    _symm_mem.rendezvous(buf, group=dist.group.WORLD.group_name)
    return buf


def _symm_barrier(tensor: torch.Tensor) -> None:
    """GPU-side barrier on a symm-mem tensor — cheaper than dist.barrier().
    Required between a producer kernel and a Triton peer-read kernel; NCCL
    collectives self-sync and don't need this."""
    _symm_mem.rendezvous(tensor, group=dist.group.WORLD.group_name).barrier()


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

_T_LOCALS = [8192, 32768]


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


def _build_nogather_recv_pos(
    expert_local_padded: torch.Tensor,
    TK_global: int,
    E_local: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute the expert-sorted recv_pos tensor that the production A2A
    nogather path uses.

    Mirrors what _moe_ep_forward_inner's A2A branch does: runs
    general_routing_router_metadata_triton on expert_local_padded to get
    s_reverse_local, which assigns each global slot a unique row in the
    expert-sorted x_compute layout (recv).
    """
    s_reverse_local = torch.empty(TK_global, dtype=torch.int32, device=device)
    s_scatter_idx = torch.empty(TK_global, dtype=torch.int32, device=device)
    expert_frequency = torch.empty(E_local, dtype=torch.int32, device=device)
    expert_frequency_offset = torch.empty(E_local + 1, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK_global, dtype=torch.int32, device=device)
    num_offset = torch.empty(TK_global + 1, dtype=torch.int32, device=device)
    token_indices = torch.arange(TK_global, dtype=torch.int32, device=device)

    general_routing_router_metadata_triton(
        token_indices,
        expert_local_padded,
        TK_global,
        E_local,
        expert_frequency,
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_local,
        num_offset,
    )
    return s_reverse_local


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

        t = bench_fn(call, warmup=args.warmup, repeat=args.repeat, cross_rank_max=False)
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


def phase_ag(rank: int, world_size: int, device: torch.device, args) -> None:
    rows = []
    for cfg in _AG_CONFIGS:
        x_symm = _alloc_symm((cfg.T_local, cfg.d), cfg.dtype, device)
        x_symm.normal_()
        out_nccl = torch.empty(world_size * cfg.T_local, cfg.d, dtype=cfg.dtype, device=device)

        def copy_engine_call():
            all_gather_copy_engine_async(x_symm, dist.group.WORLD).wait()

        def triton_call():
            all_gather_triton(x_symm, dist.group.WORLD)

        def nccl_call():
            dist.all_gather_into_tensor(out_nccl, x_symm, group=dist.group.WORLD)

        t_triton = bench_fn(triton_call, warmup=args.warmup, repeat=args.repeat)
        t_ce = bench_fn(copy_engine_call, warmup=args.warmup, repeat=args.repeat)
        t_nccl = bench_fn(nccl_call, warmup=args.warmup, repeat=args.repeat)

        # AG is a pure network op. NVLink bytes = (W-1) chunks from peers.
        nvlink_bytes = (world_size - 1) * cfg.T_local * cfg.d * x_symm.element_size()
        rows.append(
            [
                cfg.name,
                f"{cfg.T_local}",
                f"{cfg.d}",
                f"{t_triton*1e3:.1f}",
                f"{t_ce*1e3:.1f}",
                f"{t_nccl*1e3:.1f}",
                f"{t_nccl/t_triton:.2f}x",
                f"{_gbps(nvlink_bytes, t_triton):.0f}",
                f"{_gbps(nvlink_bytes, t_ce):.0f}",
                f"{_gbps(nvlink_bytes, t_nccl):.0f}",
            ]
        )
    _print_table(
        rank,
        f"AG: Triton vs NCCL (W={world_size})",
        [
            "name",
            "T_local",
            "d",
            "triton µs",
            "ce µs",
            "nccl µs",
            "nccl/triton time",
            "triton NVLink GB/s",
            "ce NVLink GB/s",
            "nccl NVLink GB/s",
        ],
        rows,
    )


# ============================================================================
# Phase: A2A pull (Triton nogather vs NCCL)
# ----------------------------------------------------------------------------
# Production A2A dispatch: the pull writes peer rows directly into the
# expert-sorted recv layout (TK_global, d) via recv_pos = s_reverse_local.
# This is the exact layout consumed by the nogather (concat_layout=True)
# GEMM in _moe_ep_forward_inner's A2A branch.
# ============================================================================


@dataclass
class A2APullCfg:
    name: str
    T_local: int
    d: int
    K: int
    E: int
    dtype: torch.dtype = torch.bfloat16


_A2A_PULL_CONFIGS = [
    A2APullCfg(name=f"{m.name}_T{_t_tag(T)}", T_local=T, d=m.d, K=m.K, E=m.E) for m in _MODELS for T in _T_LOCALS
]


def phase_a2a_pull(rank: int, world_size: int, device: torch.device, args) -> None:
    rows = []
    for cfg in _A2A_PULL_CONFIGS:
        if cfg.E % world_size != 0:
            continue
        E_local = cfg.E // world_size
        TK_local = cfg.T_local * cfg.K
        TK_global = world_size * TK_local

        x_symm = _alloc_symm((cfg.T_local, cfg.d), cfg.dtype, device)
        x_symm.normal_()
        # Nogather (contiguous, expert-sorted) recv layout.
        recv = torch.empty((TK_global, cfg.d), dtype=cfg.dtype, device=device)

        topk_idx_g = _make_balanced_topk(cfg.T_local, cfg.K, cfg.E, rank, world_size, device)
        meta = compute_dispatch_metadata(topk_idx_g, my_rank=rank, E_local=E_local)
        dst_rank_flat = meta["dst_rank_flat"]
        recv_pos = _build_nogather_recv_pos(meta["expert_local_padded"], TK_global, E_local, device)

        def triton_call():
            a2a_dispatch_pull_triton(x_symm, dst_rank_flat, recv_pos, recv, K=cfg.K, group=dist.group.WORLD)

        send_nccl = torch.empty(world_size * TK_local, cfg.d, dtype=cfg.dtype, device=device)
        send_nccl.normal_()
        recv_nccl = torch.empty_like(send_nccl)

        def nccl_call():
            dist.all_to_all_single(recv_nccl, send_nccl, group=dist.group.WORLD)

        t_triton = bench_fn(triton_call, warmup=args.warmup, repeat=args.repeat)
        t_nccl = bench_fn(nccl_call, warmup=args.warmup, repeat=args.repeat)

        # A2A pull is a pure network op.
        nvlink_bytes = (world_size - 1) / world_size * TK_local * cfg.d * x_symm.element_size()
        rows.append(
            [
                cfg.name,
                f"{cfg.T_local}",
                f"{cfg.d}",
                f"{cfg.K}",
                f"{cfg.E}",
                f"{t_triton*1e3:.1f}",
                f"{t_nccl*1e3:.1f}",
                f"{t_nccl/t_triton:.2f}x",
                f"{_gbps(nvlink_bytes, t_triton):.0f}",
                f"{_gbps(nvlink_bytes, t_nccl):.0f}",
            ]
        )

        del send_nccl, recv_nccl

    _print_table(
        rank,
        f"A2A pull (nogather layout): Triton vs NCCL (W={world_size})",
        [
            "name",
            "T_local",
            "d",
            "K",
            "E",
            "triton µs",
            "nccl µs",
            "speedup",
            "triton NVLink GB/s",
            "nccl NVLink GB/s",
        ],
        rows,
    )


# ============================================================================
# Phase: AG-mode dispatch vs A2A-mode dispatch (nogather layout)
# ----------------------------------------------------------------------------
# Both rows reflect the production EP forward layouts:
#   AG  → all_gather(x_symm) into (W*T_local, d) compute buffer.
#   A2A → a2a_dispatch_pull_triton(x_symm, recv_pos=s_reverse_local) into
#         (TK_global, d) expert-sorted compute buffer.
# ============================================================================


def phase_ag_vs_a2a(rank: int, world_size: int, device: torch.device, args) -> None:
    rows = []
    for cfg in _A2A_PULL_CONFIGS:
        if cfg.E % world_size != 0:
            continue
        E_local = cfg.E // world_size
        TK_local = cfg.T_local * cfg.K
        TK_global = world_size * TK_local

        x_symm_ag = _alloc_symm((cfg.T_local, cfg.d), cfg.dtype, device)
        x_symm_ag.normal_()

        def ag_call():
            all_gather_triton(x_symm_ag, dist.group.WORLD)

        x_symm_a2a = _alloc_symm((cfg.T_local, cfg.d), cfg.dtype, device)
        x_symm_a2a.normal_()
        # Nogather (contiguous) recv layout.
        recv = torch.empty((TK_global, cfg.d), dtype=cfg.dtype, device=device)

        topk_idx_g = _make_balanced_topk(cfg.T_local, cfg.K, cfg.E, rank, world_size, device)
        meta = compute_dispatch_metadata(topk_idx_g, my_rank=rank, E_local=E_local)
        dst_rank_flat = meta["dst_rank_flat"]
        recv_pos = _build_nogather_recv_pos(meta["expert_local_padded"], TK_global, E_local, device)

        def a2a_call():
            a2a_dispatch_pull_triton(x_symm_a2a, dst_rank_flat, recv_pos, recv, K=cfg.K, group=dist.group.WORLD)

        t_ag = bench_fn(ag_call, warmup=args.warmup, repeat=args.repeat)
        t_a2a = bench_fn(a2a_call, warmup=args.warmup, repeat=args.repeat)

        # Both are pure network ops.
        elem = x_symm_ag.element_size()
        nvlink_ag = (world_size - 1) * cfg.T_local * cfg.d * elem
        nvlink_a2a = (world_size - 1) / world_size * TK_local * cfg.d * elem

        rows.append(
            [
                cfg.name,
                f"{cfg.T_local}",
                f"{cfg.d}",
                f"{cfg.K}",
                f"{cfg.E}",
                f"{t_ag*1e3:.1f}",
                f"{t_a2a*1e3:.1f}",
                f"{t_ag/t_a2a:.2f}x",
                f"{_gbps(nvlink_ag, t_ag):.0f}",
                f"{_gbps(nvlink_a2a, t_a2a):.0f}",
            ]
        )

    _print_table(
        rank,
        f"AG mode vs A2A mode dispatch — nogather layout (W={world_size})",
        ["name", "T_local", "d", "K", "E", "AG µs", "A2A µs", "A2A speedup", "AG NVLink GB/s", "A2A NVLink GB/s"],
        rows,
    )


# ============================================================================
# Phase: A2A pull layout comparison (legacy per-rank-slot vs nogather)
# ----------------------------------------------------------------------------
# Sanity check: the production path is the nogather layout, but we verify
# that the layout choice itself doesn't change pull performance — both
# variants move the same NVLink bytes; only the destination row computation
# differs (a2a_token_indices vs s_reverse_local).
# ============================================================================


def phase_a2a_pull_layouts(rank: int, world_size: int, device: torch.device, args) -> None:
    rows = []
    for cfg in _A2A_PULL_CONFIGS:
        if cfg.E % world_size != 0:
            continue
        E_local = cfg.E // world_size
        TK_local = cfg.T_local * cfg.K
        TK_global = world_size * TK_local

        x_symm = _alloc_symm((cfg.T_local, cfg.d), cfg.dtype, device)
        x_symm.normal_()

        topk_idx_g = _make_balanced_topk(cfg.T_local, cfg.K, cfg.E, rank, world_size, device)
        meta = compute_dispatch_metadata(topk_idx_g, my_rank=rank, E_local=E_local)
        dst_rank_flat = meta["dst_rank_flat"]

        # Legacy per-rank-slot layout: recv shape (W, TK_local, d), recv_pos
        # = a2a_token_indices = src_rank * TK_local + slot_per_rank.
        recv_legacy = torch.empty((world_size, TK_local, cfg.d), dtype=cfg.dtype, device=device)
        recv_pos_legacy = meta["a2a_token_indices"]

        # Nogather layout: recv shape (TK_global, d), recv_pos = s_reverse_local.
        recv_nogather = torch.empty((TK_global, cfg.d), dtype=cfg.dtype, device=device)
        recv_pos_nogather = _build_nogather_recv_pos(meta["expert_local_padded"], TK_global, E_local, device)

        def legacy_call():
            a2a_dispatch_pull_triton(
                x_symm, dst_rank_flat, recv_pos_legacy, recv_legacy, K=cfg.K, group=dist.group.WORLD
            )

        def nogather_call():
            a2a_dispatch_pull_triton(
                x_symm, dst_rank_flat, recv_pos_nogather, recv_nogather, K=cfg.K, group=dist.group.WORLD
            )

        t_legacy = bench_fn(legacy_call, warmup=args.warmup, repeat=args.repeat)
        t_nogather = bench_fn(nogather_call, warmup=args.warmup, repeat=args.repeat)

        nvlink_bytes = (world_size - 1) / world_size * TK_local * cfg.d * x_symm.element_size()
        rows.append(
            [
                cfg.name,
                f"{cfg.T_local}",
                f"{cfg.d}",
                f"{cfg.K}",
                f"{cfg.E}",
                f"{t_legacy*1e3:.1f}",
                f"{t_nogather*1e3:.1f}",
                f"{t_legacy/t_nogather:.2f}x",
                f"{_gbps(nvlink_bytes, t_legacy):.0f}",
                f"{_gbps(nvlink_bytes, t_nogather):.0f}",
            ]
        )

    _print_table(
        rank,
        f"A2A pull: legacy vs nogather layout (W={world_size})",
        [
            "name",
            "T_local",
            "d",
            "K",
            "E",
            "legacy µs",
            "nogather µs",
            "nogather speedup",
            "legacy NVLink GB/s",
            "nogather NVLink GB/s",
        ],
        rows,
    )


# ============================================================================
# Phase: RS (Triton vs NCCL)
# ----------------------------------------------------------------------------
# Mirror of phase_ag for the reduce_scatter direction.
#
# x_symm shape: (W * T_local, d) — same total size NCCL expects.
# Output:       (T_local, d).
#
# NVLink bytes per rank: (W-1) * T_local * d * elem_size — each rank reads
# W-1 peer chunks of size (T_local, d). This matches AG's accounting since
# the wire-level traffic is symmetric.
# ============================================================================


def phase_rs(rank: int, world_size: int, device: torch.device, args) -> None:
    rows = []
    for cfg in _AG_CONFIGS:
        # Same per-rank chunk size as the AG bench: cfg.T_local rows per rank.
        # The full symm-mem buffer is W * T_local rows, matching what
        # reduce_scatter_tensor expects as input.
        x_symm = _alloc_symm((world_size * cfg.T_local, cfg.d), cfg.dtype, device)
        x_symm.normal_()
        out_nccl = torch.empty(cfg.T_local, cfg.d, dtype=cfg.dtype, device=device)

        def triton_call():
            reduce_scatter_triton(x_symm, dist.group.WORLD)

        def nccl_call():
            dist.reduce_scatter_tensor(out_nccl, x_symm, op=dist.ReduceOp.SUM, group=dist.group.WORLD)

        t_triton = bench_fn(triton_call, warmup=args.warmup, repeat=args.repeat)
        t_nccl = bench_fn(nccl_call, warmup=args.warmup, repeat=args.repeat)

        nvlink_bytes = (world_size - 1) * cfg.T_local * cfg.d * x_symm.element_size()
        rows.append(
            [
                cfg.name,
                f"{cfg.T_local}",
                f"{cfg.d}",
                f"{t_triton*1e3:.1f}",
                f"{t_nccl*1e3:.1f}",
                f"{t_nccl/t_triton:.2f}x",
                f"{_gbps(nvlink_bytes, t_triton):.0f}",
                f"{_gbps(nvlink_bytes, t_nccl):.0f}",
            ]
        )
    _print_table(
        rank,
        f"RS: Triton vs NCCL (W={world_size})",
        ["name", "T_local", "d", "triton µs", "nccl µs", "speedup", "triton NVLink GB/s", "nccl NVLink GB/s"],
        rows,
    )


# ============================================================================
# Phase: gather_aggregation_triton
# ============================================================================


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


def phase_gather_aggregation_triton(rank: int, world_size: int, device: torch.device, args) -> None:
    rows = []
    for cfg in _COMBINE_CONFIGS:
        if cfg.E % world_size != 0:
            continue
        E_local = cfg.E // world_size
        TK_local = cfg.T_local * cfg.K
        TK_global = world_size * TK_local

        y_symm = _alloc_symm((TK_global, cfg.d), cfg.dtype, device)
        y_symm.normal_()
        s_rev_symm = _alloc_symm((TK_global,), torch.int32, device)
        s_rev_symm.copy_(torch.randint(0, TK_local, (TK_global,), device=device, dtype=torch.int32))

        topk_idx_g = _make_balanced_topk(cfg.T_local, cfg.K, cfg.E, rank, world_size, device)
        meta = compute_dispatch_metadata(topk_idx_g, my_rank=rank, E_local=E_local)
        rank_2d = meta["my_dst_rank"]
        pos_2d = (torch.arange(TK_local, device=device, dtype=torch.int32) + rank * TK_local).view(cfg.T_local, cfg.K)

        topk_scores = torch.softmax(
            torch.randn(cfg.T_local, cfg.K, device=device, dtype=torch.float32),
            dim=-1,
        )
        out = torch.empty(cfg.T_local, cfg.d, dtype=cfg.dtype, device=device)

        def call():
            gather_aggregation_triton(
                y_symm, s_rev_symm, rank_2d, pos_2d, topk_scores, out, K=cfg.K, group=dist.group.WORLD
            )

        t = bench_fn(call, warmup=args.warmup, repeat=args.repeat)
        nvlink_bytes = (world_size - 1) / world_size * cfg.T_local * cfg.K * cfg.d * y_symm.element_size()
        rows.append(
            [
                cfg.name,
                f"{cfg.T_local}",
                f"{cfg.d}",
                f"{cfg.K}",
                f"{cfg.E}",
                f"{t*1e3:.1f}",
                f"{_gbps(nvlink_bytes, t):.0f}",
            ]
        )
    _print_table(
        rank,
        f"gather_aggregation_triton (W={world_size})",
        ["name", "T_local", "d", "K", "E", "µs", "NVLink GB/s"],
        rows,
    )


# ============================================================================
# Phase: rs_aggregation (producer + RS) — both NCCL RS and Triton RS
# ----------------------------------------------------------------------------
# rs_aggregation (producer) writes fp32 partials to rs_buf and is pure local
# HBM — zero NVLink. The NVLink traffic comes from the RS step, where we now
# bench two backends:
#
#   1. NCCL  reduce_scatter_tensor   — ring with internal sync.
#   2. Triton reduce_scatter         — direct peer reads + fp32 sum, no NCCL.
#                                       Needs explicit symm-mem barrier between
#                                       producer and RS to make peer writes
#                                       visible. Barrier cost is included in
#                                       the Triton pipeline number.
#
# rs_buf is allocated in symm-mem so both backends can scatter from it.
# NVLink GB/s columns use the standalone RS time (no producer included).
# ============================================================================


def phase_rs_aggregation(rank: int, world_size: int, device: torch.device, args) -> None:
    rows = []
    for cfg in _COMBINE_CONFIGS:
        if cfg.E % world_size != 0:
            continue
        E_local = cfg.E // world_size
        TK_local = cfg.T_local * cfg.K
        TK_global = world_size * TK_local

        y_symm = _alloc_symm((TK_global, cfg.d), cfg.dtype, device)
        y_symm.normal_()
        s_rev_symm = _alloc_symm((TK_global,), torch.int32, device)
        s_rev_symm.copy_(torch.arange(TK_global, dtype=torch.int32, device=device))

        topk_idx_g = _make_balanced_topk(cfg.T_local, cfg.K, cfg.E, rank, world_size, device)
        meta = compute_dispatch_metadata(topk_idx_g, my_rank=rank, E_local=E_local)
        dst_rank_flat = meta["dst_rank_flat"]

        scores_2d = torch.softmax(
            torch.randn(world_size * cfg.T_local, cfg.K, device=device, dtype=torch.float32),
            dim=-1,
        ).to(cfg.dtype)
        scores_flat = scores_2d.contiguous().view(-1)

        # Symm-mem rs_buf so both NCCL and Triton can RS it.
        rs_buf = _alloc_symm((world_size * cfg.T_local, cfg.d), torch.float32, device)
        out_rs = torch.empty(cfg.T_local, cfg.d, dtype=torch.float32, device=device)

        # 1. Producer only (local HBM).
        def call_produce():
            rs_aggregation(
                y_symm, s_rev_symm, dst_rank_flat, scores_flat, rs_buf, cfg.K, cfg.T_local, group=dist.group.WORLD
            )

        t_produce = bench_fn(call_produce, warmup=args.warmup, repeat=args.repeat)

        # Pre-fill rs_buf for standalone RS benches.
        rs_aggregation(
            y_symm, s_rev_symm, dst_rank_flat, scores_flat, rs_buf, cfg.K, cfg.T_local, group=dist.group.WORLD
        )
        _symm_barrier(rs_buf)
        torch.cuda.synchronize()

        # 2a. NCCL RS only (on pre-filled buffer).
        def call_rs_nccl():
            dist.reduce_scatter_tensor(out_rs, rs_buf, op=dist.ReduceOp.SUM, group=dist.group.WORLD)

        t_rs_nccl = bench_fn(call_rs_nccl, warmup=args.warmup, repeat=args.repeat)

        # 2b. Triton RS only (on pre-filled buffer; no producer-RS race
        # because rs_buf is read-only across iters).
        def call_rs_triton():
            reduce_scatter_triton(rs_buf, dist.group.WORLD, out=out_rs)

        t_rs_triton = bench_fn(call_rs_triton, warmup=args.warmup, repeat=args.repeat)

        # 3a. NCCL pipeline — NCCL RS self-syncs, no explicit barrier.
        def call_pipeline_nccl():
            rs_aggregation(
                y_symm, s_rev_symm, dst_rank_flat, scores_flat, rs_buf, cfg.K, cfg.T_local, group=dist.group.WORLD
            )
            dist.reduce_scatter_tensor(out_rs, rs_buf, op=dist.ReduceOp.SUM, group=dist.group.WORLD)

        t_pipe_nccl = bench_fn(call_pipeline_nccl, warmup=args.warmup, repeat=args.repeat)

        # 3b. Triton pipeline — needs symm-mem barrier between producer and
        # peer-read RS so writes are visible cross-rank.
        def call_pipeline_triton():
            rs_aggregation(
                y_symm, s_rev_symm, dst_rank_flat, scores_flat, rs_buf, cfg.K, cfg.T_local, group=dist.group.WORLD
            )
            _symm_barrier(rs_buf)
            reduce_scatter_triton(rs_buf, dist.group.WORLD, out=out_rs)

        t_pipe_triton = bench_fn(call_pipeline_triton, warmup=args.warmup, repeat=args.repeat)

        rs_nvlink_bytes = (world_size - 1) * cfg.T_local * cfg.d * 4  # fp32

        rows.append(
            [
                cfg.name,
                f"{cfg.T_local}",
                f"{cfg.d}",
                f"{cfg.K}",
                f"{cfg.E}",
                f"{t_produce*1e3:.1f}",
                f"{t_rs_nccl*1e3:.1f}",
                f"{t_rs_triton*1e3:.1f}",
                f"{t_rs_nccl/t_rs_triton:.2f}x",
                f"{t_pipe_nccl*1e3:.1f}",
                f"{t_pipe_triton*1e3:.1f}",
                f"{_gbps(rs_nvlink_bytes, t_rs_nccl):.0f}",
                f"{_gbps(rs_nvlink_bytes, t_rs_triton):.0f}",
            ]
        )

    _print_table(
        rank,
        f"rs_aggregation + RS (W={world_size})",
        [
            "name",
            "T_local",
            "d",
            "K",
            "E",
            "produce µs",
            "nccl RS µs",
            "triton RS µs",
            "tri RS speedup",
            "nccl pipe µs",
            "triton pipe µs",
            "nccl GB/s",
            "triton GB/s",
        ],
        rows,
    )


# ============================================================================
# Phase: gather_aggregation_triton vs rs_aggregation pipeline (NCCL and Triton RS)
# ----------------------------------------------------------------------------
# gather_aggregation_triton: fused NVLink-read kernel, total time = network time.
# rs_aggregation pipeline: producer (HBM) + barrier + RS (NVLink). Network
# time = RS only.
#
# We report the rs+RS pipeline cost for BOTH RS backends. Triton pipeline
# includes a symm-mem barrier between producer and RS, which is what real
# pipeline code would do. NCCL doesn't need an explicit barrier.
#
# "best speedup" = t_gather / min(t_pipe_nccl, t_pipe_triton). >1 means the
# best RS pipeline beats gather.
# ============================================================================


def phase_gather_vs_rs_aggregation(rank: int, world_size: int, device: torch.device, args) -> None:
    rows = []
    for cfg in _COMBINE_CONFIGS:
        if cfg.E % world_size != 0:
            continue
        E_local = cfg.E // world_size
        TK_local = cfg.T_local * cfg.K
        TK_global = world_size * TK_local

        y_symm = _alloc_symm((TK_global, cfg.d), cfg.dtype, device)
        y_symm.normal_()
        s_rev_symm = _alloc_symm((TK_global,), torch.int32, device)
        s_rev_symm.copy_(torch.arange(TK_global, dtype=torch.int32, device=device))

        topk_idx_g = _make_balanced_topk(cfg.T_local, cfg.K, cfg.E, rank, world_size, device)
        meta = compute_dispatch_metadata(topk_idx_g, my_rank=rank, E_local=E_local)
        dst_rank_flat = meta["dst_rank_flat"]

        scores_2d = torch.softmax(
            torch.randn(world_size * cfg.T_local, cfg.K, device=device, dtype=torch.float32),
            dim=-1,
        )
        scores_flat = scores_2d.view(-1)

        # --- gather_aggregation_triton ---
        rank_2d = meta["my_dst_rank"]
        pos_2d = (torch.arange(TK_local, device=device, dtype=torch.int32) + rank * TK_local).view(cfg.T_local, cfg.K)
        topk_scores = scores_2d[rank * cfg.T_local : (rank + 1) * cfg.T_local]
        out_gather = torch.empty(cfg.T_local, cfg.d, dtype=cfg.dtype, device=device)

        def call_gather():
            gather_aggregation_triton(
                y_symm, s_rev_symm, rank_2d, pos_2d, topk_scores, out_gather, K=cfg.K, group=dist.group.WORLD
            )

        t_gather = bench_fn(call_gather, warmup=args.warmup, repeat=args.repeat)

        # --- rs_aggregation pipelines, both backends ---
        rs_buf = _alloc_symm((world_size * cfg.T_local, cfg.d), torch.float32, device)
        out_rs = torch.empty(cfg.T_local, cfg.d, dtype=torch.float32, device=device)

        def call_pipeline_nccl():
            rs_aggregation(
                y_symm, s_rev_symm, dst_rank_flat, scores_flat, rs_buf, cfg.K, cfg.T_local, group=dist.group.WORLD
            )
            dist.reduce_scatter_tensor(out_rs, rs_buf, op=dist.ReduceOp.SUM, group=dist.group.WORLD)

        def call_pipeline_triton():
            rs_aggregation(
                y_symm, s_rev_symm, dst_rank_flat, scores_flat, rs_buf, cfg.K, cfg.T_local, group=dist.group.WORLD
            )
            _symm_barrier(rs_buf)
            reduce_scatter_triton(rs_buf, dist.group.WORLD, out=out_rs)

        t_pipe_nccl = bench_fn(call_pipeline_nccl, warmup=args.warmup, repeat=args.repeat)
        t_pipe_triton = bench_fn(call_pipeline_triton, warmup=args.warmup, repeat=args.repeat)
        t_pipe_best = min(t_pipe_nccl, t_pipe_triton)

        # Standalone RS times for the GB/s columns. Pre-fill rs_buf first,
        # barrier so peers see it, then bench.
        rs_aggregation(
            y_symm, s_rev_symm, dst_rank_flat, scores_flat, rs_buf, cfg.K, cfg.T_local, group=dist.group.WORLD
        )
        _symm_barrier(rs_buf)
        torch.cuda.synchronize()

        def call_rs_nccl():
            dist.reduce_scatter_tensor(out_rs, rs_buf, op=dist.ReduceOp.SUM, group=dist.group.WORLD)

        def call_rs_triton():
            reduce_scatter_triton(rs_buf, dist.group.WORLD, out=out_rs)

        t_rs_nccl = bench_fn(call_rs_nccl, warmup=args.warmup, repeat=args.repeat)
        t_rs_triton = bench_fn(call_rs_triton, warmup=args.warmup, repeat=args.repeat)

        gather_nvlink = (world_size - 1) / world_size * cfg.T_local * cfg.K * cfg.d * y_symm.element_size()
        rs_nvlink = (world_size - 1) * cfg.T_local * cfg.d * 4  # fp32

        rows.append(
            [
                cfg.name,
                f"{cfg.T_local}",
                f"{cfg.d}",
                f"{cfg.K}",
                f"{cfg.E}",
                f"{t_gather*1e3:.1f}",
                f"{t_pipe_nccl*1e3:.1f}",
                f"{t_pipe_triton*1e3:.1f}",
                f"{t_gather/t_pipe_best:.2f}x",
                f"{_gbps(gather_nvlink, t_gather):.0f}",
                f"{_gbps(rs_nvlink, t_rs_nccl):.0f}",
                f"{_gbps(rs_nvlink, t_rs_triton):.0f}",
            ]
        )

    _print_table(
        rank,
        f"gather vs rs+RS pipeline (W={world_size})",
        [
            "name",
            "T_local",
            "d",
            "K",
            "E",
            "gather µs",
            "rs+nccl µs",
            "rs+triton µs",
            "best rs vs gather",
            "gather GB/s",
            "nccl RS GB/s",
            "triton RS GB/s",
        ],
        rows,
    )


# ============================================================================
# Phase: end-to-end EP forward (AG mode vs A2A mode)
# ============================================================================


@dataclass
class E2ECfg:
    name: str
    T_local: int
    d: int
    n: int
    K: int
    E: int
    dtype: torch.dtype = torch.bfloat16
    activation: ActivationType = ActivationType.SWIGLU


_E2E_CONFIGS = [
    E2ECfg("olmoe", T_local=32768, d=2048, n=1024, E=64, K=8),
    E2ECfg("d2880_e64k4", T_local=32768, d=2880, n=2880, E=64, K=4),
    E2ECfg("d2304_e256k8", T_local=32768, d=2304, n=1024, E=256, K=8),
    E2ECfg("e512_k10", T_local=32768, d=2048, n=512, E=512, K=10),
    E2ECfg("d4096_e128k8", T_local=16384, d=4096, n=1536, E=128, K=8),
    E2ECfg("deepseek_v3", T_local=16384, d=7168, n=2048, E=256, K=8),
]


def _make_e2e_inputs(cfg: E2ECfg, rank: int, world_size: int, device: torch.device):
    g = torch.Generator(device=device).manual_seed(42)
    H, I, E = cfg.d, cfg.n, cfg.E
    raw_I = 2 * I
    E_local = E // world_size

    x = 0.02 * torch.randn(
        cfg.T_local,
        H,
        generator=torch.Generator(device=device).manual_seed(1000 + rank),
        device=device,
        dtype=cfg.dtype,
    )
    router_w = 0.02 * torch.randn(E, H, generator=g, device=device, dtype=cfg.dtype)
    w1_full = 0.02 * torch.randn(E, raw_I, H, generator=g, device=device, dtype=cfg.dtype)
    w2_full = 0.02 * torch.randn(E, H, I, generator=g, device=device, dtype=cfg.dtype)

    w1_local = w1_full[rank * E_local : (rank + 1) * E_local].permute(1, 2, 0)
    w2_local = w2_full[rank * E_local : (rank + 1) * E_local].permute(0, 2, 1).contiguous()
    return x, router_w, w1_local, w2_local


def phase_e2e(rank: int, world_size: int, device: torch.device, args) -> None:
    rows = []
    for cfg in _E2E_CONFIGS:
        if cfg.E % world_size != 0:
            continue
        x, router_w, w1_local, w2_local = _make_e2e_inputs(cfg, rank, world_size, device)

        results = {}
        for mode in ("ag", "a2a"):
            mgr = SymmMemManager(dist.group.WORLD, device)

            def call():
                moe_ep_TC_softmax_topk_forward(
                    x=x,
                    router_w=router_w,
                    w1=w1_local,
                    b1=None,
                    w2=w2_local,
                    b2=None,
                    K=cfg.K,
                    E=cfg.E,
                    mgr=mgr,
                    activation_type=cfg.activation,
                    is_inference_mode_enabled=True,
                    mode=mode,
                )

            t = bench_fn(call, warmup=args.warmup, repeat=args.repeat)
            results[mode] = t
            mgr.clear()

        rows.append(
            [
                cfg.name,
                f"{cfg.T_local}",
                f"{cfg.d}",
                f"{cfg.n}",
                f"{cfg.K}",
                f"{cfg.E}",
                f"{results['ag']*1e3:.0f}",
                f"{results['a2a']*1e3:.0f}",
                f"{results['ag']/results['a2a']:.2f}x",
            ]
        )

    _print_table(
        rank,
        f"End-to-end EP forward: AG (gather GEMM) vs A2A (nogather GEMM) (W={world_size})",
        ["name", "T_local", "d", "n", "K", "E", "AG µs", "A2A µs", "A2A speedup"],
        rows,
    )


# ============================================================================
# Phase dispatch
# ============================================================================

_PHASES = {
    "metadata": phase_metadata,
    "ag": phase_ag,
    "rs": phase_rs,
    "a2a_pull": phase_a2a_pull,
    "a2a_pull_layouts": phase_a2a_pull_layouts,
    "ag_vs_a2a": phase_ag_vs_a2a,
    "gather_aggregation_triton": phase_gather_aggregation_triton,
    "rs_aggregation": phase_rs_aggregation,
    "gather_vs_rs_aggregation": phase_gather_vs_rs_aggregation,
    "e2e": phase_e2e,
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
    parser.add_argument("--repeat", type=int, default=200)
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
    _symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
    device = torch.device(f"cuda:{local_rank}")

    try:
        run_phases(rank, world_size, device, args)
    finally:
        try:
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
