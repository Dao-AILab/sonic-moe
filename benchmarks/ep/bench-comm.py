# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
#
# Benchmark suite for SonicMoE EP collectives and end-to-end forward.
#
# Run:
#   EP_BENCH_WORLD_SIZE=4 python -m sonicmoe.benchmarks.ep_bench
# ********************************************************************************

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed import _symmetric_memory as _symm_mem

from sonicmoe.enums import ActivationType
from sonicmoe.ep import (
    SymmMemManager,
    moe_ep_TC_softmax_topk_forward,
    moe_ep_general_routing_forward,
)
from sonicmoe.functional.ep import (
    a2a_dispatch_pull,
    all_gather as triton_all_gather,
    compute_dispatch_metadata,
    gather_aggregation,
    rs_aggregation,
)


# ============================================================================
# Distributed setup
# ============================================================================

def _setup_dist(rank: int, world_size: int, master_port: str = "29555") -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", 
        rank=rank, 
        world_size=world_size,
        device_id=torch.device(f"cuda:{rank}")
    )
    _symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)


def _bench_subprocess(rank: int, world_size: int, args) -> None:
    _setup_dist(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(0 + rank)
    try:
        run_phases(rank, world_size, device, args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


# ============================================================================
# Timing primitive
# ============================================================================

def bench_fn(fn: Callable[[], None], *,
             warmup: int = 10, repeat: int = 50,
             cross_rank_max: bool = True) -> float:
    """Time `fn()` and return mean per-iter milliseconds.

    The returned value is the MAX across all ranks (when cross_rank_max=True
    and dist is initialized) — collective completion is gated on the slowest
    rank, so reporting the slowest is the right thing.

    A single pair of CUDA events bracketing the entire `repeat` loop avoids
    per-iter event-record overhead, which would dominate measurements of
    sub-microsecond kernels.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    local_ms = start.elapsed_time(end) / repeat

    if cross_rank_max and dist.is_initialized():
        t = torch.tensor([local_ms], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return t.item()
    return local_ms


# ============================================================================
# Pretty-printing
# ============================================================================

def _print_table(rank: int, title: str, header: List[str],
                 rows: List[List[str]]) -> None:
    if rank != 0:
        return
    widths = [max(len(str(r[i])) for r in [header] + rows)
              for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    bar = "  ".join("-" * w for w in widths)
    print(f"\n=== {title} ===")
    print(fmt.format(*header))
    print(bar)
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))


def _gbps(bytes_moved: float, ms: float) -> float:
    """bytes / ms → GB/s. (1e-9 GB/B) / (1e-3 s/ms) = 1e-6"""
    return bytes_moved / ms / 1e6


# ============================================================================
# Symm-mem allocation helper
# ============================================================================

def _alloc_symm(shape: Tuple[int, ...], dtype: torch.dtype,
                device: torch.device) -> torch.Tensor:
    buf = _symm_mem.empty(*shape, dtype=dtype, device=device)
    _symm_mem.rendezvous(buf, group=dist.group.WORLD.group_name)
    return buf


@dataclass
class _ModelCfg:
    name: str
    d: int
    E: int
    K: int


_MODELS = [
    _ModelCfg("olmoe",        d=2048, E=64,  K=8),
    _ModelCfg("d2880_e64k4",  d=2880, E=64,  K=4),
    _ModelCfg("d2304_e256k8", d=2304, E=256, K=8),
    _ModelCfg("e512_k10",     d=2048, E=512, K=10),
    _ModelCfg("d4096_e128k8", d=4096, E=128, K=8),
]

_T_LOCALS = [8192, 32768]


def _t_tag(T: int) -> str:
    return f"{T // 1024}k"


def _dedupe_by_d(models):
    """AG/RS depend only on (T_local, d) — keep first model per d so we
    don't bench the same kernel twice. Order preserved."""
    seen, out = set(), []
    for m in models:
        if m.d in seen:
            continue
        seen.add(m.d)
        out.append(m)
    return out



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
    MetaCfg(name=f"{m.name}_T{_t_tag(T)}",
               T_local=T, K=m.K, E=m.E)
    for m in _dedupe_by_d(_MODELS)
    for T in _T_LOCALS
]



def phase_metadata(rank: int, world_size: int, device: torch.device, args) -> None:
    rows = []
    for cfg in _META_CONFIGS:
        if cfg.E % world_size != 0:
            continue
        E_local = cfg.E // world_size

        topk_idx_g = _make_balanced_topk(cfg.T_local, cfg.K, cfg.E,
                                         rank, world_size, device)

        def call():
            compute_dispatch_metadata(topk_idx_g, my_rank=rank, E_local=E_local)

        t = bench_fn(call, warmup=args.warmup, repeat=args.repeat,
                     cross_rank_max=False)
        TK_global = world_size * cfg.T_local * cfg.K
        rows.append([
            cfg.name,
            f"{cfg.T_local}", f"{cfg.K}", f"{cfg.E}", f"{TK_global}",
            f"{t*1e3:.1f}",
        ])

    _print_table(
        rank, f"compute_dispatch_metadata (W={world_size})",
        ["name", "T_local", "K", "E", "TK_global", "µs"],
        rows,
    )



# ============================================================================
# Phase: AG (Triton vs NCCL all_gather_into_tensor)
# ============================================================================

@dataclass
class AGCfg:
    name: str
    T_local: int
    d: int
    dtype: torch.dtype = torch.bfloat16

_AG_CONFIGS = [
    AGCfg(name=f"{m.name}_T{_t_tag(T)}", T_local=T, d=m.d)
    for m in _dedupe_by_d(_MODELS)
    for T in _T_LOCALS
]


def phase_ag(rank: int, world_size: int, device: torch.device, args) -> None:
    rows = []
    for cfg in _AG_CONFIGS:
        x_symm = _alloc_symm((cfg.T_local, cfg.d), cfg.dtype, device)
        x_symm.normal_()
        out_nccl = torch.empty(world_size * cfg.T_local, cfg.d,
                               dtype=cfg.dtype, device=device)

        def triton_call():
            triton_all_gather(x_symm, dist.group.WORLD)

        def nccl_call():
            dist.all_gather_into_tensor(out_nccl, x_symm, group=dist.group.WORLD)

        t_triton = bench_fn(triton_call, warmup=args.warmup, repeat=args.repeat)
        t_nccl = bench_fn(nccl_call, warmup=args.warmup, repeat=args.repeat)

        bytes_moved = world_size * cfg.T_local * cfg.d * x_symm.element_size()
        rows.append([
            cfg.name,
            f"{cfg.T_local}", f"{cfg.d}",
            f"{t_triton*1e3:.1f}", f"{t_nccl*1e3:.1f}",
            f"{t_nccl/t_triton:.2f}x",
            f"{_gbps(bytes_moved, t_triton):.0f}",
            f"{_gbps(bytes_moved, t_nccl):.0f}",
        ])
    _print_table(
        rank, f"AG: Triton vs NCCL (W={world_size})",
        ["name", "T_local", "d",
         "triton µs", "nccl µs", "speedup",
         "triton GB/s", "nccl GB/s"],
        rows,
    )



# ============================================================================
# Phase: A2A pull (Triton vs NCCL all_to_all_single)
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
    A2APullCfg(name=f"{m.name}_T{_t_tag(T)}",
               T_local=T, d=m.d, K=m.K, E=m.E)
    for m in _MODELS
    for T in _T_LOCALS
]


def _make_balanced_topk(T_local: int, K: int, E: int, my_rank: int,
                        world_size: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(123 + my_rank)
    local = torch.randint(0, E, (T_local, K), generator=g,
                          device=device, dtype=torch.int32)
    full = torch.empty((world_size, T_local, K),
                       dtype=torch.int32, device=device)
    full[my_rank] = local
    dist.all_gather_into_tensor(full.view(-1), local.view(-1).contiguous(),
                                group=dist.group.WORLD)
    full = full.view(world_size, T_local, K)
    return full


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
        recv = torch.empty((world_size, TK_local, cfg.d),
                           dtype=cfg.dtype, device=device)

        topk_idx_g = _make_balanced_topk(cfg.T_local, cfg.K, cfg.E,
                                         rank, world_size, device)
        meta = compute_dispatch_metadata(topk_idx_g, my_rank=rank,
                                         E_local=E_local)
        dst_rank_flat = meta["dst_rank_flat"]
        slot_flat_per_rank = meta["slot_flat_per_rank"]

        def triton_call():
            a2a_dispatch_pull(x_symm, dst_rank_flat, slot_flat_per_rank,
                              recv, K=cfg.K, group=dist.group.WORLD)

        # ----------------------------------------------------------------
        # NCCL baseline: raw all_to_all_single on (W*TK_local, d) buffers.
        #
        # Caveat: this is *not* a bytes-moved apples-to-apples comparison.
        # The Triton kernel selectively pulls only the valid lanes routed
        # to this rank — roughly (W-1)/W * TK_local * d bytes cross-rank
        # in balanced routing. NCCL all_to_all moves the full chunk to
        # every peer regardless of routing — (W-1) * TK_local * d bytes
        # cross-rank — i.e. ~W× more NVLink traffic.
        #
        # The reported "speedup" therefore captures both (a) Triton's
        # better implementation efficiency and (b) the algorithmic win
        # of fusing the routing permute into the dispatch (NCCL has no
        # routing concept, so a real NCCL-based EP forward would also
        # need a local scatter pre-pass that this baseline excludes).
        # Time is the meaningful column; GB/s is the per-rank Triton
        # algorithmic payload (TK_local * d) divided by each kernel's
        # measured time, kept as the same base for ratio consistency.
        #
        # Memory: send_nccl + recv_nccl together cost 2 * W * TK_local *
        # d * dtype_bytes per rank. At deepseek_v3 (d=7168) T=32k W=8
        # that's ~60 GB — will OOM on smaller GPUs. Drop the offending
        # entry from _MODELS or skip this phase if you hit it.
        # ----------------------------------------------------------------
        send_nccl = torch.empty(world_size * TK_local, cfg.d,
                                dtype=cfg.dtype, device=device)
        send_nccl.normal_()
        recv_nccl = torch.empty_like(send_nccl)

        def nccl_call():
            dist.all_to_all_single(recv_nccl, send_nccl,
                                   group=dist.group.WORLD)

        t_triton = bench_fn(triton_call, warmup=args.warmup, repeat=args.repeat)
        t_nccl = bench_fn(nccl_call, warmup=args.warmup, repeat=args.repeat)

        bytes_moved = TK_local * cfg.d * x_symm.element_size()
        rows.append([
            cfg.name,
            f"{cfg.T_local}", f"{cfg.d}", f"{cfg.K}", f"{cfg.E}",
            f"{t_triton*1e3:.1f}", f"{t_nccl*1e3:.1f}",
            f"{t_nccl/t_triton:.2f}x",
            f"{_gbps(bytes_moved, t_triton):.0f}",
            f"{_gbps(bytes_moved, t_nccl):.0f}",
        ])

        # Free NCCL buffers before next config — these are huge at large
        # T_local * d. Triton's recv stays allocated till the next iter's
        # `_alloc_symm` rebinds (handled by Python GC + empty_cache below).
        del send_nccl, recv_nccl

    _print_table(
        rank, f"A2A pull: Triton vs NCCL (W={world_size})",
        ["name", "T_local", "d", "K", "E",
         "triton µs", "nccl µs", "speedup",
         "triton GB/s", "nccl GB/s"],
        rows,
    )



# ============================================================================
# Phase: AG-mode dispatch vs A2A-mode dispatch (same payload, same shapes)
# ----------------------------------------------------------------------------
# AG mode: every rank all-gathers x_symm (T_local, d) → (W*T_local, d);
#   downstream consumer indexes the gathered tensor by routing decisions.
# A2A mode: every rank pulls only the rows it actually consumes via the
#   fused dispatch kernel; output (W, TK_local, d) where slot positions
#   beyond peer_count_per_rank[:, my_rank] are unused.
#
# Bytes moved per rank:
#   AG : (W-1) * T_local * d                            (algorithmic NVLink)
#   A2A: ~(W-1)/W * TK_local * d = (W-1) * T_local * K / W * d
#
# Ratio AG / A2A  =  W / K   in balanced routing.
#   So K < W (e.g. K=2,4 at W=8): AG moves more bytes than A2A.
#   K > W (e.g. e512_k10 at W=8): A2A moves more bytes than AG.
#   K = W: they're algorithmically equivalent in payload.
#
# The reported "A2A speedup" is wall time only. Whether it's a *bytes
# efficiency* win or just a *latency* win you can read off the table by
# eyeballing K vs W.
# ============================================================================

def phase_ag_vs_a2a(rank: int, world_size: int,
                    device: torch.device, args) -> None:
    rows = []
    for cfg in _A2A_PULL_CONFIGS:
        if cfg.E % world_size != 0:
            continue
        E_local = cfg.E // world_size
        TK_local = cfg.T_local * cfg.K

        # --- AG mode setup ---
        x_symm_ag = _alloc_symm((cfg.T_local, cfg.d), cfg.dtype, device)
        x_symm_ag.normal_()

        def ag_call():
            triton_all_gather(x_symm_ag, dist.group.WORLD)

        # --- A2A mode setup ---
        x_symm_a2a = _alloc_symm((cfg.T_local, cfg.d), cfg.dtype, device)
        x_symm_a2a.normal_()
        recv = torch.empty((world_size, TK_local, cfg.d),
                           dtype=cfg.dtype, device=device)

        topk_idx_g = _make_balanced_topk(cfg.T_local, cfg.K, cfg.E,
                                         rank, world_size, device)
        meta = compute_dispatch_metadata(topk_idx_g, my_rank=rank,
                                         E_local=E_local)
        dst_rank_flat = meta["dst_rank_flat"]
        slot_flat_per_rank = meta["slot_flat_per_rank"]

        def a2a_call():
            a2a_dispatch_pull(x_symm_a2a, dst_rank_flat, slot_flat_per_rank,
                              recv, K=cfg.K, group=dist.group.WORLD)

        t_ag = bench_fn(ag_call, warmup=args.warmup, repeat=args.repeat)
        t_a2a = bench_fn(a2a_call, warmup=args.warmup, repeat=args.repeat)

        # Algorithmic per-rank bytes for each mode (see header comment).
        elem = x_symm_ag.element_size()
        bytes_ag = world_size * cfg.T_local * cfg.d * elem
        bytes_a2a = TK_local * cfg.d * elem

        rows.append([
            cfg.name,
            f"{cfg.T_local}", f"{cfg.d}", f"{cfg.K}", f"{cfg.E}",
            f"{t_ag*1e3:.1f}", f"{t_a2a*1e3:.1f}",
            f"{t_ag/t_a2a:.2f}x",
            f"{_gbps(bytes_ag, t_ag):.0f}",
            f"{_gbps(bytes_a2a, t_a2a):.0f}",
        ])

    _print_table(
        rank, f"AG mode vs A2A mode dispatch (W={world_size})",
        ["name", "T_local", "d", "K", "E",
         "AG µs", "A2A µs", "A2A speedup",
         "AG GB/s", "A2A GB/s"],
        rows,
    )
    


# ============================================================================
# Phase: gather_aggregation
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
    CombineCfg(name=f"{m.name}_T{_t_tag(T)}",
               T_local=T, d=m.d, K=m.K, E=m.E)
    for m in _MODELS
    for T in _T_LOCALS
]


# ============================================================================
# Phase: gather_aggregation
# ============================================================================

def phase_gather_aggregation(rank: int, world_size: int,
                             device: torch.device, args) -> None:
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
        s_rev_symm.copy_(torch.randint(0, TK_local, (TK_global,),
                                       device=device, dtype=torch.int32))

        topk_idx_g = _make_balanced_topk(cfg.T_local, cfg.K, cfg.E,
                                         rank, world_size, device)
        meta = compute_dispatch_metadata(topk_idx_g, my_rank=rank,
                                         E_local=E_local)
        rank_2d = meta["my_dst_rank"]
        pos_2d = (
            torch.arange(TK_local, device=device, dtype=torch.int32)
            + rank * TK_local
        ).view(cfg.T_local, cfg.K)

        topk_scores = torch.softmax(
            torch.randn(cfg.T_local, cfg.K, device=device, dtype=torch.float32),
            dim=-1,
        ).to(cfg.dtype)
        out = torch.empty(cfg.T_local, cfg.d, dtype=cfg.dtype, device=device)

        def call():
            gather_aggregation(y_symm, s_rev_symm, rank_2d, pos_2d,
                               topk_scores, out, K=cfg.K,
                               group=dist.group.WORLD)

        t = bench_fn(call, warmup=args.warmup, repeat=args.repeat)

        bytes_moved = cfg.T_local * cfg.K * cfg.d * y_symm.element_size()
        rows.append([
            cfg.name,
            f"{cfg.T_local}", f"{cfg.d}", f"{cfg.K}", f"{cfg.E}",
            f"{t*1e3:.1f}",
            f"{_gbps(bytes_moved, t):.0f}",
        ])

    _print_table(
        rank, f"gather_aggregation (W={world_size})",
        ["name", "T_local", "d", "K", "E", "triton µs", "GB/s"],
        rows,
    )



# ============================================================================
# Phase: rs_aggregation
# ============================================================================

def phase_rs_aggregation(rank: int, world_size: int,
                         device: torch.device, args) -> None:
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

        topk_idx_g = _make_balanced_topk(cfg.T_local, cfg.K, cfg.E,
                                         rank, world_size, device)
        meta = compute_dispatch_metadata(topk_idx_g, my_rank=rank,
                                         E_local=E_local)
        dst_rank_flat = meta["dst_rank_flat"]

        # scores: (W * T_local, K) -> flat (TK_global,)
        scores_2d = torch.softmax(
            torch.randn(world_size * cfg.T_local, cfg.K,
                        device=device, dtype=torch.float32),
            dim=-1,
        ).to(cfg.dtype)
        scores_flat = scores_2d.contiguous().view(-1)

        rs_buf = _alloc_symm((world_size * cfg.T_local, cfg.d),
                             torch.float32, device)

        def call():
            rs_aggregation(y_symm, s_rev_symm, dst_rank_flat,
                           scores_flat, rs_buf, cfg.K, cfg.T_local)

        t = bench_fn(call, warmup=args.warmup, repeat=args.repeat)

        # Per-rank reads: TK_local hits on local y_symm (only the ~1/W of
        # K-slots routed to this rank actually load — see kernel comment
        # on the masked-load is_mine trick). Reported BW is the
        # algorithmic payload (TK_local * d * dtype) over wall time, on the
        # same denominator basis as gather_aggregation so the two columns
        # are directly comparable.
        bytes_moved = cfg.T_local * cfg.K * cfg.d * y_symm.element_size()
        rows.append([
            cfg.name,
            f"{cfg.T_local}", f"{cfg.d}", f"{cfg.K}", f"{cfg.E}",
            f"{t*1e3:.1f}",
            f"{_gbps(bytes_moved, t):.0f}",
        ])

    _print_table(
        rank, f"rs_aggregation (W={world_size})",
        ["name", "T_local", "d", "K", "E", "triton µs", "GB/s"],
        rows,
    )



# ============================================================================
# Phase: gather_aggregation vs rs_aggregation (same payload, same shapes)
# ----------------------------------------------------------------------------
# Both kernels produce the same final result on a per-rank basis, but via
# very different paths:
#
#   gather_aggregation: each output row is built by K NVLink reads from
#     peer y_symm buffers + score-weighted sum in registers + one local
#     store. NVLink-bound. Saturates near the AG ceiling.
#
#   rs_aggregation:    each output row reads K rows from LOCAL y_symm
#     (HBM, no NVLink) + writes one fp32 row to rs_buf. Caller then runs
#     reduce_scatter on rs_buf to sum partials across ranks and scatter.
#     Per-rank work is W× larger (every rank produces a contribution for
#     every (home_rank, home_t) pair), but reads are local and HBM has
#     much more headroom than NVLink.
#
# What this phase reports is the **producer kernel only** — the
# subsequent reduce_scatter is excluded. Bench it separately if you want
# the full story.
#
# Read the time column. The two GB/s columns use the same denominator
# (algorithmic payload TK_local * K * d * dtype) but track different
# physical work, so cross-row comparisons of GB/s aren't strictly
# meaningful. Wall time is the one to compare.
# ============================================================================

def phase_gather_vs_rs_aggregation(rank: int, world_size: int,
                                   device: torch.device, args) -> None:
    rows = []
    for cfg in _COMBINE_CONFIGS:
        if cfg.E % world_size != 0:
            continue
        E_local = cfg.E // world_size
        TK_local = cfg.T_local * cfg.K
        TK_global = world_size * TK_local

        # Shared inputs.
        y_symm = _alloc_symm((TK_global, cfg.d), cfg.dtype, device)
        y_symm.normal_()
        s_rev_symm = _alloc_symm((TK_global,), torch.int32, device)
        s_rev_symm.copy_(torch.arange(TK_global, dtype=torch.int32, device=device))

        topk_idx_g = _make_balanced_topk(cfg.T_local, cfg.K, cfg.E,
                                         rank, world_size, device)
        meta = compute_dispatch_metadata(topk_idx_g, my_rank=rank,
                                         E_local=E_local)
        dst_rank_flat = meta["dst_rank_flat"]

        scores_2d = torch.softmax(
            torch.randn(world_size * cfg.T_local, cfg.K,
                        device=device, dtype=torch.float32),
            dim=-1,
        ).to(cfg.dtype)
        scores_flat = scores_2d.contiguous().view(-1)

        # --- gather_aggregation inputs ---
        rank_2d = meta["my_dst_rank"]
        pos_2d = (
            torch.arange(TK_local, device=device, dtype=torch.int32)
            + rank * TK_local
        ).view(cfg.T_local, cfg.K)
        topk_scores = scores_2d[rank * cfg.T_local : (rank + 1) * cfg.T_local]
        out_gather = torch.empty(cfg.T_local, cfg.d,
                                 dtype=cfg.dtype, device=device)

        def call_gather():
            gather_aggregation(y_symm, s_rev_symm, rank_2d, pos_2d,
                               topk_scores, out_gather, K=cfg.K,
                               group=dist.group.WORLD)

        # --- rs_aggregation inputs ---
        rs_buf = _alloc_symm((world_size * cfg.T_local, cfg.d),
                             torch.float32, device)

        def call_rs():
            rs_aggregation(y_symm, s_rev_symm, dst_rank_flat,
                           scores_flat, rs_buf, cfg.K, cfg.T_local)

        t_gather = bench_fn(call_gather, warmup=args.warmup, repeat=args.repeat)
        t_rs = bench_fn(call_rs, warmup=args.warmup, repeat=args.repeat)

        bytes_moved = cfg.T_local * cfg.K * cfg.d * y_symm.element_size()
        rows.append([
            cfg.name,
            f"{cfg.T_local}", f"{cfg.d}", f"{cfg.K}", f"{cfg.E}",
            f"{t_gather*1e3:.1f}", f"{t_rs*1e3:.1f}",
            f"{t_gather/t_rs:.2f}x",
            f"{_gbps(bytes_moved, t_gather):.0f}",
            f"{_gbps(bytes_moved, t_rs):.0f}",
        ])

    _print_table(
        rank, f"gather_aggregation vs rs_aggregation (W={world_size})",
        ["name", "T_local", "d", "K", "E",
         "gather µs", "rs µs", "rs speedup",
         "gather GB/s", "rs GB/s"],
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
    E2ECfg("olmoe",          T_local=32768, d=2048, n=1024, E=64,  K=8),
    E2ECfg("d2880_e64k4",    T_local=32768, d=2880, n=2880, E=64,  K=4),
    E2ECfg("d2304_e256k8",   T_local=32768, d=2304, n=1024, E=256, K=8),
    E2ECfg("e512_k10",       T_local=32768, d=2048, n=512,  E=512, K=10),
    E2ECfg("d4096_e128k8",   T_local=32768, d=4096, n=1536, E=128, K=8),
    E2ECfg("deepseek_v3",    T_local=16384, d=7168, n=2048, E=256, K=8),
]


def _make_e2e_inputs(cfg: E2ECfg, rank: int, world_size: int,
                     device: torch.device):
    g = torch.Generator(device=device).manual_seed(42)
    H, I, E = cfg.d, cfg.n, cfg.E
    raw_I = 2 * I  # GLU
    E_local = E // world_size

    x = 0.02 * torch.randn(cfg.T_local, H,
                           generator=torch.Generator(device=device).manual_seed(1000+rank),
                           device=device, dtype=cfg.dtype)
    router_w = 0.02 * torch.randn(E, H, generator=g, device=device, dtype=cfg.dtype)
    w1_full = 0.02 * torch.randn(E, raw_I, H, generator=g,
                                 device=device, dtype=cfg.dtype)
    w2_full = 0.02 * torch.randn(E, H, I, generator=g,
                                 device=device, dtype=cfg.dtype)

    w1_local = w1_full[rank * E_local : (rank + 1) * E_local].permute(1, 2, 0)
    w2_local = w2_full[rank * E_local : (rank + 1) * E_local].permute(1, 2, 0)
    return x, router_w, w1_local, w2_local


def phase_e2e(rank: int, world_size: int, device: torch.device, args) -> None:
    rows = []
    for cfg in _E2E_CONFIGS:
        if cfg.E % world_size != 0:
            continue
        x, router_w, w1_local, w2_local = _make_e2e_inputs(
            cfg, rank, world_size, device)

        results = {}
        for mode in ("ag", "a2a"):
            mgr = SymmMemManager(dist.group.WORLD, device)

            def call():
                moe_ep_TC_softmax_topk_forward(
                    x=x, router_w=router_w,
                    w1=w1_local, b1=None,
                    w2=w2_local, b2=None,
                    K=cfg.K, E=cfg.E, mgr=mgr,
                    activation_type=cfg.activation,
                    is_inference_mode_enabled=True,
                    mode=mode,
                )

            t = bench_fn(call, warmup=args.warmup, repeat=args.repeat)
            results[mode] = t
            mgr.clear()

        rows.append([
            cfg.name,
            f"{cfg.T_local}", f"{cfg.d}", f"{cfg.n}", f"{cfg.K}", f"{cfg.E}",
            f"{results['ag']*1e3:.0f}",
            f"{results['a2a']*1e3:.0f}",
            f"{results['ag']/results['a2a']:.2f}x",
        ])

    _print_table(
        rank, f"End-to-end EP forward: AG vs A2A (W={world_size})",
        ["name", "T_local", "d", "n", "K", "E",
         "AG µs", "A2A µs", "A2A speedup"],
        rows,
    )


# ============================================================================
# Phase dispatch
# ============================================================================

_PHASES = {
    "metadata":                 phase_metadata,
    "ag":                       phase_ag,
    "a2a_pull":                 phase_a2a_pull,
    "ag_vs_a2a":                phase_ag_vs_a2a,
    "gather_aggregation":       phase_gather_aggregation,
    "rs_aggregation":           phase_rs_aggregation,
    "gather_vs_rs_aggregation": phase_gather_vs_rs_aggregation,
    "e2e":                      phase_e2e,
}


def run_phases(rank: int, world_size: int, device: torch.device, args) -> None:
    if rank == 0:
        print(f"\nSonicMoE EP benchmark suite (W={world_size}, "
              f"warmup={args.warmup}, repeat={args.repeat})")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--phase", nargs="+", default=["all"],
        help="phases to run: " + ", ".join(_PHASES.keys()) + ", all")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()

    world_size = int(os.environ.get("EP_BENCH_WORLD_SIZE",
                     str(min(torch.cuda.device_count(), 8))))
    if world_size < 2:
        print("EP benchmark requires world_size ≥ 2 GPUs", file=sys.stderr)
        sys.exit(1)
    if world_size > torch.cuda.device_count():
        print(f"EP_BENCH_WORLD_SIZE={world_size} exceeds visible "
              f"GPUs={torch.cuda.device_count()}", file=sys.stderr)
        sys.exit(1)

    mp.spawn(_bench_subprocess, args=(world_size, args),
             nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
