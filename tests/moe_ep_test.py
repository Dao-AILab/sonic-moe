# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
#
# Multi-rank correctness test for the EP forward.
#
# Spawns W ranks, each constructs the same MoE (seeded), shards weights along
# the leading E axis, runs the EP forward on its slice of the global x, all-
# gathers the per-rank outputs, and compares against a single-rank
# PyTorch reference computed on rank 0.
#
# Coverage matrix per shape:
#   - Both public entry points: moe_ep_TC_softmax_topk_forward,
#                               moe_ep_general_routing_forward
#   - All three modes:          {"ag", "a2a", "auto"}
#   - Both bias settings:       {without, with}
#   - Both routing variants:    {softmax_then_topk, topk_then_softmax_norm}
# = 24 EP forward calls per shape.
#
# Shape list is comprehensive: smoke, K∈{1,2,4,8,10}, H/I aspect ratio sweep,
# T not divisible by common block sizes, E∈{32,64,128,256,512}.
#
# Usage:
#     EP_TEST_WORLD_SIZE=4 python tests/test_ep.py
#     EP_TEST_WORLD_SIZE=8 python tests/test_ep.py --master-port 29600
# ********************************************************************************

from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from sonicmoe import MoE
from sonicmoe.enums import ActivationType

# Private helper — used to print which mode "auto" resolves to.
# Adjust this import to where your cleaned-up ep.py lives.
from sonicmoe.ep import _select_dispatch_mode  # type: ignore
from sonicmoe.ep import SymmMemManager, moe_ep_general_routing_forward, moe_ep_TC_softmax_topk_forward


# ============================================================================
# Comprehensive shape list.
#
# Constraints honored across all shapes:
#   - T divisible by both 4 and 8 (the two world sizes we typically run).
#   - E divisible by both 4 and 8.
#
# Coverage axes:
#   K   ∈ {1, 2, 4, 8, 10}        — edges, K<W, K=W, K>W, non-pow-2
#   H,I ∈ varied                  — H<I, H=I, H>I, large-I (FFN expansion)
#   T   varied + a non-pow-2      — exercises tile-tail paths
#   E   ∈ {32, 64, 128, 256, 512} — small/medium/large expert counts
# ============================================================================


@dataclass
class Shape:
    name: str
    T: int
    H: int
    I: int
    E: int
    K: int


SHAPES: List[Shape] = [
    # Smoke / baseline
    Shape("smoke_K4_E32", T=2048, H=1024, I=512, E=32, K=4),
    # K edge cases
    Shape("K_eq_1", T=2048, H=1024, I=512, E=32, K=1),
    Shape("K_eq_2", T=2048, H=2048, I=1024, E=32, K=2),
    Shape("K_eq_4", T=2048, H=2048, I=1024, E=32, K=4),
    Shape("K_eq_8", T=4096, H=2048, I=1024, E=64, K=8),
    Shape("K_eq_10", T=2048, H=2048, I=512, E=64, K=10),
    # H / I aspect ratios
    Shape("H_gt_I", T=2048, H=4096, I=1024, E=64, K=8),
    Shape("H_eq_I", T=2048, H=1024, I=1024, E=32, K=4),
    Shape("H_lt_I_4x", T=2048, H=1024, I=4096, E=32, K=4),
    # T non-divisible by common pow-2 block sizes
    Shape("T_nondiv", T=3072, H=1024, I=512, E=32, K=4),
    # Large E
    Shape("E_eq_128", T=4096, H=4096, I=1024, E=128, K=8),
    Shape("E_eq_256", T=2048, H=2304, I=512, E=256, K=8),
    Shape("E_eq_512", T=2048, H=2048, I=512, E=512, K=10),
]


ROUTING_VARIANTS: List[Tuple[str, bool, bool]] = [
    # (name, is_softmax_over_topk, norm_topk_probs)
    ("softmax_then_topk", True, False),
    ("topk_then_softmax_norm", False, True),
]


# ============================================================================
# Reference — a single-rank, per-expert PyTorch MoE forward in fp32.
# ============================================================================


def _swiglu(h: torch.Tensor, concat_layout: bool = False) -> torch.Tensor:
    if concat_layout:
        g, u = torch.chunk(h, 2, dim=-1)
    else:
        u, g = h[..., 1::2], h[..., ::2]
    return u * F.silu(g)


def _routing_reference(
    x: torch.Tensor,
    router_w: torch.Tensor,
    K: int,
    is_softmax_over_topk: bool,
    norm_topk_probs: bool,
):
    logits = F.linear(x.float(), router_w.float())
    if is_softmax_over_topk:
        topk_logits, topk_idx = logits.topk(K, dim=-1)
        topk_scores = topk_logits.softmax(dim=-1, dtype=torch.float32)
    else:
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        topk_scores, topk_idx = probs.topk(K, dim=-1)
        if norm_topk_probs:
            topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
    return topk_scores, topk_idx


def _per_expert_reference(
    x: torch.Tensor,
    w1: torch.Tensor,  # (E, 2I, H) for SwiGLU
    w2: torch.Tensor,  # (E, H, I)
    b1: Optional[torch.Tensor],  # (E, 2I) or None
    b2: Optional[torch.Tensor],  # (E, H) or None
    topk_idx: torch.Tensor,  # (T, K)
    topk_scores: torch.Tensor,  # (T, K) fp32
    concat_layout: bool,
) -> torch.Tensor:
    T, H = x.shape
    E = w1.shape[0]
    out = torch.zeros(T, H, dtype=torch.float32, device=x.device)
    for i in range(E):
        rows_t, rows_k = (topk_idx == i).nonzero(as_tuple=True)
        if rows_t.numel() == 0:
            continue
        h = F.linear(
            x[rows_t].float(),
            w1[i].float(),
            bias=(b1[i].float() if b1 is not None else None),
        )
        h = _swiglu(h, concat_layout=concat_layout)
        y = F.linear(
            h,
            w2[i].float(),
            bias=(b2[i].float() if b2 is not None else None),
        )
        out[rows_t] += y * topk_scores[rows_t, rows_k, None]
    return out


# ============================================================================
# Distributed plumbing
# ============================================================================


def _setup(rank: int, world_size: int, master_port: int) -> torch.device:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    return torch.device(f"cuda:{rank}")


def _teardown():
    try:
        dist.barrier()
        dist.destroy_process_group()
    except Exception:
        pass


def _all_gather_y(y_local: torch.Tensor, world_size: int) -> torch.Tensor:
    T_local, H = y_local.shape
    out = torch.empty(world_size * T_local, H, dtype=y_local.dtype, device=y_local.device)
    dist.all_gather_into_tensor(out, y_local.contiguous())
    return out


def _check(
    tag: str, y_full: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float, log_prefix: str
) -> Tuple[bool, str]:
    diff = (y_full.float() - ref).abs()
    max_diff = diff.max().item()
    mean_rel = (diff / (ref.abs() + 1e-6)).mean().item()
    head = f"{log_prefix}{tag}"
    metrics = f"max={max_diff:.2e} rel={mean_rel:.2e}"
    try:
        torch.testing.assert_close(y_full.float(), ref, atol=atol, rtol=rtol)
        return True, f"{head:<70s} {metrics}  ✓ PASS"
    except AssertionError:
        return False, (f"{head:<70s} {metrics}  ✗ FAIL " f"(atol={atol} rtol={rtol})")


# ============================================================================
# Per-shape test driver
# ============================================================================


@dataclass
class ShapeStats:
    shape_name: str
    pass_count: int = 0
    fail_count: int = 0
    failures: list = field(default_factory=list)


def _run_one_shape(
    rank: int,
    world_size: int,
    device: torch.device,
    shape: Shape,
    dtype: torch.dtype,
    concat_layout: bool,
    atol: float,
    rtol: float,
    seed: int,
) -> ShapeStats:
    T, H, I, E, K = shape.T, shape.H, shape.I, shape.E, shape.K
    if T % world_size != 0:
        if rank == 0:
            print(f"[skip {shape.name}] T={T} not divisible by W={world_size}")
        return ShapeStats(shape.name)
    if E % world_size != 0:
        if rank == 0:
            print(f"[skip {shape.name}] E={E} not divisible by W={world_size}")
        return ShapeStats(shape.name)

    T_local = T // world_size
    E_local = E // world_size
    e_slc = slice(rank * E_local, (rank + 1) * E_local)
    stats = ShapeStats(shape.name)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Construct MoE WITH biases on every rank (seeded → identical across
    # ranks). For the bias=False pass we just pass b1=b2=None to both EP
    # and reference — saves one full MoE construction per shape.
    moe = (
        MoE(
            num_experts=E,
            num_experts_per_tok=K,
            hidden_size=H,
            intermediate_size=I,
            activation_function=ActivationType.SWIGLU,
            add_bias=True,
            std=0.02,
        )
        .to(dtype=dtype)
        .to(device)
    )
    torch.nn.init.normal_(moe.c_fc.bias, 0, 0.01)
    torch.nn.init.normal_(moe.c_proj.bias, 0, 0.01)

    # Belt-and-suspenders broadcast in case of any non-deterministic init.
    for p in moe.parameters():
        dist.broadcast(p.data, src=0)

    w1_full = moe.c_fc.weight  # (E, 2I, H)
    w2_full = moe.c_proj.weight  # (E, H, I)
    b1_with = moe.c_fc.bias  # (E, 2I)
    b2_with = moe.c_proj.bias  # (E, H)
    router_w = moe.router.weight  # (E, H)

    w1_local = w1_full[e_slc].transpose(1, 2).contiguous().permute(2, 1, 0)
    w2_local = w2_full[e_slc].permute(1, 2, 0).contiguous()
    b1_local_with = b1_with[e_slc].contiguous()
    b2_local_with = b2_with[e_slc].contiguous()

    if rank == 0:
        x_global = 0.2 * torch.randn(T, H, device=device, dtype=dtype)
    else:
        x_global = torch.empty(T, H, device=device, dtype=dtype)
    dist.broadcast(x_global, src=0)
    x_local = x_global[rank * T_local : (rank + 1) * T_local].contiguous()

    mgr = SymmMemManager(dist.group.WORLD, device)

    # Pre-broadcast a fixed routing decision used by general_routing_forward.
    if rank == 0:
        with torch.no_grad():
            scores_g, idx_g = F.linear(x_global, router_w).topk(K, dim=-1)
            scores_g = scores_g.softmax(dim=-1, dtype=torch.float32).to(dtype)
            idx_g = idx_g.to(torch.int64)
    else:
        scores_g = torch.empty(T, K, device=device, dtype=dtype)
        idx_g = torch.empty(T, K, device=device, dtype=torch.int64)
    dist.broadcast(scores_g, src=0)
    dist.broadcast(idx_g, src=0)
    scores_local = scores_g[rank * T_local : (rank + 1) * T_local].contiguous()
    idx_local = idx_g[rank * T_local : (rank + 1) * T_local].contiguous()

    # ------------------------------------------------------------------------
    # Sweep: bias × routing × mode × entry point.
    # ------------------------------------------------------------------------
    for use_bias in (False, True):
        b1_local = b1_local_with if use_bias else None
        b2_local = b2_local_with if use_bias else None
        b1_full = b1_with if use_bias else None
        b2_full = b2_with if use_bias else None
        log_prefix = f"[W={world_size} {shape.name} bias={int(use_bias)}] "

        # -------- entry point #1: TC_softmax_topk_forward --------
        for variant_name, is_softmax_over_topk, norm_topk_probs in ROUTING_VARIANTS:
            ref_topk_scores, ref_topk_idx = _routing_reference(
                x_global,
                router_w,
                K,
                is_softmax_over_topk=is_softmax_over_topk,
                norm_topk_probs=norm_topk_probs,
            )
            if rank == 0:
                ref = _per_expert_reference(
                    x_global,
                    w1_full,
                    w2_full,
                    b1_full,
                    b2_full,
                    topk_idx=ref_topk_idx,
                    topk_scores=ref_topk_scores,
                    concat_layout=concat_layout,
                )

            for mode in ("ag", "a2a", "auto"):
                resolved = _select_dispatch_mode(world_size, K) if mode == "auto" else mode
                y_local = moe_ep_TC_softmax_topk_forward(
                    x_local,
                    router_w,
                    w1_local,
                    b1_local,
                    w2_local,
                    b2_local,
                    K=K,
                    E=E,
                    mgr=mgr,
                    activation_type=ActivationType.SWIGLU,
                    is_inference_mode_enabled=True,
                    is_softmax_over_topk=is_softmax_over_topk,
                    norm_topk_probs=norm_topk_probs,
                    concat_layout=concat_layout,
                    mode=mode,
                )
                y_full = _all_gather_y(y_local, world_size)
                if rank == 0:
                    tag = f"TC[{variant_name},mode={mode}"
                    if mode == "auto":
                        tag += f"->{resolved}"
                    tag += "]"
                    ok, msg = _check(tag, y_full, ref, atol, rtol, log_prefix)
                    print(msg)
                    if ok:
                        stats.pass_count += 1
                    else:
                        stats.fail_count += 1
                        stats.failures.append(tag)
                dist.barrier()

        # -------- entry point #2: general_routing_forward --------
        if rank == 0:
            ref_general = _per_expert_reference(
                x_global,
                w1_full,
                w2_full,
                b1_full,
                b2_full,
                topk_idx=idx_g,
                topk_scores=scores_g.float(),
                concat_layout=concat_layout,
            )

        for mode in ("ag", "a2a", "auto"):
            resolved = _select_dispatch_mode(world_size, K) if mode == "auto" else mode
            y_local = moe_ep_general_routing_forward(
                x_local,
                idx_local,
                scores_local,
                w1_local,
                b1_local,
                w2_local,
                b2_local,
                E=E,
                mgr=mgr,
                activation_type=ActivationType.SWIGLU,
                is_inference_mode_enabled=True,
                concat_layout=concat_layout,
                mode=mode,
            )
            y_full = _all_gather_y(y_local, world_size)
            if rank == 0:
                tag = f"general[mode={mode}"
                if mode == "auto":
                    tag += f"->{resolved}"
                tag += "]"
                ok, msg = _check(tag, y_full, ref_general, atol, rtol, log_prefix)
                print(msg)
                if ok:
                    stats.pass_count += 1
                else:
                    stats.fail_count += 1
                    stats.failures.append(tag)
            dist.barrier()

    return stats


# ============================================================================
# Per-rank entry point
# ============================================================================


def _run_rank(
    rank: int, world_size: int, master_port: int, concat_layout: bool, atol: float, rtol: float, queue
) -> None:
    try:
        device = _setup(rank, world_size, master_port)
        all_stats: List[ShapeStats] = []
        for shape in SHAPES:
            try:
                stats = _run_one_shape(
                    rank,
                    world_size,
                    device,
                    shape,
                    dtype=torch.bfloat16,
                    concat_layout=concat_layout,
                    atol=atol,
                    rtol=rtol,
                    seed=1111,
                )
                all_stats.append(stats)
            except Exception as e:
                if rank == 0:
                    print(f"[ERR {shape.name}] {e}")
                    traceback.print_exc()
                stats = ShapeStats(shape.name)
                stats.fail_count = 1
                stats.failures.append(f"exception: {e}")
                all_stats.append(stats)
            torch.cuda.empty_cache()
        if rank == 0:
            queue.put(all_stats)
        _teardown()
    except Exception:
        if rank == 0:
            traceback.print_exc()
            queue.put([])
        _teardown()
        raise


# ============================================================================
# Driver
# ============================================================================


def _print_summary(all_stats: List[ShapeStats]) -> bool:
    total_pass = sum(s.pass_count for s in all_stats)
    total_fail = sum(s.fail_count for s in all_stats)
    print("\n=== Summary ===")
    name_w = max((len(s.shape_name) for s in all_stats), default=10)
    for s in all_stats:
        marker = "✓" if s.fail_count == 0 else "✗"
        print(f"  {marker} {s.shape_name:<{name_w}}  " f"pass={s.pass_count}  fail={s.fail_count}")
        for f in s.failures:
            print(f"      - {f}")
    print(f"\nTotal: pass={total_pass}  fail={total_fail}")
    return total_fail == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world-size",
        type=int,
        default=int(os.environ.get("EP_TEST_WORLD_SIZE", torch.cuda.device_count())),
    )
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument(
        "--concat-layout", action="store_true", help="Test the concat [g; u] up-proj layout instead " "of interleaved."
    )
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    args = parser.parse_args()

    if args.world_size < 2:
        print(f"SKIP: EP test needs world_size >= 2 (got {args.world_size}).")
        return 0
    if torch.cuda.device_count() < args.world_size:
        print(f"SKIP: need {args.world_size} GPUs, found " f"{torch.cuda.device_count()}.")
        return 0

    print(
        f"\nEP correctness test (W={args.world_size}, "
        f"concat_layout={args.concat_layout}, "
        f"shapes={len(SHAPES)})\n"
    )

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = []
    for rank in range(args.world_size):
        p = ctx.Process(
            target=_run_rank,
            args=(rank, args.world_size, args.master_port, args.concat_layout, args.atol, args.rtol, queue),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    bad_exit = [p.exitcode for p in procs if p.exitcode not in (0, None)]
    all_stats: List[ShapeStats] = []
    while not queue.empty():
        all_stats.extend(queue.get())

    if bad_exit:
        print(f"\nFAIL: ranks exited with {bad_exit}")
        return 1

    if not _print_summary(all_stats):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
