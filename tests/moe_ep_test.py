# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
#
# Multi-rank correctness test for the EP forward.
#
# Each rank constructs the same MoE (seeded), shards weights along the leading
# E axis, runs the EP forward on its slice of the global x, all-gathers the
# per-rank outputs, and compares against a single-rank PyTorch reference
# computed on rank 0.
#
# Usage (torchrun-launched):
#
#   torchrun --nproc_per_node=4 --standalone --local-ranks-filter 0 \
#            tests/moe_ep_test.py
#   torchrun --nproc_per_node=8 --standalone --local-ranks-filter 0 \
#            tests/moe_ep_test.py --concat-layout
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
import torch.nn.functional as F

from sonicmoe import MoE
from sonicmoe.distributed_utils import CombineMode, DispatchMode, RuntimeEPConfig  # type: ignore
from sonicmoe.enums import ActivationType
from sonicmoe.functional.ep import moe_ep_general_routing_forward, moe_ep_TC_softmax_topk_forward


@dataclass
class Shape:
    name: str
    T: int
    H: int
    I: int
    E: int
    K: int


SHAPES: List[Shape] = [
    Shape("K_eq_2", T=4096, H=2048, I=1024, E=32, K=2),
    Shape("K_eq_8", T=4096, H=2048, I=1024, E=64, K=8),
    Shape("K_eq_10", T=4096, H=2048, I=512, E=64, K=10),
]


ROUTING_VARIANTS: List[Tuple[str, bool, bool]] = [
    # (name, is_softmax_over_topk, norm_topk_probs)
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
# Distributed plumbing — torchrun-driven. dist is initialized once in main();
# everything else is plain rank-aware code.
# ============================================================================


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
    assert T % world_size == 0, f"T ({T}) must be divisible by world_size ({world_size})."
    assert E % world_size == 0, f"E ({E}) must be divisible by world_size ({world_size})."

    T_local = T // world_size
    E_local = E // world_size
    e_slc = slice(rank * E_local, (rank + 1) * E_local)
    stats = ShapeStats(shape.name)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

    # EP-sharded weights (per-rank expert slice). Layout matches the benchmark
    # in benchmarks/distributed/moe-ep.py:
    #   w1: (E_local, 2I, H) → permute(1, 2, 0) → (2I, H, E_local) view,
    #       strides (H, 1, 2I·H). Non-contiguous on purpose; the GEMM kernel
    #       requires the middle dim to have stride 1.
    #   w2: (E_local, H, I)  → permute(0, 2, 1).contiguous() → (E_local, I, H)
    #       contig.
    w1_local = w1_full[e_slc].permute(1, 2, 0)
    w2_local = w2_full[e_slc].permute(0, 2, 1).contiguous()
    b1_local_with = b1_with[e_slc].contiguous()
    b2_local_with = b2_with[e_slc].contiguous()

    if rank == 0:
        x_global = 0.2 * torch.randn(T, H, device=device, dtype=dtype)
    else:
        x_global = torch.empty(T, H, device=device, dtype=dtype)
    dist.broadcast(x_global, src=0)
    x_local = x_global[rank * T_local : (rank + 1) * T_local].contiguous()

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

            # Mode sweep: all three dispatch primitives. The final
            # iteration of the combine_mode loop below also exercises the
            # ep_config= path by wrapping a DispatchMode in a
            # RuntimeEPConfig.
            for dispatch_mode in (
                DispatchMode.AG_DISPATCH_TRITON,
                DispatchMode.A2A_DISPATCH_TRITON,
                DispatchMode.RANK_DEDUP_DISPATCH_TRITON,
            ):
                y_local = moe_ep_TC_softmax_topk_forward(
                    x_local,
                    router_w,
                    w1_local,
                    b1_local,
                    w2_local,
                    b2_local,
                    K=K,
                    E=E,
                    activation_type=ActivationType.SWIGLU,
                    is_inference_mode_enabled=True,
                    is_softmax_over_topk=is_softmax_over_topk,
                    norm_topk_probs=norm_topk_probs,
                    concat_layout=concat_layout,
                    ep_config=RuntimeEPConfig(dispatch_mode=dispatch_mode, W=world_size, K=K),
                )
                y_full = _all_gather_y(y_local, world_size)
                if rank == 0:
                    tag = f"TC[{variant_name},dispatch={dispatch_mode.value}]"
                    ok, msg = _check(tag, y_full, ref, atol, rtol, log_prefix)
                    print(msg)
                    if ok:
                        stats.pass_count += 1
                    else:
                        stats.fail_count += 1
                        stats.failures.append(tag)
                dist.barrier()

            # ep_config= round-trip: sweep both combine modes.
            # Delivers the dispatch decision via RuntimeEPConfig so the
            # validate path is exercised, and exercises both
            # A2A_COMBINE_TRITON (the fused kernel) and RS_COMBINE_TRITON
            # (the producer + reduce-scatter pipeline) end-to-end. The
            # two paths produce numerically equivalent outputs (modulo
            # fp32 vs bf16 accumulation order); the same atol/rtol
            # bounds apply.
            for combine_mode in (
                CombineMode.A2A_COMBINE_TRITON,
                CombineMode.RS_COMBINE_TRITON,
                CombineMode.RANK_DEDUP_COMBINE_TRITON,
            ):
                cfg = RuntimeEPConfig(
                    dispatch_mode=DispatchMode.AG_DISPATCH_TRITON,
                    W=world_size,
                    K=K,
                    combine_mode=combine_mode,
                )
                y_local = moe_ep_TC_softmax_topk_forward(
                    x_local,
                    router_w,
                    w1_local,
                    b1_local,
                    w2_local,
                    b2_local,
                    K=K,
                    E=E,
                    activation_type=ActivationType.SWIGLU,
                    is_inference_mode_enabled=True,
                    is_softmax_over_topk=is_softmax_over_topk,
                    norm_topk_probs=norm_topk_probs,
                    concat_layout=concat_layout,
                    ep_config=cfg,
                )
                y_full = _all_gather_y(y_local, world_size)
                if rank == 0:
                    tag = f"TC[{variant_name},dispatch={cfg.dispatch_mode.value},combine={cfg.combine_mode.value}]"
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

        for dispatch_mode in (DispatchMode.AG_DISPATCH_TRITON, DispatchMode.A2A_DISPATCH_TRITON):
            y_local = moe_ep_general_routing_forward(
                x_local,
                idx_local,
                scores_local,
                w1_local,
                b1_local,
                w2_local,
                b2_local,
                E=E,
                activation_type=ActivationType.SWIGLU,
                is_inference_mode_enabled=True,
                concat_layout=concat_layout,
                ep_config=RuntimeEPConfig(dispatch_mode=dispatch_mode, W=world_size, K=K),
            )
            y_full = _all_gather_y(y_local, world_size)
            if rank == 0:
                tag = f"general[dispatch={dispatch_mode.value}]"
                ok, msg = _check(tag, y_full, ref_general, atol, rtol, log_prefix)
                print(msg)
                if ok:
                    stats.pass_count += 1
                else:
                    stats.fail_count += 1
                    stats.failures.append(tag)
            dist.barrier()

        # ep_config= round-trip for general_routing_forward.
        # Sweep both combine modes (see TC entry above).
        for combine_mode in (
            CombineMode.A2A_COMBINE_TRITON,
            CombineMode.RS_COMBINE_TRITON,
            CombineMode.RANK_DEDUP_COMBINE_TRITON,
        ):
            cfg = RuntimeEPConfig(
                dispatch_mode=DispatchMode.AG_DISPATCH_TRITON,
                W=world_size,
                K=K,
                combine_mode=combine_mode,
            )
            y_local = moe_ep_general_routing_forward(
                x_local,
                idx_local,
                scores_local,
                w1_local,
                b1_local,
                w2_local,
                b2_local,
                E=E,
                activation_type=ActivationType.SWIGLU,
                is_inference_mode_enabled=True,
                concat_layout=concat_layout,
                ep_config=cfg,
            )
            y_full = _all_gather_y(y_local, world_size)
            if rank == 0:
                tag = f"general[ep_config=dispatch={cfg.dispatch_mode.value},combine={cfg.combine_mode.value}]"
                ok, msg = _check(tag, y_full, ref_general, atol, rtol, log_prefix)
                print(msg)
                if ok:
                    stats.pass_count += 1
                else:
                    stats.fail_count += 1
                    stats.failures.append(tag)
            dist.barrier()
        dist.barrier()

    return stats


def _print_summary(all_stats: List[ShapeStats]) -> bool:
    total_pass = sum(s.pass_count for s in all_stats)
    total_fail = sum(s.fail_count for s in all_stats)
    print("\n=== Summary ===")
    name_w = max((len(s.shape_name) for s in all_stats), default=10)
    for s in all_stats:
        marker = "✓" if s.fail_count == 0 else "✗"
        print(f"  {marker} {s.shape_name:<{name_w}}  pass={s.pass_count}  fail={s.fail_count}")
        for f in s.failures:
            print(f"      - {f}")
    print(f"\nTotal: pass={total_pass}  fail={total_fail}")
    return total_fail == 0


def _under_torchrun() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--concat-layout",
        action="store_true",
        help="Test the concat [g; u] up-proj layout instead of interleaved.",
    )
    args = parser.parse_args()

    if not _under_torchrun():
        print(
            "ERROR: this test must be launched with torchrun, e.g.:\n"
            "  torchrun --nproc_per_node=8 --standalone tests/moe_ep_test.py",
            file=sys.stderr,
        )
        return 2

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if world_size < 2:
        if rank == 0:
            print(f"SKIP: EP test needs world_size >= 2 (got {world_size}).")
        return 0
    if local_rank >= torch.cuda.device_count():
        print(
            f"[r{rank}] ERROR: LOCAL_RANK={local_rank} but only " f"{torch.cuda.device_count()} CUDA devices visible",
            file=sys.stderr,
        )
        return 2

    # Match the benchmark's numerics: disable TF32 so the fp32 reference path
    # (F.linear in _per_expert_reference) is genuinely fp32 and not silently
    # downcast on Ampere+.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(
            f"\nEP correctness test (W={world_size}, "
            f"concat_layout={args.concat_layout}, "
            f"shapes={len(SHAPES)})\n"
        )

    all_stats: List[ShapeStats] = []
    try:
        for shape in SHAPES:
            try:
                stats = _run_one_shape(
                    rank,
                    world_size,
                    device,
                    shape,
                    dtype=torch.bfloat16,
                    concat_layout=args.concat_layout,
                    atol=5e-2,
                    rtol=5e-2,
                    seed=1111,
                )
            except Exception as e:
                if rank == 0:
                    print(f"[ERR {shape.name}] {e}")
                    traceback.print_exc()
                stats = ShapeStats(shape.name)
                stats.fail_count = 1
                stats.failures.append(f"exception: {e}")
            all_stats.append(stats)
            torch.cuda.empty_cache()
    finally:
        # Decide pass/fail on rank 0, then broadcast so every rank exits the
        # same way. Without this, rank 0 might exit 1 while peers exit 0 and
        # torchrun's exit code becomes ambiguous.
        if rank == 0:
            success = _print_summary(all_stats)
            success_t = torch.tensor([1 if success else 0], device=device, dtype=torch.int32)
        else:
            success_t = torch.zeros(1, device=device, dtype=torch.int32)
        dist.broadcast(success_t, src=0)
        success = bool(success_t.item())

        try:
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            pass

    return 0 if success else 1


if __name__ == "__main__":
    # Hard-exit via os._exit to bypass the Python destructor chain on
    # paths where ``clear_ep_cache``'s atexit hook may not get to run
    # ahead of ``~CUDASymmetricMemory → cuMemUnmap`` (e.g. a test
    # exception that propagates past atexit ordering). Same pattern as
    # ``benchmarks/distributed/moe-ep.py`` and ``tests/distributed/
    # collectives_test.py``.
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
