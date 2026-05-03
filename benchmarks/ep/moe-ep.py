# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
#
# E2E benchmark for the EP forward (moe_ep_TC_softmax_topk_forward)
# with local (non-EP) baselines for exposed network latency estimation.
#
# T in --thiek is the GLOBAL token count; each rank processes T // W tokens.
# Reported EP wall-times are MAX across ranks (slowest rank dictates collective
# throughput). Local baselines run independently per rank and are reported from
# rank 0. TFLOPS use the global token count for EP and per-rank/full token count
# for local, so numbers are directly comparable in the summary table.
#
# Launch with torchrun:
#
#   torchrun --nproc-per-node=8 --standalone \
#       benchmarks/ep/moe-ep.py --thiek 131072,4096,1536,128,8
# ********************************************************************************

import argparse
import itertools
import os
import sys
import time
from functools import partial

# ─────────────── Monkey-patch: reduce SM100 autotuning ───────────────
import quack.gemm_config as _gc
import torch
import torch.distributed as dist
import torch.nn.functional as F
from quack.autotuner import AutotuneConfig
from quack.gemm_config import GemmConfig
from quack.gemm_interface import gemm_dgated_tuned, gemm_gated_tuned, gemm_tuned
from rich import print as print0
from triton.testing import do_bench

from sonicmoe import MoE
from sonicmoe.enums import ActivationType, is_glu

# Private helper — used to print which mode "auto" resolved to.
from sonicmoe.ep import _select_dispatch_mode  # type: ignore
from sonicmoe.ep import SymmMemManager, moe_ep_TC_softmax_topk_forward
from sonicmoe.functional import moe_TC_softmax_topk_layer


def _fast_sm100_configs(epilogue=None):
    tile_n_vals = [128, 160, 192, 256]
    tile_mn_cluster_vals = (
        [(128, tile_n, (1, 2)) for tile_n in tile_n_vals]
        + [(128, tile_n, (2, 1)) for tile_n in tile_n_vals]
        + [(256, tile_n, (2, 1)) for tile_n in tile_n_vals]
        + [(256, 512, (2, 1))]
    )
    swap_ab_vals = [False, True]
    if epilogue in ["lse", "gated"]:
        swap_ab_vals = [False]
    GemmConfigCls = partial(GemmConfig, pingpong=False, device_capacity=10)
    use_clc_vals = [True, False]
    use_tma_gather_vals = [True, False]
    return [
        GemmConfigCls(
            tile_m=m,
            tile_n=n,
            cluster_m=cm,
            cluster_n=cn,
            swap_ab=sab,
            max_swizzle_size=8,
            is_dynamic_persistent=use_clc,
            use_tma_gather=use_tma_gather,
        )
        for (m, n, (cm, cn)), sab, use_clc, use_tma_gather in itertools.product(
            tile_mn_cluster_vals, swap_ab_vals, use_clc_vals, use_tma_gather_vals
        )
    ]


_gc._get_sm100_configs = _fast_sm100_configs


def _patch_autotuner_configs(autotuner_fn):
    all_new = [AutotuneConfig(config=c) for c in _gc.get_all_configs()]
    autotuner_fn.configs = all_new


_patch_autotuner_configs(gemm_tuned)
_patch_autotuner_configs(gemm_gated_tuned)
_patch_autotuner_configs(gemm_dgated_tuned)
gemm_gated_tuned.configs = [AutotuneConfig(config=c) for c in _gc.get_all_configs("gated")]
gemm_dgated_tuned.configs = [AutotuneConfig(config=c) for c in _gc.get_all_configs("gated")]


def swiglu(h: torch.Tensor, concat_layout: bool = False) -> torch.Tensor:
    if concat_layout:
        g, u = torch.chunk(h, 2, dim=-1)
    else:
        u, g = h[..., 1::2], h[..., ::2]
    return u * F.silu(g)


def geglu(h: torch.Tensor, concat_layout: bool = False) -> torch.Tensor:
    if concat_layout:
        g, u = torch.chunk(h, 2, dim=-1)
    else:
        u, g = h[..., 1::2], h[..., ::2]
    return F.gelu(g.float()).to(dtype=g.dtype) * u


def reglu(h: torch.Tensor, concat_layout: bool = False) -> torch.Tensor:
    if concat_layout:
        g, u = torch.chunk(h, 2, dim=-1)
    else:
        u, g = h[..., 1::2], h[..., ::2]
    return (F.relu(g.float()) * u).to(dtype=g.dtype)


def parse_comma_separated_ints(s: str):
    try:
        return tuple([int(x.strip()) for x in s.split(",")])
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid format. Expected comma-separated integers.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EP forward benchmark with local baselines (launch with torchrun).",
    )
    parser.add_argument(
        "--thiek",
        type=parse_comma_separated_ints,
        default=(131072, 4096, 1536, 128, 8),
        help="T,H,I,E,K dimensions (comma-separated). T is GLOBAL tokens.",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--skip_test", action="store_true", default=False)
    parser.add_argument(
        "--activation",
        choices=["swiglu", "geglu", "reglu"],
        default="swiglu",
    )
    parser.add_argument("--add_bias", action="store_true", default=False)
    parser.add_argument(
        "--topk_over_softmax",
        action="store_true",
        default=False,
        help="Use topk(softmax(.)) routing (Qwen3 style) instead of " "softmax(topk(.)).",
    )
    parser.add_argument(
        "--norm_topk_probs",
        action="store_true",
        default=False,
        help="Renormalize topk probs to sum to 1 (only for softmax-then-topk).",
    )
    parser.add_argument(
        "--concat_layout",
        action="store_true",
        default=False,
        help="Use concat [gate; up] weight layout instead of interleaved.",
    )
    parser.add_argument(
        "--mode",
        choices=["ag", "a2a", "auto"],
        default="auto",
        help="Dispatch mode for the EP forward.",
    )
    parser.add_argument(
        "--skip_local_T",
        action="store_true",
        default=False,
        help="Skip local baselines with full T tokens (saves memory / time).",
    )
    args = parser.parse_args()
    if len(args.thiek) != 5:
        parser.error("--thiek must contain exactly 5 values")
    return args


def _require_torchrun_env() -> None:
    needed = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    if not all(v in os.environ for v in needed):
        sys.exit(
            "ERROR: this script must be launched with torchrun. Example:\n"
            "  torchrun --nproc-per-node=4 benchmarks/ep/bench-ep.py "
            "--thiek 32768,2048,1024,64,8"
        )


def _max_across_ranks(value_ms: float, device: torch.device) -> float:
    t = torch.tensor([value_ms], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def _all_gather_y(y_local: torch.Tensor, world_size: int) -> torch.Tensor:
    T_local, H = y_local.shape
    out = torch.empty(world_size * T_local, H, dtype=y_local.dtype, device=y_local.device)
    dist.all_gather_into_tensor(out, y_local.contiguous())
    return out


def do_bench_distributed(fn, warmup=5, rep=100, calls_per_iter=3):
    """Fixed-iteration bench for collective operations."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    local_ms = 0
    for _ in range(calls_per_iter):
        start.record()
        for _ in range(rep):
            fn()
        end.record()
        torch.cuda.synchronize()
        local_ms += start.elapsed_time(end) / rep
    return local_ms / calls_per_iter


def _bench_local_fwd(
    x,
    router_w,
    w1,
    b1,
    w2,
    b2,
    K,
    activation,
    is_softmax_over_topk,
    norm_topk_probs,
    concat_layout,
    is_inference,
    warmup,
    repeats,
):
    """Benchmark local forward (no EP communication)."""

    def fn():
        return moe_TC_softmax_topk_layer(
            x,
            router_w,
            w1,
            b1,
            w2,
            b2,
            K,
            None,
            activation,
            is_inference,
            is_softmax_over_topk=is_softmax_over_topk,
            norm_topk_probs=norm_topk_probs,
            concat_layout=concat_layout,
        )

    # Warmup (also triggers Triton autotune + CuTe compile caches)
    fn()
    torch.cuda.synchronize()
    return do_bench(fn, warmup=warmup, rep=repeats)


def _bench_local_fwd_bwd(
    x,
    router_w,
    w1_param,
    b1_param,
    w2_param,
    b2_param,
    w1_perm,
    w2_perm,
    K,
    activation,
    is_softmax_over_topk,
    norm_topk_probs,
    concat_layout,
    warmup,
    repeats,
):
    """Benchmark local forward + backward (no EP communication).

    w1_param / w2_param are the original nn.Parameter tensors whose .grad
    we zero after each iteration.  w1_perm / w2_perm are the permuted views
    passed to the kernel.
    """
    dout = 0.2 * torch.randn_like(x)

    def fn():
        o, _, _ = moe_TC_softmax_topk_layer(
            x,
            router_w,
            w1_perm,
            b1_param,
            w2_perm,
            b2_param,
            K,
            None,
            activation,
            False,  # training mode for autograd
            is_softmax_over_topk=is_softmax_over_topk,
            norm_topk_probs=norm_topk_probs,
            concat_layout=concat_layout,
        )
        o.backward(dout, retain_graph=True)
        x.grad = None
        w1_param.grad = None
        w2_param.grad = None
        router_w.grad = None
        if b1_param is not None:
            b1_param.grad = None
        if b2_param is not None:
            b2_param.grad = None

    # Warmup (training-mode path may differ from inference-mode)
    fn()
    torch.cuda.synchronize()
    return do_bench(fn, warmup=warmup, rep=repeats)


def run(args: argparse.Namespace) -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    W = world_size = int(os.environ["WORLD_SIZE"])

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}")

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    activation = ActivationType(args.activation)
    is_softmax_over_topk = not args.topk_over_softmax
    concat_layout = args.concat_layout
    norm_topk_probs = args.norm_topk_probs
    add_bias = args.add_bias
    mode = args.mode

    T, H, I, E, K = args.thiek

    assert W >= 2, f"EP world size should be greater than 2."
    assert T % world_size == 0, f"T ({T}) must be divisible by world_size ({world_size})."
    assert E % world_size == 0, f"E ({E}) must be divisible by world_size ({world_size})."

    T_local = T // world_size
    E_local = E // world_size

    resolved_mode = _select_dispatch_mode(world_size, K) if mode == "auto" else mode

    if rank == 0:
        routing_mode = "softmax_over_topk" if is_softmax_over_topk else f"topk_over_softmax (norm={norm_topk_probs})"
        layout_mode = "concat [gate; up]" if concat_layout else "interleaved [g0, u0, g1, u1, ...]"
        mode_str = f"{mode} (resolved → {resolved_mode})" if mode == "auto" else mode
        print0(
            f"[bold]EP forward + local baselines[/bold]  W {world_size}, "
            f"T {T} (T_local {T_local}), H {H}, I {I}, "
            f"E {E} (E_local {E_local}), K {K}, "
            f"dtype {args.dtype}, mode {mode_str}, "
            f"routing: {routing_mode}, w1 layout: {layout_mode}, "
            f"bias: {add_bias}"
        )

    torch.manual_seed(1111)
    torch.cuda.manual_seed_all(1111)

    # Construct same MoE on every rank (seeded → identical weights).
    moe = MoE(
        num_experts=E,
        num_experts_per_tok=K,
        hidden_size=H,
        intermediate_size=I,
        activation_function=activation,
        add_bias=add_bias,
        std=0.02,
    ).to(dtype=torch_dtype, device=device)
    if add_bias:
        torch.nn.init.normal_(moe.c_fc.bias, 0, 0.01)
        torch.nn.init.normal_(moe.c_proj.bias, 0, 0.01)
    for p in moe.parameters():
        dist.broadcast(p.data, src=0)

    w1_full = moe.c_fc.weight  # (E, 2I, H) for GLU activations
    w2_full = moe.c_proj.weight  # (E, H, I)
    b1_full = moe.c_fc.bias if add_bias else None
    b2_full = moe.c_proj.bias if add_bias else None
    router_w = moe.router.weight  # (E, H)

    # EP-sharded weights (per-rank expert slice).
    e_slc = slice(rank * E_local, (rank + 1) * E_local)
    # w1 = w1_full[e_slc].permute(0, 2, 1).contiguous().permute(2, 1, 0)
    # w2 = w2_full[e_slc].permute(0, 2, 1).contiguous()
    w1_view = w1_full[e_slc].permute(1, 2, 0)  # (2I, H, E_local), strides (H, 1, 2I·H)
    w1 = torch.empty_strided(
        w1_view.shape,
        w1_view.stride(),
        dtype=w1_view.dtype,
        device=w1_view.device,
    ).copy_(
        w1_view
    )  # same strides, own storage
    w2 = w2_full[e_slc].permute(0, 2, 1).contiguous()
    b1 = b1_full[e_slc].contiguous() if add_bias else None
    b2 = b2_full[e_slc].contiguous() if add_bias else None

    # Same x_global on all ranks.
    if rank == 0:
        x_global = 0.2 * torch.randn(T, H, device=device, dtype=torch_dtype)
    else:
        x_global = torch.empty(T, H, device=device, dtype=torch_dtype)
    dist.broadcast(x_global, src=0)
    x = x_global[rank * T_local : (rank + 1) * T_local].contiguous()

    mgr = SymmMemManager(dist.group.WORLD, device)

    # Bind common EP kwargs once.
    fwd_kwargs = dict(
        K=K,
        E=E,
        mgr=mgr,
        activation_type=activation,
        is_softmax_over_topk=is_softmax_over_topk,
        norm_topk_probs=norm_topk_probs,
        concat_layout=concat_layout,
        mode=mode,
    )

    # Common local-kernel kwargs.
    local_kwargs = dict(
        K=K,
        activation=activation,
        is_softmax_over_topk=is_softmax_over_topk,
        norm_topk_probs=norm_topk_probs,
        concat_layout=concat_layout,
    )

    # Full-weight views in the layout expected by moe_TC_softmax_topk_layer:
    #   w1: (E, 2I, H) → permute(1,2,0) → (2I, H, E)
    #   w2: (E, H, I)  → permute(1,2,0) → (H, I, E)
    # permute(1,2,0) gives strides (H, 1, 2I*H) — the middle dim has stride 1,
    # which is what QuACK's gated GEMM expects.  Do NOT call .contiguous() here:
    # that would produce strides (H*E, E, 1), breaking the kernel's stride check.
    w1_local_fmt = w1_full.permute(1, 2, 0)
    w2_local_fmt = w2_full.permute(1, 2, 0)

    # ====================================================================
    # Reference correctness check
    # ====================================================================
    if not args.skip_test:
        o_local = moe_ep_TC_softmax_topk_forward(
            x,
            router_w,
            w1,
            b1,
            w2,
            b2,
            is_inference_mode_enabled=True,
            **fwd_kwargs,
        )
        o_global = _all_gather_y(o_local, world_size)

        if rank == 0:
            # Compute logits per-rank to match EP's per-rank F.linear
            logits_chunks = []
            for r in range(world_size):
                chunk = x_global[r * T_local : (r + 1) * T_local]
                logits_chunks.append(F.linear(chunk, router_w))
            logits = torch.cat(logits_chunks, dim=0)
            # sometimes a direct F.linear(x_global, router_w) would yield inconsistent results

            if is_softmax_over_topk:
                topk_logits, topk_idx = logits.topk(K, dim=-1)
                topk_scores = topk_logits.softmax(dim=-1, dtype=torch.float32)
            else:
                probs = logits.softmax(dim=-1, dtype=torch.float32)
                topk_scores, topk_idx = probs.topk(K, dim=-1)
                if norm_topk_probs:
                    topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)

            act_func = {
                ActivationType.SWIGLU: swiglu,
                ActivationType.GEGLU: geglu,
                ActivationType.REGLU: reglu,
            }[activation]

            with torch.autocast(f"cuda:{local_rank}", torch.float32):
                ref_o_global = torch.zeros_like(x_global)
                for i in range(E):
                    rows_t, rows_k = (topk_idx == i).nonzero(as_tuple=True)
                    if rows_t.numel() > 0:
                        ref_h = F.linear(
                            x_global[rows_t],
                            w1_full[i],
                            bias=(b1_full[i] if add_bias else None),
                        )
                        ref_h = act_func(ref_h, concat_layout=concat_layout) if is_glu(activation) else act_func(ref_h)
                        ref_y = F.linear(
                            ref_h,
                            w2_full[i],
                            bias=(b2_full[i] if add_bias else None),
                        )
                        ref_o_global[rows_t] += ref_y * topk_scores[rows_t, rows_k, None]

                o_diff = (o_global.float() - ref_o_global).abs()
                print(f"max ref o val {ref_o_global.abs().max():.6f}")
                print(f"mean ref o val {ref_o_global.abs().mean():.6f}")
                print(f"max abs diff on o {o_diff.max():.6f}")
                print(f"mean rel diff on o {(o_diff / (ref_o_global.abs() + 1e-6)).mean():.6f}" + "\n")

    dist.barrier()

    # ====================================================================
    # FLOP counters
    # ====================================================================
    # Forward FLOPs (global).
    fwd_flops_global = (6 if is_glu(activation) else 4) * T * I * H * K
    fwd_flops_local = fwd_flops_global / world_size  # per-rank

    fwdbwd_flops_global = 3 * fwd_flops_global
    fwdbwd_flops_local = fwdbwd_flops_global / world_size

    # Backward-only FLOPs: fwdbwd − fwd(training).
    bwd_flops_global = fwdbwd_flops_global - fwd_flops_global
    bwd_flops_local = bwd_flops_global / world_size

    repeats = 100
    warmup = 5

    time.sleep(0.5)

    # ====================================================================
    # EP forward warmup (both inference & training mode code paths)
    # ====================================================================
    moe_ep_TC_softmax_topk_forward(
        x,
        router_w,
        w1,
        b1,
        w2,
        b2,
        is_inference_mode_enabled=True,
        **fwd_kwargs,
    )
    moe_ep_TC_softmax_topk_forward(
        x,
        router_w,
        w1,
        b1,
        w2,
        b2,
        is_inference_mode_enabled=False,
        **fwd_kwargs,
    )
    torch.cuda.synchronize()
    dist.barrier()

    # ====================================================================
    # 1) EP Fwd — inference mode (no graph)
    # ====================================================================
    time.sleep(0.5)
    torch.cuda.synchronize()
    dist.barrier()

    def ep_fwd_inference():
        return moe_ep_TC_softmax_topk_forward(
            x,
            router_w,
            w1,
            b1,
            w2,
            b2,
            is_inference_mode_enabled=True,
            **fwd_kwargs,
        )

    ep_fwd_inf_ms = do_bench_distributed(ep_fwd_inference, warmup=warmup, rep=repeats)
    if rank == 0:
        tflops = fwd_flops_global / (ep_fwd_inf_ms * 1e9)
        local_tflops = fwd_flops_local / (ep_fwd_inf_ms * 1e9)
        print0(
            f" EP Fwd (inference mode) Average time: {ep_fwd_inf_ms:.3f} ms, "
            f"Local TFLOPS: {local_tflops:.1f}, Net TFLOPS: {tflops:.1f}"
        )

    # ====================================================================
    # 2) EP Fwd — training mode (no graph)
    # ====================================================================
    time.sleep(0.5)
    torch.cuda.synchronize()
    dist.barrier()

    def ep_fwd_training():
        return moe_ep_TC_softmax_topk_forward(
            x,
            router_w,
            w1,
            b1,
            w2,
            b2,
            is_inference_mode_enabled=False,
            **fwd_kwargs,
        )

    ep_fwd_train_ms = do_bench_distributed(ep_fwd_training, warmup=warmup, rep=repeats)
    if rank == 0:
        tflops = fwd_flops_global / (ep_fwd_train_ms * 1e9)
        local_tflops = fwd_flops_local / (ep_fwd_train_ms * 1e9)
        print0(
            f" EP Fwd (training mode)  Average time: {ep_fwd_train_ms:.3f} ms, "
            f"Local TFLOPS: {local_tflops:.1f}, Net TFLOPS: {tflops:.1f}"
        )

    # ====================================================================
    # 3) Local baselines — T_local tokens (per-rank compute, no comms)
    #    FLOPs match EP per-rank: each rank routes T_local tokens through
    #    all E experts with top-K, giving the same T_local*K token-expert
    #    pairs and identical per-rank FLOP count.
    # ====================================================================
    if rank == 0:
        print0(f"\n[bold]── Local baselines (T_local={T_local}, no communication) ──[/bold]")

    # x already has T_local tokens; make a detached copy for fwd-only,
    # and a requires_grad copy for fwd+bwd.
    x_local_nograd = x.clone().detach()
    x_local_grad = x.clone().detach().requires_grad_(True)

    time.sleep(0.5)
    torch.cuda.synchronize()
    dist.barrier()

    # -- Local fwd, T_local, inference mode --
    local_fwd_inf_Tl_ms = _bench_local_fwd(
        x_local_nograd,
        router_w,
        w1_local_fmt,
        b1_full,
        w2_local_fmt,
        b2_full,
        is_inference=True,
        warmup=warmup,
        repeats=repeats,
        **local_kwargs,
    )
    if rank == 0:
        tflops = fwd_flops_local / (local_fwd_inf_Tl_ms * 1e9)
        print0(
            f" Local Fwd (T_local={T_local}, inference) "
            f"Average time: {local_fwd_inf_Tl_ms:.3f} ms, TFLOPS: {tflops:.1f}"
        )

    # -- Local fwd, T_local, training mode --
    local_fwd_train_Tl_ms = _bench_local_fwd(
        x_local_nograd,
        router_w,
        w1_local_fmt,
        b1_full,
        w2_local_fmt,
        b2_full,
        is_inference=False,
        warmup=warmup,
        repeats=repeats,
        **local_kwargs,
    )
    if rank == 0:
        tflops = fwd_flops_local / (local_fwd_train_Tl_ms * 1e9)
        print0(
            f" Local Fwd (T_local={T_local}, training)  "
            f"Average time: {local_fwd_train_Tl_ms:.3f} ms, TFLOPS: {tflops:.1f}"
        )

    # -- Local fwd+bwd, T_local --
    time.sleep(0.5)
    torch.cuda.synchronize()
    dist.barrier()

    local_fwdbwd_Tl_ms = _bench_local_fwd_bwd(
        x_local_grad,
        router_w,
        w1_full,
        b1_full,
        w2_full,
        b2_full,
        w1_local_fmt,
        w2_local_fmt,
        warmup=warmup,
        repeats=repeats,
        **local_kwargs,
    )
    local_bwd_Tl_ms = local_fwdbwd_Tl_ms - local_fwd_train_Tl_ms
    if rank == 0:
        bwd_tflops = bwd_flops_local / (local_bwd_Tl_ms * 1e9) if local_bwd_Tl_ms > 0 else 0.0
        print0(
            f" Local Bwd (T_local={T_local})    " f"Average time: {local_bwd_Tl_ms:.3f} ms, TFLOPS: {bwd_tflops:.1f}"
        )

    # ====================================================================
    # 4) Local baselines — full T tokens (single-GPU full-scale reference)
    # ====================================================================
    if not args.skip_local_T:
        if rank == 0:
            print0(f"\n[bold]── Local baselines (T={T}, single-GPU full-scale) ──[/bold]")

        x_full_nograd = x_global.clone().detach()
        x_full_grad = x_global.clone().detach().requires_grad_(True)

        time.sleep(0.5)
        torch.cuda.synchronize()
        dist.barrier()

        # -- Local fwd, T, inference mode --
        local_fwd_inf_T_ms = _bench_local_fwd(
            x_full_nograd,
            router_w,
            w1_local_fmt,
            b1_full,
            w2_local_fmt,
            b2_full,
            is_inference=True,
            warmup=warmup,
            repeats=repeats,
            **local_kwargs,
        )
        if rank == 0:
            tflops = fwd_flops_global / (local_fwd_inf_T_ms * 1e9)
            print0(
                f" Local Fwd (T={T}, inference)     "
                f"Average time: {local_fwd_inf_T_ms:.3f} ms, TFLOPS: {tflops:.1f}"
            )

        # -- Local fwd, T, training mode --
        local_fwd_train_T_ms = _bench_local_fwd(
            x_full_nograd,
            router_w,
            w1_local_fmt,
            b1_full,
            w2_local_fmt,
            b2_full,
            is_inference=False,
            warmup=warmup,
            repeats=repeats,
            **local_kwargs,
        )
        if rank == 0:
            tflops = fwd_flops_global / (local_fwd_train_T_ms * 1e9)
            print0(
                f" Local Fwd (T={T}, training)      "
                f"Average time: {local_fwd_train_T_ms:.3f} ms, TFLOPS: {tflops:.1f}"
            )

        # -- Local fwd+bwd, T --
        time.sleep(0.5)
        torch.cuda.synchronize()
        dist.barrier()

        local_fwdbwd_T_ms = _bench_local_fwd_bwd(
            x_full_grad,
            router_w,
            w1_full,
            b1_full,
            w2_full,
            b2_full,
            w1_local_fmt,
            w2_local_fmt,
            warmup=warmup,
            repeats=repeats,
            **local_kwargs,
        )
        local_bwd_T_ms = local_fwdbwd_T_ms - local_fwd_train_T_ms
        if rank == 0:
            fwdbwd_tflops = fwdbwd_flops_global / (local_fwdbwd_T_ms * 1e9)
            bwd_tflops = bwd_flops_global / (local_bwd_T_ms * 1e9) if local_bwd_T_ms > 0 else 0.0
            print0(f" Local Bwd (T={T})        " f"Average time: {local_bwd_T_ms:.3f} ms, TFLOPS: {bwd_tflops:.1f}")

    # ====================================================================
    # 5) Exposed network latency summary
    # ====================================================================
    if rank == 0:
        print0("\n[bold]══ Exposed network latency (EP vs. local T_local) ══[/bold]")

        # Inference mode comparison
        exposed_inf_ms = ep_fwd_inf_ms - local_fwd_inf_Tl_ms
        exposed_inf_pct = exposed_inf_ms / ep_fwd_inf_ms * 100 if ep_fwd_inf_ms > 0 else 0.0
        print0(
            f"  Inference mode:\n"
            f"    EP Fwd:                {ep_fwd_inf_ms:8.3f} ms\n"
            f"    Local Fwd (T_local):   {local_fwd_inf_Tl_ms:8.3f} ms\n"
            f"    Exposed network:       {exposed_inf_ms:8.3f} ms  "
            f"({exposed_inf_pct:.1f}% of EP time)"
        )

        # Training mode comparison
        exposed_train_ms = ep_fwd_train_ms - local_fwd_train_Tl_ms
        exposed_train_pct = exposed_train_ms / ep_fwd_train_ms * 100 if ep_fwd_train_ms > 0 else 0.0
        print0(
            f"  Training mode:\n"
            f"    EP Fwd:                {ep_fwd_train_ms:8.3f} ms\n"
            f"    Local Fwd (T_local):   {local_fwd_train_Tl_ms:8.3f} ms\n"
            f"    Exposed network:       {exposed_train_ms:8.3f} ms  "
            f"({exposed_train_pct:.1f}% of EP time)"
        )

        # EP scaling efficiency vs single-GPU full T
        if not args.skip_local_T:
            speedup = local_fwd_inf_T_ms / ep_fwd_inf_ms if ep_fwd_inf_ms > 0 else 0.0
            ideal_speedup = world_size
            scaling_eff = speedup / ideal_speedup * 100 if ideal_speedup > 0 else 0.0
            print0(
                f"\n  EP scaling efficiency (inference fwd):\n"
                f"    Single-GPU (T={T}):    {local_fwd_inf_T_ms:8.3f} ms\n"
                f"    EP W={world_size} (T={T}):        {ep_fwd_inf_ms:8.3f} ms\n"
                f"    Speedup:               {speedup:.2f}× "
                f"(ideal {ideal_speedup}×, efficiency {scaling_eff:.1f}%)"
            )


def main() -> int:
    _require_torchrun_env()
    args = parse_arguments()
    try:
        run(args)
    finally:
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()

    if int(os.environ.get("RANK", 0)) == 0:
        print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
