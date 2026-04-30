# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
#
# E2E benchmark for the EP forward (moe_ep_TC_softmax_topk_forward).
# Single-config-per-invocation, mirroring benchmarks/bench-moe.py:
#
#   1. (optional) per-expert fp32 reference check vs the gathered EP output
#   2. inference-mode forward, with CUDA graph capture
#   3. inference-mode forward, no graph
#   4. training-mode forward, no graph
#
# T in --thiek is the GLOBAL token count; each rank processes T // W tokens.
# Reported wall-times are MAX across ranks (slowest rank dictates collective
# throughput). TFLOPS use the global token count, so the number is total
# system throughput.
#
# Launch with torchrun (RANK / LOCAL_RANK / WORLD_SIZE / MASTER_PORT come
# from torchrun's env vars):
#
#   CUDA_VISIBLE_DEVICES=0,1,2,7 torchrun --nproc-per-node=4 \
#       benchmarks/ep/bench-ep.py --thiek 32768,2048,1024,64,8
#
#   torchrun --nproc-per-node=8 --master-port=29600 \
#       benchmarks/ep/bench-ep.py --thiek 32768,4096,1536,128,8
# ********************************************************************************

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from rich import print as print0
from triton.testing import do_bench

from sonicmoe import MoE
from sonicmoe.enums import ActivationType, is_glu

# Private helper — used to print which mode "auto" resolved to.
from sonicmoe.ep import _select_dispatch_mode  # type: ignore
from sonicmoe.ep import SymmMemManager, moe_ep_TC_softmax_topk_forward


# ============================================================================
# Activation function helpers (copied from bench-moe.py for the ref check)
# ============================================================================


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


def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x.float()).to(dtype=x.dtype)


def relu(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x)


def relu_sq(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x) ** 2


def silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def parse_comma_separated_ints(s: str):
    try:
        return tuple([int(x.strip()) for x in s.split(",")])
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid format. Expected comma-separated integers.")


# ============================================================================
# CLI
# ============================================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EP forward benchmark (launch with torchrun).",
    )
    parser.add_argument(
        "--thiek",
        type=parse_comma_separated_ints,
        default=(32768, 4096, 1024, 128, 8),
        help="T,H,I,E,K dimensions (comma-separated). T is GLOBAL tokens.",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--skip_test", action="store_true", default=False)
    parser.add_argument(
        "--activation",
        choices=["swiglu", "geglu", "reglu", "relu_sq", "relu", "silu", "gelu"],
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
        help="Dispatch mode for the EP forward. 'auto' picks per the " "device-capability heuristic in sonicmoe.ep.",
    )
    args = parser.parse_args()
    if len(args.thiek) != 5:
        parser.error("--thiek must contain exactly 5 values")
    return args


# ============================================================================
# Distributed plumbing
# ============================================================================


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


# ============================================================================
# Main run logic (executed once per rank under torchrun)
# ============================================================================


def run(args: argparse.Namespace) -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

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
    if T % world_size != 0:
        raise ValueError(f"T ({T}) must be divisible by world_size ({world_size}).")
    if E % world_size != 0:
        raise ValueError(f"E ({E}) must be divisible by world_size ({world_size}).")
    T_local = T // world_size
    E_local = E // world_size

    resolved_mode = _select_dispatch_mode(world_size, K) if mode == "auto" else mode

    if rank == 0:
        routing_mode = "softmax_over_topk" if is_softmax_over_topk else f"topk_over_softmax (norm={norm_topk_probs})"
        layout_mode = "concat [gate; up]" if concat_layout else "interleaved [g0, u0, g1, u1, ...]"
        mode_str = f"{mode} (resolved → {resolved_mode})" if mode == "auto" else mode
        print0(
            f"[bold]EP forward[/bold]  W {world_size}, "
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
    # Belt-and-suspenders broadcast in case of any non-deterministic init.
    for p in moe.parameters():
        dist.broadcast(p.data, src=0)

    w1_full = moe.c_fc.weight  # (E, 2I, H) for SwiGLU
    w2_full = moe.c_proj.weight  # (E, H, I)
    b1_full = moe.c_fc.bias if add_bias else None
    b2_full = moe.c_proj.bias if add_bias else None
    router_w = moe.router.weight  # (E, H)

    e_slc = slice(rank * E_local, (rank + 1) * E_local)
    w1 = w1_full[e_slc].permute(2, 0, 1).contiguous().permute(2, 0, 1)
    w2 = w2_full[e_slc].contiguous()
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

    # Bind common kwargs once.
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

    # ------------------------------------------------------------------------
    # Reference correctness check.
    # ------------------------------------------------------------------------
    if not args.skip_test:
        y_local = moe_ep_TC_softmax_topk_forward(
            x,
            router_w,
            w1,
            b1,
            w2,
            b2,
            is_inference_mode_enabled=True,
            **fwd_kwargs,
        )
        y_full = _all_gather_y(y_local, world_size)

        if rank == 0:
            with torch.no_grad():
                logits = F.linear(x_global.float(), router_w.float())
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
                    ActivationType.GELU: gelu,
                    ActivationType.RELU: relu,
                    ActivationType.SILU: silu,
                    ActivationType.RELU_SQ: relu_sq,
                }[activation]

                ref_o = torch.zeros(T, H, dtype=torch.float32, device=device)
                for i in range(E):
                    rows_t, rows_k = (topk_idx == i).nonzero(as_tuple=True)
                    if rows_t.numel() == 0:
                        continue
                    h = F.linear(
                        x_global[rows_t].float(),
                        w1_full[i].float(),
                        bias=(b1_full[i].float() if add_bias else None),
                    )
                    h = act_func(h, concat_layout=concat_layout) if is_glu(activation) else act_func(h)
                    y = F.linear(
                        h,
                        w2_full[i].float(),
                        bias=(b2_full[i].float() if add_bias else None),
                    )
                    ref_o[rows_t] += y * topk_scores[rows_t, rows_k, None]

                o_diff = (y_full.float() - ref_o).abs()
                print(f"max ref o val {ref_o.abs().max():.6f}")
                print(f"mean ref o val {ref_o.abs().mean():.6f}")
                print(f"max abs diff on o {o_diff.max():.6f}")
                print(f"mean rel diff on o " f"{(o_diff / (ref_o.abs() + 1e-6)).mean():.6f}\n")
        dist.barrier()

    # ------------------------------------------------------------------------
    # Throughput counters.
    # ------------------------------------------------------------------------
    if is_glu(activation):
        flops = 6 * T * I * H * K
    else:
        flops = 4 * T * I * H * K

    repeats = 100
    warmup = 5

    time.sleep(0.5)

    # Compile-cache + autotune warmup. Both inference and training mode hit
    # different code paths in the up-projection, so warm both.
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

    # ------------------------------------------------------------------------
    # Inference mode, forward only, with CUDA graph.
    # ------------------------------------------------------------------------
    try:
        cuda_graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            with torch.cuda.graph(cuda_graph, stream=capture_stream):
                _o_graph = moe_ep_TC_softmax_topk_forward(
                    x,
                    router_w,
                    w1,
                    b1,
                    w2,
                    b2,
                    is_inference_mode_enabled=True,
                    **fwd_kwargs,
                )
        torch.cuda.synchronize()
        dist.barrier()
        fwd_graph_ms = do_bench(lambda: cuda_graph.replay(), warmup=warmup, rep=repeats)
        fwd_graph_ms = _max_across_ranks(fwd_graph_ms, device)
        if rank == 0:
            tflops = flops / (fwd_graph_ms * 1e9)
            print0(
                f" EP Fwd (inference mode + cudagraph) Average time: " f"{fwd_graph_ms:.3f} ms, TFLOPS: {tflops:.1f}"
            )
    except Exception as e:
        if rank == 0:
            print0(
                f"[yellow][warn][/yellow] cudagraph capture failed "
                f"({type(e).__name__}: {e}); skipping graph timing."
            )
        # Drain any partial NCCL state across ranks before continuing.
        dist.barrier()

    # ------------------------------------------------------------------------
    # Inference mode, forward only, no graph.
    # ------------------------------------------------------------------------
    time.sleep(0.5)
    torch.cuda.synchronize()
    dist.barrier()

    def forward_only_inference_mode():
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

    fwd_inf_ms = do_bench(forward_only_inference_mode, warmup=warmup, rep=repeats)
    fwd_inf_ms = _max_across_ranks(fwd_inf_ms, device)
    if rank == 0:
        tflops = flops / (fwd_inf_ms * 1e9)
        print0(f" EP Fwd (inference mode) Average time: " f"{fwd_inf_ms:.3f} ms, TFLOPS: {tflops:.1f}")

    # ------------------------------------------------------------------------
    # Training mode, forward only, no graph.
    # ------------------------------------------------------------------------
    time.sleep(0.5)
    torch.cuda.synchronize()
    dist.barrier()

    def forward_only_training_mode():
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

    fwd_train_ms = do_bench(forward_only_training_mode, warmup=warmup, rep=repeats)
    fwd_train_ms = _max_across_ranks(fwd_train_ms, device)
    if rank == 0:
        tflops = flops / (fwd_train_ms * 1e9)
        print0(f" EP Fwd (training mode) Average time: " f"{fwd_train_ms:.3f} ms, TFLOPS: {tflops:.1f}")


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
