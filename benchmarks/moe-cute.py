# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import argparse
import random
import time
from typing import Tuple, Type

import cutlass
import torch
import torch.nn.functional as F
from rich import print as print0
from triton.testing import do_bench

from sonicmoe import MoE
from sonicmoe.enums import ActivationType, is_glu
from sonicmoe.functional import moe_TC_softmax_topk_layer


def swiglu(x: torch.Tensor) -> torch.Tensor:
    u = x[..., 1::2]
    g = x[..., ::2]
    return u * F.silu(g)


def geglu(x: torch.Tensor) -> torch.Tensor:
    u = x[..., 1::2]
    g = x[..., ::2]
    return F.gelu(g.float()).to(dtype=g.dtype) * u


def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x.float()).to(dtype=x.dtype)


def reglu(x: torch.Tensor) -> torch.Tensor:
    u = x[..., 1::2]
    g = x[..., ::2]
    return (F.relu(g.float()) * u).to(dtype=g.dtype)


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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Example of SonicMoE (TC top-K routing).")

    parser.add_argument(
        "--thiek",
        type=parse_comma_separated_ints,
        default=(32768, 4096, 1024, 128, 8),
        help="T, H, I, E, K dimensions (comma-separated)",
    )
    parser.add_argument(
        "--dtype",
        type=cutlass.dtype,
        default=cutlass.BFloat16,
    )
    parser.add_argument(
        "--skip_test",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--activation", choices=["swiglu", "geglu", "reglu", "relu_sq", "relu", "silu", "gelu"], default="swiglu"
    )
    parser.add_argument(
        "--add_bias",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    if len(args.thiek) != 5:
        parser.error("--thiek must contain exactly 5 values")

    return args


def our_e2e_fwd_bwd_call(moe, x, dout):
    o = moe(x)[0]
    w1, w2, router_w = moe.c_fc.weight, moe.c_proj.weight, moe.router.weight
    o.backward(dout, retain_graph=True)
    x.grad = w1.grad = w2.grad = router_w.grad = None


def run(
    thiek: Tuple[int, int, int, int, int],
    dtype: Type[cutlass.Numeric],
    skip_test: Type[bool],
    add_bias: Type[bool],
    activation: Type[str],
    **kwargs,
):
    torch_dtype = {cutlass.BFloat16: torch.bfloat16, cutlass.Float16: torch.float16}[dtype]

    activation = ActivationType(activation)
    # Unpack parameters
    T, H, I, E, K = thiek
    TK = T * K
    print(f"T {T}, I {I}, H {H}, E {E}, K {K}")

    random.seed(1111)
    torch.manual_seed(1111)
    torch.cuda.manual_seed_all(1111)

    # Create and permute tensor A/B/C

    moe = (
        MoE(
            num_experts=E,
            num_experts_per_tok=K,
            hidden_size=H,
            intermediate_size=I,
            activation_function=activation,
            add_bias=add_bias,
            std=0.02,
        )
        .to(dtype=torch_dtype)
        .cuda()
    )

    x = 0.2 * torch.randn(T, H, device="cuda:0", dtype=torch_dtype, requires_grad=True)
    w1, w2, router_w = moe.c_fc.weight, moe.c_proj.weight, moe.router.weight
    b1, b2 = moe.c_fc.bias, moe.c_proj.bias
    if add_bias:
        torch.nn.init.normal_(b1, 0, 0.01)
        torch.nn.init.normal_(b2, 0, 0.01)
    dout = 0.2 * torch.randn_like(x, requires_grad=True)

    # # Ref check
    if not skip_test:
        o, router_logits, expert_frequency = moe_TC_softmax_topk_layer(
            x, router_w, w1.permute(1, 2, 0), b1, w2.permute(1, 2, 0), b2, moe.top_k, moe.stream_id, activation
        )
        if add_bias:
            dx, dw1, db1, dw2, db2, drouter_w = torch.autograd.grad(
                o, [x, w1, b1, w2, b2, router_w], grad_outputs=dout
            )
        else:
            dx, dw1, dw2, drouter_w = torch.autograd.grad(o, [x, w1, w2, router_w], grad_outputs=dout)

        logits = F.linear(x, router_w)
        ref_topk_logits, ref_topk_experts = logits.topk(K, dim=-1)
        ref_topk_scores = ref_topk_logits.softmax(dim=-1, dtype=torch.float32)

        ref_topk_expert_idx, ref_s_scatter_idx = ref_topk_experts.flatten().sort()
        ref_topk_expert_idx, ref_s_scatter_idx = ref_topk_expert_idx.int(), ref_s_scatter_idx.int()

        ref_expert_frequency = ref_topk_expert_idx.view(-1).bincount(minlength=E).int()
        torch.testing.assert_close(expert_frequency.int(), ref_expert_frequency.int())

        act_func = {
            ActivationType.SWIGLU: swiglu,
            ActivationType.GEGLU: geglu,
            ActivationType.REGLU: reglu,
            ActivationType.GELU: gelu,
            ActivationType.RELU: relu,
            ActivationType.SILU: silu,
            ActivationType.RELU_SQ: relu_sq,
        }[activation]

        with torch.autocast("cuda:0", torch.float32):
            ref_o = torch.zeros_like(x)

            for i in range(E):
                T_idx, E_idx = torch.argwhere(ref_topk_experts == i).split(1, dim=1)
                T_idx, E_idx = T_idx.squeeze(-1), E_idx.squeeze(-1)

                if T_idx.numel() > 0:
                    w1_out = F.linear(x[T_idx, :], w1[i, :, :].squeeze(), bias=(b1[i] if add_bias else None))
                    w1_out = act_func(w1_out)

                    w2_out = F.linear(w1_out, w2[i, :, :].squeeze(), bias=(b2[i] if add_bias else None))

                    ref_o[T_idx, :] += w2_out * ref_topk_scores[T_idx, E_idx, None]

            o_diff = (o.float() - ref_o).abs()

            print(f"max ref o val {ref_o.abs().max():.6f}")
            print(f"mean ref o val {ref_o.abs().mean():.6f}")
            print(f"max abs diff on o {o_diff.max():.6f}")
            print(f"mean rel diff on o {(o_diff / (ref_o.abs() + 1e-6)).mean():.6f}" + "\n")

            if add_bias:
                ref_dx, ref_dw1, ref_db1, ref_dw2, ref_db2, ref_drouter_w = torch.autograd.grad(
                    ref_o, [x, w1, b1, w2, b2, router_w], grad_outputs=dout
                )
                test_triple_list = [
                    ("dx", dx, ref_dx),
                    ("dw2", dw2, ref_dw2),
                    ("db2", db2, ref_db2),
                    ("dw1", dw1, ref_dw1),
                    ("db1", db1, ref_db1),
                    ("drouter_w", drouter_w, ref_drouter_w),
                ]
            else:
                ref_dx, ref_dw1, ref_dw2, ref_drouter_w = torch.autograd.grad(
                    ref_o, [x, w1, w2, router_w], grad_outputs=dout
                )
                test_triple_list = [
                    ("dx", dx, ref_dx),
                    ("dw2", dw2, ref_dw2),
                    ("dw1", dw1, ref_dw1),
                    ("drouter_w", drouter_w, ref_drouter_w),
                ]
            for n, our, ref in test_triple_list:
                print(f"max abs ref value {n} {ref.abs().max():.6f}")
                print(f"mean abs ref value {n} {ref.abs().mean():.6f}")
                print(f"max abs diff on {n} {(our - ref).abs().max():.6f}")
                print(f"mean rel diff on {n} {((our - ref).abs() / (ref.abs() + 1e-6)).mean():.6f}" + "\n")

    if is_glu(activation):
        flops = 6 * T * I * H * K
    else:
        flops = 4 * T * I * H * K

    repeats = 500
    warmup = 5

    time.sleep(0.5)

    @torch.compile
    def forward_only(is_inference_mode_enabled):
        o, router_logits, expert_frequency = moe_TC_softmax_topk_layer(
            x,
            router_w,
            w1.permute(1, 2, 0),
            b1,
            w2.permute(1, 2, 0),
            b2,
            moe.top_k,
            moe.stream_id,
            activation,
            is_inference_mode_enabled,
        )
        return o

    fwd_timing = do_bench(lambda: forward_only(False), warmup=warmup, rep=repeats)
    tflops = flops / (fwd_timing * 1e9)  # Convert to TFlops
    print0(f"[bold green][/bold green] Cute-DSL Fwd Average time: {fwd_timing:.3f} ms, TFLOPS: {tflops:.1f}")

    time.sleep(0.5)

    timing = do_bench(lambda: forward_only(True), warmup=warmup, rep=repeats)
    tflops = flops / (timing * 1e9)  # Convert to TFlops
    print0(
        f"[bold green][/bold green] Cute-DSL Fwd, inference mode, Average time: {timing:.3f} ms, TFLOPS: {tflops:.1f}"
    )

    if is_glu(activation):
        flops = 18 * T * I * H * K
    else:
        flops = 12 * T * I * H * K

    time.sleep(0.5)

    @torch.compile
    def forward_and_backward():
        o, router_logits, expert_frequency = moe_TC_softmax_topk_layer(
            x,
            router_w,
            w1.permute(1, 2, 0),
            b1,
            w2.permute(1, 2, 0),
            b2,
            moe.top_k,
            moe.stream_id,
            activation,
            False,
        )
        o.backward(dout, retain_graph=True)
        x.grad = w1.grad = w2.grad = router_w.grad = None

    e2e_timing = do_bench(forward_and_backward, warmup=warmup, rep=repeats, grad_to_none=[x, w1, w2, router_w, dout])
    tflops = flops / (e2e_timing * 1e9)  # Convert to TFlops
    print0(f"[bold green][/bold green] Cute-DSL Fwd + Bwd Average time: {e2e_timing:.3f} ms, TFLOPS: {tflops:.1f}")

    if is_glu(activation):
        flops = 12 * T * I * H * K
    else:
        flops = 8 * T * I * H * K

    bwd_time = e2e_timing - fwd_timing
    tflops = flops / (bwd_time / 1e3) / 1e12
    print0(f"[bold green][/bold green] Cute-DSL Bwd Average time: {bwd_time:.3f} ms, TFLOPS: {tflops:.1f}")


if __name__ == "__main__":
    args = parse_arguments()
    run(args.thiek, args.dtype, args.skip_test, args.add_bias, args.activation)
    print("PASS")
