import argparse
import random
import time

import torch
import torch.nn.functional as F
from rich import print as print0
from triton.testing import do_bench
from unsloth.kernels.moe.grouped_gemm.interface import grouped_gemm
from unsloth.kernels.moe.grouped_gemm.reference.moe_ops import get_routing_indices

from sonicmoe import MoE
from sonicmoe.enums import ActivationType, is_glu


@torch.compile()
def swiglu(h):
    g, u = torch.chunk(h, 2, dim=-1)
    return u * F.silu(g)


@torch.compile()
def _route_softmax_over_topk(x, router_w, top_k: int):
    """Fastest routing: topk(logits) → softmax over K values (vs softmax over E)."""
    logits = F.linear(x, router_w)  # (T, E)
    topk_logits, topk_experts = logits.topk(top_k, dim=-1)  # (T, K)
    topk_scores = topk_logits.softmax(dim=-1, dtype=torch.float32).to(x.dtype)
    return logits, topk_scores, topk_experts


@torch.compile()
def _combine(h, topk_scores, T: int, K: int, H: int):
    return (h.view(T, K, H) * topk_scores[..., None]).sum(dim=1)  # (T, H)


def moe_unsloth_layer(x, router_w, w1, w2, top_k):
    """
    x        : (T, H)
    router_w : (E, H)
    w1       : (E, 2*I, H) for GLU, (E, I, H) otherwise   — NO permute
    w2       : (E, H, I)                                   — NO permute
    """
    T, H = x.shape
    E = router_w.shape[0]
    K = top_k

    # Router
    logits, topk_scores, topk_experts = _route_softmax_over_topk(x, router_w, K)

    m_sizes, gather_idx = get_routing_indices(topk_experts, E)

    # First grouped GEMM: gate_up (permute_x fused in prologue)
    h = grouped_gemm(
        X=x,
        W=w1,
        m_sizes=m_sizes,
        gather_indices=gather_idx,
        topk=K,
        permute_x=True,
        permute_y=False,
        autotune=True,
        is_first_gemm=True,
    )

    # Activation (compiled; uses concat chunk(2, -1) for GLU variants)
    h = swiglu(h)

    # Second grouped GEMM: down (permute_y fused in epilogue → token order)
    h = grouped_gemm(
        X=h,
        W=w2,
        m_sizes=m_sizes,
        gather_indices=gather_idx,
        topk=K,
        permute_x=False,
        permute_y=True,
        autotune=True,
        is_first_gemm=False,
    )

    out = _combine(h, topk_scores, T, K, H)

    return out, logits


def parse_comma_separated_ints(s):
    try:
        return tuple([int(x.strip()) for x in s.split(",")])
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid format. Expected comma-separated integers.")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Unsloth MoE grouped_gemm benchmark (torch.compile + fastest paths).")
    parser.add_argument(
        "--thiek",
        type=parse_comma_separated_ints,
        default=(32768, 4096, 1024, 128, 8),
        help="T, H, I, E, K dimensions (comma-separated)",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    args = parser.parse_args()
    if len(args.thiek) != 5:
        parser.error("--thiek must contain exactly 5 values")
    return args


def run(thiek, dtype):
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype]

    T, H, I, E, K = thiek
    print(
        f"T {T}, I {I}, H {H}, E {E}, K {K}, "
        f"routing: softmax_over_topk [fastest], "
        f"w1 layout: concat [gate; up] [fastest]"
    )

    random.seed(1111)
    torch.manual_seed(1111)
    torch.cuda.manual_seed_all(1111)

    moe = (
        MoE(
            num_experts=E,
            num_experts_per_tok=K,
            hidden_size=H,
            intermediate_size=I,
            activation_function=ActivationType("swiglu"),
            add_bias=False,
            std=0.02,
        )
        .to(dtype=torch_dtype)
        .cuda()
    )

    x = 0.2 * torch.randn(T, H, device="cuda:0", dtype=torch_dtype)
    x.requires_grad_(True)
    w1, w2, router_w = moe.c_fc.weight, moe.c_proj.weight, moe.router.weight
    dout = 0.2 * torch.randn_like(x)

    o, _ = moe_unsloth_layer(x, router_w, w1, w2, moe.top_k)
    o.backward(dout)
    x.grad = w1.grad = w2.grad = router_w.grad = None

    fwd_flops = 6 * T * I * H * K
    repeats, warmup_iters = 500, 5

    def fwd_training():
        o, _ = moe_unsloth_layer(x, router_w, w1, w2, moe.top_k)
        return o

    time.sleep(0.5)
    torch.cuda.synchronize()

    fwd_train = do_bench(fwd_training, warmup=warmup_iters, rep=repeats)
    print0(
        f" Unsloth Fwd (training mode)         Average time: {fwd_train:.3f} ms, "
        f"TFLOPS: {fwd_flops / (fwd_train * 1e9):.1f}"
    )

    e2e_flops = 18 * T * I * H * K
    dout = torch.randn_like(x)

    def fwd_bwd():
        o, _ = moe_unsloth_layer(x, router_w, w1, w2, moe.top_k)
        o.backward(dout)
        x.grad = w1.grad = w2.grad = router_w.grad = None

    time.sleep(0.5)
    torch.cuda.synchronize()

    e2e = do_bench(fwd_bwd, warmup=warmup_iters, rep=repeats)
    print0(
        f"[bold green][/bold green] Unsloth Fwd + Bwd Average time: {e2e:.3f} ms, "
        f"TFLOPS: {e2e_flops / (e2e * 1e9):.1f}"
    )

    bwd_flops = 12 * T * I * H * K
    bwd_time = e2e - fwd_train
    print0(
        f"[bold green][/bold green] Unsloth Bwd Average time: {bwd_time:.3f} ms, "
        f"TFLOPS: {bwd_flops / (bwd_time / 1e3) / 1e12:.1f}"
    )


if __name__ == "__main__":
    args = parse_arguments()
    run(args.thiek, args.dtype)
    print("PASS")
