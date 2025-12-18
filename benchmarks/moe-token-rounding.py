# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import argparse
import random
from typing import Tuple, Type

import cutlass
import torch
import torch.nn.functional as F
from rich import print as print0
from tqdm.auto import tqdm

from sonicmoe import MoE
from sonicmoe.functional import count_cumsum, moe_general_routing_inputs
from triton.testing import do_bench


@torch.autocast(device_type="cuda", dtype=torch.float32)
def ref_moe_token_rounding(
    x: torch.Tensor,
    router_scores_selected: torch.Tensor,
    selected_T: torch.Tensor,
    selected_E: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    E,
):
    T, D = x.shape  # # B, L, # total expert

    ref_o = torch.zeros_like(x, dtype=torch.float32)

    for i in range(E):
        pos = selected_E == i
        T_idx = selected_T[pos]

        if T_idx.numel() > 0:

            w1_out = F.linear(x[T_idx, :], w1[i, :, :].squeeze(), bias=(b1[i] if b1 is not None else None))
            w1_out = F.silu(w1_out[:, ::2]) * w1_out[:, 1::2]

            w2_out = F.linear(w1_out, w2[i, :, :].squeeze(), bias=(b2[i] if b2 is not None else None))

            ref_o[T_idx, :] += w2_out * router_scores_selected[pos, None]

    return ref_o.view(T, D)


def parse_comma_separated_ints(s: str):
    try:
        return tuple([int(x.strip()) for x in s.split(",")])
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid format. Expected comma-separated integers.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Example of SonicMoE (arbitrary routing inputs).")

    parser.add_argument(
        "--thiekq",
        type=parse_comma_separated_ints,
        default=(16384, 4096, 1024, 256, 8, 128),
        help="T, H, I, E, K, tileM dimensions (comma-separated)",
    )
    parser.add_argument(
        "--dtype",
        type=cutlass.dtype,
        default=cutlass.BFloat16,
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--routing",
        type=str,
        choices=["top_k", "nr", "up", "down"],
        default="top_k",
    )
    parser.add_argument(
        "--skip_test",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--add_bias",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    if len(args.thiekq) != 6:
        parser.error("--thiekq must contain exactly 6 values")

    return args


def our_e2e_fwd_bwd_call(x, router_scores_selected, sorted_selected_T, selected_E, w1, b1, w2, b2, E, stream_id, dout):
    o, _ = moe_general_routing_inputs(
        x, router_scores_selected, sorted_selected_T, selected_E, w1, b1, w2, b2, E, stream_id, False
    )
    torch.autograd.grad(o, [x, router_scores_selected, w1, w2], dout, retain_graph=True)
    router_scores_selected.grad = x.grad = w1.grad = w2.grad = None


def our_fwd_call(x, router_scores_selected, sorted_selected_T, selected_E, w1, b1, w2, b2, E, stream_id):
    return moe_general_routing_inputs(
        x, router_scores_selected, sorted_selected_T, selected_E, w1, b1, w2, b2, E, stream_id, False
    )


def forward_token_choice_rounding(
    x: torch.Tensor, router_w: torch.Tensor, E, K, Mtile, routing
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    T, D = x.shape  # # B, L, # total expert
    Mtile = 128

    device = x.device
    dtype = x.dtype

    router_logits = F.linear(x, router_w)
    router_scores = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(dtype)

    # first sorting, similar to TC
    topk_values, topk_indices = router_scores.topk(K, dim=-1)

    expert_freq = count_cumsum(topk_indices.view(-1), E, do_cumsum=True)[0]
    expert_freq_rounded_up = (torch.ceil(expert_freq / Mtile) * Mtile).type(torch.int32)
    expert_freq_rounded_down = expert_freq // Mtile * Mtile

    topk_values /= topk_values.sum(dim=-1, keepdim=True)

    router_scores.scatter_(-1, topk_indices, topk_values)

    router_TC_EC_combined_val = router_scores.detach().clone()
    router_TC_EC_combined_val -= 1  # make sure EC's score is lower than TC & EC keeps the score order
    router_TC_EC_combined_val.scatter_(1, topk_indices, topk_values)  # mask out original TC score

    # second sorting, similar to EC
    topk_indices = router_TC_EC_combined_val.argsort(dim=0, descending=True).int()  # type: ignore

    if routing == "down":
        expert_freq_rounded = expert_freq_rounded_down

    elif routing == "up":
        expert_freq_rounded = expert_freq_rounded_up

    elif routing == "nr":
        expert_freq_rounded = torch.round(expert_freq / Mtile).type(torch.int32) * Mtile

    else:
        raise NotImplementedError()

    expert_freq_mask = torch.arange(T, device=device, dtype=torch.int32)[:, None].expand(-1, E) < expert_freq_rounded[None, :]  # type: ignore

    selected_T = topk_indices[expert_freq_mask]
    selected_E = torch.arange(E, device=device, dtype=torch.int32)[None, :].expand(T, -1)[expert_freq_mask]  # type: ignore

    # implicit assumption: selected_T should be sorted in my reduction code
    selected_T_order = selected_T.argsort().int()
    selected_T = selected_T[selected_T_order]
    selected_E = selected_E[selected_T_order]

    return router_scores[selected_T, selected_E].contiguous(), selected_T, selected_E


def forward_topk(x: torch.Tensor, router_w: torch.Tensor, E, K) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    T = x.shape[0]

    router_logits = F.linear(x, router_w)

    top_logits, topk_indices = router_logits.topk(K, dim=1)
    router_scores = F.softmax(top_logits, dim=-1, dtype=torch.float32)

    # first sorting, similar to TC
    return (
        router_scores.view(-1),
        torch.arange(T, device="cuda", dtype=torch.int32).repeat_interleave(K),
        topk_indices.view(-1).int(),
    )


def run(
    thiekq: Tuple[int, int, int, int, int, int],
    dtype: Type[cutlass.Numeric],
    rep: int,
    routing: str,
    skip_test: Type[bool],
    add_bias: Type[bool],
    **kwargs,
):

    cutlass_dtype = dtype
    torch_dtype = {cutlass.BFloat16: torch.bfloat16, cutlass.Float16: torch.float16}[dtype]

    # Unpack parameters
    T, H, I, E, K, Mtile = thiekq
    TK = T * K
    print(f"T {T}, I {I}, H {H}, E {E}, K {K} | Routing {routing}")

    random.seed(1100)
    torch.manual_seed(1100)
    torch.cuda.manual_seed_all(1100)

    avg_time = torch.zeros(3)
    avg_tflops = torch.zeros(3)

    total_processed_tokens = 0
    total_hardware_tokens = 0

    moe = (
        MoE(
            num_experts=E,
            num_experts_per_tok=K,
            hidden_size=H,
            intermediate_size=I,
            is_glu=True,
            add_bias=add_bias,
            std=0.02,
        )
        .to(dtype=torch_dtype)
        .cuda()
    )
    moe = torch.compile(moe)

    x = 0.2 * torch.randn(T, H, device="cuda:0", dtype=torch_dtype, requires_grad=True)
    dout = 0.2 * torch.randn_like(x, requires_grad=True)

    w1, w2, router_w = moe.c_fc.weight, moe.c_proj.weight, moe.router.weight
    b1, b2 = moe.c_fc.bias, moe.c_proj.bias
    router_w = moe.router.weight

    stream_id = moe.stream_id

    if add_bias:
        torch.nn.init.normal_(b1, 0, 0.01)
        torch.nn.init.normal_(b2, 0, 0.01)

    # # Ref check
    if not skip_test:
        x_clone = x.detach().clone()
        x_clone.requires_grad_(True)
        dout_clone = dout.detach().clone()
        dout_clone.requires_grad_(True)

        if routing == "top_k":
            router_scores_selected, selected_T, selected_E = forward_topk(x, router_w, E, K)
            router_scores_selected_clone, _, _ = forward_topk(x_clone, router_w, E, K)
        else:
            router_scores_selected, selected_T, selected_E = forward_token_choice_rounding(
                x, router_w, E, K, Mtile, routing
            )
            router_scores_selected_clone, _, _ = forward_token_choice_rounding(x_clone, router_w, E, K, Mtile, routing)

        o, expert_frequency = moe_general_routing_inputs(
            x,
            router_scores_selected,
            selected_T,
            selected_E,
            w1.permute(1, 2, 0),
            b1,
            w2.permute(1, 2, 0),
            b2,
            E,
            stream_id,
            False,
        )
        if add_bias:
            dx, dw1, db1, dw2, db2, drouter_w = torch.autograd.grad(
                o, [x, w1, b1, w2, b2, router_w], grad_outputs=dout
            )
        else:
            dx, dw1, dw2, drouter_w = torch.autograd.grad(o, [x, w1, w2, router_w], grad_outputs=dout)

        ref_o = ref_moe_token_rounding(
            x_clone,
            router_scores_selected_clone,
            selected_T,
            selected_E,
            w1,
            b1,
            w2,
            b2,
            E,
        )
        ref_expert_frequency = selected_E.view(-1).bincount(minlength=E)

        torch.testing.assert_close(expert_frequency.int(), ref_expert_frequency.int())

        o_diff = (o.float() - ref_o).abs()

        print(f"max ref o val {ref_o.abs().max():.6f}")
        print(f"mean ref o val {ref_o.abs().mean():.6f}")
        print(f"max abs diff on o {o_diff.max():.6f}")
        print(f"mean rel diff on o {(o_diff / (ref_o.abs() + 1e-6)).mean():.6f}" + "\n")

        if add_bias:
            ref_dx, ref_dw1, ref_db1, ref_dw2, ref_db2, ref_drouter_w = torch.autograd.grad(
                ref_o, [x_clone, w1, b1, w2, b2, router_w], grad_outputs=dout_clone
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
                ref_o, [x_clone, w1, w2, router_w], grad_outputs=dout_clone
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

    TRIALS = 50
    for _ in tqdm(range(TRIALS)):
        torch.nn.init.normal_(w1, 0.0, 0.02)
        torch.nn.init.normal_(w2, 0.0, 0.02)

        if add_bias:
            torch.nn.init.normal_(b1, 0, 0.01)
            torch.nn.init.normal_(b2, 0, 0.01)

        x = torch.randn(T, H, device="cuda:0", dtype=torch_dtype, requires_grad=True)
        dout = 0.2 * torch.randn_like(x, requires_grad=True)

        if routing == "top_k":
            router_scores_selected, selected_T, selected_E = forward_topk(x, router_w.detach(), E, K)
        else:
            router_scores_selected, selected_T, selected_E = forward_token_choice_rounding(
                x, router_w.detach(), E, K, Mtile, routing
            )

        our_e2e_fwd_bwd_call(
            x,
            router_scores_selected,
            selected_T,
            selected_E,
            w1.permute(1, 2, 0),
            b1,
            w2.permute(1, 2, 0),
            b2,
            E,
            stream_id,
            dout,
        )

        TK = router_scores_selected.shape[0]

        forward_time = do_bench(
            lambda: our_fwd_call(
                x,
                router_scores_selected,
                selected_T,
                selected_E,
                w1.permute(1, 2, 0),
                b1,
                w2.permute(1, 2, 0),
                b2,
                E,
                stream_id,
            ),
            warmup=10,
            rep=rep,
        )
        flops = 6 * TK * I * H
        tflops = flops / (forward_time / 1e3) / 1e12

        avg_time[0] += forward_time
        avg_tflops[0] += tflops

        flops = 18 * TK * I * H
        e2e_time = do_bench(
            lambda: our_e2e_fwd_bwd_call(
                x,
                router_scores_selected,
                selected_T,
                selected_E,
                w1.permute(1, 2, 0),
                b1,
                w2.permute(1, 2, 0),
                b2,
                E,
                stream_id,
                dout,
            ),
            warmup=10,
            rep=rep,
            grad_to_none=[x, w1, w2, router_w, dout],
        )
        tflops = flops / (e2e_time / 1e3) / 1e12

        avg_time[1] += e2e_time
        avg_tflops[1] += tflops

        flops = 12 * TK * I * H
        bwd_time = e2e_time - forward_time
        tflops = flops / (bwd_time / 1e3) / 1e12

        avg_time[2] += bwd_time
        avg_tflops[2] += tflops

        expert_frequency = torch.bincount(selected_E).int()
        total_processed_tokens += expert_frequency.sum().item()
        total_hardware_tokens += (torch.ceil(expert_frequency / Mtile).to(torch.int32) * Mtile).sum().sum().item()

    avg_time /= TRIALS
    avg_tflops /= TRIALS

    print0(f"[bold green][/bold green] {routing}, Fwd Average time: {avg_time[0]:.3f} ms, TFLOPS: {avg_tflops[0]:.1f}")
    print0(f"[bold green][/bold green] {routing}, E2E Average time: {avg_time[1]:.3f} ms, TFLOPS: {avg_tflops[1]:.1f}")
    print0(f"[bold green][/bold green] {routing}, Bwd Average time: {avg_time[2]:.3f} ms, TFLOPS: {avg_tflops[2]:.1f}")
    print0(
        f"[bold green][/bold green] {routing}, processed tokens, hardware tokens {total_processed_tokens:.1f}, {total_hardware_tokens:.1f}. wasted ratio {(total_hardware_tokens-total_processed_tokens)/total_processed_tokens:.3f}"
    )


if __name__ == "__main__":
    args = parse_arguments()
    run(args.thiekq, args.dtype, args.rep, args.routing, args.skip_test, args.add_bias)
    print("PASS")
