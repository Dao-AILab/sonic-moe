# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import argparse
import itertools
import random
from functools import partial
from typing import Tuple, Type

import cutlass
import quack.autotuner
import quack.gemm_config as _gc
import torch
import torch.nn.functional as F
from quack.autotuner import AutotuneConfig
from quack.gemm_config import GemmConfig
from quack.gemm_interface import gemm_dgated_tuned, gemm_gated_tuned, gemm_tuned
from rich import print as print0
from tqdm.auto import tqdm
from triton.testing import do_bench

from sonicmoe import MoE
from sonicmoe.enums import ActivationType
from sonicmoe.functional import moe_general_routing_inputs


# ─────────────── Monkey-patch: similar M shapes map to the same cached config during QuACK autotuning ───────────────

M_QUANT = 1024


def _make_quantized_key(self, args, kwargs):
    all_args = {**dict(zip(self.arg_names, args)), **kwargs}
    _args = {k: v for k, v in all_args.items() if k in self.arg_names}
    key = [str(_args[k]) for k in self.keys if k in _args]
    for _, arg in _args.items():
        if isinstance(arg, torch.Tensor):
            s = list(arg.shape)
            # Quantize the M (first) dimension
            if s and s[0] >= M_QUANT:
                s[0] = ((s[0] + M_QUANT - 1) // M_QUANT) * M_QUANT
            key.append(str(tuple(s)))
            key.append(str([x if x in {0, 1} else 2 for x in arg.stride()]))
            key.append(str(arg.dtype))
    return tuple(key)


_orig_call = quack.autotuner.Autotuner.__call__


@torch.compiler.disable
def _patched_call(self, *args, **kwargs):
    if len(self.configs) > 1:
        qkey = _make_quantized_key(self, args, kwargs)
        if qkey in self.cache:
            # Cache hit on quantized key — skip autotuning
            config = self.cache[qkey]
            self.best_config = config
            self.nargs = dict(zip(self.arg_names, args))
            ret = self.fn.__call__(*args, **kwargs, **config.all_kwargs())
            self.nargs = None
            return ret

    # Cache miss — fall through to original autotuning
    ret = _orig_call(self, *args, **kwargs)

    # Store result under quantized key so future similar-M calls hit cache
    if len(self.configs) > 1 and hasattr(self, "best_config"):
        qkey = _make_quantized_key(self, args, kwargs)
        self.cache[qkey] = self.best_config

    return ret


quack.autotuner.Autotuner.__call__ = _patched_call
# ─────────────── Monkey-patch ends ───────────────


# ─────────────── Monkey-patch: reduce SM100 autotuning ───────────────
# !!!!!!!!!! The following code is to accelerate the autotuning process in QuACK and IS REMOVABLE (does not affect correctness) !!!!!!!!!!


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


# Patch the 3 autotuners used in MoE SwiGLU fwd+bwd
_patch_autotuner_configs(gemm_tuned)
_patch_autotuner_configs(gemm_gated_tuned)
_patch_autotuner_configs(gemm_dgated_tuned)

gemm_gated_tuned.configs = [AutotuneConfig(config=c) for c in _gc.get_all_configs("gated")]
gemm_dgated_tuned.configs = [AutotuneConfig(config=c) for c in _gc.get_all_configs("gated")]

# ─────────────── Monkey-patch ends ───────────────


@torch.autocast(device_type="cuda", dtype=torch.float32)
def ref_moe_token_rounding(
    x: torch.Tensor,
    router_scores: torch.Tensor,
    token_indices: torch.Tensor,
    expert_indices: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    E,
    concat_layout: bool = False,
):
    T, D = x.shape  # # B, L, # total expert

    ref_o = torch.zeros_like(x, dtype=torch.float32)

    for i in range(E):
        pos = expert_indices == i
        T_idx = token_indices[pos]

        if T_idx.numel() > 0:

            w1_out = F.linear(x[T_idx, :], w1[i, :, :].squeeze(), bias=(b1[i] if b1 is not None else None))
            if concat_layout:
                g, u = torch.chunk(w1_out, 2, dim=-1)
                w1_out = F.silu(g) * u
            else:
                w1_out = F.silu(w1_out[:, ::2]) * w1_out[:, 1::2]

            w2_out = F.linear(w1_out, w2[i, :, :].squeeze(), bias=(b2[i] if b2 is not None else None))

            ref_o[T_idx, :] += w2_out * router_scores[pos, None]

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
    parser.add_argument(
        "--concat_layout",
        action="store_true",
        default=False,
        help="Use concat [gate; up] weight layout instead of interleaved",
    )
    args = parser.parse_args()

    if len(args.thiekq) != 6:
        parser.error("--thiekq must contain exactly 6 values")

    return args


def our_e2e_fwd_bwd_call(
    x, router_scores, token_indices, expert_indices, w1, b1, w2, b2, E, dout, concat_layout=False
):
    o, _ = moe_general_routing_inputs(
        x,
        router_scores,
        token_indices,
        expert_indices,
        w1,
        b1,
        w2,
        b2,
        E,
        None,
        ActivationType.SWIGLU,
        False,
        concat_layout=concat_layout,
    )
    torch.autograd.grad(o, [x, router_scores, w1, w2], dout, retain_graph=True)
    router_scores.grad = x.grad = w1.grad = w2.grad = None


def our_fwd_call(x, router_scores, token_indices, expert_indices, w1, b1, w2, b2, E, concat_layout=False):
    return moe_general_routing_inputs(
        x,
        router_scores,
        token_indices,
        expert_indices,
        w1,
        b1,
        w2,
        b2,
        E,
        None,
        ActivationType.SWIGLU,
        False,
        concat_layout=concat_layout,
    )


def forward_token_choice_rounding(
    x: torch.Tensor, router_w: torch.Tensor, E, K, Mtile, routing
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    T, D = x.shape  # # B, L, # total expert
    Mtile = 128

    device = x.device

    router_logits = F.linear(x, router_w)
    router_scores = F.softmax(router_logits, dim=-1, dtype=torch.float32)

    # first sorting, similar to TC
    topk_values, topk_indices = router_scores.topk(K, dim=-1)

    expert_freq = torch.bincount(topk_indices.view(-1), minlength=E).int()
    expert_freq_rounded_up = (torch.ceil(expert_freq / Mtile) * Mtile).type(torch.int32)
    expert_freq_rounded_down = expert_freq // Mtile * Mtile

    topk_values /= topk_values.sum(dim=-1, keepdim=True)

    router_TC_EC_combined_val = router_scores.scatter(-1, topk_indices, topk_values).detach()
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

    token_indices = topk_indices[expert_freq_mask]
    expert_indices = torch.arange(E, device=device, dtype=torch.int32)[None, :].expand(T, -1)[expert_freq_mask]  # type: ignore

    # implicit assumption: selected_T should be sorted in my reduction code
    token_indices_order = token_indices.argsort().int()
    token_indices = token_indices[token_indices_order]
    expert_indices = expert_indices[token_indices_order]

    return router_scores[token_indices, expert_indices].contiguous(), token_indices, expert_indices


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
    concat_layout: bool = False,
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
            activation_function=ActivationType.SWIGLU,
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
            router_scores, token_indices, expert_indices = forward_topk(x, router_w, E, K)
            router_scores_clone, _, _ = forward_topk(x_clone, router_w, E, K)
        else:
            router_scores, token_indices, expert_indices = forward_token_choice_rounding(
                x, router_w, E, K, Mtile, routing
            )
            router_scores_clone, _, _ = forward_token_choice_rounding(x_clone, router_w, E, K, Mtile, routing)

        o, expert_frequency = moe_general_routing_inputs(
            x,
            router_scores,
            token_indices,
            expert_indices,
            w1.permute(1, 2, 0),
            b1,
            w2.permute(1, 2, 0),
            b2,
            E,
            None,
            ActivationType.SWIGLU,
            False,
            concat_layout=concat_layout,
        )
        if add_bias:
            dx, dw1, db1, dw2, db2, drouter_w = torch.autograd.grad(
                o, [x, w1, b1, w2, b2, router_w], grad_outputs=dout
            )
        else:
            dx, dw1, dw2, drouter_w = torch.autograd.grad(o, [x, w1, w2, router_w], grad_outputs=dout)

        ref_o = ref_moe_token_rounding(
            x_clone,
            router_scores_clone,
            token_indices,
            expert_indices,
            w1,
            b1,
            w2,
            b2,
            E,
            concat_layout=concat_layout,
        )
        ref_expert_frequency = expert_indices.view(-1).bincount(minlength=E)

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
            router_scores, token_indices, expert_indices = forward_topk(x, router_w.detach(), E, K)
        else:
            router_scores, token_indices, expert_indices = forward_token_choice_rounding(
                x, router_w.detach(), E, K, Mtile, routing
            )

        our_e2e_fwd_bwd_call(
            x,
            router_scores,
            token_indices,
            expert_indices,
            w1.permute(1, 2, 0),
            b1,
            w2.permute(1, 2, 0),
            b2,
            E,
            dout,
            concat_layout=concat_layout,
        )

        TK = router_scores.shape[0]

        forward_time = do_bench(
            lambda: our_fwd_call(
                x,
                router_scores,
                token_indices,
                expert_indices,
                w1.permute(1, 2, 0),
                b1,
                w2.permute(1, 2, 0),
                b2,
                E,
                concat_layout=concat_layout,
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
                router_scores,
                token_indices,
                expert_indices,
                w1.permute(1, 2, 0),
                b1,
                w2.permute(1, 2, 0),
                b2,
                E,
                dout,
                concat_layout=concat_layout,
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

        expert_frequency = torch.bincount(expert_indices).int()
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
    run(
        args.thiekq,
        args.dtype,
        args.rep,
        args.routing,
        args.skip_test,
        args.add_bias,
        concat_layout=args.concat_layout,
    )
    print("PASS")
