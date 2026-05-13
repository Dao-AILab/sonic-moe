# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
# E2E benchmark for the EP forward + backward (moe_ep_TC_softmax_topk_forward)
# with local (non-EP) baselines for exposed network latency estimation.
#
# T in --thiek is the GLOBAL token count; each rank processes T // W tokens.
#
# Mode selection: the bench profiles with ``NetworkProfiler`` to pick both
# the dispatch mode and the combine mode at runtime.
#
# Launch with torchrun:
#
#   torchrun --nproc-per-node=8 --standalone \
#       benchmarks/distributed/moe-ep.py --thiek 131072,4096,1536,128,8
# ********************************************************************************

import argparse
import datetime
import itertools
import os
import sys
import time
from functools import partial

import quack.autotuner
import torch


# !!!!!!!! sometimes there is a broken pipe issue when QuACK compile worker > 1 !!!!!!!!
os.environ["QUACK_COMPILE_WORKERS"] = "1"


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
import quack.gemm_config as _gc
import torch.distributed as dist
import torch.nn.functional as F
from quack.autotuner import AutotuneConfig
from quack.gemm_config import GemmConfig
from quack.gemm_interface import gemm_dgated_tuned, gemm_gated_tuned, gemm_tuned
from rich import print as print0
from triton.testing import do_bench


def _fast_sm100_configs(epilogue=None):
    tile_n_vals = [128, 192, 256]
    tile_mn_cluster_vals = [(256, tile_n, (2, 1)) for tile_n in tile_n_vals] + [(256, 512, (2, 1))]
    GemmConfigCls = partial(GemmConfig, pingpong=False, device_capacity=10)
    use_clc_vals = [True, False]
    use_tma_gather_vals = [True, False]
    return [
        GemmConfigCls(
            tile_m=m,
            tile_n=n,
            cluster_m=cm,
            cluster_n=cn,
            swap_ab=False,
            max_swizzle_size=8,
            is_dynamic_persistent=use_clc,
            use_tma_gather=use_tma_gather,
        )
        for (m, n, (cm, cn)), use_clc, use_tma_gather in itertools.product(
            tile_mn_cluster_vals, use_clc_vals, use_tma_gather_vals
        )
    ]


def _fast_sm90_configs(epilogue=None, tune_coop=True):
    tile_n_vals = [128, 160, 192]
    tile_mn_vals_coop = [(256, tile_n) for tile_n in tile_n_vals] + [(128, 256)]
    tile_mn_vals_pingpong = [(128, tile_n) for tile_n in tile_n_vals] + [(192, 128)]
    if epilogue in ["gated"]:
        tile_mn_vals_coop = [(m, n) for m, n in tile_mn_vals_coop if n % 32 == 0 and m != 192]
        tile_mn_vals_pingpong = [(m, n) for m, n in tile_mn_vals_pingpong if n % 32 == 0]
    tile_mn_vals = []
    if tune_coop:
        tile_mn_vals += [(m, n, False) for m, n in tile_mn_vals_coop]
    tile_mn_vals += [(m, n, True) for m, n in tile_mn_vals_pingpong]
    cluster = [(1, 2), (2, 1)]
    swap_ab_vals = [False]

    return [
        GemmConfig(
            tile_m=tile_m,
            tile_n=tile_n,
            pingpong=pingpong,
            cluster_m=cluster_m,
            cluster_n=cluster_n,
            swap_ab=swap_ab,
            device_capacity=9,
            is_dynamic_persistent=False,  # default to not use dynamic persistent on SM90
            use_tma_gather=False,  # TMA gather not supported on SM90
        )
        for (tile_m, tile_n, pingpong), (cluster_m, cluster_n), swap_ab in itertools.product(
            tile_mn_vals,
            cluster,
            swap_ab_vals,
        )
    ]


_gc._get_sm100_configs = _fast_sm100_configs
_gc._get_sm90_configs = _fast_sm90_configs


def _patch_autotuner_configs(autotuner_fn):
    all_new = [AutotuneConfig(config=c) for c in _gc.get_all_configs()]
    autotuner_fn.configs = all_new


_patch_autotuner_configs(gemm_tuned)
_patch_autotuner_configs(gemm_gated_tuned)
_patch_autotuner_configs(gemm_dgated_tuned)
gemm_gated_tuned.configs = [AutotuneConfig(config=c) for c in _gc.get_all_configs("gated")]
gemm_dgated_tuned.configs = [AutotuneConfig(config=c) for c in _gc.get_all_configs("gated")]


from sonicmoe import MoE
from sonicmoe.distributed_utils import CombineMode, DispatchMode, NetworkProfiler, RuntimeEPConfig
from sonicmoe.enums import ActivationType, is_glu
from sonicmoe.functional import TC_Softmax_Topk_Router_Function, moe_TC_softmax_topk_layer
from sonicmoe.functional.ep import moe_ep_TC_softmax_topk_forward


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
        description="EP forward+backward benchmark with local baselines (launch with torchrun).",
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
    parser.add_argument("--topk_over_softmax", action="store_true", default=False)
    parser.add_argument("--norm_topk_probs", action="store_true", default=False)
    parser.add_argument("--concat_layout", action="store_true", default=False)
    parser.add_argument(
        "--dispatch_mode",
        choices=["AG_TRITON", "A2A_TRITON", "RANK_DEDUP_DISPATCH_TRITON"],
        default=None,
        help="Manual dispatch-mode override. When unset, the profiled "
        "winner from NetworkProfiler.profile() is used.",
    )
    parser.add_argument(
        "--combine_mode",
        choices=["A2A_TRITON", "RS_COMBINE_TRITON", "RANK_DEDUP_COMBINE_TRITON"],
        default=None,
        help="Manual combine-mode override. When unset, the profiled "
        "winner from NetworkProfiler.profile() is used.",
    )
    parser.add_argument("--skip_local_T", action="store_true", default=False)
    parser.add_argument(
        "--redispatch_x_in_backward",
        action="store_true",
        default=False,
        help="Save x_local (T_local·d) instead of x_compute and re-dispatch x via "
        "AG-CE (Copy-Engine all-gather) on a side stream in backward — overlaps the "
        "rest of the backward at the cost of one extra NVLink AG of x.",
    )
    parser.add_argument(
        "--CPU_sync_on_runtime",
        action="store_true",
        default=False,
        help="One D2H sync on expert_frequency_offset[E_local] hoisted to before dispatch, so "
        "per-call h, a, and the A2A dispatch recv buffer get allocated at the actual populated "
        "row count instead of the structural ceiling. Skipped under inference mode.",
    )
    parser.add_argument("--skip_bench_bwd", action="store_true", default=False)
    args = parser.parse_args()
    if len(args.thiek) != 5:
        parser.error("--thiek must contain exactly 5 values")
    return args


def _require_torchrun_env() -> None:
    needed = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    if not all(v in os.environ for v in needed):
        sys.exit(
            "ERROR: this script must be launched with torchrun. Example:\n"
            "  torchrun --nproc-per-node=4 benchmarks/distributed/moe-ep.py "
            "--thiek 32768,2048,1024,64,8"
        )


def _all_gather_y(y_local: torch.Tensor, world_size: int) -> torch.Tensor:
    T_local, H = y_local.shape
    out = torch.empty(world_size * T_local, H, dtype=y_local.dtype, device=y_local.device)
    dist.all_gather_into_tensor(out, y_local.contiguous())
    return out


def _gather_to_rank0(t_local: torch.Tensor, world_size: int, rank: int):
    """Gather t_local from all ranks to rank 0 via all_gather_into_tensor."""
    t_local = t_local.contiguous()
    out = torch.empty(world_size, *t_local.shape, dtype=t_local.dtype, device=t_local.device)
    dist.all_gather_into_tensor(out, t_local)
    if rank == 0:
        return list(out.unbind(dim=0))
    return None


def do_bench_distributed(fn, warmup=30, rep=100, calls_per_iter=1):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    local_ms = 0.0
    for _ in range(calls_per_iter):
        start.record()
        for _ in range(rep):
            fn()
        end.record()
        torch.cuda.synchronize()
        local_ms += start.elapsed_time(end) / rep
        if dist.is_initialized():
            dist.barrier()
            torch.cuda.synchronize()
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
            False,
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

    fn()
    torch.cuda.synchronize()
    return do_bench(fn, warmup=warmup, rep=repeats)


def _bench_ep_fwd_bwd(
    x_grad,
    router_w_grad,
    w1_grad,
    b1_grad,
    w2_grad,
    b2_grad,
    grad_inputs,
    dout_local,
    fwd_kwargs,
    redispatch_x_in_backward,
    CPU_sync_on_runtime,
    warmup,
    repeats,
):
    def fn():
        o = moe_ep_TC_softmax_topk_forward(
            x_grad,
            router_w_grad,
            w1_grad,
            b1_grad,
            w2_grad,
            b2_grad,
            is_inference_mode_enabled=False,
            redispatch_x_in_backward=redispatch_x_in_backward,
            CPU_sync_on_runtime=CPU_sync_on_runtime,
            **fwd_kwargs,
        )
        torch.autograd.grad(o, grad_inputs, grad_outputs=dout_local, retain_graph=False)

    fn()
    torch.cuda.synchronize()
    return do_bench_distributed(fn, warmup=warmup, rep=repeats)


def run(args: argparse.Namespace) -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # The symm-mem workspace is owned by the process-wide cache in
    # ``sonicmoe.distributed_utils`` and freed via its atexit hook. We
    # still hard-exit through os._exit below so that an unexpected
    # exception (e.g. quack autotuner BrokenPipeError) can't race the
    # symm-mem dealloc against CUDA-context teardown in the unwind path.
    try:
        _run_impl(args, rank, local_rank, world_size)
        rc = 0
    except BaseException:
        import traceback

        traceback.print_exc()
        rc = 1

    if rank == 0:
        print("PASS" if rc == 0 else "FAIL")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)


def _run_impl(args: argparse.Namespace, rank: int, local_rank: int, world_size: int) -> None:
    W = world_size

    torch.autograd.set_multithreading_enabled(False)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(
        "nccl",
        device_id=device,
        timeout=datetime.timedelta(minutes=60),
    )

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    activation = ActivationType(args.activation)
    is_softmax_over_topk = not args.topk_over_softmax
    concat_layout = args.concat_layout
    norm_topk_probs = args.norm_topk_probs
    add_bias = args.add_bias
    redispatch_x_in_backward = args.redispatch_x_in_backward
    CPU_sync_on_runtime = args.CPU_sync_on_runtime

    T, H, I, E, K = args.thiek

    assert W >= 2, f"EP world size should be greater than 2."
    assert T % W == 0, f"T ({T}) must be divisible by world_size ({W})."
    assert E % W == 0, f"E ({E}) must be divisible by world_size ({W})."

    T_local = T // W
    E_local = E // W

    # NetworkProfiler picks dispatch + combine by default; --dispatch_mode and --combine_mode (when set) override the profiler's choice.
    dispatch_mode_override = DispatchMode(args.dispatch_mode) if args.dispatch_mode is not None else None
    combine_mode_override = CombineMode(args.combine_mode) if args.combine_mode is not None else None

    if rank == 0:
        routing_mode = "softmax_over_topk" if is_softmax_over_topk else f"topk_over_softmax (norm={norm_topk_probs})"
        layout_mode = "concat [gate; up]" if concat_layout else "interleaved [g0, u0, g1, u1, ...]"
        print0(
            f"[bold]EP forward+backward + local baselines[/bold]  EP world size W {W}, "
            f"Minibatch size T {T} (Per-rank microbatch size T_local {T_local}), H {H}, I {I}, "
            f"E {E} (E_local {E_local}), K {K}, dtype {args.dtype}, "
            f"routing: {routing_mode}, w1 layout: {layout_mode}, "
            f"bias: {add_bias}, "
            f"redispatch_x_in_backward: {redispatch_x_in_backward}, "
            f"CPU_sync_on_runtime: {CPU_sync_on_runtime}"
        )

    torch.manual_seed(1111)
    torch.cuda.manual_seed_all(1111)

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

    w1_full = moe.c_fc.weight  # (E, 2I, H) for GLU
    w2_full = moe.c_proj.weight  # (E, H, I)
    b1_full = moe.c_fc.bias if add_bias else None
    b2_full = moe.c_proj.bias if add_bias else None
    router_w = moe.router.weight  # (E, H)

    e_slc = slice(rank * E_local, (rank + 1) * E_local)
    w1_view = w1_full[e_slc].permute(1, 2, 0)
    w1 = torch.empty_strided(
        w1_view.shape,
        w1_view.stride(),
        dtype=w1_view.dtype,
        device=w1_view.device,
    ).copy_(w1_view)
    w2_view = w2_full[e_slc].permute(0, 2, 1)
    w2 = torch.empty_strided(
        w2_view.shape,
        w2_view.stride(),
        dtype=w2_view.dtype,
        device=w2_view.device,
    ).copy_(w2_view)
    b1 = b1_full[e_slc].contiguous() if add_bias else None
    b2 = b2_full[e_slc].contiguous() if add_bias else None

    if rank == 0:
        x_global = 0.2 * torch.randn(T, H, device=device, dtype=torch_dtype)
    else:
        x_global = torch.empty(T, H, device=device, dtype=torch_dtype)
    dist.broadcast(x_global, src=0)
    x = x_global[rank * T_local : (rank + 1) * T_local].contiguous()

    # Skip the NetworkProfiler entirely when both modes are user-overridden —
    # its only outputs (dispatch winner + combine winner + per-mode timings)
    # are unused in that case.
    both_overridden = dispatch_mode_override is not None and combine_mode_override is not None
    if both_overridden:
        profiler = None
        final_dispatch_mode = dispatch_mode_override
        final_combine_mode = combine_mode_override
    else:
        profiler = NetworkProfiler(
            T_local=T_local,
            H=H,
            K=K,
            dtype=torch_dtype,
            device=device,
            warmup=10,
            repeat=50,
        )
        profiled_cfg = profiler.profile()
        final_dispatch_mode = (
            dispatch_mode_override if dispatch_mode_override is not None else profiled_cfg.dispatch_mode
        )
        final_combine_mode = combine_mode_override if combine_mode_override is not None else profiled_cfg.combine_mode

    dist.barrier()
    time.sleep(0.5)

    ep_cfg = RuntimeEPConfig(dispatch_mode=final_dispatch_mode, W=W, K=K, combine_mode=final_combine_mode)
    if rank == 0:
        dispatch_src = "override" if dispatch_mode_override is not None else "profiled"
        combine_src = "override" if combine_mode_override is not None else "profiled"
        print0(
            f"[bold]Final config[/bold]: dispatch={final_dispatch_mode.value} ({dispatch_src}), "
            f"combine={final_combine_mode.value} ({combine_src})"
        )
        if profiler is not None:
            chosen_dispatch_ms = profiler.dispatch_timings_ms[final_dispatch_mode]
            chosen_combine_ms = profiler.combine_timings_ms[final_combine_mode]
            print0(
                f"Dispatch + Combine time: {chosen_dispatch_ms:.2f} + "
                f"{chosen_combine_ms:.2f} = {chosen_dispatch_ms + chosen_combine_ms:.2f} ms"
            )

    fwd_kwargs = dict(
        K=K,
        E=E,
        activation_type=activation,
        is_softmax_over_topk=is_softmax_over_topk,
        norm_topk_probs=norm_topk_probs,
        concat_layout=concat_layout,
        ep_config=ep_cfg,
    )

    local_kwargs = dict(
        K=K,
        activation=activation,
        is_softmax_over_topk=is_softmax_over_topk,
        norm_topk_probs=norm_topk_probs,
        concat_layout=concat_layout,
    )

    w1_local_fmt = w1_full.permute(1, 2, 0)
    w2_local_fmt = w2_full.permute(1, 2, 0)

    if not args.skip_test:
        x_t = x.clone().detach().requires_grad_(True)
        router_w_t = router_w.clone().detach().requires_grad_(True)
        w1_t = w1.clone().detach().requires_grad_(True)
        w2_t = w2.clone().detach().requires_grad_(True)
        b1_t = b1.clone().detach().requires_grad_(True) if add_bias else None
        b2_t = b2.clone().detach().requires_grad_(True) if add_bias else None

        o_local = moe_ep_TC_softmax_topk_forward(
            x_t,
            router_w_t,
            w1_t,
            b1_t,
            w2_t,
            b2_t,
            is_inference_mode_enabled=False,
            redispatch_x_in_backward=redispatch_x_in_backward,
            CPU_sync_on_runtime=CPU_sync_on_runtime,
            **fwd_kwargs,
        )

        if rank == 0:
            dout_global = 0.2 * torch.randn(T, H, device=device, dtype=torch_dtype)
        else:
            dout_global = torch.empty(T, H, device=device, dtype=torch_dtype)
        dist.broadcast(dout_global, src=0)
        dout_local = dout_global[rank * T_local : (rank + 1) * T_local].contiguous()

        grad_inputs = [x_t, router_w_t, w1_t, w2_t]
        if add_bias:
            grad_inputs += [b1_t, b2_t]
        grads = torch.autograd.grad(o_local, grad_inputs, grad_outputs=dout_local, retain_graph=False)
        dx_local, drouter_w, dw1_local, dw2_local = grads[:4]
        db1_local = grads[4] if add_bias else None
        db2_local = grads[5] if add_bias else None

        o_global = _all_gather_y(o_local.detach(), world_size)
        dx_global = _all_gather_y(dx_local, world_size)

        dw1_gathered = _gather_to_rank0(dw1_local, world_size, rank)
        dw2_gathered = _gather_to_rank0(dw2_local, world_size, rank)
        if add_bias:
            db1_gathered = _gather_to_rank0(db1_local, world_size, rank)
            db2_gathered = _gather_to_rank0(db2_local, world_size, rank)

        # ============================================================
        # CAPTURE EP's topk_idx_l. We use it as the test reference's
        # routing so test and EP agree on the slot ordering.
        # ============================================================
        with torch.no_grad():
            ep_router_logits = F.linear(x, router_w)
            _, ep_topk_idx_l = TC_Softmax_Topk_Router_Function.apply(
                ep_router_logits,
                world_size * E_local,
                K,
                is_softmax_over_topk,
                norm_topk_probs,
            )
        ep_topk_idx_full = torch.empty(world_size, T_local, K, dtype=torch.int32, device=device)
        dist.all_gather_into_tensor(
            ep_topk_idx_full.view(-1),
            ep_topk_idx_l.view(-1).contiguous(),
            group=dist.group.WORLD,
        )

        dist.barrier()

        if rank == 0:
            ep_dw1 = torch.cat(dw1_gathered, dim=2)
            ep_dw2 = torch.cat(dw2_gathered, dim=0)
            ep_db1 = torch.cat(db1_gathered, dim=0) if add_bias else None
            ep_db2 = torch.cat(db2_gathered, dim=0) if add_bias else None

            act_func = {
                ActivationType.SWIGLU: swiglu,
                ActivationType.GEGLU: geglu,
                ActivationType.REGLU: reglu,
            }[activation]

            ref_x = x_global.clone().detach().requires_grad_(True)
            ref_router_w = router_w.clone().detach().to(torch.float32).requires_grad_(True)
            ref_w1 = w1_full.clone().detach().requires_grad_(True)
            ref_w2 = w2_full.clone().detach().requires_grad_(True)
            ref_b1 = b1_full.clone().detach().requires_grad_(True) if add_bias else None
            ref_b2 = b2_full.clone().detach().requires_grad_(True) if add_bias else None

            ref_grad_inputs = [ref_x, ref_router_w, ref_w1, ref_w2]
            if add_bias:
                ref_grad_inputs += [ref_b1, ref_b2]

            topk_idx = ep_topk_idx_full.view(T, K).to(torch.int64)
            with torch.autocast(f"cuda:{local_rank}", torch.float32):
                topk_scores_chunks = []
                for r in range(world_size):
                    ref_x_chunk = ref_x[r * T_local : (r + 1) * T_local]
                    ref_logits_chunk = F.linear(ref_x_chunk, ref_router_w)
                    topk_idx_chunk = topk_idx[r * T_local : (r + 1) * T_local]
                    if is_softmax_over_topk:
                        ref_topk_logits_chunk = torch.gather(ref_logits_chunk, 1, topk_idx_chunk)
                        topk_scores_chunk = ref_topk_logits_chunk.softmax(dim=-1, dtype=torch.float32)
                    else:
                        ref_probs_chunk = ref_logits_chunk.softmax(dim=-1, dtype=torch.float32)
                        topk_scores_chunk = torch.gather(ref_probs_chunk, 1, topk_idx_chunk)
                        if norm_topk_probs:
                            topk_scores_chunk = topk_scores_chunk / topk_scores_chunk.sum(dim=-1, keepdim=True)
                    topk_scores_chunks.append(topk_scores_chunk)
                topk_scores = torch.cat(topk_scores_chunks, dim=0)

            with torch.autocast(f"cuda:{local_rank}", torch.float32):
                ref_o_global = torch.zeros(T, H, device=device, dtype=torch.float32)
                for i in range(E):
                    rows_t, rows_k = (topk_idx == i).nonzero(as_tuple=True)
                    if rows_t.numel() > 0:
                        ref_h = F.linear(
                            ref_x[rows_t],
                            ref_w1[i],
                            bias=(ref_b1[i] if add_bias else None),
                        )
                        ref_h = act_func(ref_h, concat_layout=concat_layout) if is_glu(activation) else act_func(ref_h)
                        ref_y = F.linear(
                            ref_h,
                            ref_w2[i],
                            bias=(ref_b2[i] if add_bias else None),
                        )
                        ref_o_global[rows_t] += ref_y * topk_scores[rows_t, rows_k, None]

                ref_grads = torch.autograd.grad(ref_o_global, ref_grad_inputs, grad_outputs=dout_global.float())

            ref_dx, ref_drouter_w, ref_dw1, ref_dw2 = ref_grads[:4]
            ref_db1 = ref_grads[4] if add_bias else None
            ref_db2 = ref_grads[5] if add_bias else None

            # All diff lines share a left-aligned 32-char label column
            # so the numbers line up regardless of variable name length.
            LBL = 32

            o_diff = (o_global.float() - ref_o_global).abs()
            print(f"{'max ref o val':<{LBL}} {ref_o_global.abs().max():.6f}")
            print(f"{'mean ref o val':<{LBL}} {ref_o_global.abs().mean():.6f}")
            print(f"{'max abs diff on o':<{LBL}} {o_diff.max():.6f}")
            print(f"{'mean rel diff on o':<{LBL}} {(o_diff / (ref_o_global.abs() + 1e-6)).mean():.6f}\n")

            test_triple_list = [
                ("dx", dx_global, ref_dx),
                ("drouter_w", drouter_w, ref_drouter_w),
                ("dw1", ep_dw1, ref_dw1.permute(1, 2, 0)),
                ("dw2", ep_dw2, ref_dw2.permute(0, 2, 1)),
            ]
            if add_bias:
                test_triple_list += [
                    ("db1", ep_db1, ref_db1),
                    ("db2", ep_db2, ref_db2),
                ]

            for n, our, ref in test_triple_list:
                diff = (our.float() - ref.float()).abs()
                print(f"{f'max abs ref value {n}':<{LBL}} {ref.abs().max():.6f}")
                print(f"{f'mean abs ref value {n}':<{LBL}} {ref.abs().mean():.6f}")
                print(f"{f'max abs diff on {n}':<{LBL}} {diff.max():.6f}")
                print(f"{f'mean rel diff on {n}':<{LBL}} {(diff / (ref.abs() + 1e-6)).mean():.6f}\n")

        del x_t, router_w_t, w1_t, w2_t, b1_t, b2_t
        del o_local, dx_local, drouter_w, dw1_local, dw2_local, db1_local, db2_local
        torch.cuda.empty_cache()

    dist.barrier()

    fwd_flops_global = (6 if is_glu(activation) else 4) * T * I * H * K
    fwd_flops_local = fwd_flops_global / world_size

    bwd_flops_global = 2 * fwd_flops_global
    bwd_flops_local = bwd_flops_global / world_size

    fwdbwd_flops_global = 3 * fwd_flops_global
    fwdbwd_flops_local = 3 * fwd_flops_local

    repeats = 100
    warmup = 30

    time.sleep(0.5)

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

    saved_info: list = []

    def _pack_hook(t):
        saved_info.append((tuple(t.shape), str(t.dtype).replace("torch.", ""), t.numel() * t.element_size()))
        return t

    def _unpack_hook(t):
        return t

    def _strided_clone(t: torch.Tensor) -> torch.Tensor:
        # quack's gemm requires the (E_local, K, N) strided layout from
        # empty_strided + permute; a plain .clone() reverts to default
        # contiguous strides and trips the gemm's stride check.
        out = torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device)
        return out.copy_(t).requires_grad_(True)

    x_audit = x.clone().detach().requires_grad_(True)
    router_w_audit = router_w.clone().detach().requires_grad_(True)
    w1_audit = _strided_clone(w1)
    w2_audit = _strided_clone(w2)
    b1_audit = b1.clone().detach().requires_grad_(True) if add_bias else None
    b2_audit = b2.clone().detach().requires_grad_(True) if add_bias else None

    with torch.autograd.graph.saved_tensors_hooks(_pack_hook, _unpack_hook):
        _o_audit = moe_ep_TC_softmax_topk_forward(
            x_audit,
            router_w_audit,
            w1_audit,
            b1_audit,
            w2_audit,
            b2_audit,
            is_inference_mode_enabled=False,
            redispatch_x_in_backward=redispatch_x_in_backward,
            CPU_sync_on_runtime=CPU_sync_on_runtime,
            **fwd_kwargs,
        )

    if rank == 0:
        h_dim2 = (2 * I) if is_glu(activation) else I

        candidates = [
            (i, shape, nbytes)
            for i, (shape, dtype_str, nbytes) in enumerate(saved_info)
            if (dtype_str.startswith("bfloat16") or dtype_str.startswith("float16")) and len(shape) == 2
        ]
        h_matches = [c for c in candidates if c[1][1] == h_dim2]
        x_matches = [c for c in candidates if c[1][1] == H]
        h_pick = max(h_matches, key=lambda c: c[2], default=(None, None, 0))
        h_idx = h_pick[0]
        x_pool = [c for c in x_matches if c[0] != h_idx]
        x_pick = max(x_pool, key=lambda c: c[2], default=(None, None, 0))

        print0(
            "\n[bold]══ Saved-activation cache audit (training, redispatch_x={}, CPU_sync={}) ══[/bold]".format(
                redispatch_x_in_backward, CPU_sync_on_runtime
            )
        )
        LBL = 18
        total_bytes = x_pick[2] + h_pick[2]
        print0(
            f"  {'X cache:':<{LBL}} {x_pick[2]/(1024 ** 3):>8.2f} GiB  shape={x_pick[1]}\n"
            f"  {'h cache:':<{LBL}} {h_pick[2]/(1024 ** 3):>8.2f} GiB  shape={h_pick[1]}\n"
            f"  {'Total:':<{LBL}} {total_bytes/(1024 ** 3):>8.2f} GiB"
        )

    del x_audit, router_w_audit, w1_audit, w2_audit, b1_audit, b2_audit, _o_audit
    saved_info.clear()
    torch.cuda.empty_cache()
    dist.barrier()

    # ====================================================================
    # 1) EP Fwd inference
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
            redispatch_x_in_backward=redispatch_x_in_backward,
            CPU_sync_on_runtime=CPU_sync_on_runtime,
            **fwd_kwargs,
        )

    ep_fwd_inf_ms = do_bench_distributed(ep_fwd_inference, warmup=warmup, rep=repeats)
    if rank == 0:
        tflops = fwd_flops_global / (ep_fwd_inf_ms * 1e9)
        local_tflops = fwd_flops_local / (ep_fwd_inf_ms * 1e9)
        print0("\n[bold]── EP throughput ──[/bold]")
        print0(
            f" EP Fwd (inference mode) Average time: {ep_fwd_inf_ms:<8.2f} ms, "
            f"Per-rank TFLOPS: {local_tflops:<5.0f}, Net EP TFLOPS: {tflops:<5.0f}"
        )

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
            redispatch_x_in_backward=redispatch_x_in_backward,
            CPU_sync_on_runtime=CPU_sync_on_runtime,
            **fwd_kwargs,
        )

    ep_fwd_train_ms = do_bench_distributed(ep_fwd_training, warmup=warmup, rep=repeats)
    if rank == 0:
        tflops = fwd_flops_global / (ep_fwd_train_ms * 1e9)
        local_tflops = fwd_flops_local / (ep_fwd_train_ms * 1e9)
        print0(
            f" EP Fwd (training mode)  Average time: {ep_fwd_train_ms:<8.2f} ms, "
            f"Per-rank TFLOPS: {local_tflops:<5.0f}, Net EP TFLOPS: {tflops:<5.0f}"
        )

    ep_fwdbwd_ms = None
    ep_bwd_ms = None
    if not args.skip_bench_bwd:
        time.sleep(0.5)
        torch.cuda.synchronize()
        dist.barrier()

        x_g = x.clone().detach().requires_grad_(True)
        router_w_g = router_w.clone().detach().requires_grad_(True)
        w1_g = w1.clone().detach().requires_grad_(True)
        w2_g = w2.clone().detach().requires_grad_(True)
        b1_g = b1.clone().detach().requires_grad_(True) if add_bias else None
        b2_g = b2.clone().detach().requires_grad_(True) if add_bias else None

        grad_inputs_bench = [x_g, router_w_g, w1_g, w2_g]
        if add_bias:
            grad_inputs_bench += [b1_g, b2_g]

        if rank == 0:
            dout_bench_global = 0.2 * torch.randn(T, H, device=device, dtype=torch_dtype)
        else:
            dout_bench_global = torch.empty(T, H, device=device, dtype=torch_dtype)
        dist.broadcast(dout_bench_global, src=0)
        dout_bench_local = dout_bench_global[rank * T_local : (rank + 1) * T_local].contiguous()

        ep_fwdbwd_ms = _bench_ep_fwd_bwd(
            x_g,
            router_w_g,
            w1_g,
            b1_g,
            w2_g,
            b2_g,
            grad_inputs_bench,
            dout_bench_local,
            fwd_kwargs,
            redispatch_x_in_backward,
            CPU_sync_on_runtime,
            warmup=warmup,
            repeats=repeats,
        )
        ep_bwd_ms = ep_fwdbwd_ms - ep_fwd_train_ms

        if rank == 0:
            bwd_tflops = bwd_flops_global / (ep_bwd_ms * 1e9) if ep_bwd_ms > 0 else 0.0
            bwd_local_tflops = bwd_flops_local / (ep_bwd_ms * 1e9) if ep_bwd_ms > 0 else 0.0
            print0(
                f" EP Bwd (derived)        Average time: {ep_bwd_ms:<8.2f} ms, "
                f"Per-rank TFLOPS: {bwd_local_tflops:<5.0f}, Net EP TFLOPS: {bwd_tflops:<5.0f}"
            )

        del x_g, router_w_g, w1_g, w2_g, b1_g, b2_g
        del dout_bench_global, dout_bench_local
        torch.cuda.empty_cache()

    if rank == 0:
        print0(f"\n[bold]── Per-rank baselines (T={T_local}, E={E}, K={K}, single-GPU with T_local tokens) ──[/bold]")

    w1_full_baseline = w1_full.clone()
    w2_full_baseline = w2_full.clone()
    b1_full_baseline = b1_full.clone() if add_bias else None
    b2_full_baseline = b2_full.clone() if add_bias else None
    router_w_baseline = router_w.clone()
    w1_local_fmt_baseline = w1_full_baseline.permute(1, 2, 0)
    w2_local_fmt_baseline = w2_full_baseline.permute(1, 2, 0)

    x_local_nograd = x.detach().clone()
    x_local_grad = x.clone().detach().requires_grad_(True)

    time.sleep(0.5)
    torch.cuda.synchronize()
    dist.barrier()

    local_fwd_train_Tl_ms = _bench_local_fwd(
        x_local_nograd,
        router_w_baseline,
        w1_local_fmt_baseline,
        b1_full_baseline,
        w2_local_fmt_baseline,
        b2_full_baseline,
        is_inference=False,
        warmup=warmup,
        repeats=repeats,
        **local_kwargs,
    )
    if rank == 0:
        tflops = fwd_flops_local / (local_fwd_train_Tl_ms * 1e9)
        print0(
            f" {f'Per-rank Fwd (T={T_local}, E={E}, K={K}, training)':<48} Average time: {local_fwd_train_Tl_ms:<8.2f} ms, TFLOPS: {tflops:<5.0f}"
        )

    local_bwd_Tl_ms = None
    if not args.skip_bench_bwd:
        time.sleep(0.5)
        torch.cuda.synchronize()
        dist.barrier()

        local_fwdbwd_Tl_ms = _bench_local_fwd_bwd(
            x_local_grad,
            router_w_baseline,
            w1_full_baseline,
            b1_full_baseline,
            w2_full_baseline,
            b2_full_baseline,
            w1_local_fmt_baseline,
            w2_local_fmt_baseline,
            warmup=warmup,
            repeats=repeats,
            **local_kwargs,
        )
        local_bwd_Tl_ms = local_fwdbwd_Tl_ms - local_fwd_train_Tl_ms
        if rank == 0:
            bwd_tflops = bwd_flops_local / (local_bwd_Tl_ms * 1e9) if local_bwd_Tl_ms > 0 else 0.0
            print0(
                f" {f'Per-rank Bwd (T={T_local}, E={E}, K={K})':<48} Average time: {local_bwd_Tl_ms:<8.2f} ms, TFLOPS: {bwd_tflops:<5.0f}"
            )

    # ── Per-rank baseline with E=E_local: matches the expert count each EP rank
    # actually owns after dispatch, so it isolates compute scaling from the
    # router/expert-fanout effect that the E=E baseline above conflates with comm.
    local_fwd_train_Tl_Elocal_ms = None
    local_bwd_Tl_E_local_ms = None
    if E_local >= K:
        if rank == 0:
            print0(
                f"\n[bold]── Per-rank baselines (T={T_local}, E={E_local}, K={K}, single-GPU with T_local tokens) ──[/bold]"
            )

        w1_E_local_src = w1_full_baseline[e_slc].clone().contiguous()
        w2_E_local_src = w2_full_baseline[e_slc].clone().contiguous()
        w1_E_local_baseline = w1_E_local_src.permute(1, 2, 0)
        w2_E_local_baseline = w2_E_local_src.permute(1, 2, 0)
        router_w_E_local_baseline = router_w_baseline[e_slc].clone().contiguous()
        b1_E_local_baseline = b1_full_baseline[e_slc].clone().contiguous() if add_bias else None
        b2_E_local_baseline = b2_full_baseline[e_slc].clone().contiguous() if add_bias else None

        x_E_local_nograd = x.detach().clone()
        x_E_local_grad = x.clone().detach().requires_grad_(True)

        time.sleep(0.5)
        torch.cuda.synchronize()
        dist.barrier()

        local_fwd_train_Tl_Elocal_ms = _bench_local_fwd(
            x_E_local_nograd,
            router_w_E_local_baseline,
            w1_E_local_baseline,
            b1_E_local_baseline,
            w2_E_local_baseline,
            b2_E_local_baseline,
            is_inference=False,
            warmup=warmup,
            repeats=repeats,
            **local_kwargs,
        )
        if rank == 0:
            tflops = fwd_flops_local / (local_fwd_train_Tl_Elocal_ms * 1e9)
            print0(
                f" {f'Per-rank Fwd (T={T_local}, E={E_local}, K={K}, training)':<48} Average time: {local_fwd_train_Tl_Elocal_ms:<8.2f} ms, TFLOPS: {tflops:<5.0f}"
            )

        if not args.skip_bench_bwd:
            time.sleep(0.5)
            torch.cuda.synchronize()
            dist.barrier()

            local_fwdbwd_Tl_E_local_ms = _bench_local_fwd_bwd(
                x_E_local_grad,
                router_w_E_local_baseline,
                w1_E_local_src,
                b1_E_local_baseline,
                w2_E_local_src,
                b2_E_local_baseline,
                w1_E_local_baseline,
                w2_E_local_baseline,
                warmup=warmup,
                repeats=repeats,
                **local_kwargs,
            )
            local_bwd_Tl_E_local_ms = local_fwdbwd_Tl_E_local_ms - local_fwd_train_Tl_Elocal_ms
            if rank == 0:
                bwd_tflops = bwd_flops_local / (local_bwd_Tl_E_local_ms * 1e9) if local_bwd_Tl_E_local_ms > 0 else 0.0
                print0(
                    f" {f'Per-rank Bwd (T={T_local}, E={E_local}, K={K})':<48} Average time: {local_bwd_Tl_E_local_ms:<8.2f} ms, TFLOPS: {bwd_tflops:<5.0f}"
                )

    local_fwd_train_T_ms = None
    local_bwd_T_ms = None
    if not args.skip_local_T:
        if rank == 0:
            print0(f"\n[bold]── Per-rank baselines (T={T}, E={E}, K={K}, single-GPU full EP scale) ──[/bold]")

        x_full_nograd = x_global.clone().detach()
        x_full_grad = x_global.clone().detach().requires_grad_(True)

        time.sleep(0.5)
        torch.cuda.synchronize()
        dist.barrier()

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
                f" {f'Per-rank Fwd (T={T}, E={E}, K={K}, training)':<48} Average time: {local_fwd_train_T_ms:<8.2f} ms, TFLOPS: {tflops:<5.0f}"
            )

        if not args.skip_bench_bwd:
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
                bwd_tflops = bwd_flops_global / (local_bwd_T_ms * 1e9) if local_bwd_T_ms > 0 else 0.0
                print0(
                    f" {f'Per-rank Bwd (T={T}, E={E}, K={K})':<48} Average time: {local_bwd_T_ms:<8.2f} ms, TFLOPS: {bwd_tflops:<5.0f}"
                )

    if rank == 0:
        print0("\n[bold]══ Exposed network latency (EP vs. local T_local) ══[/bold]")

        LBL = 40

        def _slowdown_line(ep_ms: float, base_ms: float, label: str) -> str:
            slow = ep_ms - base_ms
            pct = slow / ep_ms * 100 if ep_ms > 0 else 0.0
            return (
                f"    {label:<{LBL}} {base_ms:<8.2f} ms\n"
                f"        {'Slowdown:':<{LBL - 4}} {slow:<8.2f} ms ({pct:.1f}%)"
            )

        fwd_lines = [
            "  Training fwd:",
            f"    {'EP Fwd:':<{LBL}} {ep_fwd_train_ms:<8.2f} ms",
            _slowdown_line(ep_fwd_train_ms, local_fwd_train_Tl_ms, f"Per-rank Fwd (T_local, E={E}):"),
        ]
        if local_fwd_train_Tl_Elocal_ms is not None:
            fwd_lines.append(
                _slowdown_line(
                    ep_fwd_train_ms,
                    local_fwd_train_Tl_Elocal_ms,
                    f"Per-rank Fwd (T_local, E={E_local}):",
                )
            )
        print0("\n".join(fwd_lines) + "\n")

        if not args.skip_bench_bwd and ep_bwd_ms is not None and local_bwd_Tl_ms is not None:
            bwd_lines = [
                "  Backward:",
                f"    {'EP Bwd:':<{LBL}} {ep_bwd_ms:<8.2f} ms",
                _slowdown_line(ep_bwd_ms, local_bwd_Tl_ms, f"Per-rank Bwd (T_local, E={E}):"),
            ]
            if local_bwd_Tl_E_local_ms is not None:
                bwd_lines.append(
                    _slowdown_line(
                        ep_bwd_ms,
                        local_bwd_Tl_E_local_ms,
                        f"Per-rank Bwd (T_local, E={E_local}):",
                    )
                )
            print0("\n".join(bwd_lines))

        if not args.skip_local_T and local_fwd_train_T_ms is not None:
            print0("\n[bold]══ EP scaling efficiency ══[/bold]")
            ideal_speedup = world_size

            speedup = local_fwd_train_T_ms / ep_fwd_train_ms if ep_fwd_train_ms > 0 else 0.0
            scaling_eff = speedup / ideal_speedup * 100 if ideal_speedup > 0 else 0.0
            print0(
                f"  Training fwd:\n"
                f"    {f'Single-GPU full EP-scale (T={T}):':<{LBL}} {local_fwd_train_T_ms:<8.2f} ms\n"
                f"    {f'EP W={world_size} (T={T}):':<{LBL}} {ep_fwd_train_ms:<8.2f} ms\n"
                f"    {'Observed speedup:':<{LBL}} {speedup:<8.2f}× "
                f"(ideal {ideal_speedup}×, {scaling_eff:.1f}% over linear scaling)"
            )

            if not args.skip_bench_bwd and ep_bwd_ms is not None and local_bwd_T_ms is not None:
                bwd_speedup = local_bwd_T_ms / ep_bwd_ms if ep_bwd_ms > 0 else 0.0
                bwd_scaling_eff = bwd_speedup / ideal_speedup * 100 if ideal_speedup > 0 else 0.0
                print0(
                    f"\n  Backward:\n"
                    f"    {f'Single-GPU full EP-scale (T={T}):':<{LBL}} {local_bwd_T_ms:<8.2f} ms\n"
                    f"    {f'EP W={world_size} (T={T}):':<{LBL}} {ep_bwd_ms:<8.2f} ms\n"
                    f"    {'Observed speedup:':<{LBL}} {bwd_speedup:<8.2f}× "
                    f"(ideal {ideal_speedup}×, {bwd_scaling_eff:.1f}% over linear scaling)"
                )

    if rank == 0:
        print("PASS")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main() -> int:
    _require_torchrun_env()
    args = parse_arguments()
    run(args)
    return 0


if __name__ == "__main__":
    main()
