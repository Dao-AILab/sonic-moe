# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
#
# E2E benchmark for the EP forward + backward (moe_ep_TC_softmax_topk_forward)
# with local (non-EP) baselines for exposed network latency estimation.
#
# T in --thiek is the GLOBAL token count; each rank processes T // W tokens.
# Reported EP wall-times are rank-0's measured time; the NCCL collectives
# inside the EP forward implicitly serialize all ranks, so rank 0's time
# tracks the slowest-rank latency. Local baselines run independently per rank
# and are reported from rank 0. TFLOPS use the global token count for EP and
# per-rank / full token count for local, so numbers are directly comparable
# in the summary table.
#
# Backward correctness: a single-GPU autograd reference (token-by-expert loop
# computed in pure bf16, mirroring the EP precision exactly) is built on rank
# 0 from the same x, w1, w2, router_w used by the EP path. EP gradients (dx,
# dW1, dW2, drouter_w, and biases when present) are aggregated to rank 0 —
# dx via all-gather, drouter_w via all-reduce sum (every rank computes a
# partial drouter_w from its slice of x), dW1/dW2/db1/db2 via gather then
# concat — and compared against the reference per-tensor.
#
# Mode selection: the bench profiles with ``NetworkProfiler`` to pick both
# the dispatch mode and the combine mode at runtime. ``--mode`` and
# ``--agg_mode``, when given, override the profiler's pick (useful for
# isolating a specific primitive or reproducing a prior run).
#
# Launch with torchrun:
#
#   torchrun --nproc-per-node=8 --standalone \
#       benchmarks/distributed/moe-ep.py --thiek 131072,4096,1536,128,8
# ********************************************************************************

import argparse
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

M_QUANT = 4096


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
    tile_n_vals = [128, 160, 192, 256]
    tile_mn_cluster_vals = [(256, tile_n, (2, 1)) for tile_n in tile_n_vals] + [(256, 512, (2, 1))]
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


from sonicmoe import MoE
from sonicmoe.distributed_utils import (
    CombineMode,
    DispatchMode,
    NetworkProfiler,
    RuntimeEPConfig,
    _is_ag_dispatch_mode,
    _is_rank_dedup_dispatch_mode,
)
from sonicmoe.enums import ActivationType, is_glu
from sonicmoe.functional import TC_Softmax_Topk_Router_Function, moe_TC_softmax_topk_layer
from sonicmoe.functional.ep import SymmMemManager, moe_ep_TC_softmax_topk_forward


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
        "--mode",
        choices=["AG_TRITON", "A2A_TRITON", "RANK_DEDUP_DISPATCH_TRITON"],
        default=None,
        help="Manual dispatch-mode override. When unset, the profiled "
        "winner from NetworkProfiler.profile() is used.",
    )
    parser.add_argument(
        "--agg_mode",
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


def do_bench_distributed(fn, warmup=5, rep=100, calls_per_iter=3):
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

    # `keep_alive` is a container that lives in run()'s scope. _run_impl
    # appends `mgr` (and any other object whose destructor invokes CUDA
    # APIs that fail at process exit) into this list. When an exception
    # unwinds out of _run_impl, those objects remain referenced through
    # `keep_alive` and DO NOT destruct at the boundary — they survive
    # until run() reaches os._exit() below. os._exit terminates the
    # process atomically without invoking any C++ destructors, which is
    # the only reliable cleanup path for symm-mem allocations.
    #
    # Why this matters: the quack autotuner can hit a BrokenPipeError
    # during local benchmarks (cache-write race to a closed subprocess
    # pipe). The exception unwinds out of _run_impl. Without keep_alive,
    # `mgr` was in _run_impl's scope and destructed during the unwind;
    # ~CUDASymmetricMemory then called cuMemUnmap which threw c10::Error
    # from a destructor → std::terminate → SIGABRT *before* run()'s
    # except clause could fire. With keep_alive, mgr survives the unwind
    # and is only "released" when os._exit kills the process — at which
    # point Python's garbage collector and C++ destructors don't run.
    keep_alive: list = []

    try:
        _run_impl(args, rank, local_rank, world_size, keep_alive)
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


def _run_impl(args: argparse.Namespace, rank: int, local_rank: int, world_size: int, keep_alive: list) -> None:
    W = world_size

    torch.autograd.set_multithreading_enabled(False)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", device_id=device)

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

    # NetworkProfiler picks dispatch + combine by default; --mode and
    # --agg_mode (when set) override the profiler's choice.
    mode_override = DispatchMode(args.mode) if args.mode is not None else None
    agg_mode_override = CombineMode(args.agg_mode) if args.agg_mode is not None else None

    if rank == 0:
        routing_mode = "softmax_over_topk" if is_softmax_over_topk else f"topk_over_softmax (norm={norm_topk_probs})"
        layout_mode = "concat [gate; up]" if concat_layout else "interleaved [g0, u0, g1, u1, ...]"
        mode_status = f"override={mode_override.value}" if mode_override is not None else "auto (via NetworkProfiler)"
        agg_status = (
            f"override={agg_mode_override.value}" if agg_mode_override is not None else "auto (via NetworkProfiler)"
        )
        print0(
            f"[bold]EP forward+backward + local baselines[/bold]  W {W}, "
            f"T {T} (T_local {T_local}), H {H}, I {I}, "
            f"E {E} (E_local {E_local}), K {K}, "
            f"dtype {args.dtype}, mode {mode_status}, agg_mode {agg_status}, "
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

    mgr = SymmMemManager(dist.group.WORLD, device)
    # Pin mgr's lifetime to run()'s scope (see keep_alive comment in run()).
    # Required to survive symm-mem destructor failure during exception unwind.
    keep_alive.append(mgr)

    # Profile the three dispatch primitives once on this hardware and
    # build a RuntimeEPConfig. ``--mode`` / ``--agg_mode``, when given,
    # override the profiler's pick.
    profiler = NetworkProfiler(
        mgr,
        T_local=T_local,
        H=H,
        K=K,
        dtype=torch_dtype,
        warmup=10,
        repeat=50,
    )
    profiled_cfg = profiler.profile()
    # Settle after the profiler's many warmup + timed iterations so the
    # actual EP throughput measurements below start from a quiesced
    # state across all ranks (GPU clocks / NVLink fabric / NCCL channels
    # all benefit from a brief idle window).
    dist.barrier()
    time.sleep(0.5)

    final_mode = mode_override if mode_override is not None else profiled_cfg.mode
    final_agg_mode = agg_mode_override if agg_mode_override is not None else profiled_cfg.agg_mode
    ep_cfg = RuntimeEPConfig(mode=final_mode, W=W, K=K, agg_mode=final_agg_mode)
    resolved_mode = ep_cfg.mode
    if rank == 0:
        dispatch_src = "override" if mode_override is not None else "profiled"
        agg_src = "override" if agg_mode_override is not None else "profiled"
        print0(
            f"[bold]Final config[/bold]: dispatch={final_mode.value} ({dispatch_src}), "
            f"agg={final_agg_mode.value} ({agg_src})"
        )

    fwd_kwargs = dict(
        K=K,
        E=E,
        mgr=mgr,
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

    # ====================================================================
    # Reference correctness check
    # ====================================================================

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

            # Reference parameters: keep ref_x / ref_w1 / ref_w2 / ref_b* in
            # production dtype so the per-expert MoE forward (inside the
            # autocast(fp32) block below) sees the same input bytes EP did.
            # `ref_router_w` is promoted to FP32 explicitly so the autograd
            # backward through the routing F.linear returns ref_drouter_w in
            # fp32 regardless of autocast policy. Without this, ref_drouter_w
            # came out bf16/fp16 (matching the leaf param), the EP-vs-ref
            # comparison measured bf16-vs-bf16 summation-order noise, and
            # mean-rel diff inflated to 30%+ for bf16 due to small-value
            # elements. With fp32 ground truth, the comparison measures EP's
            # bf16 storage error against fp32 truth — cute-comparable
            # ~2-3% mean rel diff.
            ref_x = x_global.clone().detach().requires_grad_(True)
            ref_router_w = router_w.clone().detach().to(torch.float32).requires_grad_(True)
            ref_w1 = w1_full.clone().detach().requires_grad_(True)
            ref_w2 = w2_full.clone().detach().requires_grad_(True)
            ref_b1 = b1_full.clone().detach().requires_grad_(True) if add_bias else None
            ref_b2 = b2_full.clone().detach().requires_grad_(True) if add_bias else None

            ref_grad_inputs = [ref_x, ref_router_w, ref_w1, ref_w2]
            if add_bias:
                ref_grad_inputs += [ref_b1, ref_b2]

            # ========================================================
            # CHUNKED REFERENCE LOGITS *AND* SCORES.
            #
            # The EP forward computes router_logits =
            # F.linear(x_local, router_w) per rank — i.e. W cuBLAS GEMM
            # calls of shape (T_local, H) — and softmax-over-topk is
            # then applied per rank on the (T_local, E) logits. Gradients
            # flow back through each rank's softmax/gather and accumulate
            # into ref_router_w via W separate calls.
            #
            # The reference now mirrors this exactly. Two stages, both
            # chunked:
            #
            #   1) Compute router_logits in W chunks of (T_local, H), so
            #      cuBLAS picks the same algorithm as the EP for each
            #      chunk and produces bit-identical logits per chunk.
            #
            #   2) Compute topk_scores per chunk too — gather-then-softmax
            #      for softmax-over-topk routing, or softmax-then-gather
            #      for topk-over-softmax routing (same logic the EP uses
            #      internally). Each chunk's softmax is row-independent,
            #      so the *value* of topk_scores is mathematically the
            #      same whether computed chunked or globally; doing it
            #      chunked makes the autograd graph mirror the EP's exact
            #      bf16 accumulation pattern into ref_router_w (W
            #      separate GEMM-backward contributions summed in fp32),
            #      eliminating any test-side bias on drouter_w.
            #
            # ref_logits is no longer needed as a single concatenated
            # tensor — only topk_scores feeds the forward loop below.
            # ========================================================
            # NOTE: the chunked routing (and the autograd backward through it
            # that produces ref_drouter_w) used to be OUTSIDE the autocast
            # block below, so ref_drouter_w was bf16 and the EP-vs-ref
            # comparison measured bf16-vs-bf16 summation-order noise — which
            # mean-rel-amplifies to ~30%+ on bf16 because of small-value
            # elements. We now wrap the routing in the same autocast(fp32)
            # block as the MoE forward (mirroring moe-cute.py's reference),
            # so ref_drouter_w is fp32 and the EP-vs-ref comparison measures
            # EP's bf16 storage error against fp32 ground truth — directly
            # comparable to moe-cute.py's drouter_w numbers and bounded by
            # at most ~W× the single-rank bf16 ULP.
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

            # Reference forward + backward.
            #
            # Ground-truth precision: ref_o_global is FP32. This is the
            # critical detail. The autocast(fp32) wrapper makes the
            # per-expert F.linears produce fp32 ref_y, but if
            # ref_o_global were bf16, the per-expert
            # `ref_o_global[rows_t] += ref_y * scores` would silently
            # downcast the fp32 rhs to bf16 before adding, and the
            # accumulator would quantize at every step — defeating the
            # entire point of using autocast. The earlier failure mode
            # at T=32768 (forward o off by 100% magnitude) and the
            # current failure at H=I=2048 (same signature, different
            # shape) are both this same bug: bf16 += into a bf16 buffer
            # accumulates K bf16 quantization steps per token, and once
            # tokens-per-expert grows large enough, those quantizations
            # diverge from EP's gather kernel which keeps the K-sum in
            # an fp32 register and casts once.
            #
            # Allocating ref_o_global in fp32 fixes this: the entire
            # expert path stays fp32 from F.linear through the += into
            # ref_o_global. EP's bf16 output is then compared against
            # genuine fp32 ground truth, and the only divergence is
            # EP's bf16 storage precision (~1 bf16 ULP per element)
            # regardless of shape or scale.
            #
            # The routing path (chunked F.linear and softmax above) is
            # OUTSIDE this autocast on purpose: it stays bf16 so the
            # EP-vs-ref comparison on drouter_w reflects the bf16
            # storage of router_w (which is what production sees), not
            # an fp32 reference that would always show ~1% bf16 error
            # we can't reduce.
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

    # ====================================================================
    # FLOP counters
    # ====================================================================
    fwd_flops_global = (6 if is_glu(activation) else 4) * T * I * H * K
    fwd_flops_local = fwd_flops_global / world_size

    bwd_flops_global = 2 * fwd_flops_global
    bwd_flops_local = bwd_flops_global / world_size

    fwdbwd_flops_global = 3 * fwd_flops_global
    fwdbwd_flops_local = 3 * fwd_flops_local

    repeats = 100
    warmup = 5

    time.sleep(0.5)

    # ====================================================================
    # EP warmup
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
    # Saved-activation cache audit (training, no redispatch)
    # ====================================================================
    # Targets (per rank, in elements):
    #   X cache:   min(K, W) * T_local * d      (AG mode K>=W: W·T_local·d;
    #                                            A2A mode K<W: K·T_local·d)
    #   h cache:   TK_global / W  ·  h_dim2     (= TK_local · 2I for SwiGLU)
    #
    # The X target equals the dispatched x_compute under AG mode. Under A2A
    # the dispatched buffer is shape (W, TK_local, d) = TK_global·d, which
    # is a factor W over the target — only ~TK_local rows are valid.
    #
    # For h, the merged _MoeEPFunction holds (TK_global, h_dim2) but
    # gemm_gated only writes the first expert_frequency_offset[E_local]
    # (~TK_local) rows under cu_seqlens_m, so the cache is over by a
    # factor of ~W in both modes (the tradeoff that buys uniform shape
    # in the off-path E_local < K case and removes the reinflate dance).
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

        # Classify saved tensors by SHAPE + NBYTES rank. The pack_hook
        # captures every save_for_backward in the autograd graph (the
        # router's x and router_w, the topk function's saves, and the
        # EP function's saves), so save-order alone can't isolate the
        # EP's x_compute / h. The two saves that dominate cache cost
        # are h (largest with shape[1] == h_dim2) and x_compute (largest
        # with shape[1] == H, excluding h's slot when H == h_dim2).
        # Index into ``saved_info`` is the disambiguator — when H ==
        # h_dim2 the candidate pools coincide and we exclude h's
        # specific position so x picks the next-largest distinct entry.
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

    # ====================================================================
    # 2) EP Fwd training
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
            f" EP Fwd (training mode)  Average time: {ep_fwd_train_ms:<8.2f} ms, "
            f"Per-rank TFLOPS: {local_tflops:<5.0f}, Net EP TFLOPS: {tflops:<5.0f}"
        )

    # ====================================================================
    # 3) EP Fwd + Bwd
    # ====================================================================
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
            fwdbwd_tflops = fwdbwd_flops_global / (ep_fwdbwd_ms * 1e9)
            fwdbwd_local_tflops = fwdbwd_flops_local / (ep_fwdbwd_ms * 1e9)
            bwd_tflops = bwd_flops_global / (ep_bwd_ms * 1e9) if ep_bwd_ms > 0 else 0.0
            bwd_local_tflops = bwd_flops_local / (ep_bwd_ms * 1e9) if ep_bwd_ms > 0 else 0.0
            print0(
                f" EP Fwd+Bwd              Average time: {ep_fwdbwd_ms:<8.2f} ms, "
                f"Per-rank TFLOPS: {fwdbwd_local_tflops:<5.0f}, Net EP TFLOPS: {fwdbwd_tflops:<5.0f}"
            )
            print0(
                f" EP Bwd (derived)        Average time: {ep_bwd_ms:<8.2f} ms, "
                f"Per-rank TFLOPS: {bwd_local_tflops:<5.0f}, Net EP TFLOPS: {bwd_tflops:<5.0f}"
            )

        del x_g, router_w_g, w1_g, w2_g, b1_g, b2_g
        del dout_bench_global, dout_bench_local
        torch.cuda.empty_cache()

    # ====================================================================
    # 4) Local baselines T_local
    # ====================================================================
    if rank == 0:
        print0(f"\n[bold]── Per-rank baselines (T_local={T_local}, no communication) ──[/bold]")

    x_local_nograd = x.clone().detach()
    x_local_grad = x.clone().detach().requires_grad_(True)

    time.sleep(0.5)
    torch.cuda.synchronize()
    dist.barrier()

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
            f" {f'Per-rank Fwd (T_local={T_local}, training)':<42} Average time: {local_fwd_train_Tl_ms:<8.2f} ms, TFLOPS: {tflops:<5.0f}"
        )

    local_bwd_Tl_ms = None
    if not args.skip_bench_bwd:
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
                f" {f'Per-rank Bwd (T_local={T_local})':<42} Average time: {local_bwd_Tl_ms:<8.2f} ms, TFLOPS: {bwd_tflops:<5.0f}"
            )

    # ====================================================================
    # 5) Local baselines full T
    # ====================================================================
    local_fwd_inf_T_ms = None
    local_fwd_train_T_ms = None
    local_bwd_T_ms = None
    if not args.skip_local_T:
        if rank == 0:
            print0(f"\n[bold]── Per-rank baselines (T={T}, single-GPU full-scale) ──[/bold]")

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
                f" {f'Per-rank Fwd (T={T}, training)':<42} Average time: {local_fwd_train_T_ms:<8.2f} ms, TFLOPS: {tflops:<5.0f}"
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
                    f" {f'Per-rank Bwd (T={T})':<42} Average time: {local_bwd_T_ms:<8.2f} ms, TFLOPS: {bwd_tflops:<5.0f}"
                )

    # ====================================================================
    # 6) Exposed network latency summary
    # ====================================================================
    #
    # "Exposed network" = EP_time − per-rank-baseline_time. It's the
    # extra wall time EP costs vs. running just the local T_local-row
    # forward/backward — i.e., the part of comm + symm-mem orchestration
    # that wasn't hidden behind compute. Per-rank NVLink bytes for each
    # phase scale by dispatch / combine mode; see ``benchmarks/
    # distributed/bench-ep-comm.py`` for the per-primitive breakdown.
    if rank == 0:
        print0("\n[bold]══ Exposed network latency (EP vs. local T_local) ══[/bold]")

        # Left-pad labels to a 24-char column so the value column starts
        # at the same position. ms numbers use 2 decimals and left-align
        # so the digit before the decimal lines up across rows.
        LBL = 24

        exposed_train_ms = ep_fwd_train_ms - local_fwd_train_Tl_ms
        exposed_train_pct = exposed_train_ms / ep_fwd_train_ms * 100 if ep_fwd_train_ms > 0 else 0.0
        print0(
            f"  Training fwd:\n"
            f"    {'EP Fwd:':<{LBL}} {ep_fwd_train_ms:<8.2f} ms\n"
            f"    {'Per-rank Fwd (T_local):':<{LBL}} {local_fwd_train_Tl_ms:<8.2f} ms\n"
            f"    {'Exposed network:':<{LBL}} {exposed_train_ms:<8.2f} ms  "
            f"({exposed_train_pct:.1f}% of EP time)"
        )

        if not args.skip_bench_bwd and ep_bwd_ms is not None and local_bwd_Tl_ms is not None:
            exposed_bwd_ms = ep_bwd_ms - local_bwd_Tl_ms
            exposed_bwd_pct = exposed_bwd_ms / ep_bwd_ms * 100 if ep_bwd_ms > 0 else 0.0
            print0(
                f"  Backward:\n"
                f"    {'EP Bwd:':<{LBL}} {ep_bwd_ms:<8.2f} ms\n"
                f"    {'Per-rank Bwd (T_local):':<{LBL}} {local_bwd_Tl_ms:<8.2f} ms\n"
                f"    {'Exposed network:':<{LBL}} {exposed_bwd_ms:<8.2f} ms  "
                f"({exposed_bwd_pct:.1f}% of EP time)"
            )

        if not args.skip_local_T and local_fwd_train_T_ms is not None:
            speedup = local_fwd_train_T_ms / ep_fwd_train_ms if ep_fwd_train_ms > 0 else 0.0
            ideal_speedup = world_size
            scaling_eff = speedup / ideal_speedup * 100 if ideal_speedup > 0 else 0.0
            print0(
                f"\n  EP scaling efficiency (training fwd):\n"
                f"    {f'Single-GPU (T={T}):':<{LBL}} {local_fwd_train_T_ms:<8.2f} ms\n"
                f"    {f'EP W={world_size} (T={T}):':<{LBL}} {ep_fwd_train_ms:<8.2f} ms\n"
                f"    {'Observed speedup:':<{LBL}} {speedup:<8.2f}× "
                f"(ideal {ideal_speedup}×, efficiency {scaling_eff:.1f}%)"
            )

            if not args.skip_bench_bwd and ep_bwd_ms is not None and local_bwd_T_ms is not None:
                bwd_speedup = local_bwd_T_ms / ep_bwd_ms if ep_bwd_ms > 0 else 0.0
                bwd_scaling_eff = bwd_speedup / ideal_speedup * 100 if ideal_speedup > 0 else 0.0
                print0(
                    f"\n  EP scaling efficiency (backward):\n"
                    f"    {f'Single-GPU (T={T}):':<{LBL}} {local_bwd_T_ms:<8.2f} ms\n"
                    f"    {f'EP W={world_size} (T={T}):':<{LBL}} {ep_bwd_ms:<8.2f} ms\n"
                    f"    {'Observed speedup:':<{LBL}} {bwd_speedup:<8.2f}× "
                    f"(ideal {ideal_speedup}×, efficiency {bwd_scaling_eff:.1f}%)"
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
