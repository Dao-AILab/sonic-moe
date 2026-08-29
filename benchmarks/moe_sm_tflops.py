# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
"""MoE compute throughput vs SM count on one GPU (green-context partition), mirroring moe-cute.py.
Kernels autotune on the FULL device then replay confined to N SMs, so N<full includes the partition tax."""
from __future__ import annotations

import argparse
import contextlib
import itertools
from functools import partial

# ─────────────── Monkey-patch: shrink SM100 autotuning (from benchmarks/moe-cute.py) ───────────────
# Correctness-neutral; only reduces the CuTeDSL autotune search so warmup is fast.
import quack.gemm_config as _gc
import torch
from cuda.bindings import driver as cu
from quack.autotuner import AutotuneConfig
from quack.gemm_config import GemmConfig
from quack.gemm_interface import gemm_dgated_tuned, gemm_gated_tuned, gemm_tuned
from triton.testing import do_bench

from sonicmoe import MoE
from sonicmoe.enums import ActivationType, is_glu
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
            tile_m=m, tile_n=n, cluster_m=cm, cluster_n=cn, swap_ab=sab,
            max_swizzle_size=8, is_dynamic_persistent=use_clc, use_tma_gather=use_tma_gather,
        )
        for (m, n, (cm, cn)), sab, use_clc, use_tma_gather in itertools.product(
            tile_mn_cluster_vals, swap_ab_vals, use_clc_vals, use_tma_gather_vals
        )
    ]


_gc._get_sm100_configs = _fast_sm100_configs


def _patch_autotuner_configs(autotuner_fn):
    autotuner_fn.configs = [AutotuneConfig(config=c) for c in _gc.get_all_configs()]


_patch_autotuner_configs(gemm_tuned)
_patch_autotuner_configs(gemm_gated_tuned)
_patch_autotuner_configs(gemm_dgated_tuned)
gemm_gated_tuned.configs = [AutotuneConfig(config=c) for c in _gc.get_all_configs("gated")]
gemm_dgated_tuned.configs = [AutotuneConfig(config=c) for c in _gc.get_all_configs("gated")]
# ─────────────── Monkey-patch ends ───────────────

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


# ============================================================ SM-limited stream (green context) ===
def _ck(ret):
    """cuda-python returns (CUresult, *outs); assert success and unwrap the outputs."""
    err = ret[0]
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver error: {err!r}")
    outs = ret[1:]
    return outs[0] if len(outs) == 1 else outs


def device_total_sms(device_index: int) -> int:
    _ck(cu.cuInit(0))
    dev = _ck(cu.cuDeviceGet(device_index))
    res = _ck(cu.cuDeviceGetDevResource(dev, cu.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM))
    return int(res.sm.smCount)


class SMStream:
    """A torch CUDA stream confined to ~``num_sms`` SMs via a green context (``None``/default stream if
    ``num_sms >= total``). QuACK's persistent GEMMs adapt automatically — the partition just slows them."""

    def __init__(self, device_index: int, num_sms: int):
        _ck(cu.cuInit(0))
        dev = _ck(cu.cuDeviceGet(device_index))
        res = _ck(cu.cuDeviceGetDevResource(dev, cu.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM))
        self.total = int(res.sm.smCount)
        if num_sms >= self.total:
            self.sms, self.stream, self._gctx = self.total, None, None
            return
        align = int(res.sm.smCoscheduledAlignment) or 8
        n = max(align, ((num_sms + align - 1) // align) * align)
        groups, _nb, _rem = _ck(cu.cuDevSmResourceSplitByCount(1, res, 0, n))
        part = groups[0]
        self.sms = int(part.sm.smCount)
        desc = _ck(cu.cuDevResourceGenerateDesc([part], 1))
        self._gctx = _ck(cu.cuGreenCtxCreate(desc, dev, cu.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM))
        cu_stream = _ck(cu.cuGreenCtxStreamCreate(self._gctx, cu.CUstream_flags.CU_STREAM_NON_BLOCKING, 0))
        self.stream = torch.cuda.ExternalStream(int(cu_stream), device=torch.device("cuda", device_index))

    def ctx(self):
        return torch.cuda.stream(self.stream) if self.stream is not None else contextlib.nullcontext()


# ============================================================ benchmark one shape =================
def _tflops(flops: int, ms: float) -> float:
    return flops / (ms * 1e9)  # flops / (ms/1e3) / 1e12


def bench_shape(T, H, I, E, K, sms_list, torch_dtype, activation, peak, warmup, rep, device_index):
    """Return per-SM rows [(sms, fwd_inf_ms, fwd_train_ms, fwd_bwd_ms), ...] and the FLOP factors."""
    torch.manual_seed(1111)
    torch.cuda.manual_seed_all(1111)

    moe = MoE(
        num_experts=E, num_experts_per_tok=K, hidden_size=H, intermediate_size=I,
        activation_function=activation, add_bias=False, std=0.02,
    ).to(dtype=torch_dtype).cuda()
    x = 0.2 * torch.randn(T, H, device="cuda", dtype=torch_dtype, requires_grad=True)
    dout = 0.2 * torch.randn_like(x)
    w1, w2, router_w = moe.c_fc.weight, moe.c_proj.weight, moe.router.weight
    b1, b2 = moe.c_fc.bias, moe.c_proj.bias

    glu = is_glu(activation)
    fwd_flops = (6 if glu else 4) * T * I * H * K
    fb_flops = (18 if glu else 12) * T * I * H * K
    bwd_flops = (12 if glu else 8) * T * I * H * K

    def call(is_inference):
        return moe_TC_softmax_topk_layer(
            x, router_w, w1.permute(1, 2, 0), b1, w2.permute(1, 2, 0), b2,
            moe.top_k, None, activation, is_inference,
        )[0]

    def fwd_inf():
        return call(True)

    def fwd_train():
        return call(False)

    def fwd_bwd():
        o = call(False)
        o.backward(dout, retain_graph=True)
        x.grad = w1.grad = w2.grad = router_w.grad = None

    # Warmup on the FULL device: trigger autotune + CuTe compile so the config is fixed before we
    # confine to a partition (so every SM count runs the same, full-device-tuned kernel).
    for _ in range(3):
        fwd_bwd()
        fwd_inf()
    torch.cuda.synchronize()

    rows = []
    for want in sms_list:
        s = SMStream(device_index, want)
        with s.ctx():
            fi = do_bench(fwd_inf, warmup=warmup, rep=rep)
            ft = do_bench(fwd_train, warmup=warmup, rep=rep)
            fb = do_bench(fwd_bwd, warmup=warmup, rep=rep)
        rows.append((s.sms, fi, ft, fb))
    return rows, (fwd_flops, bwd_flops, fb_flops, s.total)


# ============================================================ printing ============================
def print_table(T, H, I, E, K, activation, dtype_name, rows, flops, peak):
    fwd_flops, bwd_flops, fb_flops, total_sms = flops
    params_moe = E * H + E * 2 * I * H + E * H * I
    print(
        f"\n=== T={T} H={H} I={I} E={E} K={K}  {activation.value} {dtype_name}  "
        f"| {params_moe / 1e6:.0f}M params/layer  "
        f"| fwd {fwd_flops / 1e12:.2f} / bwd {bwd_flops / 1e12:.2f} / f+b {fb_flops / 1e12:.2f} TFLOP/step ===",
        flush=True,
    )
    print(
        f"{'SMs':>4} | {'fwd_inf':>15} | {'fwd_train':>15} | {'backward':>15} | "
        f"{'fwd+bwd':>15} | {'%pk':>4} {'%pk/sm':>6} {'keep':>5}",
        flush=True,
    )
    full_fb_tf = None
    for sms, fi, ft, fb in rows:
        bw = fb - ft  # backward-only time = (fwd+bwd) - (training forward)
        fi_tf, ft_tf, bw_tf, fb_tf = (
            _tflops(fwd_flops, fi), _tflops(fwd_flops, ft), _tflops(bwd_flops, bw), _tflops(fb_flops, fb),
        )
        if full_fb_tf is None:
            full_fb_tf = fb_tf  # first row = highest SM count = the reference for "keep"
        pk = fb_tf / peak * 100
        pk_sm = fb_tf / (peak * sms / total_sms) * 100  # vs the SM-scaled peak (per-SM efficiency)
        keep = fb_tf / full_fb_tf * 100
        print(
            f"{sms:>4} | {fi:6.2f}ms {fi_tf:5.0f}TF | {ft:6.2f}ms {ft_tf:5.0f}TF | "
            f"{bw:6.2f}ms {bw_tf:5.0f}TF | {fb:6.2f}ms {fb_tf:5.0f}TF | "
            f"{pk:3.0f}% {pk_sm:5.0f}% {keep:4.0f}%",
            flush=True,
        )


def parse_shapes(s):
    shapes = []
    for spec in s.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        vals = tuple(int(v.strip()) for v in spec.split(","))
        if len(vals) != 5:
            raise argparse.ArgumentTypeError(f"shape {spec!r} must be T,H,I,E,K (5 ints)")
        shapes.append(vals)
    return shapes


def parse_sms(s):
    return [int(v.strip()) for v in s.split(",") if v.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shapes", type=parse_shapes, default="32768,2560,1280,64,4;32768,4096,1024,128,8",
                   help='semicolon-separated "T,H,I,E,K" tuples')
    p.add_argument("--sms", type=parse_sms, default="152,120,96,64",
                   help="comma-separated SM counts (>=total => full device)")
    p.add_argument("--activation", default="swiglu",
                   choices=["swiglu", "geglu", "reglu", "relu_sq", "relu", "silu", "gelu"])
    p.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    p.add_argument("--peak", type=float, default=2250.0, help="bf16 dense peak TFLOP/s for %%-of-peak (GB200=2250)")
    p.add_argument("--warmup", type=int, default=25, help="do_bench warmup budget (ms)")
    p.add_argument("--rep", type=int, default=300, help="do_bench measurement budget (ms)")
    p.add_argument("--device", type=int, default=0)
    a = p.parse_args()
    if isinstance(a.shapes, str):
        a.shapes = parse_shapes(a.shapes)
    if isinstance(a.sms, str):
        a.sms = parse_sms(a.sms)

    torch.cuda.set_device(a.device)
    torch_dtype = DTYPES[a.dtype]
    activation = ActivationType(a.activation)
    total = device_total_sms(a.device)
    name = torch.cuda.get_device_name(a.device)
    print(f"device: {name}  total SMs: {total}  peak(bf16 dense): {a.peak:.0f} TFLOP/s", flush=True)
    print(f"SM sweep: {a.sms}  (values >= {total} use the full device)", flush=True)

    for (T, H, I, E, K) in a.shapes:
        rows, flops = bench_shape(
            T, H, I, E, K, a.sms, torch_dtype, activation, a.peak, a.warmup, a.rep, a.device,
        )
        print_table(T, H, I, E, K, activation, a.dtype, rows, flops, a.peak)


if __name__ == "__main__":
    main()
