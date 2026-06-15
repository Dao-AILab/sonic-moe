# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
#
# Multi-rank correctness test for the EP forward + backward.
#
# Each rank constructs the same MoE (seeded), shards weights along the leading
# E axis, runs the EP forward AND backward on its slice of the global x,
# all-gathers the per-rank outputs and gradients, and compares against a
# single-rank PyTorch autograd reference computed on rank 0.
#
# Sweep: bias × routing-variant × {3 dispatch modes × 3 combine modes}
#        × {TC_softmax_topk (fwd + bwd), general_routing (fwd only)}
# General routing's fwd+bwd combination currently trips a QuACK DSL
# compile-time ICE in epi_ops.cute.copy on the test's smaller shapes —
# tracked in QuACK, not in EP. TC covers the backward sweep.
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
import itertools
import os
import sys
import traceback
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Tuple

import quack.autotuner
import torch


# !!!!!!!! sometimes there is a broken pipe issue when QuACK compile worker > 1 !!!!!!!!
os.environ["QUACK_COMPILE_WORKERS"] = "1"


# ─────────────── Monkey-patch: similar M shapes map to the same cached config during QuACK autotuning ───────────────
# Mirrors ``benchmarks/distributed/moe-ep.py``; without these patches the
# QuACK autotuner picks tiles for ``gemm_dgated`` at the test's smaller
# shapes that hit a bf16 alignment ICE in epi_ops. Same restricted
# config set the bench uses keeps both files consistent.

M_QUANT = 1024


def _make_quantized_key(self, args, kwargs):
    all_args = {**dict(zip(self.arg_names, args)), **kwargs}
    _args = {k: v for k, v in all_args.items() if k in self.arg_names}
    key = [str(_args[k]) for k in self.keys if k in _args]
    for _, arg in _args.items():
        if isinstance(arg, torch.Tensor):
            s = list(arg.shape)
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
            config = self.cache[qkey]
            self.best_config = config
            self.nargs = dict(zip(self.arg_names, args))
            ret = self.fn.__call__(*args, **kwargs, **config.all_kwargs())
            self.nargs = None
            return ret
    ret = _orig_call(self, *args, **kwargs)
    if len(self.configs) > 1 and hasattr(self, "best_config"):
        qkey = _make_quantized_key(self, args, kwargs)
        self.cache[qkey] = self.best_config
    return ret


quack.autotuner.Autotuner.__call__ = _patched_call


# ─────────────── Monkey-patch: restrict SM100 / SM90 autotuning to the same set the bench uses ───────────────

import quack.gemm_config as _gc
from quack.autotuner import AutotuneConfig
from quack.gemm_config import GemmConfig
from quack.gemm_interface import gemm_dgated_tuned, gemm_gated_tuned, gemm_tuned


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
            is_dynamic_persistent=False,
            use_tma_gather=False,
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
    autotuner_fn.configs = [AutotuneConfig(config=c) for c in _gc.get_all_configs()]


_patch_autotuner_configs(gemm_tuned)
_patch_autotuner_configs(gemm_gated_tuned)
_patch_autotuner_configs(gemm_dgated_tuned)
gemm_gated_tuned.configs = [AutotuneConfig(config=c) for c in _gc.get_all_configs("gated")]
gemm_dgated_tuned.configs = [AutotuneConfig(config=c) for c in _gc.get_all_configs("gated")]


import torch.distributed as dist
import torch.nn.functional as F

from sonicmoe import MoE
from sonicmoe.distributed_utils import (  # type: ignore
    CombineMode,
    DispatchMode,
    RuntimeEPConfig,
    clear_ep_cache,
    stash_overflowed,
)
from sonicmoe.enums import ActivationType
from sonicmoe.functional import TC_Softmax_Topk_Router_Function
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


# Full 3 × 3 sweep of dispatch × combine primitives.
DISPATCH_MODES: List[DispatchMode] = [
    DispatchMode.AG_DISPATCH_TRITON,
    DispatchMode.A2A_DISPATCH_TRITON,
    DispatchMode.RANK_DEDUP_DISPATCH_TRITON,
]
COMBINE_MODES: List[CombineMode] = [
    CombineMode.A2A_COMBINE_TRITON,
    CombineMode.RS_COMBINE_TRITON,
    CombineMode.RANK_DEDUP_COMBINE_TRITON,
]


# ============================================================================
# Helpers
# ============================================================================


def _swiglu(h: torch.Tensor, concat_layout: bool = False) -> torch.Tensor:
    if concat_layout:
        g, u = torch.chunk(h, 2, dim=-1)
    else:
        u, g = h[..., 1::2], h[..., ::2]
    return u * F.silu(g)


def _all_gather_y(y_local: torch.Tensor, world_size: int) -> torch.Tensor:
    """All-gather a (T_local, *trailing) tensor along dim 0 into (W*T_local, *trailing)."""
    out_shape = (world_size * y_local.shape[0],) + tuple(y_local.shape[1:])
    out = torch.empty(out_shape, dtype=y_local.dtype, device=y_local.device)
    dist.all_gather_into_tensor(out, y_local.contiguous())
    return out


def _gather_to_rank0(t_local: torch.Tensor, world_size: int, rank: int) -> Optional[List[torch.Tensor]]:
    """Gather t_local from every rank into a list of W tensors on rank 0."""
    t_local = t_local.contiguous()
    out = torch.empty((world_size,) + tuple(t_local.shape), dtype=t_local.dtype, device=t_local.device)
    dist.all_gather_into_tensor(out, t_local)
    if rank == 0:
        return list(out.unbind(dim=0))
    return None


def _strided_clone(t: torch.Tensor) -> torch.Tensor:
    """Preserve the non-contiguous strided layout that quack's grouped GEMM
    requires for ``w1`` (``(2I, H, E_local)`` view of the original contiguous
    ``(E_local, 2I, H)`` backing). A plain ``.clone()`` would reset strides to
    the contiguous layout for the new shape and trip the kernel's stride
    check."""
    out = torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device)
    return out.copy_(t)


# ============================================================================
# References — single-rank PyTorch autograd in fp32, used for fwd+bwd compare.
# ============================================================================


def _per_expert_reference_tc(
    x_global: torch.Tensor,
    router_w: torch.Tensor,
    w1_full: torch.Tensor,
    w2_full: torch.Tensor,
    b1_full: Optional[torch.Tensor],
    b2_full: Optional[torch.Tensor],
    topk_idx_global: torch.Tensor,
    is_softmax_over_topk: bool,
    norm_topk_probs: bool,
    concat_layout: bool,
    dout_global: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """TC reference: per-expert MoE forward + autograd backward in fp32.

    Uses ``topk_idx_global`` from the EP path so EP and the reference agree
    on which experts each token routes to (TC's topk tie-breaking is its
    own kernel — driving the reference off PyTorch's ``logits.topk`` would
    diverge under ties and produce incomparable backward grads).
    """
    ref_x = x_global.detach().to(torch.float32).requires_grad_(True)
    ref_router_w = router_w.detach().to(torch.float32).requires_grad_(True)
    ref_w1 = w1_full.detach().to(torch.float32).requires_grad_(True)
    ref_w2 = w2_full.detach().to(torch.float32).requires_grad_(True)
    ref_b1 = b1_full.detach().to(torch.float32).requires_grad_(True) if b1_full is not None else None
    ref_b2 = b2_full.detach().to(torch.float32).requires_grad_(True) if b2_full is not None else None

    logits = F.linear(ref_x, ref_router_w)
    if is_softmax_over_topk:
        topk_logits = torch.gather(logits, 1, topk_idx_global)
        topk_scores = topk_logits.softmax(dim=-1)
    else:
        probs = logits.softmax(dim=-1)
        topk_scores = torch.gather(probs, 1, topk_idx_global)
        if norm_topk_probs:
            topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)

    T, H = ref_x.shape
    E = ref_w1.shape[0]
    out = torch.zeros(T, H, dtype=torch.float32, device=ref_x.device)
    for i in range(E):
        rows_t, rows_k = (topk_idx_global == i).nonzero(as_tuple=True)
        if rows_t.numel() == 0:
            continue
        h = F.linear(
            ref_x[rows_t],
            ref_w1[i],
            bias=(ref_b1[i] if ref_b1 is not None else None),
        )
        h = _swiglu(h, concat_layout=concat_layout)
        y = F.linear(
            h,
            ref_w2[i],
            bias=(ref_b2[i] if ref_b2 is not None else None),
        )
        contrib = y * topk_scores[rows_t, rows_k, None]
        out = out.index_add(0, rows_t, contrib)

    inputs = [ref_x, ref_router_w, ref_w1, ref_w2]
    if ref_b1 is not None:
        inputs += [ref_b1, ref_b2]
    grads = torch.autograd.grad(out, inputs, grad_outputs=dout_global.to(torch.float32))

    ref_dx, ref_drouter_w, ref_dw1, ref_dw2 = grads[:4]
    ref_db1 = grads[4] if ref_b1 is not None else None
    ref_db2 = grads[5] if ref_b1 is not None else None
    return (
        out.detach(),
        ref_dx.detach(),
        ref_drouter_w.detach(),
        ref_dw1.detach(),
        ref_dw2.detach(),
        ref_db1.detach() if ref_db1 is not None else None,
        ref_db2.detach() if ref_db2 is not None else None,
    )


def _per_expert_reference_general_fwd(
    x_global: torch.Tensor,
    w1_full: torch.Tensor,
    w2_full: torch.Tensor,
    b1_full: Optional[torch.Tensor],
    b2_full: Optional[torch.Tensor],
    topk_idx_global: torch.Tensor,
    topk_scores_global: torch.Tensor,
    concat_layout: bool,
) -> torch.Tensor:
    """General-routing forward reference in fp32 — output only (no bwd).
    See ``_run_ep_general_one_fwd`` for why general is forward-only here."""
    ref_x = x_global.to(torch.float32)
    ref_scores = topk_scores_global.to(torch.float32)
    ref_w1 = w1_full.to(torch.float32)
    ref_w2 = w2_full.to(torch.float32)
    ref_b1 = b1_full.to(torch.float32) if b1_full is not None else None
    ref_b2 = b2_full.to(torch.float32) if b2_full is not None else None

    T, H = ref_x.shape
    E = ref_w1.shape[0]
    out = torch.zeros(T, H, dtype=torch.float32, device=ref_x.device)
    for i in range(E):
        rows_t, rows_k = (topk_idx_global == i).nonzero(as_tuple=True)
        if rows_t.numel() == 0:
            continue
        h = F.linear(
            ref_x[rows_t],
            ref_w1[i],
            bias=(ref_b1[i] if ref_b1 is not None else None),
        )
        h = _swiglu(h, concat_layout=concat_layout)
        y = F.linear(
            h,
            ref_w2[i],
            bias=(ref_b2[i] if ref_b2 is not None else None),
        )
        contrib = y * ref_scores[rows_t, rows_k, None]
        out = out.index_add(0, rows_t, contrib)
    return out


# ============================================================================
# Comparison
# ============================================================================


def _check_quantities(
    tag: str,
    log_prefix: str,
    quantities: List[Tuple[str, torch.Tensor, torch.Tensor]],
    atol: float,
    rtol: float,
) -> Tuple[bool, str]:
    """Element-wise compare each (ep, ref) pair with assert_close. Annotate
    pass/fail per quantity with max-abs / mean-rel stats for diagnostics."""
    msgs = []
    ok_all = True
    for name, ep, ref in quantities:
        ep_f = ep.float()
        ref_f = ref.float()
        diff = (ep_f - ref_f).abs()
        max_d = diff.max().item()
        mean_r = (diff / (ref_f.abs() + 1e-6)).mean().item()
        try:
            torch.testing.assert_close(ep_f, ref_f, atol=atol, rtol=rtol)
            msgs.append(f"{name}:OK")
        except AssertionError:
            ok_all = False
            msgs.append(f"{name}:FAIL[max={max_d:.2e} rel={mean_r:.2e}]")
    head = f"{log_prefix}{tag}"
    marker = "✓ PASS" if ok_all else "✗ FAIL"
    return ok_all, f"{head:<88s} {marker}  " + "  ".join(msgs)


# ============================================================================
# Per-config EP runners (forward + backward, gather grads to rank 0).
# ============================================================================


def _run_ep_tc_one(
    x_local: torch.Tensor,
    router_w: torch.Tensor,
    w1_local: torch.Tensor,
    w2_local: torch.Tensor,
    b1_local: Optional[torch.Tensor],
    b2_local: Optional[torch.Tensor],
    dout_local: torch.Tensor,
    K: int,
    E: int,
    cfg: RuntimeEPConfig,
    is_softmax_over_topk: bool,
    norm_topk_probs: bool,
    concat_layout: bool,
    world_size: int,
    rank: int,
    stash_activations: bool = False,
    stash_pool_rows: Optional[int] = None,
):
    """Run one TC EP fwd+bwd; gather per-rank grads to rank 0. Returns the
    (y_full, dx_full, drouter_w, ep_dw1, ep_dw2, ep_db1, ep_db2) tuple on
    rank 0; returns ``None`` on other ranks."""
    x_t = x_local.detach().clone().requires_grad_(True)
    router_w_t = router_w.detach().clone().requires_grad_(True)
    w1_t = _strided_clone(w1_local).requires_grad_(True)
    w2_t = w2_local.detach().clone().requires_grad_(True)
    b1_t = b1_local.detach().clone().requires_grad_(True) if b1_local is not None else None
    b2_t = b2_local.detach().clone().requires_grad_(True) if b2_local is not None else None

    y_local = moe_ep_TC_softmax_topk_forward(
        x_t,
        router_w_t,
        w1_t,
        b1_t,
        w2_t,
        b2_t,
        K=K,
        E=E,
        activation_type=ActivationType.SWIGLU,
        is_inference_mode_enabled=False,
        is_softmax_over_topk=is_softmax_over_topk,
        norm_topk_probs=norm_topk_probs,
        concat_layout=concat_layout,
        ep_config=cfg,
        stash_activations=stash_activations,
        stash_pool_rows=stash_pool_rows,
    )

    inputs = [x_t, router_w_t, w1_t, w2_t]
    if b1_t is not None:
        inputs += [b1_t, b2_t]
    grads = torch.autograd.grad(y_local, inputs, grad_outputs=dout_local, retain_graph=False)

    dx_local, drouter_w_local, dw1_local, dw2_local = grads[:4]
    db1_local = grads[4] if b1_t is not None else None
    db2_local = grads[5] if b1_t is not None else None

    y_full = _all_gather_y(y_local.detach(), world_size)
    dx_full = _all_gather_y(dx_local, world_size)
    dw1_list = _gather_to_rank0(dw1_local, world_size, rank)
    dw2_list = _gather_to_rank0(dw2_local, world_size, rank)
    db1_list = _gather_to_rank0(db1_local, world_size, rank) if db1_local is not None else None
    db2_list = _gather_to_rank0(db2_local, world_size, rank) if db2_local is not None else None

    if rank == 0:
        # dw1 per rank is (2I, H, E_local) — concat along E_local axis (dim 2)
        # to reconstruct (2I, H, E). dw2 per rank is (E_local, I, H) — concat
        # along dim 0 to reconstruct (E, I, H).
        ep_dw1 = torch.cat(dw1_list, dim=2)
        ep_dw2 = torch.cat(dw2_list, dim=0)
        ep_db1 = torch.cat(db1_list, dim=0) if db1_list is not None else None
        ep_db2 = torch.cat(db2_list, dim=0) if db2_list is not None else None
        return y_full, dx_full, drouter_w_local.detach(), ep_dw1, ep_dw2, ep_db1, ep_db2
    return None


def _run_ep_general_one_fwd(
    x_local: torch.Tensor,
    idx_local: torch.Tensor,
    scores_local: torch.Tensor,
    w1_local: torch.Tensor,
    w2_local: torch.Tensor,
    b1_local: Optional[torch.Tensor],
    b2_local: Optional[torch.Tensor],
    E: int,
    cfg: RuntimeEPConfig,
    concat_layout: bool,
    world_size: int,
    rank: int,
):
    """Run general-routing EP forward in inference mode and return y_full
    on rank 0. Inference mode is used because training-mode + general
    routing currently trips a QuACK DSL ICE in ``epi_ops.cute.copy``
    (bf16 column-vector alignment) when ``gemm_dgated`` autotunes for
    the test's smaller shapes. Forward-only is enough to exercise the
    3×3 dispatch×combine matrix on the general entry point; TC covers
    the backward sweep."""
    w1_no_grad = _strided_clone(w1_local)
    y_local = moe_ep_general_routing_forward(
        x_local,
        idx_local,
        scores_local,
        w1_no_grad,
        b1_local,
        w2_local,
        b2_local,
        E=E,
        activation_type=ActivationType.SWIGLU,
        is_inference_mode_enabled=True,
        concat_layout=concat_layout,
        ep_config=cfg,
    )
    y_full = _all_gather_y(y_local.detach(), world_size)
    if rank == 0:
        return y_full
    return None


# ============================================================================
# Per-shape test driver
# ============================================================================


@dataclass
class ShapeStats:
    shape_name: str
    pass_count: int = 0
    fail_count: int = 0
    failures: list = field(default_factory=list)


def _ep_topk_indices_global(
    x_local: torch.Tensor,
    router_w: torch.Tensor,
    K: int,
    W_E_local: int,
    is_softmax_over_topk: bool,
    norm_topk_probs: bool,
    world_size: int,
    T_local: int,
    T: int,
    device: torch.device,
) -> torch.Tensor:
    """Run the TC topk on each rank's local logits, all-gather, return the
    flat (T, K) int64 indices. Used to seed the reference so it picks the
    same experts that EP's TC topk did (deterministic tie-breaking match)."""
    with torch.no_grad():
        logits = F.linear(x_local, router_w)
        _, topk_idx_local = TC_Softmax_Topk_Router_Function.apply(
            logits,
            W_E_local,
            K,
            is_softmax_over_topk,
            norm_topk_probs,
        )
    topk_idx_global = torch.empty(world_size, T_local, K, dtype=torch.int32, device=device)
    dist.all_gather_into_tensor(
        topk_idx_global.view(-1),
        topk_idx_local.view(-1).contiguous(),
        group=dist.group.WORLD,
    )
    return topk_idx_global.view(T, K).to(torch.int64)


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

    # dout used for the backward seed — broadcast so every rank has the
    # same view and the rank-0 reference matches what each EP rank computes.
    if rank == 0:
        dout_global = 0.2 * torch.randn(T, H, device=device, dtype=dtype)
    else:
        dout_global = torch.empty(T, H, device=device, dtype=dtype)
    dist.broadcast(dout_global, src=0)
    dout_local = dout_global[rank * T_local : (rank + 1) * T_local].contiguous()

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
    idx_local = idx_g[rank * T_local : (rank + 1) * T_local].to(torch.int32).contiguous()

    # ------------------------------------------------------------------------
    # Sweep: bias × routing-variant × 3×3 (dispatch × combine) × entry point.
    # ------------------------------------------------------------------------
    for use_bias in (False, True):
        b1_local = b1_local_with if use_bias else None
        b2_local = b2_local_with if use_bias else None
        b1_full = b1_with if use_bias else None
        b2_full = b2_with if use_bias else None
        log_prefix = f"[W={world_size} {shape.name} bias={int(use_bias)}] "

        # ────────── entry point #1: TC_softmax_topk_forward ──────────
        for variant_name, is_softmax_over_topk, norm_topk_probs in ROUTING_VARIANTS:
            # Use TC's own topk decision (gathered globally) so the
            # reference picks the same experts under ties.
            topk_idx_global = _ep_topk_indices_global(
                x_local,
                router_w,
                K,
                world_size * E_local,
                is_softmax_over_topk,
                norm_topk_probs,
                world_size,
                T_local,
                T,
                device,
            )
            if rank == 0:
                ref = _per_expert_reference_tc(
                    x_global,
                    router_w,
                    w1_full,
                    w2_full,
                    b1_full,
                    b2_full,
                    topk_idx_global,
                    is_softmax_over_topk,
                    norm_topk_probs,
                    concat_layout,
                    dout_global,
                )
                ref_o, ref_dx, ref_drouter_w, ref_dw1, ref_dw2, ref_db1, ref_db2 = ref

            for dispatch_mode in DISPATCH_MODES:
                for combine_mode in COMBINE_MODES:
                    cfg = RuntimeEPConfig(
                        dispatch_mode=dispatch_mode,
                        W=world_size,
                        K=K,
                        combine_mode=combine_mode,
                    )
                    tag = f"TC[{variant_name},dispatch={dispatch_mode.value},combine={combine_mode.value}]"
                    try:
                        result = _run_ep_tc_one(
                            x_local,
                            router_w,
                            w1_local,
                            w2_local,
                            b1_local,
                            b2_local,
                            dout_local,
                            K,
                            E,
                            cfg,
                            is_softmax_over_topk,
                            norm_topk_probs,
                            concat_layout,
                            world_size,
                            rank,
                        )
                    except Exception as e:
                        if rank == 0:
                            print(f"{log_prefix}{tag:<88s} ✗ EXC   {type(e).__name__}: {str(e)[:160]}")
                            stats.fail_count += 1
                            stats.failures.append(f"{tag} (exception)")
                        dist.barrier()
                        continue
                    if rank == 0:
                        y_full, ep_dx, ep_drouter_w, ep_dw1, ep_dw2, ep_db1, ep_db2 = result
                        quantities = [
                            ("o", y_full, ref_o),
                            ("dx", ep_dx, ref_dx),
                            ("drouter_w", ep_drouter_w, ref_drouter_w),
                            ("dw1", ep_dw1, ref_dw1.permute(1, 2, 0)),  # (2I, H, E)
                            ("dw2", ep_dw2, ref_dw2.permute(0, 2, 1)),  # (E, I, H)
                        ]
                        if ep_db1 is not None:
                            quantities.append(("db1", ep_db1, ref_db1))
                            quantities.append(("db2", ep_db2, ref_db2))
                        ok, msg = _check_quantities(tag, log_prefix, quantities, atol, rtol)
                        print(msg)
                        if ok:
                            stats.pass_count += 1
                        else:
                            stats.fail_count += 1
                            stats.failures.append(tag)
                    dist.barrier()

        # ────────── entry point #2: general_routing_forward (fwd-only) ──────────
        if rank == 0:
            ref_o_g = _per_expert_reference_general_fwd(
                x_global,
                w1_full,
                w2_full,
                b1_full,
                b2_full,
                idx_g,
                scores_g,
                concat_layout,
            )

        for dispatch_mode in DISPATCH_MODES:
            for combine_mode in COMBINE_MODES:
                cfg = RuntimeEPConfig(
                    dispatch_mode=dispatch_mode,
                    W=world_size,
                    K=K,
                    combine_mode=combine_mode,
                )
                tag = f"general[dispatch={dispatch_mode.value},combine={combine_mode.value}]"
                try:
                    y_full = _run_ep_general_one_fwd(
                        x_local,
                        idx_local,
                        scores_local,
                        w1_local,
                        w2_local,
                        b1_local,
                        b2_local,
                        E,
                        cfg,
                        concat_layout,
                        world_size,
                        rank,
                    )
                except Exception as e:
                    if rank == 0:
                        print(f"{log_prefix}{tag:<88s} ✗ EXC   {type(e).__name__}: {str(e)[:160]}")
                        stats.fail_count += 1
                        stats.failures.append(f"{tag} (exception)")
                    dist.barrier()
                    continue
                if rank == 0:
                    quantities = [("o", y_full, ref_o_g)]
                    ok, msg = _check_quantities(tag, log_prefix, quantities, atol, rtol)
                    print(msg)
                    if ok:
                        stats.pass_count += 1
                    else:
                        stats.fail_count += 1
                        stats.failures.append(tag)
                dist.barrier()
        dist.barrier()

    return stats


def _run_stash_checks(
    rank: int,
    world_size: int,
    device: torch.device,
    shape: Shape,
    dtype: torch.dtype,
    concat_layout: bool,
    seed: int,
) -> ShapeStats:
    """Validate ``stash_activations`` (bump-stash of the saved up-proj h):

    (1) bit-exact (torch.equal) vs the no-stash path on output + every grad,
        across all dispatch/combine modes (single layer). stash is a pure copy
        round-trip, so any diff is a real bug, not rounding.
    (2) bit-exact across a real MULTI-LAYER autograd stack — this is what
        actually exercises the FILO pop order the single bump head relies on.
    (3) a deliberate FILO-order violation (pop an earlier layer before a later
        one) must RAISE rather than silently corrupt.
    """
    T, H, I, E, K = shape.T, shape.H, shape.I, shape.E, shape.K
    T_local = T // world_size
    E_local = E // world_size
    e_slc = slice(rank * E_local, (rank + 1) * E_local)
    max_rows_static = T_local * world_size * min(K, E_local)
    stats = ShapeStats(shape.name + "[stash]")

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
    for p in moe.parameters():
        dist.broadcast(p.data, src=0)
    w1_full, w2_full = moe.c_fc.weight, moe.c_proj.weight
    b1_with, b2_with, router_w = moe.c_fc.bias, moe.c_proj.bias, moe.router.weight
    w1_local = w1_full[e_slc].permute(1, 2, 0)
    w2_local = w2_full[e_slc].permute(0, 2, 1).contiguous()
    b1_local = b1_with[e_slc].contiguous()
    b2_local = b2_with[e_slc].contiguous()

    if rank == 0:
        x_global = 0.2 * torch.randn(T, H, device=device, dtype=dtype)
    else:
        x_global = torch.empty(T, H, device=device, dtype=dtype)
    dist.broadcast(x_global, src=0)
    x_local = x_global[rank * T_local : (rank + 1) * T_local].contiguous()
    if rank == 0:
        dout_global = 0.2 * torch.randn(T, H, device=device, dtype=dtype)
    else:
        dout_global = torch.empty(T, H, device=device, dtype=dtype)
    dist.broadcast(dout_global, src=0)
    dout_local = dout_global[rank * T_local : (rank + 1) * T_local].contiguous()

    def _record(tag: str, ok: bool) -> None:
        if rank == 0:
            print(f"[W={world_size} {shape.name} stash] {tag:<68s} {'✓ PASS' if ok else '✗ FAIL'}")
            if ok:
                stats.pass_count += 1
            else:
                stats.fail_count += 1
                stats.failures.append(tag)
        dist.barrier()

    GRAD_NAMES = ["o", "dx", "drouter_w", "dw1", "dw2", "db1", "db2"]

    # ── (1) single-layer exact equivalence across all modes ──────────────────
    for dispatch_mode in DISPATCH_MODES:
        for combine_mode in COMBINE_MODES:
            cfg = RuntimeEPConfig(dispatch_mode=dispatch_mode, W=world_size, K=K, combine_mode=combine_mode)
            tag = f"equiv[{dispatch_mode.value},{combine_mode.value}]"
            try:
                args_common = (
                    x_local,
                    router_w,
                    w1_local,
                    w2_local,
                    b1_local,
                    b2_local,
                    dout_local,
                    K,
                    E,
                    cfg,
                    True,
                    False,
                    concat_layout,
                    world_size,
                    rank,
                )
                ref = _run_ep_tc_one(*args_common, stash_activations=False)
                got = _run_ep_tc_one(*args_common, stash_activations=True, stash_pool_rows=2 * max_rows_static)
            except Exception as e:  # noqa: BLE001
                if rank == 0:
                    print(f"[W={world_size} {shape.name} stash] {tag:<68s} ✗ EXC {type(e).__name__}: {str(e)[:120]}")
                    stats.fail_count += 1
                    stats.failures.append(f"{tag} (exc)")
                clear_ep_cache()
                dist.barrier()
                continue
            ok = True
            if rank == 0:
                for name, a, b in zip(GRAD_NAMES, ref, got):
                    if a is None and b is None:
                        continue
                    if not torch.equal(a, b):
                        ok = False
                        md = (a - b).abs().max().item()
                        print(f"   {tag} {name} mismatch under stash (max|Δ|={md:.3e})")
            _record(tag, ok)

    # ── (2) multi-layer end-to-end FILO (exercises real backward pop order) ──
    def _two_layer(cfg, stash):
        xt = x_local.detach().clone().requires_grad_(True)
        rw = router_w.detach().clone().requires_grad_(True)
        w1t = _strided_clone(w1_local).requires_grad_(True)
        w2t = w2_local.detach().clone().requires_grad_(True)
        b1t = b1_local.detach().clone().requires_grad_(True)
        b2t = b2_local.detach().clone().requires_grad_(True)
        kw = dict(
            K=K,
            E=E,
            activation_type=ActivationType.SWIGLU,
            is_inference_mode_enabled=False,
            concat_layout=concat_layout,
            ep_config=cfg,
            stash_activations=stash,
            stash_pool_rows=2 * max_rows_static,
        )
        y1 = moe_ep_TC_softmax_topk_forward(xt, rw, w1t, b1t, w2t, b2t, **kw)
        y2 = moe_ep_TC_softmax_topk_forward(y1, rw, w1t, b1t, w2t, b2t, **kw)
        g = torch.autograd.grad(y2, [xt, rw, w1t, w2t, b1t, b2t], grad_outputs=dout_local)
        return [y2.detach()] + [t.detach() for t in g]

    for dispatch_mode, combine_mode in [
        (DispatchMode.AG_DISPATCH_TRITON, CombineMode.RS_COMBINE_TRITON),
        (DispatchMode.A2A_DISPATCH_TRITON, CombineMode.A2A_COMBINE_TRITON),
        (DispatchMode.RANK_DEDUP_DISPATCH_TRITON, CombineMode.RANK_DEDUP_COMBINE_TRITON),
    ]:
        cfg = RuntimeEPConfig(dispatch_mode=dispatch_mode, W=world_size, K=K, combine_mode=combine_mode)
        tag = f"multilayer[{dispatch_mode.value},{combine_mode.value}]"
        local_ok = True
        try:
            ref = _two_layer(cfg, False)
            got = _two_layer(cfg, True)
            local_ok = all(torch.equal(a, b) for a, b in zip(ref, got))
        except Exception as e:  # noqa: BLE001
            local_ok = False
            if rank == 0:
                print(f"   {tag} EXC {type(e).__name__}: {str(e)[:120]}")
        flag = torch.tensor([0 if local_ok else 1], device=device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.SUM)  # any rank mismatch -> fail
        _record(tag, int(flag.item()) == 0)

    # ── (3) a FILO-order violation must raise (not silently corrupt) ─────────
    cfg = RuntimeEPConfig(
        dispatch_mode=DispatchMode.AG_DISPATCH_TRITON, W=world_size, K=K, combine_mode=CombineMode.RS_COMBINE_TRITON
    )
    xt = x_local.detach().clone().requires_grad_(True)
    rw = router_w.detach().clone().requires_grad_(True)
    w1t = _strided_clone(w1_local).requires_grad_(True)
    w2t = w2_local.detach().clone().requires_grad_(True)
    b1t = b1_local.detach().clone().requires_grad_(True)
    b2t = b2_local.detach().clone().requires_grad_(True)
    kw = dict(
        K=K,
        E=E,
        activation_type=ActivationType.SWIGLU,
        is_inference_mode_enabled=False,
        concat_layout=concat_layout,
        ep_config=cfg,
        stash_activations=True,
        stash_pool_rows=2 * max_rows_static,
    )
    raised = False
    try:
        y1 = moe_ep_TC_softmax_topk_forward(xt, rw, w1t, b1t, w2t, b2t, **kw)  # pushes level 0
        _y2 = moe_ep_TC_softmax_topk_forward(xt, rw, w1t, b1t, w2t, b2t, **kw)  # pushes level 1 (independent)
        # Popping level-0's backward while level-1 is on top must trip the guard.
        torch.autograd.grad(y1.sum(), [xt], retain_graph=True)
    except RuntimeError:
        raised = True
    except Exception:  # noqa: BLE001
        raised = False
    clear_ep_cache()  # the bump stack is now corrupt; drop the workspace
    flag = torch.tensor([1 if raised else 0], device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)  # require ALL ranks to have raised
    _record("filo_violation_raises", int(flag.item()) == 1)

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
    parser.add_argument(
        "--stash",
        action="store_true",
        help="Also run the stash_activations validation suite (bit-exact vs no-stash "
        "across all modes, multi-layer FILO, and the FILO-violation-raises check).",
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
    # (F.linear in the references) is genuinely fp32 and not silently downcast
    # on Ampere+.
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
            f"per (bias × routing) cell exercises 3 dispatch × 3 combine modes\n"
            f"for both TC_softmax_topk and general_routing entry points,\n"
            f"comparing fwd + bwd grads against an fp32 PyTorch autograd reference.\n"
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

            if args.stash:
                try:
                    stash_stats = _run_stash_checks(
                        rank,
                        world_size,
                        device,
                        shape,
                        dtype=torch.bfloat16,
                        concat_layout=args.concat_layout,
                        seed=1111,
                    )
                except Exception as e:
                    if rank == 0:
                        print(f"[ERR {shape.name}[stash]] {e}")
                        traceback.print_exc()
                    stash_stats = ShapeStats(shape.name + "[stash]")
                    stash_stats.fail_count = 1
                    stash_stats.failures.append(f"exception: {e}")
                all_stats.append(stash_stats)
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
