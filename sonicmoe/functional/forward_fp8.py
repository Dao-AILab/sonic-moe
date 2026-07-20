# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
"""MXFP8 blockscaled MoE (SM90 / Hopper).

Activations are quantized with 1x128 K-blocks (`blockwise_quant`) and weights
with 128x128 blocks (`quantize_weight_sm90`); the grouped GEMMs use QuACK's SM90
blockscaled kernels. Builds an autograd graph (see `.backward_fp8`): gradients
w.r.t. activations reuse the same MXFP8 blockscaled grouped GEMMs as the forward,
while gradients w.r.t. weights use the bf16 grouped GEMM.
"""

import torch
import torch.nn.functional as F

from quack.quant import dqaccum_total_padded_m

from ..enums import ActivationType, is_glu
from .backward import TC_Softmax_Topk_Router_Function
from .backward_fp8 import _DownProjectionFP8, _UpProjectionFP8
from .fp8_tensor import FP8BlockwiseTensor
from .triton_kernels import TC_topk_router_metadata_triton

_SF = 128  # SM90 K-block / dQaccum row-pad granularity


def moe_TC_softmax_topk_layer_fp8(
    x: torch.Tensor,  # (T, H) bf16 — always full precision (router + weight-grad GEMM)
    router_w: torch.Tensor,  # (E, H)
    w1: torch.Tensor | FP8BlockwiseTensor,  # c_fc.weight: (E, 2*I, H) [N=2I, K=H], interleaved [gate, up]
    w2: torch.Tensor | FP8BlockwiseTensor,  # c_proj.weight: (E, H, I) [N=H, K=I]
    K: int,
    activation_type: ActivationType,
    is_softmax_over_topk: bool = True,
    norm_topk_probs: bool = False,
    concat_layout: bool = False,
    is_inference_mode: bool = False,
    x_fp8: FP8BlockwiseTensor | None = None,  # optional pre-quantized activations for the up GEMM
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MXFP8 blockscaled MoE. Returns (output (T, H), router_logits, expert_frequency).

    Weights (`w1`, `w2`) and activations may arrive already quantized:
      - Pass `w1`/`w2` as `FP8BlockwiseTensor` (e.g. `FP8BlockwiseTensor.to_weights(w)`)
        to skip the internal `quantize_weight_sm90`; a plain bf16 tensor is quantized
        internally as before.
      - Pass `x_fp8 = FP8BlockwiseTensor.to_activations(x)` (or the fused RMSNorm quant
        output) to skip the up-projection's `quantize_act(x)`. `x` itself must still be
        the bf16 activation — the router and the bf16 weight-grad GEMM need full precision.

    `is_inference_mode` skips storing the up-projection's preact (a memory/bandwidth
    optimization only) — the caller must not call `.backward()` through the output
    when it's True.
    """
    assert is_glu(activation_type), "fp8 MoE only supports GLU activations (QuACK gated GEMM)"
    assert not (concat_layout and isinstance(w1, FP8BlockwiseTensor)), (
        "concat_layout is not supported with pre-quantized FP8BlockwiseTensor weights"
    )
    if x_fp8 is not None:
        assert x_fp8._quant_block_size == (1, _SF), "x_fp8 must be a 1x128 activation quant"
    E, N1, H = w1.shape
    assert N1 % 2 == 0
    I = N1 // 2
    assert H % _SF == 0 and I % _SF == 0, (
        f"fp8 MoE requires H ({H}) and I ({I}) divisible by {_SF}"
    )
    assert w2.shape == (E, H, I), f"w2 shape {tuple(w2.shape)} != {(E, H, I)}"
    device = x.device
    T = x.size(0)
    TK = T * K

    # ── Router + top-k (bf16, autograd-tracked) ──────────────────────────────
    router_logits = F.linear(x, router_w)
    topk_scores, topk_indices = TC_Softmax_Topk_Router_Function.apply(
        router_logits, E, K, is_softmax_over_topk, norm_topk_probs
    )

    # ── Routing metadata (expert grouping / permutation) ─────────────────────
    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    expert_frequency = torch.empty(E, dtype=torch.int32, device=device)
    expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)
    total_padded_M = dqaccum_total_padded_m(TK, E)
    padded_gather_idx = torch.empty(total_padded_M, dtype=torch.int32, device=device)
    padded_grouped_idx = torch.empty(total_padded_M, dtype=torch.int32, device=device)
    TC_topk_router_metadata_triton(
        topk_indices,
        E,
        expert_frequency,
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        padded_gather_idx,
        padded_grouped_idx,
    )

    a, h = _UpProjectionFP8.apply(
        x,
        w1,
        expert_frequency_offset,
        x_gather_idx,
        s_reverse_scatter_idx,
        padded_gather_idx,
        padded_grouped_idx,
        total_padded_M,
        T,
        TK,
        K,
        activation_type,
        concat_layout,
        is_inference_mode,
        x_fp8,
    )

    o = _DownProjectionFP8.apply(
        a,
        h,
        w2,
        topk_scores,
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        padded_gather_idx,
        padded_grouped_idx,
        total_padded_M,
        T,
        K,
        activation_type,
    )

    return o, router_logits, expert_frequency
