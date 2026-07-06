# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
"""Forward-only MXFP8 blockscaled MoE (SM90 / Hopper).

Activations are quantized with 1x128 K-blocks (`blockwise_quant`) and weights
with 128x128 blocks (`quantize_weight_sm90`); the grouped GEMMs use QuACK's SM90
blockscaled kernels. This path is inference-only — it builds no autograd graph
and re-quantizes the weights each call.
"""

import torch
import torch.nn.functional as F
from quack.gemm_blockscaled_interface import (
    mxfp8_gemm_act_sm90,
    mxfp8_gemm_gated_tuned_sm90,
    quantize_weight_sm90,
)
from quack.gemm_config import GemmConfig
from quack.quant import blockwise_quant, dqaccum_total_padded_m

from ..enums import ActivationType, is_glu
from .forward import _router_forward, _topk_softmax_fwd
from .triton_kernels import TC_topk_router_metadata_triton, gather_padded_sfa

_SF = 128  # SM90 K-block / dQaccum row-pad granularity


def moe_TC_softmax_topk_layer_fp8(
    x: torch.Tensor,  # (T, H) bf16
    router_w: torch.Tensor,  # (E, H)
    w1: torch.Tensor,  # c_fc.weight: (E, 2*I, H)  [N=2I, K=H], interleaved [gate, up]
    w2: torch.Tensor,  # c_proj.weight: (E, H, I)  [N=H,  K=I]
    K: int,
    activation_type: ActivationType,
    is_softmax_over_topk: bool = True,
    norm_topk_probs: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward-only MXFP8 MoE. Returns (output (T, H), router_logits, expert_frequency)."""
    assert is_glu(activation_type), "fp8 MoE only supports GLU activations (QuACK gated GEMM)"
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

    # ── Router + top-k (bf16) ────────────────────────────────────────────────
    router_logits = F.linear(x, router_w)
    topk_scores = torch.empty(T, K, dtype=torch.float32, device=device)
    topk_indices = torch.empty(T, K, dtype=torch.int32, device=device)
    _topk_softmax_fwd(
        router_logits,
        topk_scores,
        topk_indices,
        E,
        K,
        is_softmax_over_topk=is_softmax_over_topk,
        norm_topk_probs=norm_topk_probs,
    )

    # ── Routing metadata (expert grouping / permutation) ─────────────────────
    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    expert_frequency = torch.empty(E, dtype=torch.int32, device=device)
    expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)
    total_padded_M = dqaccum_total_padded_m(TK, E)
    padded_gather_idx = torch.empty(total_padded_M, dtype=torch.int32, device=device)
    padded_grouped_idx = torch.empty(TK, dtype=torch.int32, device=device)
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

    w1_q, w1_sc = quantize_weight_sm90(w1)  # (E, 2I, H), (E, 2I/128, H/128)
    B1, B1_sc = w1_q.mT, w1_sc.mT  # (E, H, 2I), (E, H/128, 2I/128)
    w2_q, w2_sc = quantize_weight_sm90(w2)  # (E, H, I), (E, H/128, I/128)
    B2, B2_sc = w2_q.mT, w2_sc.mT  # (E, I, H), (E, I/128, H/128)

    x_q, x_sc = blockwise_quant(x, block_size=_SF, scale_transpose=True)  # (T,H) fp8, (T,H/128) f32
    sfa_up = gather_padded_sfa(x_sc, padded_gather_idx, total_padded_M)
    a = torch.empty(TK, I, dtype=x.dtype, device=device)  # postact (gated -> N//2 = I)
    gather_config = GemmConfig(
        tile_m=128,
        tile_n=128,
        cluster_m=1,
        cluster_n=1,
        pingpong=False,
        is_dynamic_persistent=False,
    )
    mxfp8_gemm_gated_tuned_sm90(
        x_q,
        B1,
        sfa_up,
        B1_sc,
        None,  # preact_out: inference, don't store
        a,  # postact_out
        None,  # C
        None,  # bias
        activation_type.value,  # "swiglu" / "geglu"
        expert_frequency_offset,  # cu_seqlens_m
        x_gather_idx,  # A_idx (gather_A)
        False,  # dynamic_scheduler
        config=gather_config,
    )

    a_q, sfa_down = blockwise_quant(
        a,
        block_size=_SF,
        scale_transpose=True,
        scale_row_idx=padded_grouped_idx,
        scale_rows=total_padded_M,
    )  # a_q (TK,I) fp8; sfa_down (total_padded_M, I/128) dQaccum-padded
    _, y = mxfp8_gemm_act_sm90(
        a_q,
        B2,
        sfa_down,
        B2_sc,
        activation=None,
        out_dtype=x.dtype,
        postact_dtype=x.dtype,
        cu_seqlens_m=expert_frequency_offset,
        store_preact=False,
        tuned=False,
    )  # y: (TK, H), expert-grouped

    # ── Combine: weighted gather-sum of each token's K expert outputs ────────
    o = torch.empty(T, H, device=device, dtype=x.dtype)
    _router_forward(
        y=y,
        o=o,
        topk_scores=topk_scores.view(-1),
        s_reverse_scatter_idx=s_reverse_scatter_idx,
        num_activated_expert_per_token_offset=None,
        varlen_K_max=K,
        H=H,
        is_varlen_K=False,
    )

    return o, router_logits, expert_frequency
