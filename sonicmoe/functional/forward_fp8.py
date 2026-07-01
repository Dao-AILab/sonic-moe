# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
"""Forward-only MXFP8 blockscaled MoE (SM90 / Hopper).

Activations are quantized with 1x128 K-blocks (`quantize_act_sm90`) and weights
with 128x128 blocks (`quantize_weight_sm90`); the grouped GEMMs use QuACK's SM90
blockscaled kernels. This path is inference-only — it builds no autograd graph
and re-quantizes the weights each call.
"""

import torch
import torch.nn.functional as F
from quack.gemm_blockscaled_interface import (
    mxfp8_gemm_act_sm90,
    mxfp8_gemm_gated_tuned_sm90,
    quantize_act_sm90,
    quantize_weight_sm90,
)
from quack.gemm_config import GemmConfig

from ..enums import ActivationType, is_glu
from .forward import _router_forward, _topk_softmax_fwd
from .triton_kernels import TC_topk_router_metadata_triton

_SF = 128  # SM90 K-block / dQaccum row-pad granularity


def _grouped_padded_sfa(
    x_scale: torch.Tensor, x_gather_idx: torch.Tensor, expert_frequency_offset: torch.Tensor
) -> torch.Tensor:
    """Gather per-token activation scales into expert-grouped order and dQaccum-pad.

    `x_scale` is (T, sf_k) in token order (from `quantize_act_sm90`). The up-proj
    gathers activations on the fly via `x_gather_idx` (grouped_pos -> token), so the
    scales must be pre-arranged to match: grouped[p] = x_scale[x_gather_idx[p]], then
    each expert's rows padded to start on a 128-row boundary (see
    quack/AI/varlen_blockscaled_sf_layout.md). Returns (total_padded_m, sf_k),
    M-innermost (stride-1 M) so the kernel's (BLOCK_M, 1) TMA burst stays contiguous.
    """
    grouped = x_scale[x_gather_idx.long()]  # (TK, sf_k), expert-sorted order
    total_m, sf_k = grouped.shape
    seqlens_m = (expert_frequency_offset[1:] - expert_frequency_offset[:-1]).cpu().tolist()
    L = len(seqlens_m)
    total_padded_m = ((total_m + _SF - 1) // _SF + (L - 1)) * _SF
    padded = grouped.new_zeros(sf_k, total_padded_m)
    row = 0
    for i, m_i in enumerate(seqlens_m):
        row_padded = (row // _SF + i) * _SF
        padded[:, row_padded : row_padded + m_i] = grouped[row : row + m_i].mT
        row += m_i
    return padded.mT  # (total_padded_m, sf_k), M innermost


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
    TC_topk_router_metadata_triton(
        topk_indices,
        E,
        expert_frequency,
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
    )

    # ── Quantize weights (128x128) ───────────────────────────────────────────
    # quantize_weight_sm90(w) : w (E, N, K) -> q (E, N, K), scale (E, N/128, K/128).
    # The kernel wants B as (K, N) K-contig, so pass q.mT / scale.mT (per QuACK convention).
    w1_q, w1_sc = quantize_weight_sm90(w1)  # (E, 2I, H), (E, 2I/128, H/128)
    B1, B1_sc = w1_q.mT, w1_sc.mT  # (E, H, 2I), (E, H/128, 2I/128)
    w2_q, w2_sc = quantize_weight_sm90(w2)  # (E, H, I), (E, H/128, I/128)
    B2, B2_sc = w2_q.mT, w2_sc.mT  # (E, I, H), (E, I/128, H/128)

    # ── Up projection: gated GEMM with on-the-fly gather of x by expert ──────
    # gather_A needs cluster_n == 1; pin an explicit config for the up-proj.
    x_q, x_sc = quantize_act_sm90(x)  # (T, H) fp8, (T, H/128) f32 (token order)
    sfa_up = _grouped_padded_sfa(x_sc, x_gather_idx, expert_frequency_offset)
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

    # ── Down projection: grouped GEMM (a already expert-grouped, no gather) ──
    a_q, sfa_down = quantize_act_sm90(a, cu_seqlens_m=expert_frequency_offset)
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
