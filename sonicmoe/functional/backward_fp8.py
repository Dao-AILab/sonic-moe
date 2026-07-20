# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
"""Backward pass for MXFP8 blockscaled MoE (SM90).

Split precision, mirroring the forward path:
  - Gradients w.r.t. activations (dx from the up-projection, da/dh from the
    down-projection) are computed with MXFP8 blockscaled grouped GEMMs, same
    kernels as the forward. The activation-grad GEMMs contract over the forward
    GEMM's N axis, so they need the weight in the transposed layout. Rather than
    re-quantizing, the forward-layout fp8 quant is saved in ctx and transposed with
    a compiled kernel (`_transpose_weight_qsc`, ~1.25x faster than re-quantizing).
    The down-projection's `dout @ w2`
    GEMM fuses the SwiGLU/GeGLU backward into its epilogue (`GemmDGatedSm90`
    composed with the blockscaled SM90 mainloop — see
    `quack.gemm_blockscaled_sm90.mxfp8_gemm_dgated_sm90`), same as the bf16
    path's fused `gemm_dgated`; the up-projection's `dh @ w1` has no
    activation to fuse, so it's a plain blockscaled GEMM.
  - Gradients w.r.t. weights (dw1, dw2) are computed with the bf16 grouped GEMM
    (`quack.gemm_interface.gemm`), same as the bf16 sonicmoe backward path.
"""

import torch
from quack.gemm_blockscaled_sm90 import (
    mxfp8_gemm_act_sm90,
    mxfp8_gemm_dgated_sm90,
    mxfp8_gemm_gated_tuned_sm90,
    quantize_act,
    quantize_weight_sm90,
)
from quack.gemm_config import GemmConfig
from quack.gemm_interface import gemm

from ..enums import ActivationType
from .backward import _token_broadcast_backward
from .forward import _router_forward
from .fp8_tensor import FP8BlockwiseTensor
from .triton_kernels import gather_padded_sfa

_GATHER_CONFIG = GemmConfig(
    tile_m=128, tile_n=256, cluster_m=2, cluster_n=1, pingpong=False, is_dynamic_persistent=False
)


def _weight_fwd_qsc(
    w: torch.Tensor | FP8BlockwiseTensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward-layout blockwise-FP8 (qdata, scale) for a weight: reuse a pre-quantized
    `FP8BlockwiseTensor`'s stored 128x128 payload, or quantize a bf16 weight."""
    if isinstance(w, FP8BlockwiseTensor):
        assert w._quant_block_size == (128, 128), "weight must be a 128x128 quant"
        return w._data, w._scale
    return quantize_weight_sm90(w)


@torch.compile(dynamic=False)
def _transpose_weight_qsc(
    q: torch.Tensor, sc: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transpose an fp8 forward-layout weight payload into the backward's
    contraction-swapped layout (the activation-grad GEMMs contract over the forward
    GEMM's N axis). Bitwise identical to `quantize_weight_sm90(w, transpose=True)`
    (square blocks -> per-block amax is transpose-invariant), but from the fp8 forward
    quant saved in ctx rather than re-reading bf16 — ~1.25x faster than re-quantizing.

    `dynamic=False`: weight shapes are static, and a dynamic-shape transpose kernel
    benchmarks ~50% slower here. Eager `.mT.contiguous()` on fp8 is catastrophic
    (~7x slower — a strided 1-byte transpose is uncoalesced), so this must be compiled."""
    return q.mT.contiguous(), sc.mT.contiguous()


def _weight_grad_dtype(w: torch.Tensor | FP8BlockwiseTensor) -> torch.dtype:
    """Autograd-visible dtype for a weight's (dense) gradient — the bf16 weight's own
    dtype, or an `FP8BlockwiseTensor`'s `_grad_dtype`."""
    return w._grad_dtype if isinstance(w, FP8BlockwiseTensor) else w.dtype


def _to_interleaved_from_concat(w: torch.Tensor, I: int) -> torch.Tensor:
    """(..., 2I, H) concat [gate(I); up(I)] -> interleaved [gate0, up0, gate1, up1, ...]."""
    gate, up = w[..., :I, :], w[..., I:, :]
    stacked = torch.stack((gate, up), dim=-2)  # (..., I, 2, H)
    return stacked.reshape(*w.shape[:-2], 2 * I, w.shape[-1])


class _UpProjectionFP8(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,  # (T, H) bf16
        w1: torch.Tensor,  # (E, 2I, H) bf16 or FP8BlockwiseTensor
        expert_frequency_offset: torch.Tensor,  # (E+1,)
        x_gather_idx: torch.Tensor,  # (TK,)
        s_reverse_scatter_idx: torch.Tensor,  # (TK,)
        padded_gather_idx: torch.Tensor,  # (total_padded_M,) token-order -> padded row
        padded_grouped_idx: torch.Tensor,  # (total_padded_M,) grouped-order -> padded row
        total_padded_M: int,
        T: int,
        TK: int,
        K: int,
        activation_type: ActivationType,
        concat_layout: bool,
        is_inference_mode: bool,
        x_fp8: FP8BlockwiseTensor | None,  # pre-quantized activations, or None to quantize x here
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        E, N1, H = w1.shape
        I = N1 // 2
        device = x.device

        if x_fp8 is not None:
            x_q, x_sc = x_fp8._data, x_fp8._scale
        else:
            x_q, x_sc = quantize_act(x)
        w1_q, w1_sc = _weight_fwd_qsc(w1)
        B1, B1_sc = w1_q.mT, w1_sc.mT
        sfa_up = gather_padded_sfa(x_sc, padded_gather_idx, total_padded_M)
        a = torch.empty(TK, I, dtype=x.dtype, device=device)
        h = torch.empty(TK, 2 * I, dtype=x.dtype, device=device) if not is_inference_mode else None
        mxfp8_gemm_gated_tuned_sm90.fn(
            x_q,
            B1,
            sfa_up,
            B1_sc,
            h,  # preact_out
            a,  # postact_out
            None,  # C
            None,  # bias
            activation_type.value,
            expert_frequency_offset,
            x_gather_idx,
            False,
            config=_GATHER_CONFIG,
            concat_layout=("B",) if concat_layout else None,
        )

        # Store the forward-layout weight quant (fp8) — the backward transposes it
        # instead of re-quantizing (~1.25x faster). Only concat_layout still needs the
        # bf16 weight, to re-interleave + re-quantize (a transpose can't reorder rows).
        ctx.save_for_backward(
            x,
            w1_q,
            w1_sc,
            w1 if concat_layout else None,
            expert_frequency_offset,
            x_gather_idx,
            s_reverse_scatter_idx,
            padded_grouped_idx,
        )
        ctx.T = T
        ctx.K = K
        ctx.H = H
        ctx.I = I
        ctx.total_padded_M = total_padded_M
        ctx.concat_layout = concat_layout
        ctx.w1_grad_dtype = _weight_grad_dtype(w1)
        ctx.mark_non_differentiable(a)
        ctx.set_materialize_grads(False)
        return a, h

    @staticmethod
    def backward(ctx, _: None, dh: torch.Tensor):
        (x, w1_q, w1_sc, w1_concat, expert_frequency_offset, x_gather_idx, s_reverse_scatter_idx, padded_grouped_idx) = (
            ctx.saved_tensors
        )
        T, K, H, I = ctx.T, ctx.K, ctx.H, ctx.I
        total_padded_M = ctx.total_padded_M
        concat_layout = ctx.concat_layout
        device = dh.device

        dh = dh.contiguous()

        # ── activations grad (fp8 GG): dx_expanded = dh @ w1 ──────────────────
        dh_q, dh_sc = quantize_act(dh)
        sfa_dx = gather_padded_sfa(dh_sc, padded_grouped_idx, total_padded_M)
        if concat_layout:
            # concat weight's backward needs the *interleaved*-layout transposed quant,
            # which a transpose of the concat-layout forward quant can't produce.
            w1_bwd_q, w1_bwd_sc = quantize_weight_sm90(
                _to_interleaved_from_concat(w1_concat, I), transpose=True
            )
        else:
            w1_bwd_q, w1_bwd_sc = _transpose_weight_qsc(w1_q, w1_sc)
        B1_bwd, B1_bwd_sc = w1_bwd_q.mT, w1_bwd_sc.mT  # (E, 2I, H) K(2I)-contig

        _, dx_expanded = mxfp8_gemm_act_sm90(
            dh_q,
            B1_bwd,
            sfa_dx,
            B1_bwd_sc,
            activation=None,
            out_dtype=dh.dtype,
            postact_dtype=dh.dtype,
            cu_seqlens_m=expert_frequency_offset,
            store_preact=False,
            tuned=False,
        )  # (TK, H)

        dx_reduced = torch.empty(T, H, dtype=dh.dtype, device=device)
        _token_broadcast_backward(
            dx_reduced=dx_reduced,
            dx_expanded=dx_expanded,
            s_reverse_scatter_idx=s_reverse_scatter_idx,
            num_activated_expert_per_token_offset=None,
            varlen_K_max=K,
            H=H,
            is_varlen_K=False,
        )

        # ── weight grad (bf16 GG): dw1 = dh^T @ x, per-expert grouped ─────────
        dw1 = torch.empty(w1_q.shape, dtype=ctx.w1_grad_dtype, device=device)
        gemm(
            x.T,
            dh,
            out=dw1.mT,  # (E, H, 2I) view of (E, 2I, H)
            cu_seqlens_k=expert_frequency_offset,
            A_idx=x_gather_idx,
            batch_idx_permute=None,
            dynamic_scheduler=False,
            concat_layout=(("out",) if concat_layout else None),
        )

        # trailing None grads: (..., x_fp8). dw1 is routed to the wrapper weight and,
        # via its straight-through ToFp8, to the underlying bf16 parameter.
        return dx_reduced, dw1, *[None] * 13


class _DownProjectionFP8(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,  # (TK, I) bf16 postact
        h: torch.Tensor,  # (TK, 2I) bf16 preact (from the up-projection)
        w2: torch.Tensor,  # (E, H, I) bf16 or FP8BlockwiseTensor
        topk_scores: torch.Tensor,  # (T, K) float32
        expert_frequency_offset: torch.Tensor,  # (E+1,)
        x_gather_idx: torch.Tensor,  # (TK,)
        s_scatter_idx: torch.Tensor,  # (TK,)
        s_reverse_scatter_idx: torch.Tensor,  # (TK,)
        padded_gather_idx: torch.Tensor,  # (total_padded_M,) token-order -> padded row
        padded_grouped_idx: torch.Tensor,  # (total_padded_M,) grouped-order -> padded row
        total_padded_M: int,
        T: int,
        K: int,
        activation_type: ActivationType,
    ) -> torch.Tensor:
        TK = a.size(0)
        E, H, I = w2.shape
        device = a.device

        w2_q, w2_sc = _weight_fwd_qsc(w2)
        B2, B2_sc = w2_q.mT, w2_sc.mT

        a_q, a_sc = quantize_act(a)
        sfa_down = gather_padded_sfa(a_sc, padded_grouped_idx, total_padded_M)
        _, y = mxfp8_gemm_act_sm90(
            a_q,
            B2,
            sfa_down,
            B2_sc,
            activation=None,
            out_dtype=a.dtype,
            postact_dtype=a.dtype,
            cu_seqlens_m=expert_frequency_offset,
            store_preact=False,
            tuned=False,
        )  # (TK, H), expert-grouped, pre-combine

        o = torch.empty(T, H, device=device, dtype=a.dtype)
        topk_scores_flat = topk_scores.reshape(-1)
        _router_forward(
            y=y,
            o=o,
            topk_scores=topk_scores_flat,
            s_reverse_scatter_idx=s_reverse_scatter_idx,
            num_activated_expert_per_token_offset=None,
            varlen_K_max=K,
            H=H,
            is_varlen_K=False,
        )

        # Store the forward-layout weight quant (fp8); the backward transposes it.
        ctx.save_for_backward(
            h, w2_q, w2_sc, topk_scores_flat, expert_frequency_offset, x_gather_idx, s_scatter_idx, padded_gather_idx
        )
        ctx.T = T
        ctx.K = K
        ctx.total_padded_M = total_padded_M
        ctx.activation_type = activation_type
        ctx.w2_grad_dtype = _weight_grad_dtype(w2)
        return o

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        (h, w2_q, w2_sc, topk_scores, expert_frequency_offset, x_gather_idx, s_scatter_idx, padded_gather_idx) = (
            ctx.saved_tensors
        )
        T, K = ctx.T, ctx.K
        total_padded_M = ctx.total_padded_M
        activation_type = ctx.activation_type
        device = dout.device

        dout = dout.contiguous()
        # per-grouped-row score: o[t] = sum_k s[t,k] * y[row(t,k)], so the grad
        # into y (and transitively into da/dh/a_prime) needs this per-row factor.
        s = topk_scores[s_scatter_idx.long()]

        # ── activations grad (fp8 GG), fused with the SwiGLU/GeGLU backward ───
        # raw_da = dout @ w2 (gathered per (token, k)); da = s * raw_da;
        # dh = dgate_bwd(h, da); a_prime = s * a (reused below for dw2, so the
        # weight-grad GEMM doesn't need its own scaling pass); ds_partial =
        # dot(raw_da, a) = dot(dout, y), reduced below into ds.
        dout_q, dout_sc = quantize_act(dout)
        sfa_da = gather_padded_sfa(dout_sc, padded_gather_idx, total_padded_M)
        w2_bwd_q, w2_bwd_sc = _transpose_weight_qsc(w2_q, w2_sc)
        B2_bwd, B2_bwd_sc = w2_bwd_q.mT, w2_bwd_sc.mT  # (E, H, I) K(H)-contig

        dh, a_prime, ds_partial = mxfp8_gemm_dgated_sm90(
            dout_q,
            B2_bwd,
            sfa_da,
            B2_bwd_sc,
            h,
            colvec_scale=s,
            activation=activation_type.value,
            colvec_reduce=True,
            cu_seqlens_m=expert_frequency_offset,
            A_idx=x_gather_idx,
            config=_GATHER_CONFIG,
        )

        # ── weight grad (bf16 GG): dw2 = dout^T @ a_prime, per-expert grouped ──
        dw2 = torch.empty(w2_q.shape, dtype=ctx.w2_grad_dtype, device=device)
        gemm(
            dout.T,
            a_prime,
            out=dw2,  # (E, H, I), already batch-leading
            cu_seqlens_k=expert_frequency_offset,
            A_idx=x_gather_idx,
            batch_idx_permute=None,
            dynamic_scheduler=False,
        )

        ds = torch.empty_like(topk_scores)
        ds[s_scatter_idx.long()] = ds_partial.to(ds.dtype)
        ds = ds.view(T, K)

        return None, dh, dw2, ds, None, None, None, None, None, None, None, None, None, None
