# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import os

import torch
import torch.nn.functional as F
from quack.gemm_interface import gemm, gemm_dgated, gemm_gated

from ..enums import ActivationType, is_glu

_FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
_FP8_E5M2_MAX = torch.finfo(torch.float8_e5m2).max    # 57344.0


def _to_fp8_e4m3(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = x.abs().amax().clamp(min=1e-12) / _FP8_E4M3_MAX
    return (x / scale).to(torch.float8_e4m3fn), scale


def _to_fp8_e5m2(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = x.abs().amax().clamp(min=1e-12) / _FP8_E5M2_MAX
    return (x / scale).to(torch.float8_e5m2), scale


def _apply_glu_act(h: torch.Tensor, activation_type: ActivationType, concat_layout: bool) -> torch.Tensor:
    if concat_layout:
        gate, up = h.chunk(2, dim=-1)
    else:
        gate, up = h[..., ::2], h[..., 1::2]
    if activation_type == ActivationType.SWIGLU:
        return F.silu(gate) * up
    elif activation_type == ActivationType.GEGLU:
        return F.gelu(gate.float()).to(gate.dtype) * up
    else:
        raise ValueError(f"unsupported activation for fp8 path: {activation_type}")
from .backward import (
    _down_projection_backward_act,
    _token_broadcast_backward,
    _topk_softmax_bwd,
    _up_projection_backward_act,
)
from .forward import _router_forward, _topk_softmax_fwd
from .triton_kernels import TC_topk_router_metadata_triton, general_routing_router_metadata_triton


class TC_Softmax_Topk_Router_Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, router_logits: torch.Tensor, E: int, K: int, is_softmax_over_topk: bool, norm_topk_probs: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T = router_logits.size(0)

        topk_router_score = torch.empty(T, K, dtype=torch.float32, device=router_logits.device)
        topk_router_indices = torch.empty(T, K, dtype=torch.int32, device=router_logits.device)

        _topk_softmax_fwd(
            router_logits,
            topk_router_score,
            topk_router_indices,
            E,
            K,
            is_softmax_over_topk=is_softmax_over_topk,
            norm_topk_probs=norm_topk_probs,
        )

        # Save router_logits for topk(softmax()) backward (recompute full softmax).
        # For softmax(topk()) it's unused but save unconditionally for simplicity.
        ctx.save_for_backward(topk_router_score, topk_router_indices, router_logits)
        ctx.E = E
        ctx.dtype = router_logits.dtype
        ctx.is_softmax_over_topk = is_softmax_over_topk
        ctx.norm_topk_probs = norm_topk_probs

        return topk_router_score, topk_router_indices

    @staticmethod
    def backward(ctx, dtopk_score: torch.Tensor, _: torch.Tensor):
        T, K = dtopk_score.size()
        E = ctx.E
        topk_router_score, topk_router_indices, router_logits = ctx.saved_tensors
        dlogits = torch.zeros(T, ctx.E, dtype=ctx.dtype, device=topk_router_score.device)

        _topk_softmax_bwd(
            router_logits,
            dlogits,
            None,
            dtopk_score,
            topk_router_score,
            topk_router_indices,
            E,
            K,
            is_softmax_over_topk=ctx.is_softmax_over_topk,
            norm_topk_probs=ctx.norm_topk_probs,
        )

        return dlogits, None, None, None, None


class _UpProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w1: torch.Tensor,
        b1: torch.Tensor | None,
        expert_frequency_offset: torch.Tensor,
        total_expert_freq: int,
        K: int,
        x_gather_idx: torch.Tensor,
        s_scatter_idx: torch.Tensor,
        s_reverse_scatter_idx: torch.Tensor,
        num_activated_expert_per_token_offset: torch.Tensor,
        is_each_token_has_variable_activated_experts: bool,
        activation_type: ActivationType,
        is_inference_mode_enabled: bool,
        concat_layout: bool = False,
        use_fp8: bool = False,
    ) -> torch.Tensor:
        T, H = x.shape
        I, H, E = w1.shape
        is_glu_activation = is_glu(activation_type)
        if is_glu_activation:
            I //= 2
        TK = total_expert_freq

        a = torch.empty(TK, I, dtype=x.dtype, device=x.device)
        h = (
            torch.empty(TK, (2 * I if is_glu_activation else I), dtype=x.dtype, device=x.device)
            if (not is_inference_mode_enabled)
            else None
        )

        assert activation_type.value in (
            "swiglu",
            "geglu",
        ), f"QuACK gemm_gated only supports glu activations, got {activation_type.value}"

        if use_fp8:
            # Use plain gemm so we can descale before the nonlinear activation.
            # gemm_gated fuses swiglu, making descaling incorrect: swiglu(x/s) ≠ swiglu(x)/s.
            x_fp8, x_sc = _to_fp8_e4m3(x)
            w1_fp8, w1_sc = _to_fp8_e4m3(w1.permute(2, 1, 0))
            h_buf = torch.empty(TK, 2 * I if is_glu_activation else I, dtype=x.dtype, device=x.device)
            gemm(
                x_fp8, w1_fp8,
                out=h_buf,
                cu_seqlens_m=expert_frequency_offset,
                A_idx=x_gather_idx,
                dynamic_scheduler=False,
            )
            h_buf.mul_(x_sc * w1_sc)
            if not is_inference_mode_enabled:
                h.copy_(h_buf)
            a[:] = _apply_glu_act(h_buf, activation_type, concat_layout)
        else:
            gemm_gated(
                x,
                w1.permute(2, 1, 0),
                activation=activation_type.value,
                cu_seqlens_m=expert_frequency_offset,
                A_idx=x_gather_idx,
                preact_out=h,
                postact_out=a,
                store_preact=(not is_inference_mode_enabled),
                bias=b1,
                concat_layout=(("B", "bias") if b1 is not None else ("B",)) if concat_layout else None,
            )

        ctx.T = T
        ctx.TK = TK
        ctx.E = E
        ctx.K = K
        ctx.H = H
        ctx.I = I
        ctx.is_each_token_has_variable_activated_experts = is_each_token_has_variable_activated_experts
        ctx.is_glu_activation = is_glu_activation
        ctx.concat_layout = concat_layout

        ctx.save_for_backward(
            x,
            w1,
            b1,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
        )

        ctx.use_fp8 = use_fp8
        ctx.mark_non_differentiable(a)
        ctx.set_materialize_grads(False)

        return a, h

    @staticmethod
    def backward(ctx, _: None, dh: torch.Tensor):
        T = ctx.T
        TK = ctx.TK
        E = ctx.E
        K = ctx.K
        H = ctx.H
        is_glu_activation = ctx.is_glu_activation
        is_each_token_has_variable_activated_experts = ctx.is_each_token_has_variable_activated_experts
        concat_layout = ctx.concat_layout

        (
            x,
            w1,
            b1,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
        ) = ctx.saved_tensors

        dx_expanded = torch.empty(TK, H, dtype=dh.dtype, device=dh.device)
        dw1 = torch.empty_like(w1)
        db1 = None if b1 is None else torch.empty_like(b1)

        if ctx.use_fp8:
            w1_fp8, w1_sc = _to_fp8_e4m3(w1)
            dh_fp8, dh_sc = _to_fp8_e5m2(dh)
            x_fp8, x_sc = _to_fp8_e4m3(x)
            _up_projection_backward_act(
                w1=w1_fp8,
                dx_expanded=dx_expanded,
                dh=dh_fp8,
                db1=db1,
                expert_frequency_offset=expert_frequency_offset,
                is_glu_activation=is_glu_activation,
                concat_layout=concat_layout,
            )
            dx_expanded.mul_(w1_sc * dh_sc)
            gemm(
                x_fp8.T, dh_fp8,
                out=dw1.permute(2, 1, 0),
                cu_seqlens_k=expert_frequency_offset,
                A_idx=x_gather_idx,
                batch_idx_permute=None,
                dynamic_scheduler=False,
                concat_layout=(("out",) if concat_layout else None),
            )
            dw1.mul_(x_sc * dh_sc)
        else:
            _up_projection_backward_act(
                w1=w1,
                dx_expanded=dx_expanded,
                dh=dh,
                db1=db1,
                expert_frequency_offset=expert_frequency_offset,
                is_glu_activation=is_glu_activation,
                concat_layout=concat_layout,
            )
            gemm(
                x.T, dh,
                out=dw1.permute(2, 1, 0),
                cu_seqlens_k=expert_frequency_offset,
                A_idx=x_gather_idx,
                batch_idx_permute=None,
                dynamic_scheduler=False,
                concat_layout=(("out",) if concat_layout else None),
            )

        dx_reduced = torch.empty(T, H, dtype=dh.dtype, device=dh.device)

        _token_broadcast_backward(
            dx_reduced=dx_reduced,
            dx_expanded=dx_expanded,
            s_reverse_scatter_idx=s_reverse_scatter_idx,
            num_activated_expert_per_token_offset=num_activated_expert_per_token_offset,
            varlen_K_max=(E if is_each_token_has_variable_activated_experts else K),
            H=H,
            is_varlen_K=is_each_token_has_variable_activated_experts,
        )

        return dx_reduced, dw1, db1, *[None] * 13


class _DownProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        h: torch.Tensor,
        w2: torch.Tensor,
        b2: torch.Tensor | None,
        topk_scores: torch.Tensor,
        expert_frequency_offset: torch.Tensor,
        T: int,
        K: int,
        x_gather_idx: torch.Tensor,
        s_scatter_idx: torch.Tensor,
        s_reverse_scatter_idx: torch.Tensor,
        num_activated_expert_per_token_offset: torch.Tensor,
        is_varlen_K: bool,
        activation_type: ActivationType,
        use_fp8: bool = False,
    ) -> torch.Tensor:
        TK = a.size(0)
        H, I, E = w2.shape

        y = torch.empty(TK, H, dtype=a.dtype, device=a.device)

        if use_fp8:
            a_fp8, a_sc = _to_fp8_e4m3(a)
            w2_fp8, w2_sc = _to_fp8_e4m3(w2.permute(2, 1, 0))
            gemm(a_fp8, w2_fp8, out=y, cu_seqlens_m=expert_frequency_offset)
            y.mul_(a_sc * w2_sc)
        else:
            gemm(a, w2.permute(2, 1, 0), out=y, cu_seqlens_m=expert_frequency_offset, bias=b2)

        o = torch.empty(T, H, device=a.device, dtype=a.dtype)
        topk_scores = topk_scores.view(-1)

        _router_forward(
            y=y,
            o=o,
            topk_scores=topk_scores,
            s_reverse_scatter_idx=s_reverse_scatter_idx,
            num_activated_expert_per_token_offset=num_activated_expert_per_token_offset,
            varlen_K_max=(E if is_varlen_K else K),
            H=H,
            is_varlen_K=is_varlen_K,
        )

        ctx.T = T
        ctx.K = K
        ctx.is_varlen_K = is_varlen_K
        ctx.activation_type = activation_type
        ctx.use_fp8 = use_fp8

        ctx.save_for_backward(
            h,
            w2,
            b2,
            topk_scores,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
        )

        return o

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        T = ctx.T
        K = ctx.K
        is_varlen_K = ctx.is_varlen_K
        activation_type = ctx.activation_type

        (
            h,
            w2,
            b2,
            topk_scores,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
        ) = ctx.saved_tensors

        dw2 = torch.empty_like(w2)
        db2 = None if b2 is None else torch.empty_like(b2)
        dh = torch.empty_like(h)

        I = w2.size(1)
        TK = x_gather_idx.size(0)

        a_prime = torch.empty(TK, I, dtype=h.dtype, device=h.device)
        ds = torch.empty_like(topk_scores)

        if ctx.use_fp8:
            dout_fp8, dout_sc = _to_fp8_e5m2(dout)
            w2_fp8, w2_sc = _to_fp8_e4m3(w2)
            _down_projection_backward_act(
                dout=dout_fp8,
                h=h,  # must stay bf16: gemm_dgated asserts PreAct.element_size() == 2
                w2=w2_fp8,
                dh=dh,
                ds=ds,
                b2=b2,
                db2=db2,
                a_prime=a_prime,
                topk_scores=topk_scores,
                expert_frequency_offset=expert_frequency_offset,
                x_gather_idx=x_gather_idx,
                s_scatter_idx=s_scatter_idx,
                activation_type=activation_type.value,
            )
            dh.mul_(dout_sc * w2_sc)
            a_prime_fp8, ap_sc = _to_fp8_e4m3(a_prime)
            gemm(
                dout_fp8.T, a_prime_fp8,
                out=dw2.permute(2, 0, 1),
                cu_seqlens_k=expert_frequency_offset,
                A_idx=x_gather_idx,
                batch_idx_permute=None,
                dynamic_scheduler=False,
            )
            dw2.mul_(dout_sc * ap_sc)
        else:
            _down_projection_backward_act(
                dout=dout,
                h=h,
                w2=w2,
                dh=dh,
                ds=ds,
                b2=b2,
                db2=db2,
                a_prime=a_prime,
                topk_scores=topk_scores,
                expert_frequency_offset=expert_frequency_offset,
                x_gather_idx=x_gather_idx,
                s_scatter_idx=s_scatter_idx,
                activation_type=activation_type.value,
            )
            gemm(
                dout.T, a_prime,
                out=dw2.permute(2, 0, 1),
                cu_seqlens_k=expert_frequency_offset,
                A_idx=x_gather_idx,
                batch_idx_permute=None,
                dynamic_scheduler=False,
            )

        # TC top-K routing
        if not is_varlen_K:
            ds = ds.view(T, K)

        return None, dh, dw2, db2, ds, *[None] * 10


def moe_TC_softmax_topk_layer(
    x: torch.Tensor,
    router_w: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    K: int,
    stream_id: int,
    activation_type: ActivationType | str = ActivationType.SWIGLU,
    is_inference_mode_enabled: bool = False,
    is_softmax_over_topk: bool = True,
    norm_topk_probs: bool = False,
    concat_layout: bool = False,
    use_fp8: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert ((b1 is None) and (b2 is None)) or (
        (b1 is not None) and (b2 is not None)
    ), "b1 and b2 has to be None or not None at the same time!"
    E = router_w.size(0)
    router_logits = F.linear(x, router_w)
    topk_scores, topk_indices = TC_Softmax_Topk_Router_Function.apply(
        router_logits, E, K, is_softmax_over_topk, norm_topk_probs
    )

    T, K = topk_indices.size()
    TK = T * K
    device = topk_indices.device

    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    expert_frequency = torch.empty(E, dtype=torch.int32, device=device)
    expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)

    TC_topk_router_metadata_triton(
        topk_indices, E, expert_frequency, expert_frequency_offset, x_gather_idx, s_scatter_idx, s_reverse_scatter_idx
    )

    if type(activation_type) == str:
        activation_type = ActivationType(activation_type)

    assert not torch.compiler.is_compiling()
    assert is_glu(activation_type), "QuACK GEMM does not support non GLU activation yet"

    a, h = _UpProjection.apply(
        x,
        w1,
        b1,
        expert_frequency_offset,
        TK,
        K,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        None,
        False,  # is_each_token_has_variable_activated_expert
        activation_type,
        is_inference_mode_enabled,
        concat_layout,
        use_fp8,
    )

    o = _DownProjection.apply(
        a,
        h,
        w2,
        b2,
        topk_scores,
        expert_frequency_offset,
        T,
        K,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        None,
        False,  # is_each_token_has_variable_activated_expert
        activation_type,
        use_fp8,
    )

    return o, router_logits, expert_frequency


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Weight format requirements:
# - w1_weight: Shape (2*I, H, E), stride order (2, 0, 1)
#     concat_layout=False (default): interleaved [gate_row0, up_row0, gate_row1, up_row1, ...]
#     concat_layout=True:            concatenated [gate_row0, ..., gate_row_{I-1}, up_row0, ..., up_row_{I-1}]
# - w2_weight: Shape (H, I, E), stride order (2, 0, 1)


# We assume token_indices is already SORTED ascendingly !!!
#   and len(token_indices) = len(expert_indices) = len(router_scores)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def moe_general_routing_inputs(
    x: torch.Tensor,
    router_scores: torch.Tensor,
    token_indices: torch.Tensor,
    expert_indices: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    E: int,
    stream_id: int,
    activation_type: ActivationType,
    is_inference_mode_enabled: bool = False,
    concat_layout: bool = False,
    use_fp8: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert ((b1 is None) and (b2 is None)) or (
        (b1 is not None) and (b2 is not None)
    ), "b1 and b2 has to be None or not None at the same time!"

    T = x.size(0)
    TK = router_scores.size(0)
    E = w2.size(-1)
    device = router_scores.device

    if router_scores.dtype != torch.float32:
        router_scores = router_scores.float()

    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    expert_frequency = torch.empty(E, dtype=torch.int32, device=device)
    expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)
    num_activated_expert_per_token_offset = torch.empty(T + 1, dtype=torch.int32, device=device)

    general_routing_router_metadata_triton(
        token_indices,
        expert_indices,
        T,
        E,
        expert_frequency,
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
    )

    assert not torch.compiler.is_compiling()
    assert is_glu(activation_type), "QuACK GEMM does not support non GLU activation yet"

    a, h = _UpProjection.apply(
        x,
        w1,
        b1,
        expert_frequency_offset,
        TK,
        None,  # K, not needed
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        True,  # is_each_token_has_variable_activated_expert
        activation_type,
        is_inference_mode_enabled,
        concat_layout,
        use_fp8,
    )

    o = _DownProjection.apply(
        a,
        h,
        w2,
        b2,
        router_scores,
        expert_frequency_offset,
        T,
        None,  # K, not needed
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        True,  # is_each_token_has_variable_activated_expert
        activation_type,
        use_fp8,
    )

    return o, expert_frequency
