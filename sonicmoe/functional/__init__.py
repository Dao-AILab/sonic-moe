# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import os

import torch
import torch.nn.functional as F
from quack.gemm_interface import gemm, gemm_dgated, gemm_gated

from ..enums import ActivationType, is_glu
from .backward import (
    _down_projection_backward_act,
    _down_projection_backward_weight,
    _token_broadcast_backward,
    _topk_softmax_bwd,
    _up_projection_backward_act,
    _up_projection_backward_weight,
)
from .forward import _down_projection_forward, _router_forward, _topk_softmax_fwd, _up_projection_forward
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

        _up_projection_forward(
            x=x,
            w1=w1,
            h=h,
            a=a,
            b1=b1,
            expert_frequency_offset=expert_frequency_offset,
            x_gather_idx=x_gather_idx,
            activation_type=activation_type.value,
            is_inference_mode_enabled=is_inference_mode_enabled,
            concat_layout=concat_layout,
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

        _up_projection_backward_act(
            w1=w1,
            dx_expanded=dx_expanded,
            dh=dh,
            db1=db1,
            expert_frequency_offset=expert_frequency_offset,
            is_glu_activation=is_glu_activation,
            concat_layout=concat_layout,
        )

        _up_projection_backward_weight(
            x=x,
            dw1=dw1,
            dh=dh,
            expert_frequency_offset=expert_frequency_offset,
            x_gather_idx=x_gather_idx,
            is_glu_activation=is_glu_activation,
            concat_layout=concat_layout,
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
    ) -> torch.Tensor:
        TK = a.size(0)
        H, I, E = w2.shape

        y = torch.empty(TK, H, dtype=a.dtype, device=a.device)

        _down_projection_forward(
            w2=w2,
            a=a,
            y=y,
            b2=b2,
            expert_frequency_offset=expert_frequency_offset,
        )

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

        _down_projection_backward_weight(
            dout=dout,
            a_prime=a_prime,
            dw2=dw2,
            expert_frequency_offset=expert_frequency_offset,
            x_gather_idx=x_gather_idx,
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
    )

    return o, expert_frequency
