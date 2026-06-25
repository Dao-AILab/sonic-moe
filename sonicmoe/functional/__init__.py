# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import torch
import torch.nn.functional as F
from quack.gemm_config import GemmConfig
from quack.gemm_interface import gemm, gemm_dgated, gemm_gated, gemm_gated_tuned, gemm_tuned
from quack.fp8_quant import fp8_quant_e4m3, fp8_quant2_e4m3

from ..enums import ActivationType, is_glu
from .backward import (
    _down_projection_backward_act,
    _token_broadcast_backward,
    _topk_softmax_bwd,
    _up_projection_backward_act,
)
from .forward import _router_forward, _topk_softmax_fwd
from .triton_kernels import TC_topk_router_metadata_triton, general_routing_router_metadata_triton

_DEFAULT_FP8_DOWN_PREQUANT_MIN = 1048576
_FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0

# Keyed by (data_ptr, perm); invalidated by _version after each optimizer step.
_fp8_weight_cache: dict[tuple, tuple[int, torch.Tensor, torch.Tensor]] = {}

_DEFAULT_FP8_SCALE_UPDATE_INTERVAL = 1024
_fp8_act_scale: dict[tuple, tuple[torch.Tensor, int]] = {}
_fp8_prequant: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def _current_fp8_scale_e4m3(x: torch.Tensor) -> torch.Tensor:
    return x.abs().amax().clamp(min=1e-12).float() / _FP8_E4M3_MAX


def _fp8_scale_e4m3(x: torch.Tensor, key: str | None = None, interval: int = _DEFAULT_FP8_SCALE_UPDATE_INTERVAL) -> torch.Tensor:
    if key is None or interval <= 1:
        return _current_fp8_scale_e4m3(x)
    cache_key = (key, tuple(x.shape), x.dtype, x.device)
    cached = _fp8_act_scale.get(cache_key)
    if cached is None:
        scale = _current_fp8_scale_e4m3(x)
        _fp8_act_scale[cache_key] = (scale, 1)
        return scale
    scale, count = cached
    if count >= interval:
        scale = _current_fp8_scale_e4m3(x)
        count = 0
    _fp8_act_scale[cache_key] = (scale, count + 1)
    return scale


def _cached_fp8_scale_e4m3(key: str, shape, dtype, device, interval: int = _DEFAULT_FP8_SCALE_UPDATE_INTERVAL) -> torch.Tensor | None:
    cache_key = (key, tuple(shape), dtype, device)
    cached = _fp8_act_scale.get(cache_key)
    if cached is None:
        return None
    scale, count = cached
    if count >= interval:
        return None
    _fp8_act_scale[cache_key] = (scale, count + 1)
    return scale


def _fp8_cast_e4m3(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if x.is_contiguous():
        return fp8_quant_e4m3(x, scale)
    return (x / scale).to(torch.float8_e4m3fn)


def _to_fp8_e4m3(x: torch.Tensor, key: str | None = None, interval: int = _DEFAULT_FP8_SCALE_UPDATE_INTERVAL) -> tuple[torch.Tensor, torch.Tensor]:
    scale = _fp8_scale_e4m3(x, key, interval)
    return _fp8_cast_e4m3(x, scale), scale


def _to_fp8_e4m3_weight(x: torch.Tensor, perm: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    key = (x.data_ptr(), perm)
    cached = _fp8_weight_cache.get(key)
    if cached is None or cached[0] != x._version:
        fp8, sc = _to_fp8_e4m3(x.permute(*perm).contiguous())
        _fp8_weight_cache[key] = (x._version, fp8, sc)
        return fp8, sc
    return cached[1], cached[2]


def _sm100_cfg(x: torch.Tensor, m: int, n: int, cm: int, cn: int) -> GemmConfig | None:
    if not x.is_cuda or torch.cuda.get_device_capability(x.device)[0] < 10:
        return None
    return GemmConfig(
        tile_m=m,
        tile_n=n,
        cluster_m=cm,
        cluster_n=cn,
        pingpong=False,
        swap_ab=False,
        max_swizzle_size=8,
        is_dynamic_persistent=False,
        use_tma_gather=False,
        device_capacity=10,
    )


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
        fp8_side_out: bool = True,
        fp8_side_out_min: int = 1048576,
        fp8_side_out_cfg: tuple[int, int, int, int] = (128, 256, 1, 1),
        fp8_scale_interval: int = _DEFAULT_FP8_SCALE_UPDATE_INTERVAL,
    ) -> torch.Tensor:
        T, H = x.shape
        I, H, E = w1.shape
        is_glu_activation = is_glu(activation_type)

        if is_glu_activation:
            I //= 2
        TK = total_expert_freq

        a_post_sc = (
            _cached_fp8_scale_e4m3("a", (TK, I), x.dtype, x.device, fp8_scale_interval)
            if use_fp8 and is_inference_mode_enabled
            else None
        )
        a_dtype = torch.float8_e4m3fn if a_post_sc is not None else x.dtype
        a = torch.empty(TK, I, dtype=a_dtype, device=x.device)
        h_shape = (TK, 2 * I if is_glu_activation else I)
        h = torch.empty(h_shape, dtype=x.dtype, device=x.device) if not is_inference_mode_enabled else None

        assert activation_type.value in (
            "swiglu",
            "geglu",
        ), f"QuACK gemm_gated only supports glu activations, got {activation_type.value}"

        if use_fp8:
            x_sc = _fp8_scale_e4m3(x, key="x", interval=fp8_scale_interval)
            w1_fp8, w1_sc = _to_fp8_e4m3_weight(w1, (2, 1, 0))
            can_fuse_x = (
                fp8_side_out
                and not is_inference_mode_enabled
                and x.numel() >= fp8_side_out_min
                and x.is_contiguous()
                and torch.cuda.get_device_capability(x.device)[0] >= 10
            )
            if can_fuse_x:
                # Fused: kernel reads bf16 x and emits fp8 side output (quant_A_out).
                x_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
                gemm_gated_tuned.fn(
                    x, w1_fp8, h, a,
                    activation=activation_type.value,
                    cu_seqlens_m=expert_frequency_offset,
                    A_idx=x_gather_idx,
                    dynamic_scheduler=False,
                    concat_layout=(("B",) if concat_layout else None),
                    alpha=x_sc, alpha2=w1_sc,
                    scale_A=x_sc, quant_A_out=x_fp8,
                    quant_A_out_idx=s_scatter_idx if K is not None else None,
                    quant_A_out_stride=K if K is not None else 1,
                    postact_scale=a_post_sc,
                    config=_sm100_cfg(x, *fp8_side_out_cfg),
                )
            else:
                x_fp8 = _fp8_cast_e4m3(x, x_sc)
                gemm_gated_tuned(
                    x_fp8, w1_fp8, h, a,
                    activation=activation_type.value,
                    cu_seqlens_m=expert_frequency_offset,
                    A_idx=x_gather_idx,
                    dynamic_scheduler=False,
                    concat_layout=(("B",) if concat_layout else None),
                    alpha=x_sc, alpha2=w1_sc,
                    postact_scale=a_post_sc,
                )
            if a_post_sc is not None:
                _fp8_prequant[a.data_ptr()] = (a, a_post_sc)
            if not is_inference_mode_enabled:
                ctx.x_fp8 = x_fp8
                ctx.x_sc = x_sc
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
        ctx.fp8_scale_interval = fp8_scale_interval
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
            assert b1 is None, "fp8 path does not support bias"
            interval = ctx.fp8_scale_interval
            # dh arrives deferred (dh_true / out_sc) from the down-proj backward;
            # the prequant carries dh_sc * out_sc so dx/dw1 fold it back via alpha2.
            prequant = _fp8_prequant.pop(dh.data_ptr(), None)
            dh_fp8, dh_sc = (
                prequant if prequant is not None else _to_fp8_e4m3(dh, key="dh", interval=interval)
            )
            # dx = dh @ w1^T
            w1_fp8, w1_sc = _to_fp8_e4m3_weight(w1, (2, 0, 1))
            gemm_tuned.fn(
                dh_fp8,
                w1_fp8,
                dx_expanded,
                alpha=w1_sc,
                alpha2=dh_sc,
                cu_seqlens_m=expert_frequency_offset,
                dynamic_scheduler=False,
                concat_layout=(("B",) if concat_layout else None),
                config=_sm100_cfg(dh, 64, 256, 1, 2),
            )
            # dw1 = x^T @ dh
            x_fp8, x_sc = ctx.x_fp8, ctx.x_sc
            gemm_tuned.fn(
                x_fp8.T,
                dh_fp8,
                dw1.permute(2, 1, 0),
                alpha=x_sc,
                alpha2=dh_sc,
                cu_seqlens_k=expert_frequency_offset,
                A_idx=x_gather_idx,
                batch_idx_permute=None,
                dynamic_scheduler=False,
                concat_layout=(("out",) if concat_layout else None),
                config=_sm100_cfg(dh, 256, 256, 2, 1),
            )
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

        return dx_reduced, dw1, db1, *[None] * 18


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
        fp8_scale_interval: int = _DEFAULT_FP8_SCALE_UPDATE_INTERVAL,
        fp8_down_prequant_min: int = _DEFAULT_FP8_DOWN_PREQUANT_MIN,
    ) -> torch.Tensor:
        TK = a.size(0)
        H, I, E = w2.shape

        y_dtype = w2.dtype if use_fp8 else a.dtype
        y = torch.empty(TK, H, dtype=y_dtype, device=a.device)

        if use_fp8:
            w2_fp8, w2_sc = _to_fp8_e4m3_weight(w2, (2, 1, 0))
            if a.dtype == torch.float8_e4m3fn:
                # a already fp8 (inference postact path); scale carried via _fp8_prequant.
                prequant = _fp8_prequant.pop(a.data_ptr(), None)
                assert prequant is not None, "missing fp8 a scale"
                a_arg, a_sc = prequant
                scale_A = None
            else:
                a_sc = _fp8_scale_e4m3(a, key="a", interval=fp8_scale_interval)
                fuse_a = (
                    torch.cuda.get_device_capability(a.device)[0] >= 10
                    and a.numel() < fp8_down_prequant_min
                )
                if fuse_a:
                    a_arg = a
                    scale_A = a_sc
                else:
                    a_arg = _fp8_cast_e4m3(a, a_sc)
                    scale_A = None
            gemm_tuned.fn(
                a_arg,
                w2_fp8,
                y,
                alpha=a_sc,
                alpha2=w2_sc,
                scale_A=scale_A,
                cu_seqlens_m=expert_frequency_offset,
                dynamic_scheduler=False,
                config=_sm100_cfg(a, 128, 256, 1, 1),
            )
        else:
            gemm(a, w2.permute(2, 1, 0), out=y, cu_seqlens_m=expert_frequency_offset, bias=b2)

        o = torch.empty(T, H, device=a.device, dtype=y.dtype)
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
        ctx.fp8_scale_interval = fp8_scale_interval

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
            assert b2 is None, "fp8 path does not support bias"
            interval = ctx.fp8_scale_interval
            E = w2.size(2)
            s = topk_scores[s_scatter_idx]
            dout_sc = _fp8_scale_e4m3(dout, key="dout", interval=interval)
            dout_fp8 = _fp8_cast_e4m3(dout, dout_sc)
            w2_fp8, w2_sc = _to_fp8_e4m3_weight(w2, (2, 0, 1))
            out_sc = dout_sc * w2_sc
            _, _, ds_scattered = gemm_dgated(
                dout_fp8,
                w2_fp8,
                PreAct=h,
                dx_out=dh,
                postact_out=a_prime,
                colvec_scale=s,
                activation=activation_type.value,
                colvec_reduce=True,
                cu_seqlens_m=expert_frequency_offset,
                A_idx=x_gather_idx,
                dynamic_scheduler=False,
                reduce_scale=out_sc,
            )
            ds[s_scatter_idx] = ds_scattered
            # dh is deferred (dh_true / out_sc); the up-proj backward folds out_sc
            # back in via alpha2. Quantize dh + a_prime for dw2/dw1.
            if (
                dh.is_contiguous()
                and a_prime.is_contiguous()
                and dh.numel() + a_prime.numel() < 2**31  # fp8_quant2 pair indexing overflows int32
            ):
                dh_sc = _fp8_scale_e4m3(dh, key="dh", interval=interval)
                ap_sc = _fp8_scale_e4m3(a_prime, key="a_prime", interval=interval)
                dh_fp8, a_prime_fp8 = fp8_quant2_e4m3(dh, dh_sc, a_prime, ap_sc)
            else:
                dh_fp8, dh_sc = _to_fp8_e4m3(dh, key="dh", interval=interval)
                a_prime_fp8, ap_sc = _to_fp8_e4m3(a_prime, key="a_prime", interval=interval)
            _fp8_prequant[dh.data_ptr()] = (dh_fp8, dh_sc * out_sc)
            # dw2 = dout^T @ a_prime per expert (per tensor)
            gemm_tuned.fn(
                dout_fp8.T,
                a_prime_fp8,
                dw2.permute(2, 0, 1),
                alpha=dout_sc,
                alpha2=ap_sc,
                cu_seqlens_k=expert_frequency_offset,
                A_idx=x_gather_idx,
                batch_idx_permute=None,
                dynamic_scheduler=False,
                config=_sm100_cfg(dout, 128, 128, 1, 1),
            )
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

        return None, dh, dw2, db2, ds, *[None] * 14


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
    fp8_side_out: bool = True,
    fp8_side_out_min: int = 1048576,
    fp8_side_out_cfg: tuple[int, int, int, int] = (128, 256, 1, 1),
    fp8_scale_interval: int = _DEFAULT_FP8_SCALE_UPDATE_INTERVAL,
    fp8_down_prequant_min: int = _DEFAULT_FP8_DOWN_PREQUANT_MIN,
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
        fp8_side_out,
        fp8_side_out_min,
        fp8_side_out_cfg,
        fp8_scale_interval,
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
        fp8_scale_interval,
        fp8_down_prequant_min,
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
    fp8_side_out: bool = True,
    fp8_side_out_min: int = 1048576,
    fp8_side_out_cfg: tuple[int, int, int, int] = (128, 256, 1, 1),
    fp8_scale_interval: int = _DEFAULT_FP8_SCALE_UPDATE_INTERVAL,
    fp8_down_prequant_min: int = _DEFAULT_FP8_DOWN_PREQUANT_MIN,
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
        fp8_side_out,
        fp8_side_out_min,
        fp8_side_out_cfg,
        fp8_scale_interval,
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
        fp8_scale_interval,
        fp8_down_prequant_min,
    )

    return o, expert_frequency
