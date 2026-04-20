# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
import triton
import triton.language as tl
from cutlass.cute.runtime import from_dlpack
from quack.cute_dsl_utils import torch2cute_dtype_map
from quack.gemm_interface import gemm, gemm_gated

from ..enums import LIBRARY_NAME
from .reduction_over_k_gather import token_gather_and_sum_varlen_K_triton
from .topk import Softmax_Over_TopK, TopK_Over_Softmax


@torch.library.custom_op(f"{LIBRARY_NAME}::_topk_fwd", mutates_args={"values", "indices"})
def _topk_fwd(
    x: torch.Tensor,
    k: int,
    values: torch.Tensor,
    indices: torch.Tensor,
    is_softmax_over_topk: bool,
    norm_topk_probs: bool,
) -> None:
    """Top-k forward pass.
    Args:
        x: Input tensor of shape (M, N)
        k: Number of top elements to return
    Returns:
        Tuple of (values tensor of shape (M, k), indices tensor of shape (M, k))
    """
    N = x.size(1)

    input_dtype = torch2cute_dtype_map[x.dtype]
    output_dtype = torch2cute_dtype_map[values.dtype]
    convert_from_dlpack = lambda tensor: (
        from_dlpack(tensor.detach(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
    )

    x_tensor, values_tensor, indices_tensor = [convert_from_dlpack(tensor) for tensor in (x, values, indices)]
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if is_softmax_over_topk:
        compile_key = (input_dtype, output_dtype, N, k, True)
    else:
        compile_key = (input_dtype, output_dtype, N, k, False, norm_topk_probs)

    if compile_key not in _topk_fwd.compile_cache:
        if is_softmax_over_topk:
            topk_op = Softmax_Over_TopK(input_dtype, output_dtype, N, k)
        else:
            topk_op = TopK_Over_Softmax(input_dtype, output_dtype, N, k, norm_topk_probs)

        _topk_fwd.compile_cache[compile_key] = cute.compile(
            topk_op, x_tensor, values_tensor, indices_tensor, current_stream
        )
    _topk_fwd.compile_cache[compile_key](x_tensor, values_tensor, indices_tensor, current_stream)


_topk_fwd.compile_cache = {}


@torch.library.custom_op(f"{LIBRARY_NAME}::_up_projection_forward", mutates_args={"h", "a"})
def _up_projection_forward(
    x: torch.Tensor,
    w1: torch.Tensor,
    h: torch.Tensor,
    a: torch.Tensor,
    b1: torch.Tensor | None,
    expert_frequency_offset: torch.Tensor,
    x_gather_idx: torch.Tensor,
    activation_type: str,
    is_inference_mode_enabled: bool = False,
    concat_layout: bool = False,
) -> None:
    assert activation_type in (
        "swiglu",
        "geglu",
    ), f"QuACK gemm_gated only supports glu activations, got {activation_type}"
    gemm_gated(
        x,
        w1.permute(2, 1, 0),
        activation=activation_type,
        cu_seqlens_m=expert_frequency_offset,
        A_idx=x_gather_idx,
        preact_out=h,
        postact_out=a,
        store_preact=(not is_inference_mode_enabled),
        bias=b1,
        concat_layout=(("B", "bias") if b1 is not None else ("B",)) if concat_layout else None,
    )


_up_projection_forward.compile_cache = {}


@torch.library.custom_op(f"{LIBRARY_NAME}::_down_projection_forward", mutates_args={"y"})
def _down_projection_forward(
    w2: torch.Tensor,
    a: torch.Tensor,
    y: torch.Tensor,
    b2: torch.Tensor | None,
    expert_frequency_offset: torch.Tensor,
) -> None:
    gemm(a, w2.permute(2, 1, 0), out=y, cu_seqlens_m=expert_frequency_offset, bias=b2)


_down_projection_forward.compile_cache = {}


@torch.library.custom_op(f"{LIBRARY_NAME}::_router_forward", mutates_args={"o"})
def _router_forward(
    y: torch.Tensor,
    o: torch.Tensor,
    topk_scores: torch.Tensor,
    s_reverse_scatter_idx: torch.Tensor,
    num_activated_expert_per_token_offset: torch.Tensor,
    varlen_K_max: int,
    H: int,
    is_varlen_K: bool,
) -> None:
    token_gather_and_sum_varlen_K_triton(
        y,
        topk_scores,
        o,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        o.size(0),
        varlen_K_max,
        H,
        is_varlen_K,
    )


@triton.jit
def _softmax_fwd_small_kernel(
    logits_ptr, stride_lm: tl.constexpr, stride_ln: tl.constexpr, K: tl.constexpr, BLOCK_K: tl.constexpr
):
    row = tl.program_id(axis=0)

    # tl.assume(K <= BLOCK_K)
    k_offs = tl.arange(0, BLOCK_K)
    k_mask = k_offs < K

    # load full row (all columns) in one go (N is small)
    x = tl.load(logits_ptr + row * stride_lm + k_offs * stride_ln, mask=k_mask, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    ex = tl.exp(x)
    y = ex / tl.sum(ex, axis=0)

    tl.store(logits_ptr + row * stride_lm + k_offs * stride_ln, y, mask=k_mask)


@torch.library.custom_op(
    f"{LIBRARY_NAME}::_softmax_topk_fwd", mutates_args={"topk_router_score", "topk_router_indices"}
)
def _topk_softmax_fwd(
    router_logits: torch.Tensor,
    topk_router_score: torch.Tensor,
    topk_router_indices: torch.Tensor,
    E: int,
    K: int,
    is_softmax_over_topk: bool,
    norm_topk_probs: bool,
) -> None:
    if E <= 4096 and K <= 16 and E % 8 == 0:
        _topk_fwd(
            router_logits,
            K,
            topk_router_score,
            topk_router_indices,
            is_softmax_over_topk=is_softmax_over_topk,
            norm_topk_probs=norm_topk_probs,
        )
    else:
        if is_softmax_over_topk:
            topk_results = router_logits.topk(K, dim=-1)
            vals = topk_results.values.softmax(dim=-1, dtype=torch.float32)
            topk_router_score.copy_(vals.to(topk_router_score.dtype))
            topk_router_indices.copy_(topk_results.indices.to(topk_router_indices.dtype))
        else:
            probs = router_logits.softmax(dim=-1, dtype=torch.float32)
            topk_results = probs.topk(K, dim=-1)
            vals = topk_results.values
            if norm_topk_probs:
                vals = vals / vals.sum(dim=-1, keepdim=True)
            topk_router_score.copy_(vals.to(topk_router_score.dtype))
            topk_router_indices.copy_(topk_results.indices.to(topk_router_indices.dtype))
