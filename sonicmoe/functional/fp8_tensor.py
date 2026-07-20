# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
"""Blockwise-FP8 tensor wrapper for the SM90 MXFP8 MoE.

`FP8BlockwiseTensor` carries an already-quantized payload (`_data` in fp8_e4m3fn
plus per-block `_scale` in float32) while presenting bf16/fp16/fp32 metadata to
autograd, so quantization is a transparent (straight-through) boundary. Callers
build one with `to_activations` (1x128 K-blocks, matching `quantize_act`) or
`to_weights` (128x128 blocks, matching `quantize_weight_sm90`) and hand it to
`moe_TC_softmax_topk_layer_fp8`, which then skips the internal re-quantization.

This is a trimmed adaptation of the pattern used in `ydt_core/fp8/tensor.py`,
backed by QuACK's SM90 quantizers instead of a generic blockwise kernel and
holding only the dispatch cases the MoE path exercises.
"""

from __future__ import annotations

from typing import Any

import torch
from quack.gemm_blockscaled_sm90 import quantize_act, quantize_weight_sm90

ACTIVATION_QUANT_SIZE = (1, 128)
WEIGHT_QUANT_SIZE = (128, 128)

_META_DATA_OPS = frozenset(
    {
        torch.ops.aten.sym_size.default,
        torch.ops.aten.sym_stride.default,
        torch.ops.aten.sym_numel.default,
    }
)

_GRAD_DTYPES = {torch.bfloat16, torch.float16, torch.float32}


def _quant(x: torch.Tensor, quant_block_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize `x` with the SM90 quantizer matching `quant_block_size`."""
    if quant_block_size == ACTIVATION_QUANT_SIZE:
        return quantize_act(x)  # 1x128 K-blocks, M-contiguous f32 scales
    if quant_block_size == WEIGHT_QUANT_SIZE:
        return quantize_weight_sm90(x)  # 128x128 blocks
    raise ValueError(f"unsupported quant_block_size {quant_block_size}")


def _dequant_weight(
    data: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Dequantize a 128x128 blockwise weight payload."""
    return (
        data.float()
        * scale.float().repeat_interleave(128, dim=-2).repeat_interleave(128, dim=-1)
    ).to(dtype)


class FP8BlockwiseTensor(torch.Tensor):
    """Tensor wrapper storing a blockwise-FP8 payload but exposing `grad_dtype`.

    Storage splits across ``_data`` (quantized fp8 values) and ``_scale`` (per-block
    f32 scales). The wrapper's public ``dtype`` is ``grad_dtype`` (not ``_data.dtype``)
    so autograd flows in the requested precision while the forward payload stays
    quantized.
    """

    _data: torch.Tensor
    _scale: torch.Tensor
    _quant_block_size: tuple[int, int]
    _grad_dtype: torch.dtype | None

    @staticmethod
    def __new__(
        cls: type[FP8BlockwiseTensor],
        data: torch.Tensor,
        scale: torch.Tensor,
        quant_block_size: tuple[int, int],
        grad_dtype: torch.dtype | None,
        **kwargs: Any,
    ) -> FP8BlockwiseTensor:
        assert data.dtype == torch.float8_e4m3fn
        assert grad_dtype is None or grad_dtype in _GRAD_DTYPES, f"{grad_dtype=}"
        return torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=grad_dtype if grad_dtype is not None else data.dtype,
            layout=data.layout,
            requires_grad=data.requires_grad,
            device=data.device,
            **kwargs,
        )

    def __init__(
        self,
        data: torch.Tensor,
        scale: torch.Tensor,
        quant_block_size: tuple[int, int],
        grad_dtype: torch.dtype | None,
    ) -> None:
        self._data = data
        self._scale = scale
        self._quant_block_size = quant_block_size
        self._grad_dtype = grad_dtype

    @classmethod
    def __torch_dispatch__(
        cls: type[FP8BlockwiseTensor],
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if kwargs is None:
            kwargs = {}
        self = args[0]

        if func in _META_DATA_OPS:
            return func(self._data, *args[1:], **kwargs)

        match func:
            case torch.ops.aten.detach.default | torch.ops.aten.clone.default:
                return FP8BlockwiseTensor(
                    func(self._data, *args[1:], **kwargs),
                    func(self._scale, *args[1:], **kwargs),
                    self._quant_block_size,
                    self._grad_dtype,
                )
            case torch.ops.aten._to_copy.default:
                data_kwargs = dict(kwargs)
                scale_kwargs = dict(kwargs)
                data_kwargs["dtype"] = torch.float8_e4m3fn
                scale_kwargs["dtype"] = torch.float32
                return FP8BlockwiseTensor(
                    func(self._data, *args[1:], **data_kwargs),
                    func(self._scale, *args[1:], **scale_kwargs),
                    self._quant_block_size,
                    self._grad_dtype,
                )
            case torch.ops.aten.transpose.int:
                # Only valid for square blocks: transposing a 128x128 block keeps the
                # same per-block scale, so we transpose data and scale together.
                assert self._quant_block_size[0] == self._quant_block_size[1], (
                    "transpose only supported for square quant blocks (weights)"
                )
                ndim = self._data.ndim
                a, b = (x + ndim if x < 0 else x for x in args[1:])
                assert a == ndim - 2 and b == ndim - 1
                return FP8BlockwiseTensor(
                    func(self._data, *args[1:], **kwargs),
                    func(self._scale, *args[1:], **kwargs),
                    self._quant_block_size,
                    self._grad_dtype,
                )
            case torch.ops.aten.view.default:
                tensor, shape = args
                assert tuple(tensor.shape) == tuple(shape)
                return FP8BlockwiseTensor(
                    self._data, self._scale, self._quant_block_size, self._grad_dtype
                )
        raise NotImplementedError(f"FP8BlockwiseTensor dispatch: {func} not implemented")

    # ── constructors ────────────────────────────────────────────────────────
    @classmethod
    def to_activations(cls, x: torch.Tensor) -> FP8BlockwiseTensor:
        """Quantize an activation (1x128 K-blocks), straight-through differentiable."""
        return _ToFp8.apply(x, ACTIVATION_QUANT_SIZE)  # type: ignore[no-any-return]

    @classmethod
    def to_weights(cls, x: torch.Tensor) -> FP8BlockwiseTensor:
        """Quantize a weight (128x128 blocks), straight-through differentiable."""
        return _ToFp8.apply(x, WEIGHT_QUANT_SIZE)  # type: ignore[no-any-return]

    @classmethod
    def from_float(
        cls, x: torch.Tensor, *, quant_block_size: tuple[int, int]
    ) -> FP8BlockwiseTensor:
        return _ToFp8.apply(x, quant_block_size)  # type: ignore[no-any-return]

    def to_dtype(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """Dequantize back to a dense tensor (straight-through differentiable)."""
        return _ToDtype.apply(self, dtype)  # type: ignore[no-any-return]

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return (
            f"FP8BlockwiseTensor(shape={tuple(self.shape)}, "
            f"quant_block_size={self._quant_block_size}, grad_dtype={self._grad_dtype})"
        )


type _Ctx = torch.autograd.function.FunctionCtx


class _ToFp8(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: _Ctx, x: torch.Tensor, quant_block_size: tuple[int, int]
    ) -> FP8BlockwiseTensor:
        data, scale = _quant(x, quant_block_size)
        ctx.dtype = x.dtype  # type: ignore[attr-defined]
        return FP8BlockwiseTensor(data, scale, quant_block_size, x.dtype)

    @staticmethod
    def backward(ctx: _Ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        # Straight-through: quantization is a transparent boundary for gradients.
        assert grad.dtype == ctx.dtype  # type: ignore[attr-defined]
        return grad, None


class _ToDtype(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _Ctx, x: FP8BlockwiseTensor, dtype: torch.dtype) -> torch.Tensor:
        if x._quant_block_size == WEIGHT_QUANT_SIZE:
            return _dequant_weight(x._data, x._scale, dtype)
        assert x._quant_block_size == ACTIVATION_QUANT_SIZE
        return (
            x._data.float() * x._scale.float().repeat_interleave(x._quant_block_size[1], dim=-1)
        ).to(dtype)

    @staticmethod
    def backward(ctx: _Ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad, None
