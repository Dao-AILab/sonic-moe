# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from typing import Any, Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op
from torch.utils._pytree import tree_map


def make_contiguous(x: Any) -> Any:
    return x.contiguous() if isinstance(x, torch.Tensor) else x


def ensure_contiguous(func: Callable) -> Callable:
    def inner(*args, **kwargs):
        args = tree_map(make_contiguous, args)
        kwargs = tree_map(make_contiguous, kwargs)
        return func(*args, **kwargs)

    return inner


def ceil_divide(x: int, y: int) -> int:
    return (x + y - 1) // y


def check_power_of_2(n: int) -> bool:
    return n & (n - 1) == 0 and n != 0


def get_powers_of_2(start: int, end: int) -> list[int]:
    assert check_power_of_2(start), "start is not a power of 2"
    assert check_power_of_2(end), "end is not a power of 2"

    output = []
    n = start
    while n <= end:
        output.append(n)
        n = n << 1

    return output


@dsl_user_op
def domain_offset_i64(coord: cute.Coord, tensor: cute.Tensor, *, loc=None, ip=None) -> cute.Tensor:
    flat_coord_i64 = tuple(cutlass.Int64(c) for c in cute.flatten(coord))
    flat_stride = cute.flatten_to_tuple(tensor.stride)
    assert len(flat_coord_i64) == len(flat_stride), "Coordinate and stride must have the same length"
    offset = sum(c * s for c, s in zip(flat_coord_i64, flat_stride))
    assert isinstance(tensor.iterator, cute.Pointer)
    # HACK: we assume that applying the offset does not change the pointer alignment
    new_ptr = cute.make_ptr(
        tensor.element_type,
        tensor.iterator.toint() + offset * tensor.element_type.width // 8,
        tensor.memspace,
        assumed_align=tensor.iterator.max_alignment,
    )
    return cute.make_tensor(new_ptr, tensor.layout)


def divide_if_divisible(dividend: int, divisor: int, msg: str = "") -> int:
    assert dividend % divisor == 0, msg
    return dividend // divisor


def get_next_power_of_2(x: int) -> int:
    x -= 1
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    x |= x >> 32
    x += 1
    return x


class _TensorWithStream:
    """Wrapper to pass stream parameter to __dlpack__() for CUDA graph compatibility.

    This wrapper allows us to pass a stream parameter to the tensor's __dlpack__() method
    when cutlass's from_dlpack() calls it, preventing cross-stream synchronization during
    CUDA graph capture.
    """

    def __init__(self, tensor: torch.Tensor, stream: int):
        self._tensor = tensor
        # Convert CUDA stream pointer to PyTorch's __dlpack__ convention:
        # - stream=0 (null/default stream) -> use -1 to disable synchronization
        # - stream=non-zero -> use the raw pointer value
        # This prevents "unsupported stream on CUDA: 0" error
        self._stream = -1 if stream == 0 else stream

    def __dlpack__(self, stream=None):  # noqa: ARG002
        # Use the wrapped stream to prevent cross-stream synchronization
        # The stream parameter is required by the DLPack protocol but ignored here
        return self._tensor.__dlpack__(stream=self._stream)

    def __dlpack_device__(self):
        return self._tensor.__dlpack_device__()


def convert_torch_tensor_to_cute_tensor(
    x: torch.Tensor,
    stride_order,
    leading_dim: int,
    alignment: int,
    divisibility: int,
    stream: int | None = None,
):
    # Wrap tensor with stream if provided to prevent cross-stream synchronization during CUDA graph capture
    tensor_input = _TensorWithStream(x, stream) if stream is not None else x

    return (
        from_dlpack(tensor_input, assumed_align=alignment)
        .mark_layout_dynamic(leading_dim=leading_dim)
        .mark_compact_shape_dynamic(mode=leading_dim, stride_order=stride_order, divisibility=divisibility)
    )
