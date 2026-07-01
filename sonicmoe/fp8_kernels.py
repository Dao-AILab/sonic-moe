import torch
import triton
import triton.language as tl

def fp8_quantize_weights(bf16_weights: torch.Tensor, quant_block_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    G, M, K = bf16_weights.shape
    fp8_weight = torch.empty(G, M, K, dtype=torch.float8_e4m3fn, device=bf16_weights.device)
    weight_scale = torch.empty(G, M // quant_block_size, K // quant_block_size, device=bf16_weights.device)
    quant_grouped_matrix_to_mxe4m3(bf16_weights, fp8_weight, weight_scale, quant_block_size)
    return fp8_weight, weight_scale


_MAX_E4M3_VAL = torch.finfo(torch.float8_e4m3fn).max
_MAX_FP32_VAL = torch.finfo(torch.float32).max

@triton.jit
def compute_scale_from_amax(
    amax: tl.tensor,
    _MAX_E4M3_VAL: tl.constexpr,
    _MAX_FP32_VAL: tl.constexpr,
) -> tl.tensor:
    """
    Compute FP8 scaling factor from absolute maximum values.

    Returns:
        A scaling factors
    """
    scale = tl.where(amax == 0, 1.0, _MAX_E4M3_VAL / amax)

    is_inf = tl.cast(scale, tl.int32, bitcast=True) == 0x7F800000
    scale = tl.where(is_inf, _MAX_FP32_VAL, scale)

    # round to the power-of-2
    scale_bits = tl.cast(scale, tl.uint32, bitcast=True)
    scale = tl.cast(scale_bits & 0xFF800000, tl.float32, bitcast=True)

    return scale

@triton.jit
def _quant_grouped_matrix_to_mxe4m3_kernel(
    src_ptr,
    dst_ptr,
    dst_transpose_ptr,
    scale_ptr,
    scale_transpose_ptr,
    num_groups: int,
    M: int,
    N: int,
    stride_g: int,
    stride_m: int,
    stride_g_t: int,
    stride_m_t: int,
    stride_scale_g: int,
    stride_scale_m: int,
    stride_scale_g_t: int,
    stride_scale_m_t: int,
    BLOCK_SIZE: tl.constexpr,
    _MAX_E4M3_VAL: tl.constexpr,
    _MAX_FP32_VAL: tl.constexpr,
    STORE_TRANSPOSE: tl.constexpr = False,
):
    """
    Perform (BLOCK_SIZE, BLOCK_SIZE)-block quantization of the grouped input matrix
    `src_ptr` and stores the result in `dst_ptr` and the scaling factor in `scale_ptr`.

    Args:
        src_ptr: Pointer to the input matrix of shape (num_groups, M, N).
        dst_ptr: Pointer to the output matrix.
        scale_ptr: Pointer to the scaling factors.
        num_groups (int): Number of groups/batches in the matrix.
        M (int): Number of rows in each matrix.
        N (int): Number of columns in each matrix.
        stride_g (int): Stride between groups.
        stride_m (int): Stride between rows.
        BLOCK_SIZE (tl.constexpr): Size of the block for tiling.
    """
    pid_g = tl.program_id(axis=0).to(tl.int64)
    pid_m = tl.program_id(axis=1).to(tl.int64)
    pid_n = tl.program_id(axis=2).to(tl.int64)

    m_blocks = tl.cdiv(M, BLOCK_SIZE)
    n_blocks = tl.cdiv(N, BLOCK_SIZE)

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    offs_m_grid = offs_m[:, None]
    offs_n_grid = offs_n[None, :]

    mask = (offs_m_grid < M) & (offs_n_grid < N)

    offs = pid_g * stride_g
    offs += offs_m_grid * stride_m + offs_n_grid

    src = tl.load(src_ptr + offs, mask=mask).to(tl.float32)

    amax = tl.max(tl.abs(src))
    scale = compute_scale_from_amax(amax, _MAX_E4M3_VAL, _MAX_FP32_VAL)
    scale_inv = 1.0 / scale

    dst = (src * scale).to(dst_ptr.dtype.element_ty)

    tl.store(dst_ptr + offs, dst, mask=mask)

    scale_idx = pid_g * m_blocks * n_blocks + pid_m * n_blocks + pid_n
    tl.store(scale_ptr + scale_idx, scale_inv)

    if STORE_TRANSPOSE:
        abs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        abs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

        transpose_offs = pid_g * stride_g_t + abs_n[:, None] * stride_m_t + abs_m[None, :]

        dst_t = tl.trans(dst)
        mask_t = (abs_n[:, None] < N) & (abs_m[None, :] < M)

        tl.store(dst_transpose_ptr + transpose_offs, dst_t, mask=mask_t)

        scale_offs_t = pid_g * stride_scale_g_t + pid_n * stride_scale_m_t + pid_m
        tl.store(scale_transpose_ptr + scale_offs_t, scale_inv)

def quant_grouped_matrix_to_mxe4m3(
    src: torch.Tensor,
    dst: torch.Tensor,
    scale_dst: torch.Tensor,
    block_size: int = 128,
    dst_transpose: torch.Tensor = None,
    scale_transpose: torch.Tensor = None,
):
    """
    Quantize a grouped matrix to FP8 format with per-block scaling.

    Args:
        src: Input tensor of shape (num_groups, M, N)
        dst: Output tensor for quantized values of shape (num_groups, M, N)
        scale_dst: Output tensor for scaling factors of shape
                  (num_groups, ceil_div(M, block_size), ceil_div(N, block_size))
        block_size: Quantization block size (default: 128)
    """
    assert src.is_contiguous()
    assert dst.is_contiguous()
    assert scale_dst.is_contiguous()
    store_transpose = dst_transpose is not None
    if store_transpose:
        assert dst_transpose.is_contiguous()
        assert scale_transpose.is_contiguous()

    assert len(src.shape) == 3
    num_groups, M, N = src.size()

    assert src.size() == dst.size()
    m_blocks = triton.cdiv(M, block_size)
    n_blocks = triton.cdiv(N, block_size)
    assert scale_dst.shape == (num_groups, m_blocks, n_blocks)

    if store_transpose:
        assert dst_transpose.shape == (num_groups, N, M)
        assert scale_transpose.shape == (num_groups, n_blocks, m_blocks)

    assert dst.dtype == torch.float8_e4m3fn
    assert scale_dst.dtype == torch.float32

    grid = (num_groups, m_blocks, n_blocks)

    _quant_grouped_matrix_to_mxe4m3_kernel[grid](
        src,
        dst,
        dst_transpose if store_transpose else dst,  # dummy pointer if not used
        scale_dst,
        scale_transpose if store_transpose else scale_dst,  # dummy pointer if not used
        num_groups,
        M,
        N,
        src.stride(0),
        src.stride(1),
        dst_transpose.stride(0) if store_transpose else 0,
        dst_transpose.stride(1) if store_transpose else 0,
        scale_dst.stride(0),
        scale_dst.stride(1),
        scale_transpose.stride(0) if store_transpose else 0,
        scale_transpose.stride(1) if store_transpose else 0,
        STORE_TRANSPOSE=store_transpose,
        BLOCK_SIZE=block_size,
        _MAX_E4M3_VAL=_MAX_E4M3_VAL,
        _MAX_FP32_VAL=_MAX_FP32_VAL,
    )
