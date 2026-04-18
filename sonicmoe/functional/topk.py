# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

# this impl is adapted from QuACK's topk https://github.com/Dao-AILab/quack/blob/main/quack/topk.py
import math
from enum import Enum
from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import quack.copy_utils as copy_utils
import quack.utils as utils
from cutlass import const_expr
from quack.sort.bitonic_sort import bitonic_topk
from triton import next_power_of_2

from sonicmoe.utils import domain_offset_i64


class _TopKMode(Enum):
    SOFTMAX_OVER_TOPK = "softmax_over_topk"  # most common choice: softmax(topk(x))
    TOPK_OVER_SOFTMAX = "topk_over_softmax"  # Qwen3:              topk(softmax(x))
    TOPK_NO_FUSION = "topk"


class _TopK:
    """Private base class. Use TopK_Softmax, Softmax_TopK, or TopK instead."""

    def __init__(
        self,
        input_dtype: Type[cutlass.Numeric],
        output_dtype: Type[cutlass.Numeric],
        N: int,
        k: int,
        mode: _TopKMode,
        norm_topk_prob: bool = False,
    ):
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.N = N
        self.input_vecsize = 128 // input_dtype.width
        self.output_vecsize = 128 // output_dtype.width
        self.k = k
        self.next_power_of_2_N = next_power_of_2(N)
        self.next_power_of_2_K = next_power_of_2(k)
        assert k <= 128 and k <= N
        assert N <= 4096 and N % 8 == 0
        assert input_dtype.width <= output_dtype.width, "input bitwidth must <= output bitwidth"

        self.mode = mode
        if norm_topk_prob:
            assert mode == _TopKMode.TOPK_OVER_SOFTMAX, "`norm_topk_prob` only works with softmax-then-topk"

        self.norm_topk_prob = norm_topk_prob

    def _calculate_threads_per_row(self):
        N = self.next_power_of_2_N
        num_threads_per_row = max(min(N // self.k, 32, N // 64), 1)
        return num_threads_per_row

    def _get_tv_layout(self, vecsize):
        N = self.next_power_of_2_N
        num_threads = 128 if N <= 16384 else 256
        threads_per_row = self._calculate_threads_per_row()
        cols_per_block = num_threads // threads_per_row
        num_blocks_N = cute.ceil_div(min(N, 16384) // vecsize, threads_per_row)
        tiler_mn = (cols_per_block, vecsize * num_blocks_N * threads_per_row)
        tv_layout = cute.make_layout(
            ((threads_per_row, cols_per_block), (vecsize, num_blocks_N)),
            stride=(
                (vecsize * cols_per_block, 1),
                (cols_per_block, cols_per_block * vecsize * threads_per_row),
            ),
        )
        return tiler_mn, tv_layout

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mValues: cute.Tensor,
        mIndices: cute.Tensor,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.input_dtype
        assert mValues.element_type == self.output_dtype
        assert mIndices.element_type == cutlass.Int32
        input_tiler_mn, input_tv_layout = self._get_tv_layout(self.input_vecsize)
        output_tiler_mn, output_tv_layout = self._get_tv_layout(self.output_vecsize)

        num_threads = cute.size(input_tv_layout, mode=[0])
        self.kernel(mX, mValues, mIndices, input_tv_layout, input_tiler_mn, output_tv_layout).launch(
            grid=[cute.ceil_div(mX.shape[0], input_tiler_mn[0]), 1, 1],
            block=[num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mValues: cute.Tensor,
        mIndices: cute.Tensor,
        input_tv_layout: cute.Layout,
        input_tiler_mn: cute.Shape,
        output_tv_layout: cute.Layout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        # slice for CTAs
        # We use domain_offset_i64 to deal with tensors larger than 2^31 elements
        mX = domain_offset_i64((bidx * input_tiler_mn[0], 0), mX)
        gX = cute.local_tile(mX, input_tiler_mn, (0, 0))
        cX = cute.local_tile(idX, input_tiler_mn, (bidx, 0))

        copy_atom_load_X = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type, num_bits_per_copy=128)
        thr_copy_X = cute.make_tiled_copy(copy_atom_load_X, input_tv_layout, input_tiler_mn).get_slice(tidx)
        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        # allocate fragments for gmem->rmem
        tXrX = cute.make_rmem_tensor_like(tXgX)

        is_even_N = const_expr(shape[1] == input_tiler_mn[1])
        tXpX = (
            copy_utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
            if const_expr((not is_even_N) or (self.N != self.next_power_of_2_N))
            else None
        )
        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_load_X, tXgX, tXrX, pred=tXpX)
        tXrX_f32 = cute.make_rmem_tensor(tXrX.shape, cutlass.Float32)
        tXrX_f32.store(tXrX.load().to(cutlass.Float32))

        # ------------------------------------------------------------------
        # Softmax-then-TopK: full-row softmax → in-place log-prob transform.
        # ------------------------------------------------------------------
        if const_expr(self.mode == _TopKMode.TOPK_OVER_SOFTMAX):
            if const_expr((not is_even_N) or (self.N != self.next_power_of_2_N)):
                utils.fill_oob(tXrX_f32, tXpX, -tXrX_f32.element_type.inf)

            threads_per_row_red = const_expr(self._calculate_threads_per_row())
            num_threads_cta = const_expr(128 if self.next_power_of_2_N <= 16384 else 256)

            # ---- thread-local (max, sum_exp) pair ----
            local_max = -cutlass.Float32.inf
            for i in cutlass.range_constexpr(cute.size(tXrX_f32)):
                local_max = cute.arch.fmax(tXrX_f32[i], local_max)

            local_sum = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(cute.size(tXrX_f32)):
                local_sum = local_sum + cute.math.exp(tXrX_f32[i] - local_max)

            if const_expr(threads_per_row_red == 1):
                row_max = local_max
                row_sum = local_sum
            else:
                smem = cutlass.utils.SmemAllocator()
                smem_layout = cute.make_ordered_layout((num_threads_cta,), order=(0,))
                smem_max = smem.allocate_tensor(
                    cutlass.Float32,
                    smem_layout,
                    byte_alignment=16,
                )
                smem_sum = smem.allocate_tensor(
                    cutlass.Float32,
                    smem_layout,
                    byte_alignment=16,
                )
                row_in_blk = tidx // threads_per_row_red

                smem_max[tidx] = local_max
                smem_sum[tidx] = local_sum
                cute.arch.barrier()

                # Peel first partner: no exp needed
                base = row_in_blk * threads_per_row_red
                row_max = smem_max[base]
                row_sum = smem_sum[base]

                for p in cutlass.range_constexpr(1, self._calculate_threads_per_row()):
                    p_max = smem_max[base + p]
                    p_sum = smem_sum[base + p]
                    if p_max > row_max:
                        row_sum = row_sum * cute.math.exp(row_max - p_max) + p_sum
                        row_max = p_max
                    else:
                        row_sum = row_sum + p_sum * cute.math.exp(p_max - row_max)

            # In-place logit → log-probability
            log_normalizer = row_max + cute.math.log(row_sum)
            for i in cutlass.range_constexpr(cute.size(tXrX_f32)):
                tXrX_f32[i] = tXrX_f32[i] - log_normalizer

        # Encode indices into mantissa low bits.
        log_N = int(math.log2(self.next_power_of_2_N))
        idx_mask = const_expr((1 << log_N) - 1)
        input_vecsize = cutlass.const_expr(input_tv_layout.shape[1][0])
        tXrX_u32 = cute.recast_tensor(tXrX_f32, cutlass.Uint32)
        # Encode indices into the last log_N bits of tXrX_u32
        for i in cutlass.range(cute.size(tXrX_u32), unroll_full=True):
            # tXcX only keeps track of the indices for every @vecsize elements
            col_idx = cutlass.Uint32(tXcX[i // input_vecsize][1] + i % input_vecsize)
            # If positive, invert the bits of the index, so that if there's a tie,
            # indices coming from a earlier column will win.
            encoded_idx = ~col_idx if tXrX_f32[i] >= 0 else col_idx
            # Mask to keep only the last log_N bits of the encoded index
            encoded_idx = encoded_idx & idx_mask
            # Clear the last log_N bits and set them to our encoded index
            tXrX_u32[i] = (tXrX_u32[i] & ~idx_mask) | encoded_idx

        # Fill OOB values with -inf for top-k
        if const_expr((not is_even_N) or (self.N != self.next_power_of_2_N)):
            utils.fill_oob(tXrX_f32, tXpX, -tXrX_f32.element_type.inf)

        threads_per_row = input_tv_layout.shape[0][0]
        topk_vals = bitonic_topk(tXrX_f32, self.next_power_of_2_K, warp_width=threads_per_row)

        # Extract indices and clean values
        topk_vals_u32 = cute.recast_tensor(topk_vals, cutlass.Uint32)
        topk_indices = cute.make_rmem_tensor(self.k, cutlass.Int32)
        for i in cutlass.range_constexpr(self.k):
            # Extract the encoded index from the last log_N bits
            encoded_idx = topk_vals_u32[i] & idx_mask
            # Check if original value was positive by looking at the cleaned value
            topk_vals_u32[i] = topk_vals_u32[i] & ~idx_mask  # Clear last log_N bits
            # If positive, we need to invert the bits back to get original index
            col_idx = ~encoded_idx if topk_vals[i] >= 0 else encoded_idx
            topk_indices[i] = cutlass.Int32(col_idx & idx_mask)

        # TopK-then-Softmax
        if const_expr(self.mode == _TopKMode.SOFTMAX_OVER_TOPK):
            topk_vals_max = -cutlass.Float32.inf
            for i in cutlass.range_constexpr(self.k):
                topk_vals_max = cute.arch.fmax(topk_vals[i], topk_vals_max)

            topk_exp_sum = cutlass.Int32(0.0)
            for i in cutlass.range_constexpr(self.k):
                topk_vals[i] = cute.math.exp(topk_vals[i] - topk_vals_max)
                topk_exp_sum = topk_exp_sum + topk_vals[i]

            for i in cutlass.range_constexpr(self.k):
                topk_vals[i] = topk_vals[i] / topk_exp_sum

        # Softmax-then-TopK: recover probabilities from log-probs.
        if const_expr(self.mode == _TopKMode.TOPK_OVER_SOFTMAX):
            for i in cutlass.range_constexpr(self.k):
                topk_vals[i] = cute.math.exp(topk_vals[i])

            if const_expr(self.norm_topk_prob):
                topk_sum = cutlass.Float32(0.0)
                for i in cutlass.range_constexpr(self.k):
                    topk_sum = topk_sum + topk_vals[i]
                for i in cutlass.range_constexpr(self.k):
                    topk_vals[i] = topk_vals[i] / topk_sum

        topk_vals_out = cute.make_rmem_tensor_like(topk_indices, mValues.element_type)
        for i in cutlass.range_constexpr(self.k):
            topk_vals_out[i] = topk_vals[i].to(mValues.element_type)

        row = tXcX[0][0]
        # Only the 1st thread in this row writes the top-k values and indices
        output_vecsize = cutlass.const_expr(output_tv_layout.shape[1][0])
        if row < shape[0] and tXcX[0][1] == 0:
            # Vectorized write
            elems_per_store = const_expr(math.gcd(output_vecsize, self.k))
            mValues_store = cute.tiled_divide(mValues[row, None], (elems_per_store,))
            mIndices_store = cute.tiled_divide(mIndices[row, None], (elems_per_store,))
            topk_vals_out_store = cute.tiled_divide(topk_vals_out, (elems_per_store,))
            topk_indices_store = cute.tiled_divide(topk_indices, (elems_per_store,))
            for i in cutlass.range_constexpr(cute.size(topk_vals_out_store.shape, [1])):
                cute.autovec_copy(topk_vals_out_store[None, i], mValues_store[None, i])
                cute.autovec_copy(topk_indices_store[None, i], mIndices_store[None, i])


class Softmax_Over_TopK(_TopK):
    """softmax(topk(x))"""

    def __init__(
        self,
        input_dtype: Type[cutlass.Numeric],
        output_dtype: Type[cutlass.Numeric],
        N: int,
        k: int,
    ):
        mode = _TopKMode.SOFTMAX_OVER_TOPK
        super().__init__(
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            N=N,
            k=k,
            mode=mode,
        )


class TopK_Over_Softmax(_TopK):
    """Qwen3: topk(softmax(x))
    When norm_topk_prob=True, renormalizes the K selected probabilities to sum to 1.
    """

    def __init__(
        self,
        input_dtype: Type[cutlass.Numeric],
        output_dtype: Type[cutlass.Numeric],
        N: int,
        k: int,
        norm_topk_prob: bool = True,
    ):
        super().__init__(
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            N=N,
            k=k,
            mode=_TopKMode.TOPK_OVER_SOFTMAX,
            norm_topk_prob=norm_topk_prob,
        )


class TopK(_TopK):
    """Raw topk — no softmax."""

    def __init__(
        self,
        input_dtype: Type[cutlass.Numeric],
        output_dtype: Type[cutlass.Numeric],
        N: int,
        k: int,
    ):
        super().__init__(
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            N=N,
            k=k,
            mode=_TopKMode.TOPK_NO_FUSION,
        )
