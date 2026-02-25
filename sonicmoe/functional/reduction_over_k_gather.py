# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from typing import Optional

import torch
import triton
import triton.language as tl

from ..utils import get_powers_of_2


### This triton impl is equivalent as the cute-dsl impl shown above,
# and also achieves similar memory bandwidth on H100 for large K and H.
# However, for small K and H, this impl is better by autotuning so we use it as the default.
def _get_triton_autotune_configs() -> list[triton.Config]:
    configs = []
    for BLOCK_H in get_powers_of_2(256, 4096):
        for BLOCK_K in get_powers_of_2(1, 128):
            for num_warps in [4, 8]:
                if BLOCK_K * BLOCK_H <= 32768:
                    configs.append(
                        triton.Config({"BLOCK_H": BLOCK_H, "BLOCK_K": BLOCK_K}, num_warps=num_warps, num_stages=4)
                    )
    return configs


def _prune_triton_autotune_config(configs, nargs, **kw):
    pruned_configs = []
    for c in configs:
        BLOCK_H = c.kwargs["BLOCK_H"]
        BLOCK_K = c.kwargs["BLOCK_K"]
        H = kw["H"]
        MAX_K = kw["MAX_K"]
        if (
            BLOCK_H <= triton.next_power_of_2(H)
            and BLOCK_K <= triton.next_power_of_2(MAX_K)
            and min(H * MAX_K, 1024) <= (BLOCK_H * BLOCK_K)
        ):
            pruned_configs.append(c)

    if len(pruned_configs) == 0:
        return configs
    else:
        return pruned_configs


@triton.autotune(
    configs=_get_triton_autotune_configs(),
    key=["H", "MAX_K", "w_is_None", "is_varlen_K"],
    prune_configs_by={"early_config_prune": _prune_triton_autotune_config},
)
@triton.jit
def token_gather_sum_kernel(
    x_ptr,  # (Mtotal, H)
    w_ptr,  # (Mtotal,)
    M_perm_ptr,  # (Mtotal,) int32
    M_offset_ptr,  # (T+1,)   int32
    out_ptr,  # (T, H)
    T,
    H: tl.constexpr,
    MAX_K: tl.constexpr,
    # strides
    stride_xM: tl.constexpr,
    stride_xH: tl.constexpr,
    stride_outT: tl.constexpr,
    stride_outH: tl.constexpr,
    # tile sizes
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    w_is_None: tl.constexpr,
    is_varlen_K: tl.constexpr,
):
    # 1D tiling over T only
    pid_t = tl.program_id(axis=0)
    t_idx = pid_t.to(tl.uint32)

    # Load segment starts and ends for this token
    if is_varlen_K:
        Ms = tl.load(M_offset_ptr + t_idx).to(tl.uint32)
        Me = tl.load(M_offset_ptr + t_idx + 1).to(tl.uint32)
        K_this_token = Me - Ms  # actual K for this token
    else:
        Ms = MAX_K * t_idx
        K_this_token: tl.constexpr = MAX_K

    # Outer loop over H tiles
    for h_tile in tl.static_range(triton.cdiv(H, BLOCK_H)):
        h_idx = (h_tile * BLOCK_H + tl.arange(0, BLOCK_H)).to(tl.uint32)  # [BLOCK_H]
        m_h = h_idx < H

        # Initialize accumulator for this H tile
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)  # [BLOCK_H]

        # Inner loop over K tiles
        for k_tile in tl.range(tl.cdiv(K_this_token, BLOCK_K)):
            k_offset = k_tile * BLOCK_K

            k_idx = (k_offset + tl.arange(0, BLOCK_K)).to(tl.uint32)  # [BLOCK_K]

            # Mask for valid K indices
            m_k = k_idx < K_this_token  # [BLOCK_K]

            # Absolute positions into M_perm and w
            m_abs = Ms + k_idx  # [BLOCK_K]

            # Gather permuted indices
            perm_idx = tl.load(M_perm_ptr + m_abs, mask=m_k, other=0).to(tl.uint32)  # [BLOCK_K]

            # Load x values: [BLOCK_K, BLOCK_H]
            x_ptrs = x_ptr + perm_idx[:, None] * stride_xM + h_idx[None, :] * stride_xH
            x_mask = m_k[:, None] & m_h[None, :]
            x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

            # Reduce along K dimension and add to accumulator
            if w_is_None:
                acc += tl.sum(x_vals, axis=0)  # [BLOCK_H]
            else:
                w_vals = tl.load(w_ptr + m_abs, mask=m_k, other=0.0).to(tl.float32)  # [BLOCK_K]
                acc += tl.sum(x_vals * w_vals[:, None], axis=0)  # [BLOCK_H]

        # Store final result for this H tile (only once!)
        out_ptrs = out_ptr + t_idx * stride_outT + h_idx * stride_outH
        tl.store(out_ptrs, acc, mask=m_h)


def token_gather_and_sum_varlen_K_triton(
    x: torch.Tensor,  # (Mtotal, H)
    w: Optional[torch.Tensor],  # (Mtotal,)
    out: torch.Tensor,  # (T, H)
    M_perm: torch.Tensor,  # (Mtotal,) int32
    M_offset: torch.Tensor,  # (T+1,)   int32, variable K per token
    T: int,
    MAX_K: int,  # maximum K across all tokens
    H: int,
    is_varlen_K: bool,
):
    """
    1D parallelization over T, with iterative accumulation over K tiles and H tiles.
    Supports variable K per token.

    out[i, :] = sum_{j=0..K[i]-1}  x[M_perm[M_offset[i] + j], :] * w[M_offset[i] + j]

    where K[i] = M_offset[i+1] - M_offset[i] can vary per token.
    """

    # 1D grid over T only
    token_gather_sum_kernel[(T,)](
        x,
        w,
        M_perm,
        M_offset,
        out,
        T=T,
        H=H,
        MAX_K=MAX_K,
        stride_xM=x.stride(0),
        stride_xH=x.stride(1),
        stride_outT=out.stride(0),
        stride_outH=out.stride(1),
        w_is_None=(w is None),
        is_varlen_K=is_varlen_K,
    )
