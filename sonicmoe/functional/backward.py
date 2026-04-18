# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
import triton
import triton.language as tl
from quack.gemm_interface import gemm, gemm_dgated

from ..enums import LIBRARY_NAME
from ..utils import get_powers_of_2
from .reduction_over_k_gather import token_gather_and_sum_varlen_K_triton


def _get_autotune_configs_for_db2_and_ds() -> list[triton.Config]:
    configs = []
    for BLOCK_TK in get_powers_of_2(4, 32):
        configs.append(triton.Config({"BLOCK_TK": BLOCK_TK}, num_warps=8, num_stages=4))
    return configs


@triton.autotune(
    configs=_get_autotune_configs_for_db2_and_ds(),
    key=["H", "E"],
)
@triton.jit
def db2_and_ds_kernel(
    dout_ptr,  # (T, H)
    s_ptr,  # (TK,)
    new_ds_partial_ptr,  # (TK, n_h_blocks)
    old_ds_partial_ptr,  # (TK, OLD_DS_PARTIAL_N)
    b2_ptr,  # (E, H),
    db2_ptr,  # (E, H),
    x_gather_idx_ptr,  # (TK,), maps grouped -> token index
    s_scatter_idx_ptr,  # (TK,), maps grouped -> scatter index
    expert_offset_ptr,  # (E+1,), offsets in grouped layout
    H: tl.constexpr,
    E: tl.constexpr,
    OLD_DS_PARTIAL_N: tl.constexpr,
    BLOCK_H: tl.constexpr,  # Block size for H dimension
    BLOCK_TK: tl.constexpr,  # Block size for token dimension
    BLOCK_OLD_DS_PARTIAL_N: tl.constexpr,
):
    Eidx = tl.program_id(0)  # expert id
    Hidx = tl.program_id(1)  # h-block id
    NUM_H_BLOCKS: tl.constexpr = tl.num_programs(1)

    # Hidden dimension indices for this block
    h_offsets = Hidx * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    E_count_start = tl.load(expert_offset_ptr + Eidx)
    E_count_end = tl.load(expert_offset_ptr + Eidx + 1)
    n_tokens = E_count_end - E_count_start

    b2 = tl.load(b2_ptr + Eidx * H + h_offsets, mask=h_mask, other=0.0).to(tl.float32)

    db2_acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Process tokens in blocks of BLOCK_TK
    for block_start in tl.range(0, n_tokens, BLOCK_TK):
        # Token offsets within this block
        tk_offsets = block_start + tl.arange(0, BLOCK_TK)
        tk_mask = tk_offsets < n_tokens
        tk_grouped = E_count_start + tk_offsets

        # Gather token indices: [BLOCK_TK]
        token_indices = tl.load(x_gather_idx_ptr + tk_grouped, mask=tk_mask, other=0).to(tl.uint32)

        # Get scatter indices: [BLOCK_TK]
        scatter_indices = tl.load(s_scatter_idx_ptr + tk_grouped, mask=tk_mask, other=0).to(tl.uint32)

        s = tl.load(s_ptr + scatter_indices, mask=tk_mask, other=0.0).to(tl.float32)

        # Gather dout: [BLOCK_TK, BLOCK_H]
        dout_offsets = token_indices[:, None] * H + h_offsets[None, :]
        dout_mask = tk_mask[:, None] & h_mask[None, :]
        dout = tl.load(dout_ptr + dout_offsets, mask=dout_mask, other=0.0).to(tl.float32)

        # Accumulate db2: sum over tokens of (dout * s)
        db2_acc += tl.sum(dout * s[:, None], axis=0)  # Sum over BLOCK_TK dimension

        # Compute ds: dot(dout, b2) for this H-block
        ds_partial = tl.sum(dout * b2[None, :], axis=1)  # [BLOCK_TK]

        # On first H-block, add old_ds_partial.sum(dim=1)
        if Hidx == 0:
            n_offsets = tl.arange(0, BLOCK_OLD_DS_PARTIAL_N)
            old_ds_partial_offsets = scatter_indices[:, None] * OLD_DS_PARTIAL_N + n_offsets[None, :]
            old_ds_partial_mask = tk_mask[:, None] & (n_offsets[None, :] < OLD_DS_PARTIAL_N)
            old_ds_partial_vals = tl.load(
                old_ds_partial_ptr + old_ds_partial_offsets, mask=old_ds_partial_mask, other=0.0
            ).to(tl.float32)
            ds_partial += tl.sum(old_ds_partial_vals, axis=1)

        tl.store(new_ds_partial_ptr + scatter_indices * NUM_H_BLOCKS + Hidx, ds_partial, mask=tk_mask)

    tl.store(db2_ptr + Eidx * H + h_offsets, db2_acc, mask=h_mask)


def _get_autotune_configs_for_db1() -> list[triton.Config]:
    configs = []
    for BLOCK_TK in get_powers_of_2(4, 128):
        for BLOCK_I in get_powers_of_2(64, 4096):
            if 4096 <= BLOCK_I * BLOCK_TK <= 16384:
                configs.append(triton.Config({"BLOCK_I": BLOCK_I, "BLOCK_TK": BLOCK_TK}, num_warps=8, num_stages=4))
    return configs


def _prune_triton_autotune_config(configs, nargs, **kw):
    pruned_configs = []
    for c in configs:
        if c.kwargs["BLOCK_I"] <= triton.next_power_of_2(nargs["I"]):
            pruned_configs.append(c)
    return pruned_configs


@triton.autotune(
    configs=_get_autotune_configs_for_db1(),
    key=["I", "E"],
    prune_configs_by={"early_config_prune": _prune_triton_autotune_config},
)
@triton.jit
def db1_kernel(
    dh_ptr,  # (TK, I)  — always interleaved
    db1_ptr,  # (E, I)
    expert_offset_ptr,  # (E+1,)
    I: tl.constexpr,
    E: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_TK: tl.constexpr,
    CONCAT_LAYOUT: tl.constexpr = False,
):
    Eidx = tl.program_id(0)

    E_count_start = tl.load(expert_offset_ptr + Eidx).to(tl.int64)
    E_count_end = tl.load(expert_offset_ptr + Eidx + 1).to(tl.int64)
    n_tokens = E_count_end - E_count_start

    NUM_I_BLOCKS: tl.constexpr = triton.cdiv(I, BLOCK_I)
    I_HALF: tl.constexpr = I // 2
    for Iidx in tl.static_range(0, NUM_I_BLOCKS, 1):
        i_offsets = Iidx * BLOCK_I + tl.arange(0, BLOCK_I)
        i_mask = i_offsets < I

        db1_acc = tl.zeros([BLOCK_I], dtype=tl.float32)

        for block_start in tl.range(0, n_tokens, BLOCK_TK):
            # Token offsets within this block
            tk_offsets = block_start + tl.arange(0, BLOCK_TK)
            tk_mask = tk_offsets < n_tokens
            tk_grouped = E_count_start + tk_offsets

            dz_offsets = tk_grouped[:, None] * I + i_offsets[None, :]
            dz_mask = tk_mask[:, None] & i_mask[None, :]
            dz = tl.load(dh_ptr + dz_offsets, mask=dz_mask, other=0.0).to(tl.float32)

            db1_acc += tl.sum(dz, axis=0)

        # Write: remap interleaved → concat if needed
        if CONCAT_LAYOUT:
            out_offsets = i_offsets // 2 + (i_offsets % 2) * I_HALF
        else:
            out_offsets = i_offsets
        db1_offsets = Eidx.to(tl.int64) * I + out_offsets
        tl.store(db1_ptr + db1_offsets, db1_acc, mask=i_mask)


@torch.library.custom_op(f"{LIBRARY_NAME}::_up_projection_backward_act", mutates_args={"dx_expanded", "db1"})
def _up_projection_backward_act(
    w1: torch.Tensor,
    dx_expanded: torch.Tensor,
    dh: torch.Tensor,
    db1: torch.Tensor | None,
    expert_frequency_offset: torch.Tensor,
    is_glu_activation: bool,
    concat_layout: bool = False,
) -> None:
    I, H, E = w1.size()
    if is_glu_activation:
        I //= 2

    gemm(
        dh,
        w1.permute(2, 0, 1),
        cu_seqlens_m=expert_frequency_offset,
        dynamic_scheduler=False,
        out=dx_expanded,
        concat_layout=(("B",) if concat_layout else None),
    )

    # db1 computation
    if db1 is not None:
        db1_kernel[(E,)](
            dh,
            db1,
            expert_frequency_offset,
            (2 * I if is_glu_activation else I),
            E,
            CONCAT_LAYOUT=concat_layout and is_glu_activation,
        )


_up_projection_backward_act.compile_cache = {}


@torch.library.custom_op(f"{LIBRARY_NAME}::_up_projection_backward_weight", mutates_args={"dw1"})
def _up_projection_backward_weight(
    x: torch.Tensor,
    dw1: torch.Tensor,
    dh: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    x_gather_idx: torch.Tensor,
    is_glu_activation: bool,
    concat_layout: bool = False,
) -> None:
    I, H, E = dw1.size()
    if is_glu_activation:
        I //= 2

    gemm(
        x.T,
        dh,
        out=dw1.permute(2, 1, 0),
        cu_seqlens_k=expert_frequency_offset,
        A_idx=x_gather_idx,
        batch_idx_permute=None,
        dynamic_scheduler=False,
        concat_layout=(("out",) if concat_layout else None),
    )


_up_projection_backward_weight.compile_cache = {}


@torch.library.custom_op(f"{LIBRARY_NAME}::_down_projection_backward_act", mutates_args={"dh", "ds", "db2", "a_prime"})
def _down_projection_backward_act(
    dout: torch.Tensor,
    h: torch.Tensor,
    w2: torch.Tensor,
    dh: torch.Tensor,
    ds: torch.Tensor,
    b2: torch.Tensor | None,
    db2: torch.Tensor | None,  # add impl later
    a_prime: torch.Tensor,
    topk_scores: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    x_gather_idx: torch.Tensor,
    s_scatter_idx: torch.Tensor,
    activation_type: str,
) -> None:
    assert activation_type in (
        "swiglu",
        "geglu",
    ), f"QuACK gemm_gated only supports glu activations, got {activation_type}"

    s = topk_scores[s_scatter_idx]
    _, _, ds_scattered = gemm_dgated(
        dout,
        w2.permute(2, 0, 1),
        PreAct=h,
        activation=activation_type,
        dx_out=dh,
        postact_out=a_prime,
        colvec_scale=s,
        colvec_reduce=True,
        cu_seqlens_m=expert_frequency_offset,
        A_idx=x_gather_idx,
        dynamic_scheduler=False,
    )
    ds[s_scatter_idx] = ds_scattered

    if db2 is None:
        ds[s_scatter_idx] = ds_scattered
    else:
        H = w2.size(0)
        E = expert_frequency_offset.size(0) - 1
        TK = x_gather_idx.size(0)

        old_ds_partial = torch.empty(TK, 1, device=ds_scattered.device, dtype=ds_scattered.dtype)
        old_ds_partial[s_scatter_idx, 0] = ds_scattered

        BLOCK_H = min(triton.next_power_of_2(H), 2048)
        NUM_H_BLOCKS = triton.cdiv(H, BLOCK_H)
        new_ds_partial = torch.empty(TK, NUM_H_BLOCKS, dtype=torch.float32, device=ds.device)

        db2_and_ds_kernel[(E, NUM_H_BLOCKS)](
            dout,
            topk_scores,
            new_ds_partial,
            old_ds_partial,
            b2,
            db2,
            x_gather_idx,
            s_scatter_idx,
            expert_frequency_offset,
            H,
            E,
            1,  # OLD_DS_PARTIAL_N = 1
            BLOCK_H=BLOCK_H,
            BLOCK_OLD_DS_PARTIAL_N=1,
        )

        if NUM_H_BLOCKS == 1:
            ds.copy_(new_ds_partial.view(-1).to(dtype=ds.dtype))
        else:
            ds.copy_(new_ds_partial.sum(dim=-1, dtype=ds.dtype))


_down_projection_backward_act.compile_cache = {}


@torch.library.custom_op(f"{LIBRARY_NAME}::_down_projection_backward_weight", mutates_args={"dw2"})
def _down_projection_backward_weight(
    dout: torch.Tensor,
    a_prime: torch.Tensor,
    dw2: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    x_gather_idx: torch.Tensor,
) -> None:
    gemm(
        dout.T,
        a_prime,
        out=dw2.permute(2, 0, 1),
        cu_seqlens_k=expert_frequency_offset,
        A_idx=x_gather_idx,
        batch_idx_permute=None,
        dynamic_scheduler=False,
    )


_down_projection_backward_weight.compile_cache = {}


@torch.library.custom_op(f"{LIBRARY_NAME}::_token_broadcast_backward", mutates_args={"dx_reduced"})
def _token_broadcast_backward(
    dx_reduced: torch.Tensor,
    dx_expanded: torch.Tensor,
    s_reverse_scatter_idx: torch.Tensor,
    num_activated_expert_per_token_offset: Optional[torch.Tensor],
    varlen_K_max: int,
    H: int,
    is_varlen_K: bool,
) -> None:
    if num_activated_expert_per_token_offset is None:
        assert not is_varlen_K, "`num_activated_expert_per_token_offset` as None requires fixed top-K routing"
    token_gather_and_sum_varlen_K_triton(
        dx_expanded,
        None,
        dx_reduced,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        dx_reduced.size(0),
        varlen_K_max,
        H,
        is_varlen_K,
    )


@triton.jit
def _softmax_over_topk_bwd_kernel(
    dlogits_ptr,
    dlogits_full_ptr,
    score_ptr,
    dscore_ptr,
    idx_ptr,
    stride_dm: tl.constexpr,
    stride_dn: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_gm: tl.constexpr,
    stride_gk: tl.constexpr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    dlogits_is_none: tl.constexpr,
):
    row = tl.program_id(axis=0)

    # tl.assume(K <= BLOCK_K)
    k_offs = tl.arange(0, BLOCK_K)
    k_mask = k_offs < K

    idx = tl.load(idx_ptr + row * stride_im + k_offs * stride_ik, mask=k_mask, other=0).to(tl.int32)
    s_sel = tl.load(score_ptr + row * stride_sm + k_offs * stride_sn, mask=k_mask, other=0).to(tl.float32)
    g_sel = tl.load(dscore_ptr + row * stride_gm + k_offs * stride_gk, mask=k_mask, other=0).to(tl.float32)

    # dot = sum_j g_j * y_j over selected columns
    dot = tl.sum(g_sel * s_sel, axis=0)

    # scatter-only: dx[idx] += y_sel * (g_sel - dot)
    add_vals = s_sel * (g_sel - dot)

    indices = row * stride_dm + idx * stride_dn
    if not dlogits_is_none:
        add_vals += tl.load(dlogits_ptr + indices, mask=k_mask)
    tl.store(dlogits_full_ptr + indices, add_vals, mask=k_mask)


@triton.jit
def _topk_over_softmax_bwd_kernel(
    logits_ptr,  # (T, N) saved router logits
    dlogits_ptr,  # (T, N) output gradient
    dscore_ptr,  # (T, K) upstream gradient
    idx_ptr,  # (T, K) selected indices (int32)
    score_ptr,  # (T, K) forward scores (only used for renorm)
    stride_lm: tl.constexpr,
    stride_le: tl.constexpr,
    stride_dm: tl.constexpr,
    stride_dn: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    stride_scm: tl.constexpr,
    stride_scn: tl.constexpr,
    E: tl.constexpr,
    K: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
    norm_topk_probs: tl.constexpr,
):
    """
    Full topk(softmax()) backward over ALL E indices.

    Forward: logits → p = softmax(logits) → [raw, idx] = topk(p, K)
             → scores = raw / sum(raw)  (if norm_topk_probs)

    Backward:
      1. Recompute p = softmax(logits) over all E
      2. If renorm: dp_sel = (dscore - dot_s) / S
         Else:      dp_sel = dscore
      3. dot = Σ dp_sel_j * p_sel_j
      4. Scatter dp_sel into E-wide dp (zero at non-selected)
      5. dlogits = p * (dp - dot)  for all E
    """
    row = tl.program_id(axis=0)

    e_offs = tl.arange(0, BLOCK_E)
    e_mask = e_offs < E
    logits = tl.load(logits_ptr + row * stride_lm + e_offs * stride_le, mask=e_mask, other=-float("inf")).to(
        tl.float32
    )
    row_max = tl.max(logits, axis=0)
    exp_vals = tl.exp(logits - row_max)
    row_sum = tl.sum(exp_vals, axis=0)
    p = exp_vals / row_sum  # (BLOCK_E,)

    # --- Load K selected indices and upstream gradient ---
    k_offs = tl.arange(0, BLOCK_K)
    k_mask = k_offs < K
    idx = tl.load(
        idx_ptr + row * stride_im + k_offs * stride_ik,
        mask=k_mask,
        other=0,
    ).to(tl.int32)
    g_sel = tl.load(
        dscore_ptr + row * stride_sm + k_offs * stride_sn,
        mask=k_mask,
        other=0,
    ).to(tl.float32)

    # p at selected indices (gather from global mem; can't index register tensor)
    sel_logits = tl.load(
        logits_ptr + row * stride_lm + idx * stride_le,
        mask=k_mask,
        other=-float("inf"),
    ).to(tl.float32)
    p_sel = tl.exp(sel_logits - row_max) / row_sum  # (BLOCK_K,)

    # --- Backward through optional renormalization ---
    if norm_topk_probs:
        scores = tl.load(
            score_ptr + row * stride_scm + k_offs * stride_scn,
            mask=k_mask,
            other=0,
        ).to(tl.float32)
        dot_s = tl.sum(g_sel * scores, axis=0)
        S = tl.sum(p_sel, axis=0)
        dp_sel = (g_sel - dot_s) / S
    else:
        dp_sel = g_sel

    # dot = Σ dp_sel_j * p_sel_j
    dot = tl.sum(dp_sel * p_sel, axis=0)

    # --- Scatter dp_sel into N-wide dp ---
    # dp[i] = dp_sel[k] if i == idx[k], else 0
    # Loop over K (unrolled at compile time since K is constexpr)
    dp = tl.zeros([BLOCK_E], dtype=tl.float32)
    for k_iter in tl.static_range(K):
        cur_dp = tl.sum(tl.where(k_offs == k_iter, dp_sel, 0.0))
        cur_idx = tl.sum(tl.where(k_offs == k_iter, idx, 0))
        dp = tl.where(e_offs == cur_idx, cur_dp, dp)

    # --- dlogits = p * (dp - dot) for all E ---
    dlogits = p * (dp - dot)
    tl.store(
        dlogits_ptr + row * stride_dm + e_offs * stride_dn,
        dlogits,
        mask=e_mask,
    )


@torch.library.custom_op(f"{LIBRARY_NAME}::_topk_softmax_bwd", mutates_args={"dlogits_full"})
def _topk_softmax_bwd(
    router_logits: torch.Tensor,
    dlogits_full: torch.Tensor,
    dlogits: Optional[torch.Tensor],
    dtopk_score: torch.Tensor,
    topk_router_score: torch.Tensor,
    topk_router_indices: torch.Tensor,
    E: int,
    K: int,
    is_softmax_over_topk: bool = True,
    norm_topk_probs: bool = False,
) -> None:
    T = dtopk_score.shape[0]

    if is_softmax_over_topk:
        # non-selected gradient is zero.
        _softmax_over_topk_bwd_kernel[T,](
            dlogits,
            dlogits_full,
            topk_router_score,
            dtopk_score,
            topk_router_indices,
            dlogits_full.stride(0),
            dlogits_full.stride(1),
            topk_router_score.stride(0),
            topk_router_score.stride(1),
            dtopk_score.stride(0),
            dtopk_score.stride(1),
            topk_router_indices.stride(0),
            topk_router_indices.stride(1),
            K,
            triton.next_power_of_2(K),
            (dlogits is None),
        )
    else:
        # topk(softmax(.)): non-selected gradient is -p_i * dot, NOT zero.
        # must recompute full softmax for the complete Jacobian.
        _topk_over_softmax_bwd_kernel[T,](
            router_logits,
            dlogits_full,
            dtopk_score,
            topk_router_indices,
            topk_router_score,
            router_logits.stride(0),
            router_logits.stride(1),
            dlogits_full.stride(0),
            dlogits_full.stride(1),
            dtopk_score.stride(0),
            dtopk_score.stride(1),
            topk_router_indices.stride(0),
            topk_router_indices.stride(1),
            topk_router_score.stride(0),
            topk_router_score.stride(1),
            E,
            K,
            triton.next_power_of_2(E),
            triton.next_power_of_2(K),
            norm_topk_probs,
        )


@triton.jit
def _topk_bwd_scatter_small_kernel(
    dlogits_full_ptr,
    dscore_ptr,
    idx_ptr,
    stride_dm: tl.constexpr,
    stride_dn: tl.constexpr,
    stride_gm: tl.constexpr,
    stride_gk: tl.constexpr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(axis=0)

    # tl.assume(K <= BLOCK_K)
    k_offs = tl.arange(0, BLOCK_K)
    k_mask = k_offs < K

    idx = tl.load(idx_ptr + row * stride_im + k_offs * stride_ik, mask=k_mask, other=0).to(tl.int32)
    g_sel = tl.load(dscore_ptr + row * stride_gm + k_offs * stride_gk, mask=k_mask, other=0).to(tl.float32)

    # scatter-only: dx[idx] += y_sel * (g_sel - dot)
    add_vals = g_sel

    indices = row * stride_dm + idx * stride_dn
    tl.store(dlogits_full_ptr + indices, add_vals, mask=k_mask)


@torch.library.custom_op(f"{LIBRARY_NAME}::_topk_bwd", mutates_args={"dlogits_full"})
def _topk_bwd(
    dlogits_full: torch.Tensor,
    dtopk_values: torch.Tensor,
    topk_indices: torch.Tensor,
    K: int,
) -> None:
    T = dtopk_values.shape[0]

    _topk_bwd_scatter_small_kernel[T,](
        dlogits_full,
        dtopk_values,
        topk_indices,
        dlogits_full.stride(0),
        dlogits_full.stride(1),
        dtopk_values.stride(0),
        dtopk_values.stride(1),
        topk_indices.stride(0),
        topk_indices.stride(1),
        K,
        triton.next_power_of_2(K),
    )
