# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
#
# SonicMoE EP — dispatch metadata kernels + compute_dispatch_metadata(), the only public entry
# point (see its docstring for I/O shapes). Triton-only, host-sync-free, CUDA-graph safe.
# ********************************************************************************

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _metadata_a2a_phase1_reduce_kernel(
    topk_idx_g_ptr,  # (W, T_local, K) int32, contiguous
    out_dst_rank_flat_ptr,  # (TK_global,) int32
    out_tile_count_ptr,  # (W, n_tiles, W) int32
    out_my_dst_rank_ptr,  # (T_local, K) int32
    out_my_expert_local_ptr,  # (T_local, K) int32
    out_expert_local_padded_ptr,  # (TK_global,) int32
    n_tiles,  # runtime stride for tile_count
    my_rank: tl.constexpr,
    W: tl.constexpr,
    W_PAD: tl.constexpr,
    TK_local: tl.constexpr,
    E_local: tl.constexpr,
    BLOCK_TK: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_tile = tl.program_id(1)
    is_mine = pid_r == my_rank

    tile_offs = pid_tile * BLOCK_TK + tl.arange(0, BLOCK_TK)
    valid = tile_offs < TK_local
    flat_offs = pid_r * TK_local + tile_offs

    expert_global = tl.load(topk_idx_g_ptr + flat_offs, mask=valid, other=0)
    dst = expert_global // E_local

    peer_axis = tl.arange(0, W_PAD)
    one_hot = (dst[:, None] == peer_axis[None, :]).to(tl.int32)
    one_hot = tl.where(valid[:, None], one_hot, 0)

    tile_count_p = tl.sum(one_hot, axis=0)

    local_expert = expert_global - dst * E_local
    padded = tl.where(dst == my_rank, local_expert, E_local)

    tl.store(out_dst_rank_flat_ptr + flat_offs, dst, mask=valid)
    tl.store(
        out_tile_count_ptr + pid_r * n_tiles * W + pid_tile * W + peer_axis,
        tile_count_p,
        mask=peer_axis < W,
    )
    tl.store(out_expert_local_padded_ptr + flat_offs, padded, mask=valid)
    if is_mine:
        tl.store(out_my_dst_rank_ptr + tile_offs, dst, mask=valid)
        tl.store(out_my_expert_local_ptr + tile_offs, local_expert, mask=valid)


@triton.jit
def _metadata_a2a_phase2_scan_kernel(
    tile_count_ptr,  # (W, n_tiles, W) int32, contiguous
    out_tile_prefix_ptr,  # (W, n_tiles, W) int32, contiguous
    out_peer_count_per_rank_ptr,  # (W, W) int32, contiguous
    n_tiles,  # runtime: stride and mask bound
    W: tl.constexpr,
    BLOCK_NTILES: tl.constexpr,  # next_pow2(n_tiles), >= 2
):
    """1-D cumsum over n_tiles for one (src_rank, peer) pair; grid (W, W)."""
    pid_r = tl.program_id(0)
    pid_p = tl.program_id(1)

    t_offs = tl.arange(0, BLOCK_NTILES)
    t_mask = t_offs < n_tiles

    addr = tile_count_ptr + pid_r * n_tiles * W + t_offs * W + pid_p
    tc = tl.load(addr, mask=t_mask, other=0)

    incl = tl.cumsum(tc, axis=0)
    excl = incl - tc

    peer_count = tl.sum(tc, axis=0)

    tl.store(
        out_tile_prefix_ptr + pid_r * n_tiles * W + t_offs * W + pid_p,
        excl,
        mask=t_mask,
    )
    tl.store(out_peer_count_per_rank_ptr + pid_r * W + pid_p, peer_count)


@triton.jit
def _metadata_a2a_phase3_emit_kernel(
    dst_rank_flat_ptr,  # (TK_global,) int32
    tile_prefix_ptr,  # (W, n_tiles, W) int32, contiguous
    peer_count_per_rank_ptr,  # (W, W) int32, contiguous
    out_slot_per_rank_ptr,  # (TK_global,) int32
    out_slot_global_ptr,  # (TK_global,) int32
    out_a2a_token_indices_ptr,  # (TK_global,) int32
    n_tiles,  # runtime stride
    W: tl.constexpr,
    W_PAD: tl.constexpr,
    TK_local: tl.constexpr,
    BLOCK_TK: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_tile = tl.program_id(1)

    r_axis = tl.arange(0, W_PAD)
    p_axis = tl.arange(0, W_PAD)
    rs = r_axis[:, None]
    ps = p_axis[None, :]
    pc_mask = (rs < W) & (ps < W)
    pc = tl.load(peer_count_per_rank_ptr + rs * W + ps, mask=pc_mask, other=0)
    excl_pc = tl.cumsum(pc, axis=0) - pc
    row_mask = r_axis == pid_r
    ec_row = tl.sum(tl.where(row_mask[:, None], excl_pc, 0), axis=0)

    tile_offs = pid_tile * BLOCK_TK + tl.arange(0, BLOCK_TK)
    valid = tile_offs < TK_local
    flat_offs = pid_r * TK_local + tile_offs

    dst = tl.load(dst_rank_flat_ptr + flat_offs, mask=valid, other=0)

    p_offs = tl.arange(0, W_PAD)
    one_hot = (dst[:, None] == p_offs[None, :]).to(tl.int32)
    one_hot = tl.where(valid[:, None], one_hot, 0)
    cumsum = tl.cumsum(one_hot, axis=0)
    within_pos = tl.sum(cumsum * one_hot, axis=1) - 1

    tp_row = tl.load(tile_prefix_ptr + pid_r * n_tiles * W + pid_tile * W + p_offs, mask=p_offs < W, other=0)

    tp_at_dst = tl.sum(one_hot * tp_row[None, :], axis=1)
    ec_at_dst = tl.sum(one_hot * ec_row[None, :], axis=1)

    slot_pr = tp_at_dst + within_pos
    slot_gl = slot_pr + ec_at_dst

    a2a_ti = pid_r * TK_local + slot_pr

    tl.store(out_slot_per_rank_ptr + flat_offs, slot_pr, mask=valid)
    tl.store(out_slot_global_ptr + flat_offs, slot_gl, mask=valid)
    tl.store(out_a2a_token_indices_ptr + flat_offs, a2a_ti, mask=valid)


@triton.jit
def _metadata_dedup_phase1_reduce_kernel(
    dst_rank_flat_ptr,  # (TK_global,) int32
    out_present_tile_count_ptr,  # (W, n_token_tiles, W) int32
    out_pair_present_mask_ptr,  # (TK_global,) int8 (0/1)
    n_token_tiles,  # runtime stride
    T_local: tl.constexpr,
    K: tl.constexpr,
    K_PAD: tl.constexpr,  # next_pow2(K)
    W: tl.constexpr,
    W_PAD: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Canonical-slot mask + per-(token-tile, q) presence histogram.
    Invalid lanes get dst=W (sentinel) so one-hot entries auto-zero — no extra mask needed."""
    pid_r = tl.program_id(0)
    pid_tile = tl.program_id(1)

    t_offs_local = tl.arange(0, BLOCK_T)
    t_offs_global = pid_tile * BLOCK_T + t_offs_local
    valid_t = t_offs_global < T_local

    k_axis = tl.arange(0, K_PAD)
    valid_k = k_axis < K
    flat_slot_offs = (pid_r * T_local + t_offs_global)[:, None] * K + k_axis[None, :]
    valid_2d = valid_t[:, None] & valid_k[None, :]

    dst = tl.load(dst_rank_flat_ptr + flat_slot_offs, mask=valid_2d, other=W)

    q_axis = tl.arange(0, W_PAD)
    oh = (dst[:, :, None] == q_axis[None, None, :]).to(tl.int32)

    cum_k = tl.cumsum(oh, axis=1)
    cnt_at_dst = tl.sum(cum_k * oh, axis=2)
    canonical = (cnt_at_dst == 1).to(tl.int8)
    canonical = tl.where(valid_2d, canonical, tl.zeros_like(canonical))

    sum_oh = tl.sum(oh, axis=1)
    present_tq = (sum_oh > 0).to(tl.int32)
    tile_count_q = tl.sum(present_tq, axis=0)

    tl.store(out_pair_present_mask_ptr + flat_slot_offs, canonical, mask=valid_2d)
    tl.store(
        out_present_tile_count_ptr + pid_r * n_token_tiles * W + pid_tile * W + q_axis,
        tile_count_q,
        mask=q_axis < W,
    )


@triton.jit
def _metadata_dedup_phase2_scan_kernel(
    present_tile_count_ptr,
    out_tile_token_prefix_ptr,
    out_pair_count_ptr,
    n_token_tiles,
    W: tl.constexpr,
    BLOCK_NTILES: tl.constexpr,
):
    """1-D cumsum across n_token_tiles for one (src_rank, q) pair."""
    pid_r = tl.program_id(0)
    pid_q = tl.program_id(1)

    t_offs = tl.arange(0, BLOCK_NTILES)
    t_mask = t_offs < n_token_tiles

    addr = present_tile_count_ptr + pid_r * n_token_tiles * W + t_offs * W + pid_q
    tc = tl.load(addr, mask=t_mask, other=0)

    incl = tl.cumsum(tc, axis=0)
    excl = incl - tc
    pair_count = tl.sum(tc, axis=0)

    tl.store(
        out_tile_token_prefix_ptr + pid_r * n_token_tiles * W + t_offs * W + pid_q,
        excl,
        mask=t_mask,
    )
    tl.store(out_pair_count_ptr + pid_r * W + pid_q, pair_count)


@triton.jit
def _metadata_dedup_phase3_emit_kernel(
    dst_rank_flat_ptr,
    pair_count_ptr,
    tile_token_prefix_ptr,
    out_pair_offset_ptr,
    out_rank_dedup_recv_pos_ptr,
    n_token_tiles,
    T_local: tl.constexpr,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
    W: tl.constexpr,
    W_PAD: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Emit pair_offset (W, W) + per-slot rank_dedup_recv_pos (TK_global,)."""
    pid_r = tl.program_id(0)
    pid_tile = tl.program_id(1)

    r_axis = tl.arange(0, W_PAD)
    q_cols = tl.arange(0, W_PAD)
    rs = r_axis[:, None]
    qs = q_cols[None, :]
    pc = tl.load(pair_count_ptr + rs * W + qs, mask=(rs < W) & (qs < W), other=0)
    pair_off = tl.cumsum(pc, axis=0) - pc
    row_mask_r = r_axis == pid_r
    pair_off_row = tl.sum(tl.where(row_mask_r[:, None], pair_off, 0), axis=0)

    if pid_tile == 0:
        q_axis_init = tl.arange(0, W_PAD)
        tl.store(out_pair_offset_ptr + pid_r * W + q_axis_init, pair_off_row, mask=q_axis_init < W)

    t_offs_local = tl.arange(0, BLOCK_T)
    t_offs_global = pid_tile * BLOCK_T + t_offs_local
    valid_t = t_offs_global < T_local

    k_axis = tl.arange(0, K_PAD)
    valid_k = k_axis < K
    flat_slot_offs = (pid_r * T_local + t_offs_global)[:, None] * K + k_axis[None, :]
    valid_2d = valid_t[:, None] & valid_k[None, :]
    dst = tl.load(dst_rank_flat_ptr + flat_slot_offs, mask=valid_2d, other=W)

    q_axis = tl.arange(0, W_PAD)
    oh = (dst[:, :, None] == q_axis[None, None, :]).to(tl.int32)

    sum_oh = tl.sum(oh, axis=1)
    present_tq = (sum_oh > 0).to(tl.int32)
    incl_t = tl.cumsum(present_tq, axis=0)
    excl_t = incl_t - present_tq

    tile_pref = tl.load(tile_token_prefix_ptr + pid_r * n_token_tiles * W + pid_tile * W + q_axis, mask=q_axis < W, other=0)

    within_pq = excl_t + tile_pref[None, :]

    pair_off_per_tk = tl.sum(oh * pair_off_row[None, None, :], axis=2)
    within_per_tk = tl.sum(oh * within_pq[:, None, :], axis=2)
    dedup_pos = pair_off_per_tk + within_per_tk
    tl.store(out_rank_dedup_recv_pos_ptr + flat_slot_offs, dedup_pos, mask=valid_2d)


# HIER inter-node dispatch metadata: node-axis dedup (dst_node = dst_rank // node_size, remote-only),
# producing dst_slot + dst_recv_count for the GIN put/wait + NVLink gather. Oracle: hier_ep_reference.py.
@triton.jit
def _metadata_node_dedup_phase1_reduce_kernel(
    dst_node_flat_ptr,  # (TK_global,) int32  (= dst_rank_flat // node_size)
    out_present_tile_count_ptr,  # (W, n_token_tiles, num_nodes) int32
    out_node_present_mask_ptr,  # (TK_global,) int8 (0/1)
    n_token_tiles,
    T_local: tl.constexpr,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
    num_nodes: tl.constexpr,
    NUM_NODES_PAD: tl.constexpr,
    node_size: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Mirrors _metadata_dedup_phase1_reduce_kernel over num_nodes, with LOCAL (same-node) slots
    forced to one-hot 0 — they use NVLink not GIN, so can never be node-canonical."""
    pid_r = tl.program_id(0)
    pid_tile = tl.program_id(1)
    src_node = pid_r // node_size

    t_offs_local = tl.arange(0, BLOCK_T)
    t_offs_global = pid_tile * BLOCK_T + t_offs_local
    valid_t = t_offs_global < T_local

    k_axis = tl.arange(0, K_PAD)
    valid_k = k_axis < K
    flat_slot_offs = (pid_r * T_local + t_offs_global)[:, None] * K + k_axis[None, :]
    valid_2d = valid_t[:, None] & valid_k[None, :]

    dst_node = tl.load(dst_node_flat_ptr + flat_slot_offs, mask=valid_2d, other=num_nodes)

    n_axis = tl.arange(0, NUM_NODES_PAD)
    is_remote = n_axis != src_node
    oh = ((dst_node[:, :, None] == n_axis[None, None, :]) & is_remote[None, None, :]).to(tl.int32)

    cum_k = tl.cumsum(oh, axis=1)
    cnt_at_node = tl.sum(cum_k * oh, axis=2)
    canonical = (cnt_at_node == 1).to(tl.int8)
    canonical = tl.where(valid_2d, canonical, tl.zeros_like(canonical))

    sum_oh = tl.sum(oh, axis=1)
    present_tn = (sum_oh > 0).to(tl.int32)
    tile_count_n = tl.sum(present_tn, axis=0)

    tl.store(out_node_present_mask_ptr + flat_slot_offs, canonical, mask=valid_2d)
    tl.store(
        out_present_tile_count_ptr + pid_r * n_token_tiles * num_nodes + pid_tile * num_nodes + n_axis,
        tile_count_n,
        mask=n_axis < num_nodes,
    )


@triton.jit
def _metadata_node_stripe_base_kernel(
    node_token_count_ptr,  # (W, num_nodes) int32 — #tokens at src r with >=1 slot to remote node n
    out_stripe_base_ptr,  # (W, num_nodes) int32
    out_dst_recv_count_ptr,  # (W,) int32 — inbound puts per receiving GPU R = n*node_size + m
    num_nodes: tl.constexpr,
    node_size: tl.constexpr,
    NUM_NODES_PAD: tl.constexpr,
):
    """stripe_base[rank_of(s,m),n] = excl-cumsum of node_token_count over s' (same lane m); diagonal
    s'==n is naturally 0 (own node is local) so no special-casing. dst_recv_count[R] = inclusive total (GIN wait count)."""
    pid_m = tl.program_id(0)  # local lane m in [0, node_size)
    pid_n = tl.program_id(1)  # dst node n in [0, num_nodes)

    s = tl.arange(0, NUM_NODES_PAD)
    valid = s < num_nodes
    ranks = s * node_size + pid_m  # rank_of(s, m)
    vals = tl.load(node_token_count_ptr + ranks * num_nodes + pid_n, mask=valid, other=0)

    incl = tl.cumsum(vals, axis=0)
    excl = incl - vals
    total = tl.sum(vals, axis=0)

    tl.store(out_stripe_base_ptr + ranks * num_nodes + pid_n, excl, mask=valid)
    tl.store(out_dst_recv_count_ptr + pid_n * node_size + pid_m, total)  # receiving GPU R = n*node_size+m


@triton.jit
def _metadata_node_dedup_phase3_emit_kernel(
    dst_node_flat_ptr,  # (TK_global,) int32
    stripe_base_ptr,  # (W, num_nodes) int32
    tile_token_prefix_ptr,  # (W, n_token_tiles, num_nodes) int32
    out_dst_slot_ptr,  # (TK_global,) int32 — row in the dst_node_buffer, -1 on local/invalid slots
    n_token_tiles,
    T_local: tl.constexpr,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
    num_nodes: tl.constexpr,
    NUM_NODES_PAD: tl.constexpr,
    node_size: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """dst_slot = stripe_base + within-node prefix, set on EVERY remote slot (not just node-canonical)
    so all K slots of a (src,token)->same dst node resolve to ONE shared row. Local/invalid slots get -1."""
    pid_r = tl.program_id(0)
    pid_tile = tl.program_id(1)
    src_node = pid_r // node_size

    t_offs_local = tl.arange(0, BLOCK_T)
    t_offs_global = pid_tile * BLOCK_T + t_offs_local
    valid_t = t_offs_global < T_local

    k_axis = tl.arange(0, K_PAD)
    valid_k = k_axis < K
    flat_slot_offs = (pid_r * T_local + t_offs_global)[:, None] * K + k_axis[None, :]
    valid_2d = valid_t[:, None] & valid_k[None, :]
    dst_node = tl.load(dst_node_flat_ptr + flat_slot_offs, mask=valid_2d, other=num_nodes)

    n_axis = tl.arange(0, NUM_NODES_PAD)
    is_remote = n_axis != src_node
    oh = ((dst_node[:, :, None] == n_axis[None, None, :]) & is_remote[None, None, :]).to(tl.int32)

    # within-node exclusive token prefix (presence), + cross-tile prefix
    sum_oh = tl.sum(oh, axis=1)
    present_tn = (sum_oh > 0).to(tl.int32)
    incl_t = tl.cumsum(present_tn, axis=0)
    excl_t = incl_t - present_tn
    tile_pref = tl.load(
        tile_token_prefix_ptr + pid_r * n_token_tiles * num_nodes + pid_tile * num_nodes + n_axis,
        mask=n_axis < num_nodes, other=0)
    within_tn = excl_t + tile_pref[None, :]  # (BLOCK_T, num_nodes)

    sb_row = tl.load(stripe_base_ptr + pid_r * num_nodes + n_axis, mask=n_axis < num_nodes, other=0)  # (num_nodes,) stripe_base[r, :]

    # one-hot select the dst node's stripe_base + within-node prefix, per slot
    dst_slot_per_tk = (tl.sum(oh * sb_row[None, None, :], axis=2)
                       + tl.sum(oh * within_tn[:, None, :], axis=2))
    has_remote = tl.sum(oh, axis=2)  # 1 if the slot's dst node is remote, else 0

    dst_slot_out = tl.where(has_remote > 0, dst_slot_per_tk, -1)
    dst_slot_out = tl.where(valid_2d, dst_slot_out, -1)

    tl.store(out_dst_slot_ptr + flat_slot_offs, dst_slot_out, mask=valid_2d)


@triton.jit
def _build_peer_present_mask_kernel(
    dst_rank_flat_ptr,  # (TK_global,) int32
    out_mask_ptr,  # (W, T_local) int8
    T_local: tl.constexpr,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
    W: tl.constexpr,
    W_PAD: tl.constexpr,
    my_rank: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Build (W, T_local) presence mask: peer q has any expert for my (t)
    iff ∃k: dst_rank_flat[my_rank * TK_local + t * K + k] == q."""
    pid = tl.program_id(0)
    t_offs = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    valid_t = t_offs < T_local

    k_axis = tl.arange(0, K_PAD)
    valid_k = k_axis < K
    flat_slot_offs = (my_rank * T_local + t_offs)[:, None] * K + k_axis[None, :]
    valid_2d = valid_t[:, None] & valid_k[None, :]

    dst = tl.load(dst_rank_flat_ptr + flat_slot_offs, mask=valid_2d, other=W)

    q_axis = tl.arange(0, W_PAD)
    oh = (dst[:, :, None] == q_axis[None, None, :]).to(tl.int32)
    present_tq = (tl.sum(oh, axis=1) > 0).to(tl.int8)

    store_offs = q_axis[None, :] * T_local + t_offs[:, None]
    store_mask = valid_t[:, None] & (q_axis < W)[None, :]
    tl.store(out_mask_ptr + store_offs, present_tq, mask=store_mask)


@triton.jit
def _build_combine_peer_present_all_kernel(
    dst_rank_flat_ptr,  # (TK_global,) int32
    out_all_ptr,        # (W, W, T_local) int8 = [origin R][peer q][token t]
    T_local: tl.constexpr,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
    W: tl.constexpr,
    W_PAD: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """(W,W,T_local) ALL-origins version of peer_present_mask: the HIER gateway reduces OTHER origins'
    partials too, so it needs everyone's masks, not just my_rank's (origin reads its own [my_rank] slice)."""
    R = tl.program_id(0)
    pid_t = tl.program_id(1)
    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    valid_t = t_offs < T_local

    k_axis = tl.arange(0, K_PAD)
    valid_k = k_axis < K
    flat_slot_offs = (R * T_local + t_offs)[:, None] * K + k_axis[None, :]
    valid_2d = valid_t[:, None] & valid_k[None, :]
    dst = tl.load(dst_rank_flat_ptr + flat_slot_offs, mask=valid_2d, other=W)

    q_axis = tl.arange(0, W_PAD)
    oh = (dst[:, :, None] == q_axis[None, None, :]).to(tl.int32)  # (BLOCK_T, K_PAD, W_PAD)
    present_tq = (tl.sum(oh, axis=1) > 0).to(tl.int8)             # (BLOCK_T, W_PAD)

    # store to out[R, q, t] = R*(W*T_local) + q*T_local + t
    store_offs = R * (W * T_local) + q_axis[None, :] * T_local + t_offs[:, None]
    store_mask = valid_t[:, None] & (q_axis < W)[None, :]
    tl.store(out_all_ptr + store_offs, present_tq, mask=store_mask)


@triton.jit
def _build_combine_single_k_kernel(
    dst_rank_flat_ptr,  # (TK_global,) int32
    out_single_k_ptr,  # (W, T_local) int8 — k of the SINGLE contributor for (q,t), else -1
    T_local: tl.constexpr,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
    W: tl.constexpr,
    W_PAD: tl.constexpr,
    my_rank: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """single_k[q,t] = the k-slot if EXACTLY ONE of t's K experts routes to peer q, else -1. Lets the
    selective gather read single contributors directly (their pre-reduce would save no NVLink)."""
    pid = tl.program_id(0)
    t_offs = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    valid_t = t_offs < T_local

    k_axis = tl.arange(0, K_PAD)
    valid_k = k_axis < K
    flat_slot_offs = (my_rank * T_local + t_offs)[:, None] * K + k_axis[None, :]
    valid_2d = valid_t[:, None] & valid_k[None, :]

    dst = tl.load(dst_rank_flat_ptr + flat_slot_offs, mask=valid_2d, other=W)  # (BLOCK_T, K_PAD)

    q_axis = tl.arange(0, W_PAD)
    oh = (dst[:, :, None] == q_axis[None, None, :]).to(tl.int32)  # (BLOCK_T, K_PAD, W_PAD)
    cnt = tl.sum(oh, axis=1)  # (BLOCK_T, W_PAD) experts of t on q
    # k-slot of the single contributor: sum of k over matching slots, valid only
    # when cnt==1 (exactly one slot matches); -1 for absent (0) and multi (>=2).
    k_contrib = tl.sum(oh * k_axis[None, :, None], axis=1)  # (BLOCK_T, W_PAD)
    single_k = tl.where(cnt == 1, k_contrib, -1).to(tl.int8)  # (BLOCK_T, W_PAD)

    store_offs = q_axis[None, :] * T_local + t_offs[:, None]
    store_mask = valid_t[:, None] & (q_axis < W)[None, :]
    tl.store(out_single_k_ptr + store_offs, single_k, mask=store_mask)


@triton.jit
def _build_mine_slot_idx_kernel(
    dst_rank_flat_ptr,  # (TK_global,) int32 — destination rank per slot
    out_mine_slot_idx_ptr,  # (W*T_local, C) int32 — packed flat slot indices of mine-slots
    out_mine_count_ptr,  # (W*T_local,) int32 — number of mine-slots per row, 0..C
    WT_local,  # runtime bound = W * T_local
    K: tl.constexpr,
    K_PAD: tl.constexpr,  # next_pow2(K)
    C: tl.constexpr,  # = min(K, E_local), column stride of mine_slot_idx
    my_rank: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Packs the slot indices routing to my_rank into mine_slot_idx[g,:] + mine_count[g], so the
    tight-loop combine producer iterates only contributing slots instead of all K. C = min(K, E_local)."""
    pid = tl.program_id(0)
    g_offs = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    valid_g = g_offs < WT_local

    k_axis = tl.arange(0, K_PAD)
    valid_k = k_axis < K
    flat_slot = g_offs[:, None] * K + k_axis[None, :]  # (BLOCK_T, K_PAD)
    valid_2d = valid_g[:, None] & valid_k[None, :]

    dst = tl.load(dst_rank_flat_ptr + flat_slot, mask=valid_2d, other=-1)
    is_mine = ((dst == my_rank) & valid_2d).to(tl.int32)  # (BLOCK_T, K_PAD)

    # Exclusive cumsum along k → packed column for each mine-slot.
    incl = tl.cumsum(is_mine, axis=1)
    excl = incl - is_mine
    count = tl.sum(is_mine, axis=1)  # (BLOCK_T,)

    # Scatter mine-slots into their packed column. Non-mine cells are masked
    # out; excl < count ≤ C for mine cells ⇒ in-bounds.
    store_off = g_offs[:, None] * C + excl
    tl.store(out_mine_slot_idx_ptr + store_off, flat_slot, mask=(is_mine != 0))
    tl.store(out_mine_count_ptr + g_offs, count, mask=valid_g)


# Work-list stream compaction: replaces an ~11-op torch cumsum/scatter chain (270us eager, ~= the
# whole combine win) with 3 fused Triton launches (~55us eager); rows land in increasing-g order (byte-identical).


@triton.jit
def _worklist_count_kernel(
    mine_count_ptr,  # (WT_local,) int32
    block_counts_ptr,  # (num_blocks,) int32 — live count per block
    WT_local,
    T_local,
    my_rank,
    BLOCK: tl.constexpr,
    MIN_COUNT: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < WT_local
    mc = tl.load(mine_count_ptr + offs, mask=mask, other=0)
    is_self = (offs // T_local) == my_rank
    # MIN_COUNT=1 = all contributing rows; MIN_COUNT=2 (selective dedup) keeps only MULTI-peer rows —
    # singles are dropped since the home rank reads their y_symm directly. Self stripe always live.
    live = ((mc >= MIN_COUNT) | is_self) & mask
    tl.store(block_counts_ptr + pid, tl.sum(live.to(tl.int32)))


@triton.jit
def _worklist_blockscan_kernel(
    block_counts_ptr,  # (num_blocks,) int32
    block_offsets_ptr,  # (num_blocks,) int32 — exclusive prefix of block_counts
    work_count_ptr,  # (1,) int32 — total live count (device scalar)
    num_blocks,
    BLOCK_NB: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_NB)
    mask = offs < num_blocks
    cnt = tl.load(block_counts_ptr + offs, mask=mask, other=0)
    excl = tl.cumsum(cnt, axis=0) - cnt
    tl.store(block_offsets_ptr + offs, excl, mask=mask)
    tl.store(work_count_ptr, tl.sum(cnt))


@triton.jit
def _worklist_scatter_kernel(
    mine_count_ptr,  # (WT_local,) int32
    block_offsets_ptr,  # (num_blocks,) int32
    work_list_ptr,  # (WT_local,) int32 — compact live rows, increasing-g order
    WT_local,
    T_local,
    my_rank,
    BLOCK: tl.constexpr,
    MIN_COUNT: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < WT_local
    mc = tl.load(mine_count_ptr + offs, mask=mask, other=0)
    is_self = (offs // T_local) == my_rank
    live = ((mc >= MIN_COUNT) | is_self) & mask  # MIN_COUNT must match _worklist_count_kernel
    live_i = live.to(tl.int32)
    local_excl = tl.cumsum(live_i, axis=0) - live_i
    base = tl.load(block_offsets_ptr + pid)
    pos = base + local_excl
    tl.store(work_list_ptr + pos, offs.to(tl.int32), mask=live)


def compute_dispatch_metadata(
    topk_idx_global: torch.Tensor,
    my_rank: int,
    E_local: int,
    *,
    emit_dedup: bool = True,
    emit_combine: bool = True,
    emit_hier: bool = False,
    node_size: int = 1,
):
    """Host-sync-free, CUDA-graph-safe. emit_hier adds HIER inter-node metadata (dst_slot/dst_recv_count
    for the GIN path); requires emit_dedup. All other keys are byte-identical when emit_hier=False."""
    W, T_local, K = topk_idx_global.shape
    TK_local = T_local * K
    TK_global = W * TK_local
    device = topk_idx_global.device

    # Pad the peer/q (W) axis to a power of 2 so tl.arange works for non-pow2 W; the
    # padded lanes [W, W_PAD) are masked on every W-axis load/store/reduction below.
    W_PAD = max(triton.next_power_of_2(W), 1)

    BLOCK_TK = max(triton.next_power_of_2(TK_local), 64) if TK_local <= 256 else 512
    n_tiles = (TK_local + BLOCK_TK - 1) // BLOCK_TK

    dst_rank_flat = torch.empty(TK_global, dtype=torch.int32, device=device)
    tile_count = torch.empty((W, n_tiles, W), dtype=torch.int32, device=device)
    my_dst_rank = torch.empty((T_local, K), dtype=torch.int32, device=device)
    my_expert_local = torch.empty((T_local, K), dtype=torch.int32, device=device)
    expert_local_padded = torch.empty(TK_global, dtype=torch.int32, device=device)

    _metadata_a2a_phase1_reduce_kernel[(W, n_tiles)](
        topk_idx_global,
        dst_rank_flat,
        tile_count,
        my_dst_rank,
        my_expert_local,
        expert_local_padded,
        n_tiles,
        my_rank=my_rank,
        W=W,
        W_PAD=W_PAD,
        TK_local=TK_local,
        E_local=E_local,
        BLOCK_TK=BLOCK_TK,
    )

    tile_prefix = torch.empty_like(tile_count)
    peer_count_per_rank = torch.empty((W, W), dtype=torch.int32, device=device)
    BLOCK_NTILES = max(triton.next_power_of_2(n_tiles), 2)
    _metadata_a2a_phase2_scan_kernel[(W, W)](
        tile_count,
        tile_prefix,
        peer_count_per_rank,
        n_tiles,
        W=W,
        BLOCK_NTILES=BLOCK_NTILES,
    )

    slot_per_rank = torch.empty(TK_global, dtype=torch.int32, device=device)
    slot_global = torch.empty(TK_global, dtype=torch.int32, device=device)
    a2a_token_indices = torch.empty(TK_global, dtype=torch.int32, device=device)
    _metadata_a2a_phase3_emit_kernel[(W, n_tiles)](
        dst_rank_flat,
        tile_prefix,
        peer_count_per_rank,
        slot_per_rank,
        slot_global,
        a2a_token_indices,
        n_tiles,
        W=W,
        W_PAD=W_PAD,
        TK_local=TK_local,
        BLOCK_TK=BLOCK_TK,
    )

    my_pos_on_peer = slot_global[my_rank * TK_local : (my_rank + 1) * TK_local].view(T_local, K)
    my_pos_per_rank = slot_per_rank[my_rank * TK_local : (my_rank + 1) * TK_local].view(T_local, K)

    out = {
        "dst_rank_flat": dst_rank_flat,
        "slot_flat_per_rank": slot_per_rank,
        "slot_flat_global": slot_global,
        "my_dst_rank": my_dst_rank,
        "my_pos_on_peer": my_pos_on_peer,
        "my_pos_per_rank": my_pos_per_rank,
        "my_expert_local": my_expert_local,
        "peer_count_per_rank": peer_count_per_rank,
        "expert_local_padded": expert_local_padded,
        "a2a_token_indices": a2a_token_indices,
    }

    if not emit_dedup:
        return out

    K_PAD = max(triton.next_power_of_2(K), 1)
    # Cap the (BLOCK_T, K_PAD, W) one-hot at ~8K elems/CTA — larger blows B200's per-CTA SMEM
    # budget at large W*K.
    MAX_OH_ELEMS = 8192
    oh_per_t = K_PAD * max(W, 1)
    BLOCK_T = max(
        16,
        min(
            triton.next_power_of_2(max(T_local, 16)),
            64,
            triton.next_power_of_2(max(MAX_OH_ELEMS // max(oh_per_t, 1), 16)),
        ),
    )
    n_token_tiles = (T_local + BLOCK_T - 1) // BLOCK_T

    present_tile_count = torch.empty((W, n_token_tiles, W), dtype=torch.int32, device=device)
    pair_present_mask = torch.empty(TK_global, dtype=torch.int8, device=device)

    _metadata_dedup_phase1_reduce_kernel[(W, n_token_tiles)](
        dst_rank_flat,
        present_tile_count,
        pair_present_mask,
        n_token_tiles,
        T_local=T_local,
        K=K,
        K_PAD=K_PAD,
        W=W,
        W_PAD=W_PAD,
        BLOCK_T=BLOCK_T,
    )

    tile_token_prefix = torch.empty_like(present_tile_count)
    pair_count = torch.empty((W, W), dtype=torch.int32, device=device)
    BLOCK_NTOKEN_TILES = max(triton.next_power_of_2(n_token_tiles), 2)
    _metadata_dedup_phase2_scan_kernel[(W, W)](
        present_tile_count,
        tile_token_prefix,
        pair_count,
        n_token_tiles,
        W=W,
        BLOCK_NTILES=BLOCK_NTOKEN_TILES,
    )

    pair_offset = torch.empty((W, W), dtype=torch.int32, device=device)
    rank_dedup_recv_pos = torch.empty(TK_global, dtype=torch.int32, device=device)
    _metadata_dedup_phase3_emit_kernel[(W, n_token_tiles)](
        dst_rank_flat,
        pair_count,
        tile_token_prefix,
        pair_offset,
        rank_dedup_recv_pos,
        n_token_tiles,
        T_local=T_local,
        K=K,
        K_PAD=K_PAD,
        W=W,
        W_PAD=W_PAD,
        BLOCK_T=BLOCK_T,
    )

    peer_present_mask = torch.empty((W, T_local), dtype=torch.int8, device=device)
    oh_per_t = K_PAD * W
    BLOCK_T_MASK = max(
        16,
        min(
            triton.next_power_of_2(max(T_local, 16)),
            triton.next_power_of_2(max(MAX_OH_ELEMS // max(oh_per_t, 1), 16)),
        ),
    )
    _build_peer_present_mask_kernel[(triton.cdiv(T_local, BLOCK_T_MASK),)](
        dst_rank_flat,
        peer_present_mask,
        T_local=T_local,
        K=K,
        K_PAD=K_PAD,
        W=W,
        W_PAD=W_PAD,
        my_rank=my_rank,
        BLOCK_T=BLOCK_T_MASK,
    )

    out["pair_count"] = pair_count
    out["pair_offset"] = pair_offset
    out["pair_present_mask"] = pair_present_mask
    out["rank_dedup_recv_pos"] = rank_dedup_recv_pos
    out["peer_present_mask"] = peer_present_mask

    # HIER metadata: node-dedup pipeline (remote-masked) + strided stripe-base cumsum. Still writes
    # recv_packed[rank_dedup_recv_pos] (layout unchanged) — only how remote rows ARRIVE differs, GEMM gather is identical.
    if emit_hier:
        assert emit_dedup, \
            "emit_hier requires emit_dedup (hierarchical gather writes recv_packed[rank_dedup_recv_pos])"
        assert W % node_size == 0, f"node_size {node_size} must divide W {W}"
        num_nodes = W // node_size
        # Pad the node axis to a power of 2 (mirrors W_PAD) so tl.arange works for non-pow2 num_nodes.
        NUM_NODES_PAD = max(triton.next_power_of_2(num_nodes), 1)

        # dst node + local/remote split (elementwise, host-sync-free, graph-capturable)
        dst_node_flat = torch.div(dst_rank_flat, node_size, rounding_mode="floor").to(torch.int32)
        src_rank_flat = torch.div(
            torch.arange(TK_global, device=device, dtype=torch.int32), TK_local, rounding_mode="floor")
        is_local_slot = (
            torch.div(src_rank_flat, node_size, rounding_mode="floor") == dst_node_flat).to(torch.int8)

        oh_per_t_node = K_PAD * max(num_nodes, 1)
        BLOCK_T_NODE = max(
            16,
            min(
                triton.next_power_of_2(max(T_local, 16)),
                64,
                triton.next_power_of_2(max(MAX_OH_ELEMS // max(oh_per_t_node, 1), 16)),
            ),
        )
        n_node_tiles = (T_local + BLOCK_T_NODE - 1) // BLOCK_T_NODE

        present_tile_count_node = torch.empty((W, n_node_tiles, num_nodes), dtype=torch.int32, device=device)
        node_present_mask = torch.empty(TK_global, dtype=torch.int8, device=device)
        _metadata_node_dedup_phase1_reduce_kernel[(W, n_node_tiles)](
            dst_node_flat,
            present_tile_count_node,
            node_present_mask,
            n_node_tiles,
            T_local=T_local,
            K=K,
            K_PAD=K_PAD,
            num_nodes=num_nodes,
            NUM_NODES_PAD=NUM_NODES_PAD,
            node_size=node_size,
            BLOCK_T=BLOCK_T_NODE,
        )

        # Reuse the dedup token-tile scan over the node axis (W := num_nodes columns).
        tile_token_prefix_node = torch.empty_like(present_tile_count_node)
        node_token_count = torch.empty((W, num_nodes), dtype=torch.int32, device=device)
        BLOCK_NTILES_NODE = max(triton.next_power_of_2(n_node_tiles), 2)
        _metadata_dedup_phase2_scan_kernel[(W, num_nodes)](
            present_tile_count_node,
            tile_token_prefix_node,
            node_token_count,
            n_node_tiles,
            W=num_nodes,
            BLOCK_NTILES=BLOCK_NTILES_NODE,
        )

        stripe_base = torch.empty((W, num_nodes), dtype=torch.int32, device=device)
        dst_recv_count = torch.empty(W, dtype=torch.int32, device=device)
        _metadata_node_stripe_base_kernel[(node_size, num_nodes)](
            node_token_count,
            stripe_base,
            dst_recv_count,
            num_nodes=num_nodes,
            node_size=node_size,
            NUM_NODES_PAD=NUM_NODES_PAD,
        )

        dst_slot = torch.empty(TK_global, dtype=torch.int32, device=device)
        _metadata_node_dedup_phase3_emit_kernel[(W, n_node_tiles)](
            dst_node_flat,
            stripe_base,
            tile_token_prefix_node,
            dst_slot,
            n_node_tiles,
            T_local=T_local,
            K=K,
            K_PAD=K_PAD,
            num_nodes=num_nodes,
            NUM_NODES_PAD=NUM_NODES_PAD,
            node_size=node_size,
            BLOCK_T=BLOCK_T_NODE,
        )

        out["dst_node_flat"] = dst_node_flat        # node_idx per slot (the put's target node)
        out["is_local_slot"] = is_local_slot
        out["node_present_mask"] = node_present_mask
        out["node_token_count"] = node_token_count
        out["stripe_base"] = stripe_base            # (W, num_nodes) my tokens→node n land at [stripe_base[r,n], +node_token_count[r,n]) in (n,m)'s buffer — contiguous ⇒ ONE coalesced put per remote node
        out["dst_slot"] = dst_slot                  # row in the receiving GPU's dst_node_buffer
        out["dst_recv_count"] = dst_recv_count       # inbound puts per receiving GPU (signal-wait count)

        # Combine reads back exactly the nodes dispatch sent to, so contrib_node_mask + inbound count are
        # FREE from node_token_count (diagonal already 0). combine_recv stripe is closed-form, no tensor needed.
        contrib_node_mask = (node_token_count > 0).to(torch.int8)              # (W, num_nodes)
        expected_count_combine = contrib_node_mask.sum(dim=1).to(torch.int32)  # (W,) inbound combine puts per R
        # The gateway reduces node-local peers' partials on behalf of OTHER origins ⇒ needs ALL origins'
        # present masks (per-rank peer_present_mask only covers my_rank); origin final reduce reads [my_rank].
        combine_peer_present_all = torch.empty((W, W, T_local), dtype=torch.int8, device=device)
        _build_combine_peer_present_all_kernel[(W, triton.cdiv(T_local, BLOCK_T_MASK))](
            dst_rank_flat, combine_peer_present_all,
            T_local=T_local, K=K, K_PAD=K_PAD, W=W, W_PAD=W_PAD, BLOCK_T=BLOCK_T_MASK,
        )
        out["contrib_node_mask"] = contrib_node_mask
        out["expected_count_combine"] = expected_count_combine
        out["combine_peer_present_all"] = combine_peer_present_all

    # Combine-producer metadata: only RANK_DEDUP combine reads it (~136us eager/~17us cudagraph). Skipped
    # otherwise — _do_combine falls back to all-K when these keys are absent.
    if not emit_combine:
        return out

    # Tight-loop producer metadata: per token row g, the packed slot indices routing to my_rank + count,
    # so the producer iterates only contributing slots (~1 at high W) instead of all K. Host-sync-free.
    WT_local = W * T_local
    C_contrib = min(K, E_local)
    mine_slot_idx = torch.zeros((WT_local, C_contrib), dtype=torch.int32, device=device)
    mine_count = torch.zeros(WT_local, dtype=torch.int32, device=device)
    BLOCK_T_MINE = max(16, min(triton.next_power_of_2(max(T_local, 16)), 256))
    _build_mine_slot_idx_kernel[(triton.cdiv(WT_local, BLOCK_T_MINE),)](
        dst_rank_flat,
        mine_slot_idx,
        mine_count,
        WT_local,
        K=K,
        K_PAD=K_PAD,
        C=C_contrib,
        my_rank=my_rank,
        BLOCK_T=BLOCK_T_MINE,
    )
    out["mine_slot_idx"] = mine_slot_idx
    out["mine_count"] = mine_count
    out["combine_contrib_C"] = C_contrib

    # Selective-dedup gather metadata (see _build_combine_single_k_kernel). Always emitted under
    # emit_combine; only the selective gather path consumes it.
    combine_single_k = torch.empty((W, T_local), dtype=torch.int8, device=device)
    BLOCK_T_SK = max(16, min(triton.next_power_of_2(max(T_local, 16)), 256))
    _build_combine_single_k_kernel[(triton.cdiv(T_local, BLOCK_T_SK),)](
        dst_rank_flat,
        combine_single_k,
        T_local=T_local,
        K=K,
        K_PAD=K_PAD,
        W=W,
        W_PAD=W_PAD,
        my_rank=my_rank,
        BLOCK_T=BLOCK_T_SK,
    )
    out["combine_single_k"] = combine_single_k

    # Live work-list = (count>0) OR self-stripe (so the gather consumer's out[t] init is never stale).
    # combine_work_list is zero-init; producer only reads [0:combine_work_count].
    WL_BLOCK = 2048
    num_wl_blocks = triton.cdiv(WT_local, WL_BLOCK)
    BLOCK_NB = max(triton.next_power_of_2(num_wl_blocks), 1)
    block_counts = torch.empty(num_wl_blocks, dtype=torch.int32, device=device)
    block_offsets = torch.empty(num_wl_blocks, dtype=torch.int32, device=device)
    combine_work_list = torch.zeros(WT_local, dtype=torch.int32, device=device)
    combine_work_count = torch.empty(1, dtype=torch.int32, device=device)  # device scalar
    _worklist_count_kernel[(num_wl_blocks,)](
        mine_count, block_counts, WT_local, T_local, my_rank, BLOCK=WL_BLOCK
    )
    _worklist_blockscan_kernel[(1,)](
        block_counts, block_offsets, combine_work_count, num_wl_blocks, BLOCK_NB=BLOCK_NB
    )
    _worklist_scatter_kernel[(num_wl_blocks,)](
        mine_count, block_offsets, combine_work_list, WT_local, T_local, my_rank, BLOCK=WL_BLOCK
    )
    out["combine_work_list"] = combine_work_list
    out["combine_work_count"] = combine_work_count

    # MULTI-ONLY work-list (MIN_COUNT=2): drops single-peer rows (home rank reads y_symm directly),
    # shrinking the producer 48/71/85% at W=8/16/32. Output is identical — dropped partials were never read.
    combine_work_list_multi = torch.zeros(WT_local, dtype=torch.int32, device=device)
    combine_work_count_multi = torch.empty(1, dtype=torch.int32, device=device)
    _worklist_count_kernel[(num_wl_blocks,)](
        mine_count, block_counts, WT_local, T_local, my_rank, BLOCK=WL_BLOCK, MIN_COUNT=2
    )
    _worklist_blockscan_kernel[(1,)](
        block_counts, block_offsets, combine_work_count_multi, num_wl_blocks, BLOCK_NB=BLOCK_NB
    )
    _worklist_scatter_kernel[(num_wl_blocks,)](
        mine_count, block_offsets, combine_work_list_multi, WT_local, T_local, my_rank, BLOCK=WL_BLOCK, MIN_COUNT=2
    )
    out["combine_work_list_multi"] = combine_work_list_multi
    out["combine_work_count_multi"] = combine_work_count_multi
    return out
