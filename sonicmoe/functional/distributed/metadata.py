# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
#
# SonicMoE EP — dispatch metadata kernels and a high-level wrapper.
#
# `compute_dispatch_metadata(topk_idx_global, my_rank, E_local, emit_dedup=True)`
# is the only public entry point. It runs a fixed all-Triton pipeline (no
# host syncs, CUDA-graph safe) and returns a dict of int32/int8 metadata
# tensors that the dispatch and combine primitives consume.
#
# ============================================================================
# Inputs
# ============================================================================
#   topk_idx_global : (W, T_local, K) int32
#         Global expert ID (∈ [0, W·E_local)) for each (rank, token, slot).
#         Built externally — typically dist.all_gather(topk_local).
#   my_rank : int — this rank's index in [0, W).
#   E_local : int — experts per rank. E_global == W · E_local.
#   emit_dedup : bool — when True, additionally emit RANK_DEDUP dispatch /
#         combine metadata. Existing keys are byte-identical regardless.
#
# ============================================================================
# Outputs (always)
# ============================================================================
#   dst_rank_flat        (TK_global,) int32 — destination rank per slot.
#                                       dst_rank_flat[r*TK_local + t*K + k] =
#                                       topk_idx_global[r, t, k] // E_local.
#   slot_flat_per_rank   (TK_global,) int32 — within source-r's stripe of
#                                       the destination buffer, the row
#                                       index of this slot. Counts how many
#                                       prior slots in source r have the
#                                       same dst as this one (i.e. cumcount
#                                       per (src_rank, dst_rank) bucket).
#   slot_flat_global     (TK_global,) int32 — global packed-receive row at
#                                       the destination. Source ranks
#                                       concatenated in order:
#                                       slot_flat_global[f] = slot_flat_per_rank[f]
#                                                            + early_count[r, dst[f]]
#                                       where early_count is the exclusive
#                                       cumsum across source ranks of
#                                       peer_count_per_rank[:, dst].
#   my_dst_rank          (T_local, K) int32 — my_rank's slice of dst_rank_flat,
#                                       reshaped. dst rank for each of my
#                                       (t, k) slots.
#   my_pos_on_peer       (T_local, K) int32 — my_rank's slice of
#                                       slot_flat_global, reshaped.
#   my_pos_per_rank      (T_local, K) int32 — my_rank's slice of
#                                       slot_flat_per_rank, reshaped.
#   my_expert_local      (T_local, K) int32 — my_rank's slice of (expert
#                                       global − dst·E_local), reshaped.
#                                       The local expert index on the dst rank.
#   peer_count_per_rank  (W, W) int32 — peer_count_per_rank[r, p] = #slots
#                                       at source r heading to peer p.
#                                       Histogram of (src_rank, dst_rank).
#   expert_local_padded  (TK_global,) int32 — local expert index when
#                                       dst==my_rank, sentinel E_local
#                                       otherwise. Lets dispatch consumers
#                                       skip "not for me" slots without a
#                                       separate mask.
#   a2a_token_indices    (TK_global,) int32 — src_rank·TK_local +
#                                       slot_flat_per_rank. Per-rank-slot
#                                       layout used by A2A dispatch.
#
# ============================================================================
# Outputs (when ``emit_dedup=True``)
# ============================================================================
#   pair_count           (W, W) int32 — pair_count[p, q] = #tokens at source
#                                       p with at least one slot routed to q
#                                       (so ≤ T_local; #unique destinations
#                                       per-token ≤ K).
#   pair_offset          (W, W) int32 — exclusive cumsum across source-rank
#                                       axis of pair_count[:, q]. Source p's
#                                       stripe in dest q's packed receive
#                                       buffer starts at row pair_offset[p, q]
#                                       and has pair_count[p, q] rows.
#   pair_present_mask    (TK_global,) int8 — 1 at the lowest-k canonical
#                                       slot of each (p, t, dst[f]) triple,
#                                       0 elsewhere. The RANK_DEDUP dispatch
#                                       kernel only fires on canonical slots,
#                                       collapsing K duplicate peer reads
#                                       to one row per (token, peer) pair.
#   rank_dedup_recv_pos   (TK_global,) int32 — packed-receive row at
#                                       destination dst[f]. All K slots of
#                                       (p, t) that route to the same dst q
#                                       share the same recv pos, so a peer
#                                       gathering combine can de-duplicate.
#   peer_present_mask    (W, T_local) int8 — peer_present_mask[q, t] = 1 iff
#                                       peer q has at least one expert for
#                                       MY token t (∃k: dst_rank_flat[my_rank
#                                       · TK_local + t · K + k] == q). Used
#                                       by the RANK_DEDUP combine gather
#                                       to skip peer reads where the partial
#                                       sum is structurally zero.
#
# ============================================================================
# Plain PyTorch reference
# ----------------------------------------------------------------------------
# The fully-vectorized reference below produces byte-identical outputs to
# the Triton pipeline. Useful for unit tests and as documentation of the
# semantics. Slow (full one-hot tensors of shape (W, T_local, K, W) — fine
# at the parity-test scale used in tests/, not for production hot path).
# ----------------------------------------------------------------------------
#
#   import torch
#   import torch.nn.functional as F
#
#   def compute_dispatch_metadata_ref(topk_idx_global, my_rank, E_local,
#                                     *, emit_dedup=True):
#       W, T_local, K = topk_idx_global.shape
#       TK_local = T_local * K
#       TK_global = W * TK_local
#       device = topk_idx_global.device
#       i32 = torch.int32
#
#       # ── Phase 1 ─ basic per-slot quantities ─────────────────────────
#       expert_global_flat = topk_idx_global.reshape(-1).to(i32)
#       dst_rank_flat = (expert_global_flat // E_local).to(i32)
#       local_expert_flat = expert_global_flat - dst_rank_flat * E_local
#
#       expert_local_padded = torch.where(
#           dst_rank_flat == my_rank,
#           local_expert_flat,
#           torch.full_like(local_expert_flat, E_local),
#       )
#
#       my_slice = slice(my_rank * TK_local, (my_rank + 1) * TK_local)
#       my_dst_rank = dst_rank_flat[my_slice].view(T_local, K)
#       my_expert_local = local_expert_flat[my_slice].view(T_local, K)
#
#       # peer_count_per_rank[r, p] = #slots from source r heading to p.
#       dst_2d = dst_rank_flat.view(W, TK_local)                       # (W, TK_local)
#       one_hot_2d = F.one_hot(dst_2d.long(), num_classes=W).to(i32)   # (W, TK_local, W)
#       peer_count_per_rank = one_hot_2d.sum(dim=1).to(i32)            # (W, W)
#
#       # ── Phase 2 / 3 ─ slot positions ────────────────────────────────
#       # slot_per_rank = cumcount within each (src_rank, dst_rank) bucket.
#       cumsum_2d = one_hot_2d.cumsum(dim=1)                            # (W, TK_local, W)
#       slot_per_rank_2d = (cumsum_2d * one_hot_2d).sum(dim=-1) - 1     # (W, TK_local)
#       slot_flat_per_rank = slot_per_rank_2d.flatten().to(i32)
#
#       # early_count[r, p] = exclusive cumsum across r of peer_count[:, p].
#       early_count = (peer_count_per_rank.cumsum(dim=0)
#                      - peer_count_per_rank).to(i32)
#       early_count_per_slot = early_count.gather(1, dst_2d.long())     # (W, TK_local)
#       slot_flat_global = (slot_per_rank_2d
#                            + early_count_per_slot).flatten().to(i32)
#
#       src_rank_flat = torch.arange(
#           W, device=device, dtype=i32
#       ).repeat_interleave(TK_local)
#       a2a_token_indices = (src_rank_flat * TK_local
#                             + slot_flat_per_rank).to(i32)
#
#       my_pos_on_peer = slot_flat_global[my_slice].view(T_local, K)
#       my_pos_per_rank = slot_flat_per_rank[my_slice].view(T_local, K)
#
#       out = {
#           "dst_rank_flat":       dst_rank_flat,
#           "slot_flat_per_rank":  slot_flat_per_rank,
#           "slot_flat_global":    slot_flat_global,
#           "my_dst_rank":         my_dst_rank,
#           "my_pos_on_peer":      my_pos_on_peer,
#           "my_pos_per_rank":     my_pos_per_rank,
#           "my_expert_local":     my_expert_local,
#           "peer_count_per_rank": peer_count_per_rank,
#           "expert_local_padded": expert_local_padded,
#           "a2a_token_indices":   a2a_token_indices,
#       }
#       if not emit_dedup:
#           return out
#
#       # ── Dedup ─ RANK_DEDUP dispatch metadata ─────────────────────────
#       dst_3d = dst_rank_flat.view(W, T_local, K)                      # (W, T, K)
#       oh_3d = F.one_hot(dst_3d.long(), num_classes=W).to(i32)          # (W, T, K, W)
#
#       # canonical[r, t, k] = (cumsum_k(oh)[r, t, k, dst[r,t,k]] == 1)
#       # — true iff this k is the lowest k routing (r, t) → dst[r,t,k].
#       cumsum_k = oh_3d.cumsum(dim=2)
#       cnt_at_dst = (cumsum_k * oh_3d).sum(dim=-1)                     # (W, T, K)
#       pair_present_mask = (cnt_at_dst == 1).to(torch.int8).reshape(-1)
#
#       # pair_count[r, q] = #tokens at r with ≥1 slot routed to q.
#       present_tq = (oh_3d.sum(dim=2) > 0).to(i32)                     # (W, T, W)
#       pair_count = present_tq.sum(dim=1).to(i32)                      # (W, W)
#       pair_offset = (pair_count.cumsum(dim=0) - pair_count).to(i32)
#
#       # rank_dedup_recv_pos[f] = pair_offset[r, q] + within_p_token_rank[r, t, q]
#       # where within_p_token_rank = exclusive cumsum across t of present_tq.
#       excl_t = (present_tq.cumsum(dim=1) - present_tq).to(i32)        # (W, T, W)
#       # Pick at q = dst[r, t, k] via one-hot (oh_3d).
#       within_per_tk = (oh_3d * excl_t.unsqueeze(2)).sum(dim=-1)       # (W, T, K)
#       pair_off_per_tk = (oh_3d
#                            * pair_offset.unsqueeze(1).unsqueeze(2)
#                          ).sum(dim=-1)                                  # (W, T, K)
#       rank_dedup_recv_pos = (pair_off_per_tk
#                              + within_per_tk).reshape(-1).to(i32)
#
#       # peer_present_mask[q, t] = 1 iff peer q has any expert for my (t).
#       my_dst_2d = dst_3d[my_rank]                                     # (T, K)
#       my_oh = F.one_hot(my_dst_2d.long(), num_classes=W).to(i32)      # (T, K, W)
#       peer_present_t = (my_oh.sum(dim=1) > 0).to(torch.int8)          # (T, W)
#       peer_present_mask = peer_present_t.t().contiguous()             # (W, T)
#
#       out["pair_count"]         = pair_count
#       out["pair_offset"]        = pair_offset
#       out["pair_present_mask"]  = pair_present_mask
#       out["rank_dedup_recv_pos"] = rank_dedup_recv_pos
#       out["peer_present_mask"]  = peer_present_mask
#       return out
#
# ============================================================================
# Triton implementation — three-phase scan, then optional dedup phases.
# ----------------------------------------------------------------------------
# Phases 1/2/3 are always run. Phases D1/D2/D3 + the peer_present_mask
# kernel run only when ``emit_dedup=True``. All kernels share `dst_rank_flat`
# via HBM (written in phase 1, read in phase 3 / D1 / D3 / mask).
#
#   Phase 1 (`_metadata_a2a_phase1_reduce_kernel`, grid (W, n_tiles)):
#     One program per (src_rank, BLOCK_TK-tile). Builds the (BLOCK_TK, W)
#     one-hot of `dst == peer`, reduces over BLOCK_TK to get per-(rank,
#     tile, peer) histogram tile_count. Emits dst_rank_flat, my_dst_rank,
#     my_expert_local, expert_local_padded inline.
#     within_tile_slot is intentionally NOT materialized — phase 3
#     recomputes it from dst_rank_flat in registers (saves an 8 MB HBM
#     round-trip at T=32k K=8 W=8).
#
#   Phase 2 (`_metadata_a2a_phase2_scan_kernel`, grid (W, W)):
#     One program per (src_rank, peer). 1-D cumsum of tile_count[r, :, p]
#     of length n_tiles (BLOCK_NTILES = next_pow2(n_tiles)). Emits
#     tile_prefix (exclusive scan) and peer_count_per_rank (sum over
#     n_tiles). Strided-W loads, L2-friendly for W ≤ 16.
#
#   Phase 3 (`_metadata_a2a_phase3_emit_kernel`, grid (W, n_tiles)):
#     Same grid as phase 1. Reloads dst tile, recomputes within_tile_slot
#     in registers via the same one-hot cumsum idiom. Loads tile_prefix
#     for this (src, tile) and computes early_count from peer_count_per_rank
#     in registers. Emits slot_per_rank, slot_global, a2a_token_indices.
#
#   Phase D1 (`_metadata_dedup_phase1_reduce_kernel`, grid (W, n_token_tiles)):
#     Per (src_rank, BLOCK_T-token-tile). Builds the (BLOCK_T, K_PAD, W)
#     one-hot, cumsum-along-K to find canonical slots, reduces over
#     BLOCK_T for the per-(p, n_token_tile, q) presence histogram.
#
#   Phase D2 (`_metadata_dedup_phase2_scan_kernel`, grid (W, W)):
#     Mirror of phase 2 on the token-tile axis. Emits tile_token_prefix
#     and pair_count.
#
#   Phase D3 (`_metadata_dedup_phase3_emit_kernel`, grid (W, n_token_tiles)):
#     Recomputes within-tile per-(t, q) prefix in registers. Loads the
#     full (W, W) pair_count, builds pair_offset (exclusive cumsum across
#     p) in registers; the (pid_tile == 0) program publishes pair_offset
#     to HBM. Emits rank_dedup_recv_pos.
#
#   peer_present_mask kernel (`_build_peer_present_mask_kernel`, grid
#   (cdiv(T_local, BLOCK_T_MASK),)):
#     Reads only my_rank's stripe of dst_rank_flat. Same one-hot trick
#     as D1 but per-token, no cumsum / cross-tile coordination.
# ============================================================================

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

    peer_axis = tl.arange(0, W)
    one_hot = (dst[:, None] == peer_axis[None, :]).to(tl.int32)
    one_hot = tl.where(valid[:, None], one_hot, 0)

    tile_count_p = tl.sum(one_hot, axis=0)

    local_expert = expert_global - dst * E_local
    padded = tl.where(dst == my_rank, local_expert, E_local)

    tl.store(out_dst_rank_flat_ptr + flat_offs, dst, mask=valid)
    tl.store(
        out_tile_count_ptr + pid_r * n_tiles * W + pid_tile * W + peer_axis,
        tile_count_p,
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
    """1-D cumsum over n_tiles for one (src_rank, peer) pair.

    Grid (W, W). Each program loads tile_count[pid_r, :, pid_p] of length
    n_tiles into registers (masked beyond n_tiles), runs tl.cumsum, and
    writes back tile_prefix[pid_r, :, pid_p] (exclusive) and a single
    scalar peer_count_per_rank[pid_r, pid_p].
    """
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
    TK_local: tl.constexpr,
    BLOCK_TK: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_tile = tl.program_id(1)

    rs = tl.arange(0, W)[:, None]
    ps = tl.arange(0, W)[None, :]
    pc = tl.load(peer_count_per_rank_ptr + rs * W + ps)
    excl_pc = tl.cumsum(pc, axis=0) - pc
    row_mask = tl.arange(0, W) == pid_r
    ec_row = tl.sum(tl.where(row_mask[:, None], excl_pc, 0), axis=0)

    tile_offs = pid_tile * BLOCK_TK + tl.arange(0, BLOCK_TK)
    valid = tile_offs < TK_local
    flat_offs = pid_r * TK_local + tile_offs

    dst = tl.load(dst_rank_flat_ptr + flat_offs, mask=valid, other=0)

    p_offs = tl.arange(0, W)
    one_hot = (dst[:, None] == p_offs[None, :]).to(tl.int32)
    one_hot = tl.where(valid[:, None], one_hot, 0)
    cumsum = tl.cumsum(one_hot, axis=0)
    within_pos = tl.sum(cumsum * one_hot, axis=1) - 1

    tp_row = tl.load(tile_prefix_ptr + pid_r * n_tiles * W + pid_tile * W + p_offs)

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
    BLOCK_T: tl.constexpr,
):
    """Canonical-slot mask + per-(token-tile, q) presence histogram.

    Invalid lanes get dst=W (sentinel outside [0, W)) so all one-hot
    entries naturally zero — no extra mask needed on the reductions.
    """
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

    q_axis = tl.arange(0, W)
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
    BLOCK_T: tl.constexpr,
):
    """Emit pair_offset (W, W) + per-slot rank_dedup_recv_pos (TK_global,)."""
    pid_r = tl.program_id(0)
    pid_tile = tl.program_id(1)

    rs = tl.arange(0, W)[:, None]
    qs = tl.arange(0, W)[None, :]
    pc = tl.load(pair_count_ptr + rs * W + qs)
    pair_off = tl.cumsum(pc, axis=0) - pc
    row_mask_r = tl.arange(0, W) == pid_r
    pair_off_row = tl.sum(tl.where(row_mask_r[:, None], pair_off, 0), axis=0)

    if pid_tile == 0:
        q_axis_init = tl.arange(0, W)
        tl.store(out_pair_offset_ptr + pid_r * W + q_axis_init, pair_off_row)

    t_offs_local = tl.arange(0, BLOCK_T)
    t_offs_global = pid_tile * BLOCK_T + t_offs_local
    valid_t = t_offs_global < T_local

    k_axis = tl.arange(0, K_PAD)
    valid_k = k_axis < K
    flat_slot_offs = (pid_r * T_local + t_offs_global)[:, None] * K + k_axis[None, :]
    valid_2d = valid_t[:, None] & valid_k[None, :]
    dst = tl.load(dst_rank_flat_ptr + flat_slot_offs, mask=valid_2d, other=W)

    q_axis = tl.arange(0, W)
    oh = (dst[:, :, None] == q_axis[None, None, :]).to(tl.int32)

    sum_oh = tl.sum(oh, axis=1)
    present_tq = (sum_oh > 0).to(tl.int32)
    incl_t = tl.cumsum(present_tq, axis=0)
    excl_t = incl_t - present_tq

    tile_pref = tl.load(tile_token_prefix_ptr + pid_r * n_token_tiles * W + pid_tile * W + q_axis)

    within_pq = excl_t + tile_pref[None, :]

    pair_off_per_tk = tl.sum(oh * pair_off_row[None, None, :], axis=2)
    within_per_tk = tl.sum(oh * within_pq[:, None, :], axis=2)
    dedup_pos = pair_off_per_tk + within_per_tk
    tl.store(out_rank_dedup_recv_pos_ptr + flat_slot_offs, dedup_pos, mask=valid_2d)


@triton.jit
def _build_peer_present_mask_kernel(
    dst_rank_flat_ptr,  # (TK_global,) int32
    out_mask_ptr,  # (W, T_local) int8
    T_local: tl.constexpr,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
    W: tl.constexpr,
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

    q_axis = tl.arange(0, W)
    oh = (dst[:, :, None] == q_axis[None, None, :]).to(tl.int32)
    present_tq = (tl.sum(oh, axis=1) > 0).to(tl.int8)

    store_offs = q_axis[None, :] * T_local + t_offs[:, None]
    store_mask = valid_t[:, None] & tl.full((1, W), True, tl.int1)
    tl.store(out_mask_ptr + store_offs, present_tq, mask=store_mask)


def compute_dispatch_metadata(
    topk_idx_global: torch.Tensor,
    my_rank: int,
    E_local: int,
    *,
    emit_dedup: bool = True,
):
    """Compute SonicMoE EP dispatch / combine metadata.

    See module docstring for input/output shapes and the plain-PyTorch
    reference. Triton-only, CUDA-graph safe (no host syncs).
    """
    W, T_local, K = topk_idx_global.shape
    TK_local = T_local * K
    TK_global = W * TK_local
    device = topk_idx_global.device

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
    # D1/D3 build a (BLOCK_T, K_PAD, W) one-hot int32 plus an equally-sized
    # cumsum in registers / SMEM. Cap one-hot at ~8K elements per CTA so we
    # stay under B200's per-CTA SMEM budget at large W·K.
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
        BLOCK_T=BLOCK_T,
    )

    peer_present_mask = torch.empty((W, T_local), dtype=torch.int8, device=device)
    MAX_OH_ELEMS = 8192
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
        my_rank=my_rank,
        BLOCK_T=BLOCK_T_MASK,
    )

    out["pair_count"] = pair_count
    out["pair_offset"] = pair_offset
    out["pair_present_mask"] = pair_present_mask
    out["rank_dedup_recv_pos"] = rank_dedup_recv_pos
    out["peer_present_mask"] = peer_present_mask
    return out
