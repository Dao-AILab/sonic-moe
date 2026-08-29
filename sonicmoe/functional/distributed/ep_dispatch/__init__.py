# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from __future__ import annotations

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from ..collectives import _CUDA_MAX_GRID_Y, _prune_block_d_vs_d, rendezvous

# torch dtype -> triton pointer element dtype, for int64->pointer runtime-peer-addressing
# (the hierarchical gather reads peer GIN windows by LSA base address; mirrors ep_combine's map).
_TL_PTR_DTYPE = {
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.int32: tl.int32,
    torch.int64: tl.int64,
}


_A2A_DISPATCH_CONFIGS = [
    triton.Config({"BLOCK_D": BLOCK_D}, num_warps=nw, num_stages=ns)
    for BLOCK_D in [128, 256, 512, 1024, 2048, 4096]
    for nw in [2, 4, 8]
    for ns in [3, 4]
    if BLOCK_D // nw >= 32  # min lanes per warp
    if not (BLOCK_D <= 256 and nw == 8)  # over-paralleled small tile
    if not (BLOCK_D >= 2048 and ns == 4)  # huge memory-bound tile already saturates at ns=3
]


# BLOCK_SLOT batches adjacent slots per program: rank-dedup has a high dead-program (early-return)
# rate, so batching amortizes launch cost; BLOCK_SLOT=1 stays in the sweep as the escape hatch.
_RANK_DEDUP_DISPATCH_CONFIGS = [
    triton.Config(
        {"BLOCK_D": BLOCK_D, "BLOCK_SLOT": BLOCK_SLOT},
        num_warps=nw,
        num_stages=ns,
    )
    for BLOCK_D in [128, 256, 512, 1024, 2048, 4096]
    for BLOCK_SLOT in [1, 2, 4]
    for nw in [2, 4, 8]
    for ns in [3, 4]
    if BLOCK_D // nw >= 32
    if not (BLOCK_D <= 256 and nw == 8)
    if not (BLOCK_D >= 2048 and ns == 4)
]


def _prune_block_d_keep_single_tile(configs, named_args, **kwargs):
    """Like `_prune_block_d_vs_d` but also keeps the single-tile BLOCK_D>d config: avoids re-issuing
    per-slot metadata n_dtiles× (+7% measured d=2304,K=8,W=8: 571 vs 534 GB/s; bit-exact, masked d-tail)."""
    d = kwargs.get("d", named_args.get("d"))
    if d is None:
        return list(configs)
    configs = list(configs)
    over = [c.kwargs["BLOCK_D"] for c in configs if c.kwargs["BLOCK_D"] > max(d, 1)]
    single_tile_bd = min(over) if over else None
    kept = []
    for cfg in configs:
        bd = cfg.kwargs["BLOCK_D"]
        if bd > max(d, 1) and bd != single_tile_bd:
            continue  # wastes lanes; keep only the single-d-tile option
        if triton.cdiv(d, bd) > _CUDA_MAX_GRID_Y:
            continue  # CUDA grid_y limit
        kept.append(cfg)
    if not kept:
        valid_for_d = [c for c in configs if c.kwargs["BLOCK_D"] <= max(d, 1)]
        kept = [max(valid_for_d or configs, key=lambda c: c.kwargs["BLOCK_D"])]
    return kept


# ============================================================================
# Fused A2A dispatch: caller-supplied recv_pos encodes the output layout (legacy per-rank-slot
# or expert-sorted); slots with dst != my_rank are no-ops, so downstream GEMM must tolerate garbage there.
# ============================================================================
@triton.autotune(
    configs=_A2A_DISPATCH_CONFIGS,
    key=["d", "world_size"],
    prune_configs_by={"early_config_prune": _prune_block_d_vs_d},
)
@triton.heuristics({"EVEN_D": lambda args: args["d"] % args["BLOCK_D"] == 0})
@triton.jit
def _a2a_dispatch_kernel(
    x_peer_tuple,  # tuple[(T_local, d) tensor, ...] for each peer
    dst_rank_flat_ptr,  # (TK_global,) int32
    recv_pos_ptr,  # (TK_global,) int32 — destination row in recv per global slot
    recv_ptr,  # flat (>= TK_global, d) output
    TK_local,  # = T_local * K
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    K: tl.constexpr,
    d: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    pid_orig = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1)

    # Peer-interleaved decomposition: adjacent programs target distinct peers.
    src_rank = (pid_orig % world_size).to(tl.int32)
    pid_tk = pid_orig // world_size  # ∈ [0, TK_local)

    # Metadata layout is unchanged (src_rank * TK_local + tk). We read it
    # at the strided offset corresponding to this program's (src_rank, tk).
    orig_idx = src_rank.to(tl.int64) * TK_local + pid_tk

    dst = tl.load(dst_rank_flat_ptr + orig_idx)
    if dst != my_rank:
        return  # invalid lane: no-op for this program

    # Generic recv layout: caller-supplied per-slot destination row.
    pos = tl.load(recv_pos_ptr + orig_idx).to(tl.int64)
    t_local = pid_tk // K

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    src_offs = t_local * d + offs_d
    dst_offs = pos * d + offs_d

    # Static dispatch over peers; only the matching i actually loads/stores.
    for i in tl.static_range(world_size):
        if src_rank == i:
            if EVEN_D:
                row = tl.load(x_peer_tuple[i] + src_offs)
                tl.store(recv_ptr + dst_offs, row)
            else:
                d_mask = offs_d < d
                row = tl.load(x_peer_tuple[i] + src_offs, mask=d_mask)
                tl.store(recv_ptr + dst_offs, row, mask=d_mask)


def a2a_dispatch_triton(
    x_symm: torch.Tensor,
    dst_rank_flat: torch.Tensor,
    recv_pos: torch.Tensor,
    recv: torch.Tensor,
    K: int,
    group,
    hdl=None,
    peer_bufs=None,
    my_rank=None,
):
    """A2A dispatch: NVLink-reads peer x_symm rows for slots routed to my_rank into
    recv[recv_pos[f]]. recv_pos layout is caller-defined (legacy per-rank-slot vs expert-sorted); untouched elsewhere."""
    T_local, d = x_symm.shape
    if peer_bufs is None:
        if hdl is None:
            hdl = rendezvous(x_symm, group)
        W = hdl.world_size
        my_rank = hdl.rank if my_rank is None else my_rank
        peer_bufs = tuple(hdl.get_buffer(r, (T_local, d), x_symm.dtype) for r in range(W))
    else:
        W = len(peer_bufs)
        if my_rank is None:
            my_rank = hdl.rank if hdl is not None else dist.get_rank(group)
    TK_local = T_local * K
    TK_global = W * TK_local

    recv_flat = recv.view(-1, d)

    grid = lambda META: (TK_global, triton.cdiv(d, META["BLOCK_D"]))
    _a2a_dispatch_kernel[grid](
        peer_bufs,
        dst_rank_flat,
        recv_pos,
        recv_flat,
        TK_local=TK_local,
        my_rank=my_rank,
        world_size=W,
        K=K,
        d=d,
    )
    return recv


# ============================================================================
# RANK_DEDUP dispatch: single-pass peer-pull of canonical slots only, packed by source
# rank; GEMM reads it directly via build_rank_dedup_a_idx's expert-row -> packed-row map.
# ============================================================================


@triton.autotune(
    configs=_RANK_DEDUP_DISPATCH_CONFIGS,
    key=["d", "world_size", "K"],
    prune_configs_by={"early_config_prune": _prune_block_d_keep_single_tile},
)
@triton.heuristics({"EVEN_D": lambda args: args["d"] % args["BLOCK_D"] == 0})
@triton.jit
def _rank_dedup_dispatch_kernel(
    x_peer_tuple,  # tuple[(T_local, d) tensor, ...] for each peer
    pair_present_mask_ptr,  # (TK_global,) int8
    dst_rank_flat_ptr,  # (TK_global,) int32
    rank_dedup_recv_pos_ptr,  # (TK_global,) int32
    recv_packed_ptr,  # flat (>= MAX_PAIR_COUNT, d)
    TK_local,
    TK_global,  # tail bounds check for the BLOCK_SLOT walk
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    K: tl.constexpr,
    d: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_SLOT: tl.constexpr,
    EVEN_D: tl.constexpr,
    node_size: tl.constexpr = 0,
):
    pid_block = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1)

    base_pid_orig = pid_block * BLOCK_SLOT

    # node_size>0: HIERARCHICAL same-node gate (only pull local-node sources); cross-node rows
    # come via the GIN put+expand instead. node_size==0 disables the gate (byte-identical to flat).
    for j in tl.static_range(BLOCK_SLOT):
        pid_orig = base_pid_orig + j
        # Tail guard: TK_global may not divide BLOCK_SLOT evenly.
        if pid_orig < TK_global:
            src_rank = (pid_orig % world_size).to(tl.int32)
            pid_tk = pid_orig // world_size
            orig_idx = src_rank.to(tl.int64) * TK_local + pid_tk

            dst = tl.load(dst_rank_flat_ptr + orig_idx)
            is_canonical = tl.load(pair_present_mask_ptr + orig_idx)
            # Combined predicate: both loads issue unconditionally (int8 dominated by int32 cost),
            # folding the old two-stage early-return into one warp-uniform branch — avoids a divergence point.
            if node_size == 0:
                same_node = True
            else:
                same_node = (src_rank // node_size) == (my_rank // node_size)

            if (dst == my_rank) & (is_canonical != 0) & same_node:
                pos = tl.load(rank_dedup_recv_pos_ptr + orig_idx).to(tl.int64)
                t_local = pid_tk // K

                offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
                src_offs = t_local * d + offs_d
                dst_offs = pos * d + offs_d

                for i in tl.static_range(world_size):
                    if src_rank == i:
                        if EVEN_D:
                            row = tl.load(x_peer_tuple[i] + src_offs)
                            tl.store(recv_packed_ptr + dst_offs, row)
                        else:
                            d_mask = offs_d < d
                            row = tl.load(x_peer_tuple[i] + src_offs, mask=d_mask)
                            tl.store(recv_packed_ptr + dst_offs, row, mask=d_mask)


@triton.jit
def _build_rank_dedup_a_idx_kernel(
    dst_rank_flat_ptr,  # (TK_global,) int32
    s_reverse_local_ptr,  # (TK_global,) int32 — slot → expert-grouped row (permutation on my-rank slots)
    rank_dedup_recv_pos_ptr,  # (TK_global,) int32 — slot → packed row at my_rank
    a_idx_ptr,  # (MAX_ROWS_PER_RANK_STATIC,) int32 — output
    TK_global,
    my_rank: tl.constexpr,
    BLOCK_SLOT: tl.constexpr,
):
    """Scatters rank_dedup_recv_pos[f] -> a_idx[s_reverse_local[f]] for slots routed to my_rank.
    PRECONDITION (unenforced): s_reverse_local/rank_dedup_recv_pos must stay pair-consistent (guaranteed by general_routing_router_metadata_triton) or this races and silently corrupts a_idx."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SLOT + tl.arange(0, BLOCK_SLOT)
    valid = offs < TK_global

    dst = tl.load(dst_rank_flat_ptr + offs, mask=valid, other=-1)
    is_mine = (dst == my_rank) & valid

    e = tl.load(s_reverse_local_ptr + offs, mask=is_mine, other=0).to(tl.int64)
    p = tl.load(rank_dedup_recv_pos_ptr + offs, mask=is_mine, other=0)
    tl.store(a_idx_ptr + e, p, mask=is_mine)


def rank_dedup_dispatch_triton(
    x_symm: torch.Tensor,
    dst_rank_flat: torch.Tensor,
    pair_present_mask: torch.Tensor,
    rank_dedup_recv_pos: torch.Tensor,
    recv_packed: torch.Tensor,
    K: int,
    group,
    hdl=None,
    peer_bufs=None,
    my_rank=None,
    node_size: int = 0,
):
    """RANK_DEDUP dispatch: one NVLink read per (src,t,my_rank) triple with >=1 routed slot, packed
    by source rank. Caller must barrier after writing x_symm before calling; no barrier needed after (local-only read)."""
    T_local, d = x_symm.shape
    if peer_bufs is None:
        if hdl is None:
            hdl = rendezvous(x_symm, group)
        W = hdl.world_size
        my_rank = hdl.rank if my_rank is None else my_rank
        peer_bufs = tuple(hdl.get_buffer(r, (T_local, d), x_symm.dtype) for r in range(W))
    else:
        W = len(peer_bufs)
        if my_rank is None:
            my_rank = hdl.rank if hdl is not None else dist.get_rank(group)
    TK_local = T_local * K
    TK_global = W * TK_local

    recv_packed_flat = recv_packed.view(-1, d)
    grid = lambda META: (
        triton.cdiv(TK_global, META["BLOCK_SLOT"]),
        triton.cdiv(d, META["BLOCK_D"]),
    )
    _rank_dedup_dispatch_kernel[grid](
        peer_bufs,
        pair_present_mask,
        dst_rank_flat,
        rank_dedup_recv_pos,
        recv_packed_flat,
        TK_local=TK_local,
        TK_global=TK_global,
        my_rank=my_rank,
        world_size=W,
        K=K,
        d=d,
        node_size=node_size,
    )
    return recv_packed


# ── Hierarchical inter-node dispatch: COALESCED-put staging gather ───────────────
# Replaces per-token GIN puts (~512 serial ~d*2B RDMA @ ~1 GB/s) with ONE coalesced put per remote node:
# scatter into a compact per-node staging buffer first (disjoint blocks — dst_slot is per-receiver, so a single buffer would alias across nodes).
@triton.autotune(
    configs=_A2A_DISPATCH_CONFIGS,
    key=["d"],
    prune_configs_by={"early_config_prune": _prune_block_d_vs_d},
)
@triton.heuristics({"EVEN_D": lambda args: args["d"] % args["BLOCK_D"] == 0})
@triton.jit
def _hier_stage_coalesced_kernel(
    x_ptr,             # (T_local, d) bf16 — my token rows (the x_gin window)
    node_present_ptr,  # (TK_global,) int8 — 1 on the node-canonical slot per (token, dst node)
    dst_slot_ptr,      # (TK_global,) int32 — receiver dst_node_buffer row (>=0 on remote slots)
    dst_node_ptr,      # (TK_global,) int32 — dst node index per slot
    stripe_base_row_ptr,  # (num_nodes,) int32 — stripe_base[my_rank, :]
    staging_ptr,       # (num_nodes*T_local, d) bf16 — COMPACT PER-NODE staging
    my_base,           # rank * TK_local (start of my slots in the global arrays)
    T_local,
    K: tl.constexpr,
    d: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    pid_slot = tl.program_id(0)  # 0..TK_local (my slot index)
    pid_d = tl.program_id(1)
    orig = my_base + pid_slot
    present = tl.load(node_present_ptr + orig)
    if present != 0:  # node-canonical remote slot: one write per (token, dst node)
        ds = tl.load(dst_slot_ptr + orig)
        n = tl.load(dst_node_ptr + orig)
        sb = tl.load(stripe_base_row_ptr + n)  # stripe_base[r, n]: my stripe base IN RECEIVER n
        # Compact per-node row: disjoint blocks avoid dst_slot aliasing across nodes (see banner above).
        staging_row = (n * T_local + (ds - sb)).to(tl.int64)
        t = (pid_slot // K).to(tl.int64)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        if EVEN_D:
            row = tl.load(x_ptr + t * d + offs_d)
            tl.store(staging_ptr + staging_row * d + offs_d, row)
        else:
            d_mask = offs_d < d
            row = tl.load(x_ptr + t * d + offs_d, mask=d_mask)
            tl.store(staging_ptr + staging_row * d + offs_d, row, mask=d_mask)


def hier_stage_coalesced_triton(x_gin, node_present_mask, dst_slot, dst_node_flat, stripe_base_row,
                                staging, *, rank, T_local, K, d):
    """Scatters x_gin into compact per-node staging (row = dst_node*T_local + within-node offset) for
    ONE coalesced put per remote node; disjoint per-node blocks avoid dst_slot aliasing across nodes."""
    TK_local = T_local * K
    my_base = rank * TK_local
    x_flat = x_gin.view(-1, d)
    staging_flat = staging.view(-1, d)
    grid = lambda META: (TK_local, triton.cdiv(d, META["BLOCK_D"]))
    _hier_stage_coalesced_kernel[grid](
        x_flat, node_present_mask, dst_slot, dst_node_flat, stripe_base_row, staging_flat,
        my_base, T_local, K=K, d=d)
    return staging


# ── Hierarchical inter-node dispatch: remote NVLink expand ───────────────────────
# Remote NVLink expand: after GIN lands rows in the receiver's dst_node_buffer, each destination GPU
# pulls its rows into the same recv_packed layout the same-node pull writes (disjoint by construction).
@triton.autotune(
    configs=_RANK_DEDUP_DISPATCH_CONFIGS,
    key=["d", "world_size", "K"],
    prune_configs_by={"early_config_prune": _prune_block_d_keep_single_tile},
)
@triton.heuristics({"EVEN_D": lambda args: args["d"] % args["BLOCK_D"] == 0})
@triton.jit
def _expand_dispatch_kernel(
    dst_node_bufs,  # tuple[(dst_node_buf_rows, d) tensor, ...] — peer dst_node_buffers (one per same-node GPU)
    pair_present_mask_ptr,  # (TK_global,) int8 — rank-level canonical
    is_local_slot_ptr,  # (TK_global,) int8 — 1 iff src node == dst rank's node
    dst_rank_flat_ptr,  # (TK_global,) int32
    dst_slot_ptr,  # (TK_global,) int32 — row in the receiving GPU's dst_node_buffer
    rank_dedup_recv_pos_ptr,  # (TK_global,) int32 — final packed recv row
    recv_packed_ptr,  # flat (>= MAX_PAIR_COUNT, d)
    TK_local,
    TK_global,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    node_size: tl.constexpr,
    K: tl.constexpr,
    d: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_SLOT: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    pid_block = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1)
    base_pid_orig = pid_block * BLOCK_SLOT
    my_node = my_rank // node_size  # constexpr

    for j in tl.static_range(BLOCK_SLOT):
        pid_orig = base_pid_orig + j
        if pid_orig < TK_global:
            src_rank = (pid_orig % world_size).to(tl.int32)
            pid_tk = pid_orig // world_size
            orig_idx = src_rank.to(tl.int64) * TK_local + pid_tk

            dst = tl.load(dst_rank_flat_ptr + orig_idx)
            is_canonical = tl.load(pair_present_mask_ptr + orig_idx)
            is_local = tl.load(is_local_slot_ptr + orig_idx)
            # Fires on REMOTE rank-canonical slots routed to me. A token w/ 2 experts on the same remote
            # node's different ranks fires twice here (once per dst rank), both reading the SAME landed row.
            if (dst == my_rank) & (is_canonical != 0) & (is_local == 0):
                # The row landed on the GPU in MY node sharing the source's local index.
                recv_gpu = my_node * node_size + (src_rank % node_size)
                ds = tl.load(dst_slot_ptr + orig_idx).to(tl.int64)
                pos = tl.load(rank_dedup_recv_pos_ptr + orig_idx).to(tl.int64)

                offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
                src_offs = ds * d + offs_d
                dst_offs = pos * d + offs_d

                for i in tl.static_range(world_size):
                    if recv_gpu == i:
                        if EVEN_D:
                            row = tl.load(dst_node_bufs[i] + src_offs)
                            tl.store(recv_packed_ptr + dst_offs, row)
                        else:
                            d_mask = offs_d < d
                            row = tl.load(dst_node_bufs[i] + src_offs, mask=d_mask)
                            tl.store(recv_packed_ptr + dst_offs, row, mask=d_mask)


# ── Hierarchical dispatch (GIN-native): unified runtime-peer-addressed gather ──────────────
# Unified runtime-peer-addressed gather (GIN-native): one kernel fills recv_packed from BOTH same-node
# (NVLink, via x_gin) and remote (GIN-landed dst_node_buffer) rows, addressed via LSA base ptrs — no torch-symm-mem copies.
@triton.autotune(
    configs=_RANK_DEDUP_DISPATCH_CONFIGS,
    key=["d", "world_size", "K"],
    prune_configs_by={"early_config_prune": _prune_block_d_keep_single_tile},
)
@triton.heuristics({"EVEN_D": lambda args: args["d"] % args["BLOCK_D"] == 0})
@triton.jit
def _hier_gather_rt_kernel(
    x_lsa_base_ptr,  # int64[node_size]: same-node peers' x_gin window base addrs (LSA peer ptrs)
    dst_node_buf_lsa_base_ptr,  # int64[node_size]: same-node peers' dst_node_buffer base addrs
    pair_present_mask_ptr,  # (TK_global,) int8 — rank-level canonical
    is_local_slot_ptr,  # (TK_global,) int8 — 1 iff src node == my node
    dst_rank_flat_ptr,  # (TK_global,) int32
    dst_slot_ptr,  # (TK_global,) int32 — row in the rail's dst_node_buffer (remote slots)
    rank_dedup_recv_pos_ptr,  # (TK_global,) int32 — final packed recv row
    recv_packed_ptr,  # flat (>= MAX_PAIR_COUNT, d), local HBM
    TK_local,
    TK_global,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    node_size: tl.constexpr,
    K: tl.constexpr,
    d: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_SLOT: tl.constexpr,
    EVEN_D: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid_block = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1)
    base_pid_orig = pid_block * BLOCK_SLOT
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = offs_d < d

    for j in tl.static_range(BLOCK_SLOT):
        pid_orig = base_pid_orig + j
        if pid_orig < TK_global:
            src_rank = (pid_orig % world_size).to(tl.int32)
            pid_tk = pid_orig // world_size
            orig_idx = src_rank.to(tl.int64) * TK_local + pid_tk

            dst = tl.load(dst_rank_flat_ptr + orig_idx)
            is_canonical = tl.load(pair_present_mask_ptr + orig_idx)
            if (dst == my_rank) & (is_canonical != 0):
                is_local = tl.load(is_local_slot_ptr + orig_idx)
                pos = tl.load(rank_dedup_recv_pos_ptr + orig_idx).to(tl.int64)
                peer_lsa = src_rank % node_size  # the same-node peer (source local index)
                # same-node: read source's x_gin[t_local]; remote: read rail's dst_node_buffer[dst_slot]
                if is_local != 0:
                    base = tl.load(x_lsa_base_ptr + peer_lsa)
                    row_idx = (pid_tk // K).to(tl.int64)
                else:
                    base = tl.load(dst_node_buf_lsa_base_ptr + peer_lsa)
                    row_idx = tl.load(dst_slot_ptr + orig_idx).to(tl.int64)

                src_ptr = base.to(tl.pointer_type(DTYPE))
                src_offs = row_idx * d + offs_d
                dst_offs = pos * d + offs_d
                if EVEN_D:
                    tl.store(recv_packed_ptr + dst_offs, tl.load(src_ptr + src_offs))
                else:
                    tl.store(recv_packed_ptr + dst_offs,
                             tl.load(src_ptr + src_offs, mask=d_mask), mask=d_mask)


def hier_gather_rt_triton(
    x_lsa_base: torch.Tensor,  # int64[node_size]
    dst_node_buf_lsa_base: torch.Tensor,  # int64[node_size]
    pair_present_mask: torch.Tensor,
    is_local_slot: torch.Tensor,
    dst_rank_flat: torch.Tensor,
    dst_slot: torch.Tensor,
    rank_dedup_recv_pos: torch.Tensor,
    recv_packed: torch.Tensor,
    K: int,
    my_rank: int,
    world_size: int,
    node_size: int,
):
    """Hierarchical gather: fills recv_packed from same-node x_gin + remote dst_node_buffers via LSA
    base addrs. DTYPE CONTRACT: recv_packed.dtype must equal the peer windows' dtype or reads silently return garbage."""
    d = recv_packed.view(-1, recv_packed.shape[-1]).shape[-1]
    TK_local = dst_rank_flat.numel() // world_size
    TK_global = dst_rank_flat.numel()
    recv_packed_flat = recv_packed.view(-1, d)
    dtype = _TL_PTR_DTYPE[recv_packed.dtype]
    grid = lambda META: (
        triton.cdiv(TK_global, META["BLOCK_SLOT"]),
        triton.cdiv(d, META["BLOCK_D"]),
    )
    _hier_gather_rt_kernel[grid](
        x_lsa_base, dst_node_buf_lsa_base, pair_present_mask, is_local_slot, dst_rank_flat,
        dst_slot, rank_dedup_recv_pos, recv_packed_flat,
        TK_local=TK_local, TK_global=TK_global, my_rank=my_rank, world_size=world_size,
        node_size=node_size, K=K, d=d, DTYPE=dtype)
    return recv_packed


def expand_dispatch_triton(
    dst_node_bufs,  # tuple/list of W peer dst_node_buffers, each (dst_node_buf_rows, d)
    pair_present_mask: torch.Tensor,
    is_local_slot: torch.Tensor,
    dst_rank_flat: torch.Tensor,
    dst_slot: torch.Tensor,
    rank_dedup_recv_pos: torch.Tensor,
    recv_packed: torch.Tensor,
    K: int,
    my_rank: int,
    node_size: int,
):
    """Remote NVLink expand (pull) half of hierarchical dispatch: reads landed rows from the receiving
    GPU's dst_node_buffer into recv_packed. Pairs with rank_dedup_dispatch_triton(node_size=...) (same-node pull); disjoint writers."""
    W = len(dst_node_bufs)
    d = recv_packed.view(-1, recv_packed.shape[-1]).shape[-1]
    TK_local = (dst_rank_flat.numel() // W)
    TK_global = dst_rank_flat.numel()
    recv_packed_flat = recv_packed.view(-1, d)
    dst_node_bufs_tuple = tuple(b.view(-1, d) for b in dst_node_bufs)
    grid = lambda META: (
        triton.cdiv(TK_global, META["BLOCK_SLOT"]),
        triton.cdiv(d, META["BLOCK_D"]),
    )
    _expand_dispatch_kernel[grid](
        dst_node_bufs_tuple,
        pair_present_mask,
        is_local_slot,
        dst_rank_flat,
        dst_slot,
        rank_dedup_recv_pos,
        recv_packed_flat,
        TK_local=TK_local,
        TK_global=TK_global,
        my_rank=my_rank,
        world_size=W,
        node_size=node_size,
        K=K,
        d=d,
    )
    return recv_packed


def build_rank_dedup_a_idx(
    dst_rank_flat: torch.Tensor,
    s_reverse_local: torch.Tensor,
    rank_dedup_recv_pos: torch.Tensor,
    my_rank: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Builds the dedup up-proj A_idx in-place: out[s_reverse_local[f]] = rank_dedup_recv_pos[f] for every
    f routed to my_rank. Tail beyond #routed rows is left uninitialized (matches x_gather_idx's AG convention)."""
    TK_global = dst_rank_flat.shape[0]
    BLOCK_SLOT = 1024
    grid = (triton.cdiv(TK_global, BLOCK_SLOT),)
    _build_rank_dedup_a_idx_kernel[grid](
        dst_rank_flat,
        s_reverse_local,
        rank_dedup_recv_pos,
        out,
        TK_global=TK_global,
        my_rank=my_rank,
        BLOCK_SLOT=BLOCK_SLOT,
    )
    return out
