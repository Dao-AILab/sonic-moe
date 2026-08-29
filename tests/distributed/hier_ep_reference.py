# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
"""Pure-functional CPU oracle (no Triton/GPU/torch.distributed) for hierarchical inter-node EP
dispatch, derived only from ``topk_idx_global`` — ground truth the real kernels are validated against."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


# ── closed-form rank/node geometry ────────────────────────────────────────────
def node_of(r: int, node_size: int) -> int:
    return r // node_size


def local_of(r: int, node_size: int) -> int:
    return r % node_size


def rank_of(n: int, m: int, node_size: int) -> int:
    return n * node_size + m


def recv_gpu(src: int, dst_node: int, node_size: int) -> int:
    """The GPU that RECEIVES source ``src``'s put to ``dst_node``: same local index as ``src``."""
    return dst_node * node_size + (src % node_size)


@dataclass
class HierDispatchRef:
    """Per-slot tensors flat over ``f = r*TK_local + t*K + k``, matching ``metadata.dst_rank_flat``;
    sentinel ``-1`` marks not-applicable (e.g. local/invalid ``dst_slot``)."""

    W: int
    num_nodes: int
    node_size: int
    T_local: int
    K: int
    E_local: int

    # per-slot (TK_global,) int64 / int8
    dst_rank_flat: torch.Tensor          # global destination rank (expert owner) of each routed slot
    dst_node_flat: torch.Tensor          # dst_rank // node_size = node_idx (the put's target node)
    pair_present_mask: torch.Tensor      # 1 at lowest-k canonical slot per (r,t,dst_rank)
    rank_dedup_recv_pos: torch.Tensor    # final packed recv row at the dst rank (existing semantics)
    is_local_slot: torch.Tensor          # 1 iff src node == dst rank's node (NVLink, no GIN)
    node_present_mask: torch.Tensor      # 1 at lowest-k canonical slot per (r,t,REMOTE dst_node)
    dst_slot: torch.Tensor               # row in the receiving GPU's dst_node_buffer, -1 on local/invalid

    # rank/node aggregates
    pair_count: torch.Tensor             # (W, W) int64 — #tokens at src r with >=1 slot to dst q
    pair_offset: torch.Tensor            # (W, W) int64 — exclusive cumsum over src of pair_count[:,q]
    node_token_count: torch.Tensor       # (W, num_nodes) int64 — #tokens at src r with >=1 slot to node n
    dst_recv_count: torch.Tensor         # (W,) int64 — inbound puts per receiving GPU (signal-wait count)
    recv_rows_per_rank: torch.Tensor     # (W,) int64 — total recv_packed rows at each dst rank
    DST_NODE_BUF_ROWS: int                    # static per-GPU dst_node_buffer row capacity = T_local*(num_nodes-1)


def compute_hier_dispatch_reference(
    topk_idx_global: torch.Tensor,  # (W, T_local, K) int — global expert id in [0, W*E_local)
    num_nodes: int,
    node_size: int,
    E_local: int,
) -> HierDispatchRef:
    """Derive every hierarchical-dispatch index from the global routing decision (CPU)."""
    assert topk_idx_global.dim() == 3, "topk_idx_global must be (W, T_local, K)"
    W, T_local, K = (int(s) for s in topk_idx_global.shape)
    assert W == num_nodes * node_size, f"W={W} != num_nodes*node_size={num_nodes*node_size}"
    assert W % node_size == 0

    idx = topk_idx_global.to(torch.int64).cpu()
    TK_local = T_local * K
    TK_global = W * TK_local

    dst_rank = idx // E_local                       # (W, T_local, K) global dst rank
    assert int(dst_rank.max()) < W and int(dst_rank.min()) >= 0, "dst rank out of range"
    dst_node = dst_rank // node_size                  # (W, T_local, K) dst node
    src_node = (torch.arange(W) // node_size)[:, None, None].expand(W, T_local, K)

    # ── rank-level dedup (replicates metadata.py semantics exactly) ───────────
    # routed[r,t,q] = (∃k: dst_rank[r,t,k] == q)
    q_axis = torch.arange(W)
    onehot_rank = (dst_rank[..., None] == q_axis).to(torch.int64)        # (W,T,K,W)
    routed = (onehot_rank.sum(dim=2) > 0).to(torch.int64)               # (W,T,W)
    pair_count = routed.sum(dim=1)                                      # (W,W)
    # pair_offset[r,q] = exclusive cumsum over SOURCE rank of pair_count[:,q]
    pair_offset = torch.cumsum(pair_count, dim=0) - pair_count          # (W,W)
    # within[r,t,q] = # tokens t'<t on rank r that also route to q (exclusive cumsum over t)
    within_rank = torch.cumsum(routed, dim=1) - routed                  # (W,T,W)
    # canonical (lowest-k) slot per (r,t,q): cumsum over k of one-hot == 1
    cum_k = torch.cumsum(onehot_rank, dim=2)                            # (W,T,K,W)
    cnt_at_dst = (cum_k * onehot_rank).sum(dim=3)                       # (W,T,K)
    pair_present = (cnt_at_dst == 1).to(torch.int8)                    # (W,T,K)

    # per-slot rank_dedup_recv_pos = pair_offset[r,dst] + within[r,t,dst]
    # (gather along the q axis at q = dst_rank[r,t,k])
    pair_off_slot = torch.gather(
        pair_offset[:, None, :].expand(W, T_local, W), 2, dst_rank).contiguous()      # (W,T,K)
    within_slot = torch.gather(within_rank, 2, dst_rank).contiguous()                 # (W,T,K)
    rank_dedup_recv_pos = pair_off_slot + within_slot                                 # (W,T,K)

    is_local_slot = (src_node == dst_node).to(torch.int8)                             # (W,T,K)

    # ── node-level dedup (the inter-node GIN dimension) ───────────────────────
    n_axis = torch.arange(num_nodes)
    onehot_node = (dst_node[..., None] == n_axis).to(torch.int64)       # (W,T,K,num_nodes)
    routed_node = (onehot_node.sum(dim=2) > 0).to(torch.int64)         # (W,T,num_nodes)
    remote_node_mask = (n_axis[None, None, :] != src_node[:, :, :1]).to(torch.int64)  # (W,T,num_nodes)
    routed_node_remote = routed_node * remote_node_mask               # (W,T,num_nodes): remote only
    node_token_count = routed_node_remote.sum(dim=1)                  # (W, num_nodes): per-(src rank, node)

    # node-canonical: lowest k per (r,t,dst_node) AND the dst_node is remote
    cum_k_node = torch.cumsum(onehot_node, dim=2)                      # (W,T,K,num_nodes)
    cnt_at_node = (cum_k_node * onehot_node).sum(dim=3)               # (W,T,K)
    is_remote_slot = (1 - is_local_slot).to(torch.int64)             # (W,T,K)
    node_present = ((cnt_at_node == 1) & (is_remote_slot == 1)).to(torch.int8)        # (W,T,K)

    # dst_node_buffer is source-node-major: stripe_base(R,s) = cumsum of node_token_count over
    # contributing source nodes s != n; dst_slot = stripe_base + within-stripe token offset.
    DST_NODE_BUF_ROWS = T_local * max(num_nodes - 1, 0)
    dst_recv_count = torch.zeros(W, dtype=torch.int64)
    # stripe_base[r, n] = base offset of source rank r's stripe in recv_gpu(r,n)'s dst_node_buffer
    stripe_base = torch.zeros(W, num_nodes, dtype=torch.int64)
    for m in range(node_size):
        for n in range(num_nodes):
            R = rank_of(n, m, node_size)
            base = 0
            for s in range(num_nodes):
                if s == n:
                    continue

                r = rank_of(s, m, node_size)
                stripe_base[r, n] = base
                base += int(node_token_count[r, n])

            dst_recv_count[R] = base  # total puts landing on receiving GPU R

    # within-stripe position of token t among source rank r's tokens going to node n (exclusive)
    within_node_excl = torch.cumsum(routed_node_remote, dim=1) - routed_node_remote   # (W,T,num_nodes)

    # ── assemble per-slot dst_slot (only on remote slots) ─
    dst_slot = torch.full((W, T_local, K), -1, dtype=torch.int64)
    # dst_slot = stripe_base[r, dst_node] + within_node_excl[r, t, dst_node]
    sb_slot = torch.gather(stripe_base[:, None, :].expand(W, T_local, num_nodes), 2, dst_node)
    wre_slot = torch.gather(within_node_excl, 2, dst_node)
    dst_slot_all = sb_slot + wre_slot
    # dst_slot is set on EVERY remote slot, not just the node-canonical one: all rank-canonical
    # slots of a (src,token)->dst_node share ONE landed row and must resolve to the same value.
    sel = is_remote_slot.to(torch.bool)
    dst_slot[sel] = dst_slot_all[sel]

    recv_rows_per_rank = pair_count.sum(dim=0)                         # (W,) total rows per dst rank

    flat = lambda x: x.reshape(TK_global).contiguous()
    return HierDispatchRef(
        W=W, num_nodes=num_nodes, node_size=node_size, T_local=T_local, K=K, E_local=E_local,
        dst_rank_flat=flat(dst_rank), dst_node_flat=flat(dst_node),
        pair_present_mask=flat(pair_present), rank_dedup_recv_pos=flat(rank_dedup_recv_pos),
        is_local_slot=flat(is_local_slot), node_present_mask=flat(node_present),
        dst_slot=flat(dst_slot),
        pair_count=pair_count, pair_offset=pair_offset, node_token_count=node_token_count,
        dst_recv_count=dst_recv_count, recv_rows_per_rank=recv_rows_per_rank,
        DST_NODE_BUF_ROWS=DST_NODE_BUF_ROWS,
    )


@dataclass
class HierCombineRef:
    """Hierarchical-COMBINE metadata: a per-node REDUCTION (gateway NVLink-reduces node-N peers, then
    GIN-puts one stripe to the origin), the reverse-mirror of dispatch — never an RDMA float atomic."""

    W: int
    num_nodes: int
    node_size: int
    T_local: int
    K: int
    E_local: int

    peer_present_mask: torch.Tensor      # (W_origin, W_peer, T_local) int8: peer q has an expert for R's token t
    contrib_node_mask: torch.Tensor      # (W_origin, num_nodes) int8: any q in node N is present for origin R (remote only)
    expected_count_combine: torch.Tensor # (W,) int64: # remote nodes that GIN-put a partial back to R (signal-wait count)
    COMBINE_RECV_ROWS: int               # static per-GPU combine_recv_buf row capacity = T_local*(num_nodes-1)


def compute_hier_combine_reference(
    topk_idx_global: torch.Tensor,  # (W, T_local, K) int — same routing the dispatch ref consumes
    num_nodes: int,
    node_size: int,
    E_local: int,
) -> HierCombineRef:
    """Derive every hierarchical-COMBINE index from the global routing decision (CPU)."""
    assert topk_idx_global.dim() == 3, "topk_idx_global must be (W, T_local, K)"
    W, T_local, K = (int(s) for s in topk_idx_global.shape)
    assert W == num_nodes * node_size, f"W={W} != num_nodes*node_size={num_nodes*node_size}"

    idx = topk_idx_global.to(torch.int64).cpu()
    dst_rank = idx // E_local                                            # (W,T,K)
    dst_node = dst_rank // node_size                                     # (W,T,K)

    # peer_present[R, q, t] = ∃k: dst_rank[R,t,k] == q  (== the dispatch ref's `routed`, transposed to (R,q,t))
    q_axis = torch.arange(W)
    onehot_rank = (dst_rank[..., None] == q_axis).to(torch.int64)        # (W,T,K,W)
    routed = (onehot_rank.sum(dim=2) > 0)                                # (W,T,W) bool: (R,t,q)
    peer_present_mask = routed.permute(0, 2, 1).contiguous().to(torch.int8)  # (W,W,T): (R,q,t)

    # contrib_node_mask[R, N] = ∃ q∈N, t: present, for REMOTE N only
    n_axis = torch.arange(num_nodes)
    onehot_node = (dst_node[..., None] == n_axis).to(torch.int64)        # (W,T,K,num_nodes)
    routed_node = (onehot_node.sum(dim=2) > 0)                           # (W,T,num_nodes): (R,t,N)
    src_node = (torch.arange(W) // node_size)                            # (W,)
    remote_node = (n_axis[None, :] != src_node[:, None])                 # (W,num_nodes): N != node_of(R)
    contrib_node = (routed_node.any(dim=1) & remote_node)                # (W,num_nodes) bool
    contrib_node_mask = contrib_node.to(torch.int8)

    expected_count_combine = contrib_node.sum(dim=1).to(torch.int64)     # (W,) inbound combine puts per R

    return HierCombineRef(
        W=W, num_nodes=num_nodes, node_size=node_size, T_local=T_local, K=K, E_local=E_local,
        peer_present_mask=peer_present_mask, contrib_node_mask=contrib_node_mask,
        expected_count_combine=expected_count_combine, COMBINE_RECV_ROWS=T_local * max(num_nodes - 1, 0),
    )
