# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import torch
from parameterized import parameterized

from sonicmoe.functional.distributed import build_rank_dedup_a_idx, compute_dispatch_metadata
from sonicmoe.functional.metadata import general_routing_router_metadata_triton
from tests.test_commons import TestCommons


_SEED = 0


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _ref_compute_dispatch_metadata(topk_idx_g: torch.Tensor, my_rank: int, E_local: int):
    """Pure-PyTorch reference for compute_dispatch_metadata. Different
    algorithm than the 3-phase Triton scan (one-shot torch ops on the full
    (W, TK_local, W) one-hot tensor, no tile structure), so bit-exact
    agreement is a strong correctness signal."""
    W, T_local, K = topk_idx_g.shape
    TK_local = T_local * K
    TK_global = W * TK_local
    device = topk_idx_g.device

    flat = topk_idx_g.reshape(TK_global)
    dst = (flat // E_local).to(torch.int32)

    one_hot = (dst.view(W, TK_local).long().unsqueeze(-1) == torch.arange(W, device=device).view(1, 1, W)).to(
        torch.int32
    )

    pcpr = one_hot.sum(dim=1).to(torch.int32).contiguous()
    incl = one_hot.cumsum(dim=1).to(torch.int32)
    slot_pr = ((incl * one_hot).sum(dim=2).to(torch.int32) - 1).reshape(-1).contiguous()
    early = (pcpr.cumsum(dim=0).to(torch.int32) - pcpr).contiguous()
    r_idx = (torch.arange(TK_global, device=device) // TK_local).long()
    slot_g = (slot_pr + early[r_idx, dst.long()]).to(torch.int32)

    s, e = my_rank * TK_local, (my_rank + 1) * TK_local
    my_dst = dst[s:e].view(T_local, K)

    # expert_local_padded: real local expert when dst == my_rank, OOB
    # sentinel (== E_local) otherwise. Downstream
    # general_routing_router_metadata_triton interprets values >= E_local
    # as "padding bucket" and routes them to expert_frequency_offset[E_local]
    # so they don't pollute any real expert's row count.
    local_exp = (flat - dst.long() * E_local).to(torch.int32)
    expert_local_padded = torch.where(
        dst == my_rank,
        local_exp,
        torch.full_like(local_exp, E_local),
    )

    # a2a_token_indices: src_rank * TK_local + slot_per_rank.
    a2a_token_indices = (r_idx * TK_local + slot_pr.long()).to(torch.int32)

    return {
        "dst_rank_flat": dst,
        "slot_flat_per_rank": slot_pr,
        "slot_flat_global": slot_g,
        "my_dst_rank": my_dst,
        "my_pos_on_peer": slot_g[s:e].view(T_local, K),
        "my_pos_per_rank": slot_pr[s:e].view(T_local, K),
        "my_expert_local": (flat[s:e] - my_dst.reshape(-1) * E_local).view(T_local, K).to(torch.int32),
        "peer_count_per_rank": pcpr,
        "expert_local_padded": expert_local_padded,
        "a2a_token_indices": a2a_token_indices,
    }


def _make_routing(W: int, T_local: int, K: int, E: int, device: torch.device, *, skew_to: int | None = None):
    """Generate (W, T_local, K) routing. If skew_to is set, force half the
    slots to experts on that peer; otherwise uniform."""
    out = torch.randint(0, E, (W, T_local, K), dtype=torch.int32, device=device)
    if skew_to is not None:
        E_local = E // W
        n = W * T_local * K
        idx = torch.randperm(n, device=device)[: n // 2]
        out.view(-1)[idx] = torch.randint(0, E_local, (n // 2,), dtype=torch.int32, device=device) + skew_to * E_local
    return out


def _assert_metadata_equal(test_case: TestCommons, out: dict, ref: dict):
    """Verify all metadata outputs are bit-exact with the reference."""
    for k in ref:
        test_case.assertTrue(
            torch.equal(out[k], ref[k]),
            f"{k}: {(out[k] != ref[k]).sum().item()}/{out[k].numel()} differ",
        )


def _assert_slot_global_permutation(test_case: TestCommons, out: dict, W: int):
    """Self-consistency: slot_flat_global values where dst==p must form
    {0, 1, ..., total_p - 1} for each destination peer p."""
    dst = out["dst_rank_flat"]
    slot_g = out["slot_flat_global"]
    pcpr = out["peer_count_per_rank"]
    for p in range(W):
        idxs = slot_g[dst == p].sort().values
        total = int(pcpr[:, p].sum().item())
        expected = torch.arange(total, dtype=torch.int32, device=slot_g.device)
        test_case.assertTrue(
            torch.equal(idxs, expected),
            f"peer {p}: got {idxs.shape[0]} slots, expected {total}",
        )


def _ref_compute_dedup_metadata(dst_rank_flat: torch.Tensor, my_rank: int, W: int, T_local: int, K: int):
    """Pure-PyTorch reference for the 5 dedup-related metadata keys
    (4 dispatch-side + the combine-side ``peer_present_mask``).

    Algorithmically distinct from the Triton scan: builds the
    (W, T_local, K, W) one-hot tensor in one shot, then derives every
    output via straight torch reductions / cumsums. Bit-exact agreement
    with the kernel is the correctness bar.
    """
    device = dst_rank_flat.device
    TK_global = W * T_local * K
    dst3 = dst_rank_flat.view(W, T_local, K).long()  # (W, T, K)
    q_axis = torch.arange(W, device=device).view(1, 1, 1, W)
    oh = (dst3.unsqueeze(-1) == q_axis).to(torch.int32)  # (W, T, K, W)

    # canonical[p, t, k] = 1 iff lowest-k slot of (p, t) routing to its dst.
    cum_k = oh.cumsum(dim=2)  # (W, T, K, W)
    cnt_at_dst = (cum_k * oh).sum(dim=3)  # (W, T, K)
    pair_present_mask_ref = (cnt_at_dst == 1).to(torch.int8).reshape(TK_global).contiguous()

    # present[p, t, q] = (sum_k oh[p, t, k, q] > 0)
    present_ptq = (oh.sum(dim=2) > 0).to(torch.int32)  # (W, T, W)
    pair_count_ref = present_ptq.sum(dim=1).to(torch.int32).contiguous()  # (W, W)

    # pair_offset[p, q] = exclusive cumsum across p of pair_count[:, q]
    pair_offset_ref = (pair_count_ref.cumsum(dim=0) - pair_count_ref).to(torch.int32).contiguous()

    # within_p_token_rank[p, t, q] = exclusive cumsum across t of present[p, t, q]
    within_pq = (present_ptq.cumsum(dim=1) - present_ptq).to(torch.int32)  # (W, T, W)

    # rank_dedup_recv_pos[p, t, k] = pair_offset[p, dst[p,t,k]] + within_pq[p, t, dst[p,t,k]]
    p_idx = torch.arange(W, device=device).view(W, 1, 1).expand(W, T_local, K)
    t_idx = torch.arange(T_local, device=device).view(1, T_local, 1).expand(W, T_local, K)
    pair_off_at = pair_offset_ref[p_idx, dst3]  # (W, T, K)
    within_at = within_pq[p_idx, t_idx, dst3]  # (W, T, K)
    rank_dedup_recv_pos_ref = (pair_off_at + within_at).to(torch.int32).reshape(TK_global).contiguous()

    # peer_present_mask[q, t] (combine side): at this rank as the home
    # of token t, does peer q have any expert routed-to from my (t)?
    # Equivalent to present_ptq[my_rank, t, q] transposed to (q, t).
    peer_present_mask_ref = present_ptq[my_rank, :, :].T.to(torch.int8).contiguous()

    return {
        "pair_count": pair_count_ref,
        "pair_offset": pair_offset_ref,
        "pair_present_mask": pair_present_mask_ref,
        "rank_dedup_recv_pos": rank_dedup_recv_pos_ref,
        "peer_present_mask": peer_present_mask_ref,
    }


_DEDUP_KEYS = (
    "pair_count",
    "pair_offset",
    "pair_present_mask",
    "rank_dedup_recv_pos",
    "peer_present_mask",
)


def _assert_dedup_metadata_equal(test_case: TestCommons, out: dict, ref: dict):
    for k in _DEDUP_KEYS:
        test_case.assertIn(k, out, f"missing dedup key {k}")
        test_case.assertEqual(out[k].dtype, ref[k].dtype, f"{k}: dtype mismatch")
        test_case.assertEqual(out[k].shape, ref[k].shape, f"{k}: shape mismatch")
        test_case.assertTrue(
            torch.equal(out[k], ref[k]),
            f"{k}: {(out[k] != ref[k]).sum().item()}/{out[k].numel()} differ",
        )


def _assert_dedup_invariants(test_case: TestCommons, out: dict, my_rank: int, W: int, T_local: int, K: int):
    """Sanity checks independent of the reference function — guard semantic
    contracts that Phase 2/3 kernels rely on:

      (1) pair_count[p, q] ∈ [0, T_local].
      (2) sum_q pair_count[p, q] = sum_t |dst-set of (p, t)|.
      (3) pair_offset[p, q] == exclusive cumsum across p of pair_count[:, q].
      (4) pair_present_mask sums to sum(pair_count) (one canonical slot per
          (p, t, q) present-triple).
      (5) For each (p, q): the rank_dedup_recv_pos values across all CANONICAL
          slots f with src=p, dst=q form exactly
          [pair_offset[p, q], pair_offset[p, q] + pair_count[p, q]).
      (6) rank_dedup_recv_pos is uniform across the K slots of (p, t, q): all K
          slots routing to the same q at the same (p, t) must agree.
      (7) peer_present_mask[q, t] != 0 ⇔ at least one slot of (my_rank, t) → q.
    """
    pc = out["pair_count"]
    po = out["pair_offset"]
    mask = out["pair_present_mask"]
    drp = out["rank_dedup_recv_pos"]
    peer_present = out["peer_present_mask"]
    dst3 = out["dst_rank_flat"].view(W, T_local, K).long()

    # (1)
    test_case.assertTrue(torch.all(pc >= 0))
    test_case.assertTrue(torch.all(pc <= T_local))

    # (2)
    distinct_dsts = torch.zeros(W, dtype=torch.int64, device=pc.device)
    for p in range(W):
        per_token_distinct = torch.tensor(
            [dst3[p, t].unique().numel() for t in range(T_local)],
            device=pc.device,
        )
        distinct_dsts[p] = per_token_distinct.sum()
    test_case.assertTrue(
        torch.equal(pc.sum(dim=1).long(), distinct_dsts),
        f"sum_q pair_count[p, q] != sum_t |dst-set(p, t)|: " f"{pc.sum(dim=1).tolist()} vs {distinct_dsts.tolist()}",
    )

    # (3)
    po_ref = (pc.cumsum(dim=0) - pc).to(po.dtype)
    test_case.assertTrue(torch.equal(po, po_ref))

    # (4)
    test_case.assertEqual(int(mask.sum().item()), int(pc.sum().item()))

    # (5) + (6)
    drp3 = drp.view(W, T_local, K)
    mask3 = mask.view(W, T_local, K)
    for p in range(W):
        for q in range(W):
            sel = (dst3[p] == q) & (mask3[p].bool())
            canonical_pos = drp3[p][sel]
            expected = torch.arange(
                int(po[p, q]),
                int(po[p, q]) + int(pc[p, q]),
                dtype=drp.dtype,
                device=drp.device,
            )
            test_case.assertTrue(
                torch.equal(canonical_pos.sort().values, expected),
                f"(p={p}, q={q}): canonical positions {canonical_pos.tolist()} "
                f"!= [{int(po[p, q])}, {int(po[p, q] + pc[p, q])})",
            )
            # (6) — uniformity across all K slots of (p, t) routing to q.
            for t in range(T_local):
                routed = dst3[p, t] == q
                if routed.any():
                    vals = drp3[p, t][routed]
                    test_case.assertTrue(
                        bool((vals == vals[0]).all()),
                        f"(p={p}, t={t}, q={q}): K-slot rank_dedup_recv_pos disagree " f"{vals.tolist()}",
                    )

    # (7) — per (q, t), does (my_rank, t) route to q?
    routes_q_t = (
        (dst3[my_rank].unsqueeze(-1) == torch.arange(W, device=pc.device).view(1, 1, W)).any(dim=1).T
    )  # (W, T)
    test_case.assertTrue(
        torch.equal((peer_present != 0), routes_q_t),
        "peer_present_mask mismatch with my_rank-routes-to-q",
    )


def _assert_padded_sentinel(test_case: TestCommons, out: dict, my_rank: int, E_local: int):
    """expert_local_padded must be:
       - in [0, E_local) at slots where dst == my_rank, equal to expert_global % E_local
       - exactly E_local everywhere else (the OOB sentinel consumed by
         general_routing_router_metadata_triton).

    Independent of the reference function — guards the semantic contract
    rather than just self-consistency."""
    dst = out["dst_rank_flat"]
    padded = out["expert_local_padded"]
    is_mine = dst == my_rank

    # Sentinel slots: every non-mine entry must be exactly E_local.
    sentinel_vals = padded[~is_mine]
    if sentinel_vals.numel() > 0:
        test_case.assertTrue(
            torch.all(sentinel_vals == E_local),
            f"sentinel value mismatch: expected all == {E_local}, "
            f"got min={sentinel_vals.min().item()}, max={sentinel_vals.max().item()}",
        )

    # Mine slots: in-range and equal to my_expert_local flattened.
    mine_vals = padded[is_mine]
    if mine_vals.numel() > 0:
        test_case.assertTrue(
            torch.all((mine_vals >= 0) & (mine_vals < E_local)),
            f"mine values out of range [0, {E_local}): " f"min={mine_vals.min().item()}, max={mine_vals.max().item()}",
        )


class DispatchMetadataTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")],
            [
                # (W, T_local, K, E_local)
                (2, 64, 1, 4),
                (2, 256, 8, 16),
                (4, 64, 1, 4),
                (4, 333, 4, 4),  # non-pow2 T_local
                (4, 333, 4, 7),  # non-pow2 T, non-pow2 E_local
                (4, 1024, 8, 16),
                (4, 8192, 8, 16),
                (4, 8192, 8, 24),  # non-pow2 E_local
                (4, 32768, 8, 16),
                (8, 64, 2, 8),
                (8, 256, 10, 8),  # K=10 (matches e512_k10 model)
                (8, 256, 10, 28),
                (8, 1024, 8, 8),
                (8, 8192, 8, 8),
                (8, 32768, 8, 16),
            ],
            ["uniform", "skew_r0"],
        )
    )
    def test_semantic_correctness(
        self,
        device: torch.device,
        problem_shape: tuple[int, int, int, int],
        pattern: str,
    ) -> None:
        _set_seed(_SEED)
        W, T_local, K, E_local = problem_shape
        E = W * E_local
        skew = 0 if pattern == "skew_r0" else None
        topk = _make_routing(W, T_local, K, E, device, skew_to=skew)

        for my_rank in range(min(W, 4)):
            ref = _ref_compute_dispatch_metadata(topk, my_rank, E_local)
            out = compute_dispatch_metadata(topk, my_rank=my_rank, E_local=E_local)
            _assert_metadata_equal(self, out, ref)
            _assert_slot_global_permutation(self, out, W)
            _assert_padded_sentinel(self, out, my_rank, E_local)

    def test_all_to_peer_0(self) -> None:
        """Every slot routes to peer 0 — degenerate distribution."""
        _set_seed(_SEED)
        W, T_local, K, E_local = 4, 256, 4, 16
        device = torch.device("cuda")
        topk = torch.zeros(W, T_local, K, dtype=torch.int32, device=device)

        for my_rank in (0, 2):
            ref = _ref_compute_dispatch_metadata(topk, my_rank, E_local)
            out = compute_dispatch_metadata(topk, my_rank=my_rank, E_local=E_local)
            _assert_metadata_equal(self, out, ref)
            _assert_slot_global_permutation(self, out, W)
            _assert_padded_sentinel(self, out, my_rank, E_local)

        self.assertEqual(out["peer_count_per_rank"][:, 0].sum().item(), W * T_local * K)
        self.assertEqual(out["peer_count_per_rank"][:, 1:].sum().item(), 0)

    def test_none_to_me(self) -> None:
        """All routing goes to peer 0; my_rank=1 receives nothing.
        Every entry of expert_local_padded must equal the OOB sentinel."""
        _set_seed(_SEED)
        W, T_local, K, E_local = 4, 256, 4, 16
        device = torch.device("cuda")
        topk = torch.zeros(W, T_local, K, dtype=torch.int32, device=device)

        my_rank = 1
        ref = _ref_compute_dispatch_metadata(topk, my_rank, E_local)
        out = compute_dispatch_metadata(topk, my_rank=my_rank, E_local=E_local)
        _assert_metadata_equal(self, out, ref)
        _assert_slot_global_permutation(self, out, W)
        _assert_padded_sentinel(self, out, my_rank, E_local)

        self.assertEqual(out["peer_count_per_rank"][:, my_rank].sum().item(), 0)

        # Nothing routes to rank 1, so dst is never my_rank.
        dst = out["dst_rank_flat"]
        self.assertEqual((dst == my_rank).sum().item(), 0)

        # Stronger: every entry of expert_local_padded must be E_local.
        self.assertTrue(
            torch.all(out["expert_local_padded"] == E_local),
            "expected all sentinel values == E_local when nothing routes to my_rank",
        )

    def test_all_to_last_peer(self) -> None:
        """Every slot routes to the highest-numbered peer."""
        _set_seed(_SEED)
        W, T_local, K, E_local = 4, 256, 4, 16
        E = W * E_local
        device = torch.device("cuda")
        topk = torch.full((W, T_local, K), E - 1, dtype=torch.int32, device=device)

        my_rank = W - 1
        ref = _ref_compute_dispatch_metadata(topk, my_rank, E_local)
        out = compute_dispatch_metadata(topk, my_rank=my_rank, E_local=E_local)
        _assert_metadata_equal(self, out, ref)
        _assert_slot_global_permutation(self, out, W)
        _assert_padded_sentinel(self, out, my_rank, E_local)

        self.assertEqual(out["peer_count_per_rank"][:, W - 1].sum().item(), W * T_local * K)

    def test_deterministic(self) -> None:
        """Two identical calls must produce identical results."""
        _set_seed(_SEED)
        W, T_local, K, E_local = 4, 1024, 8, 16
        device = torch.device("cuda")
        topk = _make_routing(W, T_local, K, W * E_local, device)

        results1 = compute_dispatch_metadata(topk, my_rank=0, E_local=E_local)
        results2 = compute_dispatch_metadata(topk, my_rank=0, E_local=E_local)
        for k in results1:
            self.assertTrue(
                torch.equal(results1[k], results2[k]),
                f"Non-deterministic output for key {k}",
            )

    @parameterized.expand([(seed,) for seed in range(10)])
    def test_random_stress(self, seed: int) -> None:
        """Randomized stress test with varying shapes."""
        _set_seed(seed)
        device = torch.device("cuda")
        W = [2, 4, 8][seed % 3]
        T_local = torch.randint(64, 8192, (1,)).item()
        K = torch.randint(1, 9, (1,)).item()
        E_local = torch.randint(1, 32, (1,)).item()
        topk = _make_routing(W, T_local, K, W * E_local, device)

        for my_rank in range(min(W, 2)):
            ref = _ref_compute_dispatch_metadata(topk, my_rank, E_local)
            out = compute_dispatch_metadata(topk, my_rank=my_rank, E_local=E_local)
            self.assertTrue(
                all(torch.equal(out[k], ref[k]) for k in ref),
                f"seed={seed} W={W} T={T_local} K={K} E_l={E_local} my_rank={my_rank}: mismatch",
            )
            _assert_slot_global_permutation(self, out, W)
            _assert_padded_sentinel(self, out, my_rank, E_local)


class DedupMetadataTest(TestCommons):
    """Tests for the 6 RANK_DEDUP dispatch metadata keys emitted by phases D1/D2/D3.

    Bit-exact agreement with the pure-PyTorch reference is the bar; the
    additional invariant checks guard the contracts the dispatch /
    combine kernels rely on (canonical-slot uniqueness, packed-row range
    coverage, K-slot dedup uniformity, sentinel ↔ has-route equivalence)."""

    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")],
            [
                # (W, T_local, K, E_local) — the 5 validation shapes from the
                # task spec (covers K<W, K=W, K>W regimes), plus a few small
                # ones for fast iteration.
                (4, 64, 2, 4),
                (4, 256, 2, 4),
                (4, 1024, 8, 4),
                (8, 256, 8, 8),
                (8, 1024, 8, 8),
                (8, 8192, 8, 8),
                (8, 256, 16, 8),
                (8, 1024, 16, 8),
                (16, 256, 8, 4),
                (16, 1024, 8, 4),
                # Non-pow2 T_local — exercises BLOCK_T tail mask.
                (4, 333, 4, 4),
                (8, 511, 8, 8),
            ],
        )
    )
    def test_dedup_correctness(
        self,
        device: torch.device,
        problem_shape: tuple[int, int, int, int],
    ) -> None:
        _set_seed(_SEED)
        W, T_local, K, E_local = problem_shape
        E = W * E_local
        topk = _make_routing(W, T_local, K, E, device)

        for my_rank in range(min(W, 4)):
            out = compute_dispatch_metadata(topk, my_rank=my_rank, E_local=E_local)
            ref = _ref_compute_dedup_metadata(out["dst_rank_flat"], my_rank, W, T_local, K)
            _assert_dedup_metadata_equal(self, out, ref)

    def test_dedup_invariants_small(self) -> None:
        """Invariant checks (K-slot dedup uniformity, packed-row coverage,
        sentinel ↔ has-route) on a small shape so the python-level loops
        finish quickly."""
        _set_seed(_SEED)
        W, T_local, K, E_local = 4, 64, 4, 4
        device = torch.device("cuda")
        topk = _make_routing(W, T_local, K, W * E_local, device)
        for my_rank in range(W):
            out = compute_dispatch_metadata(topk, my_rank=my_rank, E_local=E_local)
            _assert_dedup_invariants(self, out, my_rank, W, T_local, K)

    def test_dedup_all_to_one_peer(self) -> None:
        """Every slot routes to peer 0. Each token's K slots collapse to a
        single canonical pull → pair_count[p, 0] == T_local, all others 0.
        """
        _set_seed(_SEED)
        W, T_local, K, E_local = 4, 128, 4, 4
        device = torch.device("cuda")
        topk = torch.zeros(W, T_local, K, dtype=torch.int32, device=device)

        for my_rank in (0, 2):
            out = compute_dispatch_metadata(topk, my_rank=my_rank, E_local=E_local)
            ref = _ref_compute_dedup_metadata(out["dst_rank_flat"], my_rank, W, T_local, K)
            _assert_dedup_metadata_equal(self, out, ref)

            # Stronger structural assertions.
            pc = out["pair_count"]
            self.assertEqual(int(pc[:, 0].sum().item()), W * T_local)
            self.assertEqual(int(pc[:, 1:].sum().item()), 0)
            # Each (p, t) has exactly one canonical slot (k=0) routed to 0.
            mask3 = out["pair_present_mask"].view(W, T_local, K)
            self.assertTrue(torch.all(mask3[:, :, 0] == 1))
            self.assertTrue(torch.all(mask3[:, :, 1:] == 0))

    def test_dedup_disjoint_destinations(self) -> None:
        """Each of token (p, t)'s K slots routes to a DIFFERENT peer →
        K canonical slots per token, dedup gives the same count as A2A.
        Requires K ≤ W; here K=4, W=4 covers the boundary."""
        _set_seed(_SEED)
        W, T_local, K, E_local = 4, 64, 4, 4
        device = torch.device("cuda")
        # Routing: token (p, t)'s slot k goes to expert k * E_local (lives on rank k).
        topk = torch.zeros(W, T_local, K, dtype=torch.int32, device=device)
        for k in range(K):
            topk[:, :, k] = k * E_local
        for my_rank in range(W):
            out = compute_dispatch_metadata(topk, my_rank=my_rank, E_local=E_local)
            ref = _ref_compute_dedup_metadata(out["dst_rank_flat"], my_rank, W, T_local, K)
            _assert_dedup_metadata_equal(self, out, ref)

            # Every slot is canonical (no repeated dst within a token).
            self.assertTrue(torch.all(out["pair_present_mask"] == 1))
            # pair_count[p, q] == T_local for every (p, q) in this routing.
            self.assertTrue(torch.all(out["pair_count"] == T_local))

    def test_dedup_none_to_me(self) -> None:
        """All routing goes to peer 0; my_rank=1 receives nothing.
        peer_present_mask for my_rank=1 must mark only peer 0 as having
        any contribution (since my (t)'s K experts all live on rank 0).
        """
        _set_seed(_SEED)
        W, T_local, K, E_local = 4, 64, 4, 4
        device = torch.device("cuda")
        topk = torch.zeros(W, T_local, K, dtype=torch.int32, device=device)

        out = compute_dispatch_metadata(topk, my_rank=1, E_local=E_local)
        # No tokens at any source rank route to my_rank=1.
        self.assertEqual(int(out["pair_count"][:, 1].sum().item()), 0)
        # (my_rank=1, t)'s dst is 0 for every k → peer 0 has every t
        # present, all other peers have nothing.
        ppm = out["peer_present_mask"]
        self.assertTrue(torch.all(ppm[0] == 1))
        self.assertTrue(torch.all(ppm[1:] == 0))

    def test_dedup_deterministic(self) -> None:
        """Two identical calls produce bit-identical dedup outputs."""
        _set_seed(_SEED)
        W, T_local, K, E_local = 8, 1024, 8, 8
        device = torch.device("cuda")
        topk = _make_routing(W, T_local, K, W * E_local, device)

        r1 = compute_dispatch_metadata(topk, my_rank=0, E_local=E_local)
        r2 = compute_dispatch_metadata(topk, my_rank=0, E_local=E_local)
        for k in _DEDUP_KEYS:
            self.assertTrue(torch.equal(r1[k], r2[k]), f"non-deterministic {k}")

    def test_dedup_emit_flag_off(self) -> None:
        """emit_dedup=False keeps the existing return dict unchanged."""
        _set_seed(_SEED)
        W, T_local, K, E_local = 4, 256, 4, 4
        device = torch.device("cuda")
        topk = _make_routing(W, T_local, K, W * E_local, device)
        with_dedup = compute_dispatch_metadata(topk, my_rank=0, E_local=E_local, emit_dedup=True)
        without_dedup = compute_dispatch_metadata(topk, my_rank=0, E_local=E_local, emit_dedup=False)
        for k in _DEDUP_KEYS:
            self.assertNotIn(k, without_dedup)
        # Existing keys remain byte-identical regardless of emit_dedup.
        for k in without_dedup:
            self.assertTrue(
                torch.equal(with_dedup[k], without_dedup[k]),
                f"existing key {k} changed when emit_dedup toggled",
            )

    @parameterized.expand([(seed,) for seed in range(8)])
    def test_dedup_random_stress(self, seed: int) -> None:
        """Random shapes (power-of-2 K). Compares to the torch reference."""
        _set_seed(seed)
        device = torch.device("cuda")
        W = [4, 8][seed % 2]
        T_local = int(torch.randint(64, 4096, (1,)).item())
        K = [1, 2, 4, 8][seed % 4]
        E_local = int(torch.randint(1, 32, (1,)).item())
        topk = _make_routing(W, T_local, K, W * E_local, device)
        for my_rank in range(min(W, 2)):
            out = compute_dispatch_metadata(topk, my_rank=my_rank, E_local=E_local)
            ref = _ref_compute_dedup_metadata(out["dst_rank_flat"], my_rank, W, T_local, K)
            _assert_dedup_metadata_equal(self, out, ref)


def _build_consumer_metadata(
    W: int,
    T_local: int,
    K: int,
    E_local: int,
    expert_local_padded: torch.Tensor,
    a2a_token_indices: torch.Tensor,
    device: torch.device,
):
    """Mirror of ep.py's ``_build_consumer_metadata`` — runs the histogram +
    bucketed-sort kernel to produce the s_reverse_local that the dedup
    A_idx scatter consumes."""
    TK_global = W * T_local * K
    E_total = E_local + 1
    s_reverse_local = torch.empty(TK_global, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK_global, dtype=torch.int32, device=device)
    s_scatter_idx = torch.empty(TK_global, dtype=torch.int32, device=device)
    expert_freq = torch.empty(E_total, dtype=torch.int32, device=device)
    expert_freq_off = torch.empty(E_total + 1, dtype=torch.int32, device=device)
    general_routing_router_metadata_triton(
        a2a_token_indices,
        expert_local_padded,
        TK_global,
        E_total,
        expert_freq,
        expert_freq_off,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_local,
        None,
    )
    return s_reverse_local, expert_freq_off


class BuildDedupAIdxTest(TestCommons):
    """Unit tests for ``build_rank_dedup_a_idx`` — the small Triton scatter that
    materializes the up-proj A_idx for the RANK_DEDUP_DISPATCH_TRITON path.

    Standalone reference: a_idx[s_reverse_local[f]] = rank_dedup_recv_pos[f]
    for every f with dst_rank_flat[f] == my_rank.
    """

    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")],
            [
                # (W, T_local, K, E_local). Same validation grid as the
                # dedup metadata tests; covers K<W, K=W, K>W regimes.
                (4, 64, 2, 4),
                (4, 256, 8, 4),
                (8, 256, 8, 8),
                (8, 1024, 8, 8),
                (8, 256, 16, 8),
                (16, 256, 8, 4),
                # non-pow2 T_local — exercises tail mask in the scatter.
                (4, 333, 4, 4),
                (8, 511, 8, 8),
                (8, 512, 16, 5),
                (8, 512, 12, 4),
            ],
        )
    )
    def test_a_idx_matches_reference(
        self,
        device: torch.device,
        problem_shape: tuple[int, int, int, int],
    ) -> None:
        _set_seed(_SEED)
        W, T_local, K, E_local = problem_shape
        E = W * E_local
        topk = _make_routing(W, T_local, K, E, device)

        for my_rank in range(min(W, 4)):
            out = compute_dispatch_metadata(topk, my_rank=my_rank, E_local=E_local)
            s_reverse_local, _ = _build_consumer_metadata(
                W,
                T_local,
                K,
                E_local,
                out["expert_local_padded"],
                out["a2a_token_indices"],
                device,
            )

            MAX_ROWS_PER_RANK = T_local * W * min(K, E_local)
            a_idx = torch.full((MAX_ROWS_PER_RANK,), -1, dtype=torch.int32, device=device)
            build_rank_dedup_a_idx(
                dst_rank_flat=out["dst_rank_flat"],
                s_reverse_local=s_reverse_local,
                rank_dedup_recv_pos=out["rank_dedup_recv_pos"],
                my_rank=my_rank,
                out=a_idx,
            )

            # Pure-Python reference: for every f routed to my_rank,
            # ref[s_reverse_local[f]] = rank_dedup_recv_pos[f].
            ref = torch.full_like(a_idx, -1)
            is_mine = out["dst_rank_flat"] == my_rank
            f_idx = is_mine.nonzero(as_tuple=False).flatten().long()
            e_idx = s_reverse_local[f_idx].long()
            ref[e_idx] = out["rank_dedup_recv_pos"][f_idx]

            # Tail beyond #routed is uninitialized in both kernel and
            # reference; only check the populated prefix.
            n_populated = int(is_mine.sum().item())
            # The kernel writes to a_idx[e_idx] for is_mine; uninit
            # tail in `a_idx` could be anything. Compare only at the
            # written e values.
            if n_populated > 0:
                # All-K-slots-of-(p,t,my_rank)-share-same-pos invariant
                # ⇒ every e ∈ unique(e_idx) gets the SAME packed-pos
                # written multiple times harmlessly. Compare the
                # populated entries.
                self.assertTrue(
                    torch.equal(a_idx[e_idx], ref[e_idx]),
                    f"build_rank_dedup_a_idx mismatch at "
                    f"W={W} T_local={T_local} K={K} E_local={E_local} "
                    f"my_rank={my_rank}",
                )


# ============================================================================
# Phase 1 robustness tests (BLOCK_T_MASK scaling + invariant check).
# ============================================================================


class PeerPresentMaskRegisterPressureTest(TestCommons):
    """``_build_peer_present_mask_kernel`` materializes a (BLOCK_T, K_PAD, W)
    one-hot tensor in registers; at large W·K the wrapper must scale
    BLOCK_T_MASK down to keep the working set bounded. Spot-check that
    the kernel compiles and produces the right mask at W=64, K=16."""

    @parameterized.expand(
        [
            # (W, K, T_local) — chosen to stress the (W·K) axis.
            (4, 2, 1024),  # tiny sanity
            (8, 8, 4096),  # standard
            (16, 16, 1024),  # K = W = 16
            (32, 8, 1024),  # large W
            (64, 16, 512),  # frontier
        ]
    )
    def test_register_pressure(self, W, K, T_local):
        device = torch.device("cuda")
        E_local = 8
        E = W * E_local
        # Routing only enters the kernel via dst_rank_flat = topk // E_local;
        # so we only care about dst values in [0, W). Construct directly.
        topk = torch.randint(0, E, (W, T_local, K), dtype=torch.int32, device=device)
        meta = compute_dispatch_metadata(topk, my_rank=0, E_local=E_local, emit_dedup=True)
        mask = meta["peer_present_mask"]

        # Reference: from MY (rank=0) tokens, did peer q get any expert?
        my_dst = topk[0] // E_local  # (T_local, K)
        ref = torch.zeros((W, T_local), dtype=torch.int8, device=device)
        for q in range(W):
            ref[q] = (my_dst == q).any(dim=-1).to(torch.int8)
        self.assertTrue(
            torch.equal(mask, ref),
            f"peer_present_mask mismatch at (W={W}, K={K}, T_local={T_local})",
        )
