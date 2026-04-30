# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import torch
from parameterized import parameterized

from sonicmoe.functional.ep import compute_dispatch_metadata
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

    # expert_local_padded: real local expert when dst == my_rank,
    # sentinel (global_idx % E_local) otherwise.
    local_exp = (flat - dst.long() * E_local).to(torch.int32)
    sentinel = (torch.arange(TK_global, device=device) % E_local).to(torch.int32)
    expert_local_padded = torch.where(dst == my_rank, local_exp, sentinel)

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

        self.assertEqual(out["peer_count_per_rank"][:, 0].sum().item(), W * T_local * K)
        self.assertEqual(out["peer_count_per_rank"][:, 1:].sum().item(), 0)

    def test_none_to_me(self) -> None:
        """All routing goes to peer 0; my_rank=1 receives nothing."""
        _set_seed(_SEED)
        W, T_local, K, E_local = 4, 256, 4, 16
        device = torch.device("cuda")
        topk = torch.zeros(W, T_local, K, dtype=torch.int32, device=device)

        my_rank = 1
        ref = _ref_compute_dispatch_metadata(topk, my_rank, E_local)
        out = compute_dispatch_metadata(topk, my_rank=my_rank, E_local=E_local)
        _assert_metadata_equal(self, out, ref)
        _assert_slot_global_permutation(self, out, W)

        self.assertEqual(out["peer_count_per_rank"][:, my_rank].sum().item(), 0)

        # expert_local_padded should be all sentinels for my_rank=1
        # (nothing routes to rank 1).
        dst = out["dst_rank_flat"]
        self.assertEqual((dst == my_rank).sum().item(), 0)

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
