# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************


import torch
from parameterized import parameterized

from sonicmoe.functional.triton_kernels import TC_topk_router_metadata_triton, general_routing_router_metadata_triton

from .test_commons import TestCommons


_SEED = 0


def _ref_TC_topk_router_metadata(topk_router_indices: torch.Tensor, E: int):
    """Pure-PyTorch reference for TC (token-choice) top-K metadata."""
    T, K = topk_router_indices.shape
    TK = T * K
    device = topk_router_indices.device
    flat = topk_router_indices.reshape(-1)

    expert_frequency = torch.bincount(flat.long(), minlength=E).int()
    expert_frequency_offset = torch.zeros(E + 1, dtype=torch.int32, device=device)
    expert_frequency_offset[1:] = torch.cumsum(expert_frequency, 0)

    s_scatter_idx = torch.argsort(flat.long(), stable=True).int()

    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx[s_scatter_idx] = torch.arange(TK, dtype=torch.int32, device=device)

    x_gather_idx = (s_scatter_idx // K).int()

    return expert_frequency, expert_frequency_offset, x_gather_idx, s_scatter_idx, s_reverse_scatter_idx


def _ref_general_routing_metadata(
    sorted_selected_T: torch.Tensor,
    selected_E: torch.Tensor,
    T: int,
    E: int,
):
    """Pure-PyTorch reference for general routing metadata."""
    TK = selected_E.size(0)
    device = selected_E.device

    expert_frequency = torch.bincount(selected_E.long(), minlength=E).int()
    expert_frequency_offset = torch.zeros(E + 1, dtype=torch.int32, device=device)
    expert_frequency_offset[1:] = torch.cumsum(expert_frequency, 0)

    s_scatter_idx = torch.argsort(selected_E.long(), stable=True).int()

    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx[s_scatter_idx] = torch.arange(TK, dtype=torch.int32, device=device)

    x_gather_idx = sorted_selected_T[s_scatter_idx.long()]

    token_counts = torch.bincount(sorted_selected_T.long(), minlength=T).int()
    num_activated_expert_per_token_offset = torch.zeros(T + 1, dtype=torch.int32, device=device)
    num_activated_expert_per_token_offset[1:] = torch.cumsum(token_counts, 0)

    return (
        expert_frequency,
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
    )


def _make_topk_indices(T: int, E: int, K: int, device: torch.device):
    """Generate random topk_router_indices [T, K] with unique experts per token."""
    indices = torch.zeros(T, K, dtype=torch.int32, device=device)
    for t in range(T):
        perm = torch.randperm(E, device=device)[:K].sort().values
        indices[t] = perm.int()
    return indices


def _make_general_routing_inputs(T: int, E: int, avg_K: float, device: torch.device):
    """Generate random general routing inputs with variable experts per token."""
    max_k = min(int(2 * avg_K), E)
    token_list = []
    expert_list = []
    for t in range(T):
        k = torch.randint(1, max_k + 1, (1,)).item()
        experts = torch.randperm(E, device=device)[:k].int()
        token_list.append(torch.full((k,), t, dtype=torch.int32, device=device))
        expert_list.append(experts)

    sorted_selected_T = torch.cat(token_list)
    selected_E = torch.cat(expert_list)

    sort_idx = torch.argsort(sorted_selected_T.long(), stable=True)
    sorted_selected_T = sorted_selected_T[sort_idx]
    selected_E = selected_E[sort_idx]

    return sorted_selected_T, selected_E


def _assert_inverse_permutation(test_case: TestCommons, s_scatter_idx, s_reverse_scatter_idx):
    """Verify that s_scatter_idx and s_reverse_scatter_idx are inverse permutations."""
    TK = s_scatter_idx.size(0)
    test_case.assertEqual(s_reverse_scatter_idx.size(0), TK)

    expected = torch.arange(TK, dtype=torch.int32, device=s_scatter_idx.device)

    recon = s_reverse_scatter_idx[s_scatter_idx.long()]
    test_case.assertTrue(torch.equal(recon, expected), "s_reverse_scatter_idx[s_scatter_idx] != identity")

    recon2 = s_scatter_idx[s_reverse_scatter_idx.long()]
    test_case.assertTrue(torch.equal(recon2, expected), "s_scatter_idx[s_reverse_scatter_idx] != identity")


def _assert_expert_grouping(test_case: TestCommons, flat_expert_ids, expert_frequency_offset, s_scatter_idx, E):
    """Verify entries in sorted order are correctly grouped by expert."""
    for e in range(E):
        start = expert_frequency_offset[e].item()
        end = expert_frequency_offset[e + 1].item()
        if start == end:
            continue
        flat_indices = s_scatter_idx[start:end].long()
        experts = flat_expert_ids[flat_indices]
        test_case.assertTrue(
            (experts == e).all(),
            f"Expert {e}: expected all entries in [{start},{end}) to be expert {e}, got {experts.unique().tolist()}",
        )


class TCTopkRouterMetadataTest(TestCommons):

    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")],
            [
                # (T, E, K)
                (10, 8, 2),
                (236, 16, 4),
                (1024, 128, 8),
                (8192, 128, 8),  # matches moe_test.py T=8192, E=128, K=8
                (8192, 64, 4),  # matches moe_test.py T=8192, E=64, K=4
                (8192, 32, 2),  # matches moe_test.py T=8192, E=32, K=2
                (8192, 256, 16),  # matches moe_test.py T=8192, E=256, K=16
                (32768, 128, 8),  # large scale
                (1000, 96, 7),  # K=7, non-power-of-2, many tokens
                (50, 384, 3),  # K=1, degenerate
                (2048, 512, 10),  # many experts
            ],
        )
    )
    def test_semantic_correctness(
        self,
        device: torch.device,
        problem_shape: tuple[int, int, int],
    ) -> None:
        self.set_seed(_SEED)

        T, E, K = problem_shape
        TK = T * K
        topk_indices = _make_topk_indices(T, E, K, device)

        expert_frequency = torch.empty(E, dtype=torch.int32, device=device)
        expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
        x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)
        s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
        s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)

        TC_topk_router_metadata_triton(
            topk_indices,
            E,
            expert_frequency,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
        )

        ref_freq, ref_offset, ref_gather, ref_scatter, ref_reverse = _ref_TC_topk_router_metadata(topk_indices, E)

        # expert_frequency exact match
        self.assertTrue(torch.equal(expert_frequency, ref_freq), "expert_frequency mismatch")

        # expert_frequency_offset exact match
        self.assertTrue(torch.equal(expert_frequency_offset, ref_offset), "expert_frequency_offset mismatch")

        # inverse permutation
        _assert_inverse_permutation(self, s_scatter_idx, s_reverse_scatter_idx)

        # correct expert grouping
        _assert_expert_grouping(self, topk_indices.reshape(-1), expert_frequency_offset, s_scatter_idx, E)

        # x_gather_idx == s_scatter_idx // K
        self.assertTrue(torch.equal(x_gather_idx, s_scatter_idx // K), "x_gather_idx != s_scatter_idx // K")

        # offset[-1] == TK, freq sums to TK
        self.assertEqual(expert_frequency_offset[-1].item(), TK)
        self.assertEqual(expert_frequency.sum().item(), TK)

    def test_all_same_expert(self) -> None:
        """Every token picks expert 0 and expert 1."""
        self.set_seed(_SEED)

        T, E, K = 64, 8, 2
        device = torch.device("cuda")
        topk_indices = torch.zeros(T, K, dtype=torch.int32, device=device)
        topk_indices[:, 0] = 0
        topk_indices[:, 1] = 1
        TK = T * K

        expert_frequency = torch.empty(E, dtype=torch.int32, device=device)
        expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
        x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)
        s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
        s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)

        TC_topk_router_metadata_triton(
            topk_indices,
            E,
            expert_frequency,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
        )

        self.assertEqual(expert_frequency[0].item(), T)
        self.assertEqual(expert_frequency[1].item(), T)
        self.assertEqual(expert_frequency[2:].sum().item(), 0)
        _assert_inverse_permutation(self, s_scatter_idx, s_reverse_scatter_idx)
        _assert_expert_grouping(self, topk_indices.reshape(-1), expert_frequency_offset, s_scatter_idx, E)

    def test_single_token(self) -> None:
        """T=1, minimal input."""
        self.set_seed(_SEED)

        T, E, K = 1, 8, 2
        device = torch.device("cuda")
        topk_indices = torch.tensor([[3, 5]], dtype=torch.int32, device=device)
        TK = T * K

        expert_frequency = torch.empty(E, dtype=torch.int32, device=device)
        expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
        x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)
        s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
        s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)

        TC_topk_router_metadata_triton(
            topk_indices,
            E,
            expert_frequency,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
        )

        ref_freq, ref_offset, _, _, _ = _ref_TC_topk_router_metadata(topk_indices, E)
        self.assertTrue(torch.equal(expert_frequency, ref_freq))
        self.assertTrue(torch.equal(expert_frequency_offset, ref_offset))
        _assert_inverse_permutation(self, s_scatter_idx, s_reverse_scatter_idx)
        self.assertTrue(torch.equal(x_gather_idx, s_scatter_idx // K))

    def test_deterministic(self) -> None:
        """Two identical calls must produce identical results."""
        self.set_seed(_SEED)

        T, E, K = 512, 64, 4
        device = torch.device("cuda")
        topk_indices = _make_topk_indices(T, E, K, device)
        TK = T * K

        def _run():
            ef = torch.empty(E, dtype=torch.int32, device=device)
            efo = torch.empty(E + 1, dtype=torch.int32, device=device)
            xg = torch.empty(TK, dtype=torch.int32, device=device)
            ss = torch.empty(TK, dtype=torch.int32, device=device)
            sr = torch.empty(TK, dtype=torch.int32, device=device)
            TC_topk_router_metadata_triton(topk_indices, E, ef, efo, xg, ss, sr)
            return ef, efo, xg, ss, sr

        results1 = _run()
        results2 = _run()
        for a, b in zip(results1, results2):
            self.assertTrue(torch.equal(a, b), "Non-deterministic metadata output")

    @parameterized.expand([(seed,) for seed in range(10)])
    def test_random_stress(self, seed: int) -> None:
        """Randomized stress test with varying shapes."""
        self.set_seed(seed)

        T = torch.randint(1, 32768, (1,)).item()
        E = torch.randint(2, 128, (1,)).item()
        K = torch.randint(1, min(E, 8) + 1, (1,)).item()
        device = torch.device("cuda")

        topk_indices = _make_topk_indices(T, E, K, device)
        TK = T * K

        ef = torch.empty(E, dtype=torch.int32, device=device)
        efo = torch.empty(E + 1, dtype=torch.int32, device=device)
        xg = torch.empty(TK, dtype=torch.int32, device=device)
        ss = torch.empty(TK, dtype=torch.int32, device=device)
        sr = torch.empty(TK, dtype=torch.int32, device=device)

        TC_topk_router_metadata_triton(topk_indices, E, ef, efo, xg, ss, sr)

        ref_freq, ref_offset, _, _, _ = _ref_TC_topk_router_metadata(topk_indices, E)

        self.assertTrue(torch.equal(ef, ref_freq), f"seed={seed}, T={T}, E={E}, K={K}: freq mismatch")
        self.assertTrue(torch.equal(efo, ref_offset), f"seed={seed}, T={T}, E={E}, K={K}: offset mismatch")
        _assert_inverse_permutation(self, ss, sr)
        _assert_expert_grouping(self, topk_indices.reshape(-1), efo, ss, E)
        self.assertTrue(torch.equal(xg, ss // K))


class GeneralRoutingRouterMetadataTest(TestCommons):

    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")],
            [
                # (T, E, avg_K)
                (10, 8, 2),
                (236, 16, 4),
                (1024, 128, 8),
                (8192, 128, 8),  # matches moe_test.py T=8192, E=128, K=8
                (8192, 64, 4),  # matches moe_test.py T=8192, E=64, K=4
                (8192, 32, 2),  # matches moe_test.py T=8192, E=32, K=2
                (8192, 256, 16),  # matches moe_test.py T=8192, E=256, K=16
                (32768, 128, 8),  # large scale
                (1000, 96, 7),  # K=7, non-power-of-2, many tokens
                (50, 384, 3),  # K=1, degenerate
                (2048, 512, 10),  # many experts
            ],
        )
    )
    def test_semantic_correctness(
        self,
        device: torch.device,
        problem_shape: tuple[int, int, int],
    ) -> None:
        self.set_seed(_SEED)

        T, E, avg_K = problem_shape
        sorted_selected_T, selected_E = _make_general_routing_inputs(T, E, avg_K, device)
        TK = selected_E.size(0)

        # --- Triton kernel ---
        expert_frequency = torch.empty(E, dtype=torch.int32, device=device)
        expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
        x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)
        s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
        s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
        num_activated_expert_per_token_offset = torch.empty(T + 1, dtype=torch.int32, device=device)

        general_routing_router_metadata_triton(
            sorted_selected_T,
            selected_E,
            T,
            E,
            expert_frequency,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
        )

        # --- PyTorch reference ---
        ref_freq, ref_offset, ref_gather, ref_scatter, ref_reverse, ref_token_offset = _ref_general_routing_metadata(
            sorted_selected_T, selected_E, T, E
        )

        self.assertTrue(torch.equal(expert_frequency, ref_freq), "expert_frequency mismatch")
        self.assertTrue(torch.equal(expert_frequency_offset, ref_offset), "expert_frequency_offset mismatch")
        _assert_inverse_permutation(self, s_scatter_idx, s_reverse_scatter_idx)
        _assert_expert_grouping(self, selected_E, expert_frequency_offset, s_scatter_idx, E)

        expected_gather = sorted_selected_T[s_scatter_idx.long()]
        self.assertTrue(torch.equal(x_gather_idx, expected_gather), "x_gather_idx mismatch")

        self.assertTrue(
            torch.equal(num_activated_expert_per_token_offset, ref_token_offset),
            "num_activated_expert_per_token_offset mismatch",
        )

        self.assertEqual(num_activated_expert_per_token_offset[0].item(), 0)
        self.assertEqual(num_activated_expert_per_token_offset[-1].item(), TK)
        self.assertEqual(expert_frequency_offset[-1].item(), TK)

    def test_uniform_K(self) -> None:
        """Every token has exactly K experts — token offset should be uniform."""
        self.set_seed(_SEED)

        T, E, K = 64, 16, 4
        device = torch.device("cuda")
        topk_indices = _make_topk_indices(T, E, K, device)

        token_list = []
        expert_list = []
        for t in range(T):
            token_list.append(torch.full((K,), t, dtype=torch.int32, device=device))
            expert_list.append(topk_indices[t])
        sorted_selected_T = torch.cat(token_list)
        selected_E = torch.cat(expert_list)

        TK = T * K
        expert_frequency = torch.empty(E, dtype=torch.int32, device=device)
        expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
        x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)
        s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
        s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
        num_activated_expert_per_token_offset = torch.empty(T + 1, dtype=torch.int32, device=device)

        general_routing_router_metadata_triton(
            sorted_selected_T,
            selected_E,
            T,
            E,
            expert_frequency,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
        )

        expected_token_offset = torch.arange(0, T * K + 1, K, dtype=torch.int32, device=device)
        self.assertTrue(torch.equal(num_activated_expert_per_token_offset, expected_token_offset))
        _assert_inverse_permutation(self, s_scatter_idx, s_reverse_scatter_idx)
        _assert_expert_grouping(self, selected_E, expert_frequency_offset, s_scatter_idx, E)

    def test_single_expert_per_token(self) -> None:
        """Each token picks exactly 1 expert — token_offset = [0, 1, ..., T]."""
        self.set_seed(_SEED)

        T, E = 128, 32
        device = torch.device("cuda")
        selected_E = torch.randint(0, E, (T,), dtype=torch.int32, device=device)
        sorted_selected_T = torch.arange(T, dtype=torch.int32, device=device)

        TK = T
        expert_frequency = torch.empty(E, dtype=torch.int32, device=device)
        expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
        x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)
        s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
        s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
        num_activated_expert_per_token_offset = torch.empty(T + 1, dtype=torch.int32, device=device)

        general_routing_router_metadata_triton(
            sorted_selected_T,
            selected_E,
            T,
            E,
            expert_frequency,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
        )

        ref = _ref_general_routing_metadata(sorted_selected_T, selected_E, T, E)
        self.assertTrue(torch.equal(expert_frequency, ref[0]))
        self.assertTrue(torch.equal(expert_frequency_offset, ref[1]))
        self.assertTrue(torch.equal(num_activated_expert_per_token_offset, ref[5]))
        _assert_inverse_permutation(self, s_scatter_idx, s_reverse_scatter_idx)

        expected = torch.arange(T + 1, dtype=torch.int32, device=device)
        self.assertTrue(torch.equal(num_activated_expert_per_token_offset, expected))

    def test_empty_experts(self) -> None:
        """More experts than token*K entries — many experts get zero tokens."""
        self.set_seed(_SEED)

        T, E, K = 32, 64, 2
        device = torch.device("cuda")
        topk_indices = _make_topk_indices(T, E, K, device)

        token_list = []
        expert_list = []
        for t in range(T):
            token_list.append(torch.full((K,), t, dtype=torch.int32, device=device))
            expert_list.append(topk_indices[t])
        sorted_selected_T = torch.cat(token_list)
        selected_E = torch.cat(expert_list)

        TK = T * K
        expert_frequency = torch.empty(E, dtype=torch.int32, device=device)
        expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
        x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)
        s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
        s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
        num_activated_expert_per_token_offset = torch.empty(T + 1, dtype=torch.int32, device=device)

        general_routing_router_metadata_triton(
            sorted_selected_T,
            selected_E,
            T,
            E,
            expert_frequency,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
        )

        ref = _ref_general_routing_metadata(sorted_selected_T, selected_E, T, E)
        self.assertTrue(torch.equal(expert_frequency, ref[0]))
        self.assertTrue(torch.equal(expert_frequency_offset, ref[1]))
        self.assertTrue(torch.equal(num_activated_expert_per_token_offset, ref[5]))
        _assert_inverse_permutation(self, s_scatter_idx, s_reverse_scatter_idx)
        self.assertTrue((expert_frequency == 0).any(), "Expected some experts to have zero tokens")

    def test_deterministic(self) -> None:
        """Two identical calls must produce identical results."""
        self.set_seed(_SEED)

        T, E, avg_K = 256, 32, 4
        device = torch.device("cuda")
        sorted_selected_T, selected_E = _make_general_routing_inputs(T, E, avg_K, device)
        TK = selected_E.size(0)

        def _run():
            ef = torch.empty(E, dtype=torch.int32, device=device)
            efo = torch.empty(E + 1, dtype=torch.int32, device=device)
            xg = torch.empty(TK, dtype=torch.int32, device=device)
            ss = torch.empty(TK, dtype=torch.int32, device=device)
            sr = torch.empty(TK, dtype=torch.int32, device=device)
            tok_off = torch.empty(T + 1, dtype=torch.int32, device=device)
            general_routing_router_metadata_triton(
                sorted_selected_T,
                selected_E,
                T,
                E,
                ef,
                efo,
                xg,
                ss,
                sr,
                tok_off,
            )
            return ef, efo, xg, ss, sr, tok_off

        results1 = _run()
        results2 = _run()
        for a, b in zip(results1, results2):
            self.assertTrue(torch.equal(a, b), "Non-deterministic general routing metadata output")

    @parameterized.expand([(seed,) for seed in range(10)])
    def test_random_stress(self, seed: int) -> None:
        """Randomized stress test with varying shapes."""
        self.set_seed(seed)

        T = torch.randint(1, 32768, (1,)).item()
        E = torch.randint(2, 128, (1,)).item()
        avg_K = torch.randint(1, min(E, 6) + 1, (1,)).item()
        device = torch.device("cuda")

        sorted_selected_T, selected_E = _make_general_routing_inputs(T, E, avg_K, device)
        TK = selected_E.size(0)

        ef = torch.empty(E, dtype=torch.int32, device=device)
        efo = torch.empty(E + 1, dtype=torch.int32, device=device)
        xg = torch.empty(TK, dtype=torch.int32, device=device)
        ss = torch.empty(TK, dtype=torch.int32, device=device)
        sr = torch.empty(TK, dtype=torch.int32, device=device)
        tok_off = torch.empty(T + 1, dtype=torch.int32, device=device)

        general_routing_router_metadata_triton(
            sorted_selected_T,
            selected_E,
            T,
            E,
            ef,
            efo,
            xg,
            ss,
            sr,
            tok_off,
        )

        ref = _ref_general_routing_metadata(sorted_selected_T, selected_E, T, E)
        self.assertTrue(torch.equal(ef, ref[0]), f"seed={seed}: freq mismatch")
        self.assertTrue(torch.equal(efo, ref[1]), f"seed={seed}: offset mismatch")
        self.assertTrue(torch.equal(tok_off, ref[5]), f"seed={seed}: token offset mismatch")
        _assert_inverse_permutation(self, ss, sr)
        _assert_expert_grouping(self, selected_E, efo, ss, E)
        self.assertTrue(torch.equal(xg, sorted_selected_T[ss.long()]))
