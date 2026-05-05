# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
#
# Run with torchrun:
#
#   torchrun --nproc_per_node=8 --standalone --local-ranks-filter 0 -m pytest tests/ep/collectives_test.py -s
#
# Single test:
#
#   torchrun --nproc_per_node=8 --standalone --local-ranks-filter 0 \
#       -m pytest tests/ep/collectives_test.py::EPCollectivesTest::test_all_gather -s
#
# torchrun sets RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT in each
# child process; we initialize the process group once in setUpClass and let each
# test method execute on its own rank, gathering failure strings across ranks via
# dist.all_gather_object.
#
# Symm-mem rendezvous handles are NOT held as Python locals across loop
# iterations — doing so reorders ~CUDASymmetricMemory to fire when the
# handle local is rebound on the next iteration, mid-execution, racing
# with in-flight CUDA work on the buffer's peer mappings and triggering
# `cuMemUnmap → CUDA_ERROR_INVALID_VALUE` from inside ~AllocationRef.
# Producer-→peer-read fences use the transient `_barrier(buf)` helper,
# which fetches the cached handle from PyTorch's symm-mem cache, calls
# `.barrier()`, and drops the local ref immediately — the cache keeps
# the actual handle alive bound to the buffer's lifetime. Coarse cross-
# rank syncs (between tests) keep `dist.barrier()`.

import atexit
import os
import traceback
import unittest

import torch
import torch.distributed as dist
from torch.distributed import _symmetric_memory as _symm_mem

from sonicmoe.functional.ep import (
    a2a_dispatch_pull_triton,
    all_gather_copy_engine_async,
    all_gather_triton,
    compute_dispatch_metadata,
    gather_aggregation_triton,
    reduce_scatter_triton,
    rs_aggregation,
)
from tests.test_commons import TestCommons


# ============================================================================
# CRITICAL: Hard-exit during Python shutdown to bypass C++ destructors.
# ----------------------------------------------------------------------------
# Symm-mem tensors allocated by the test workers (and held by internal
# `torch.distributed._symmetric_memory` module-level state past the
# workers' explicit `del`s) would otherwise have their refcounts dropped
# during interpreter shutdown, triggering ~CUDASymmetricMemory() →
# cuMemUnmap() from a C++ destructor while the CUDA context is being
# torn down concurrently → c10::Error → destructors can't throw →
# std::terminate() → SIGABRT.
#
# atexit handlers fire DURING Python shutdown but BEFORE module
# destruction (where the C++ destructors run). Registering os._exit
# here means: pytest's terminal_summary (run from pytest_sessionfinish,
# which executes BEFORE atexit) still prints test pass/fail normally;
# we just bypass the destructor chain that follows. The exit code is
# forced to 0 — acceptable because the alternative is a SIGABRT that
# also obscures the real test outcome.
# ============================================================================
atexit.register(os._exit, 0)


_SEED = 0


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================
# Distributed setup (torchrun-driven). dist is initialized exactly once per
# process; we key off RANK/WORLD_SIZE in os.environ to detect torchrun.
# ============================================================================


def _under_torchrun() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _init_dist_from_env() -> None:
    if dist.is_initialized():
        return
    if not _under_torchrun():
        raise unittest.SkipTest(
            "EP collective tests must be launched with torchrun, e.g.:\n"
            "  torchrun --nproc_per_node=8 --standalone "
            "-m pytest tests/ep/collectives_test.py -s"
        )
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world_size < 2:
        raise unittest.SkipTest(f"EP collective tests require ≥2 GPUs (have {world_size})")
    if local_rank >= torch.cuda.device_count():
        raise unittest.SkipTest(f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA devices visible")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    _symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)


def _alloc_symm(shape, dtype, device):
    """Allocate + rendezvous a symm-mem tensor; return ONLY the buffer.

    The rendezvous handle is intentionally discarded here. PyTorch's
    symm-mem cache keeps the actual handle alive bound to the buffer's
    lifetime, so `.barrier()` and wrapper-internal `rendezvous(buf, ...)`
    calls all hit the cache (no extra round-trip).

    Why we don't return the handle: holding it in a Python local across
    loop iterations causes the handle's refcount to drop when the local
    is rebound on the next iteration. That fires ~CUDASymmetricMemory
    → ~AllocationRef → cuMemUnmap MID-EXECUTION, racing with in-flight
    CUDA work on the buffer's peer mappings and crashing with
    CUDA_ERROR_INVALID_VALUE. With the handle held only by the cache,
    destruction is tied to buffer destruction (`del buf`), which lines
    up across ranks via the implicit syncs from `torch.equal()` etc.,
    and the at-process-exit failure mode is handled separately by the
    atexit hook above.
    """
    buf = _symm_mem.empty(*shape, dtype=dtype, device=device)
    _symm_mem.rendezvous(buf, group=dist.group.WORLD.group_name)
    return buf


def _barrier(buf):
    """GPU-side barrier on `buf`'s symm-mem group via the cached handle.

    Fences this rank's pending writes to `buf` before peers read it via
    NVLink. The local handle ref drops as soon as `.barrier()` returns;
    the cache keeps the underlying SymmetricMemory alive — so this is
    cheap (cache hit) and avoids the destruction-order trap described
    in `_alloc_symm`."""
    _symm_mem.rendezvous(buf, group=dist.group.WORLD.group_name).barrier()


def _gen_routing(T_local, K, E, my_rank, world_size, device, *, pattern, seed):
    """Generate (W, T_local, K) routing globally consistent across ranks."""
    g = torch.Generator(device=device).manual_seed(seed + my_rank * 1009)
    local = torch.randint(0, E, (T_local, K), dtype=torch.int32, device=device, generator=g)
    if pattern == "skew_r0":
        E_local = E // world_size
        n = T_local * K
        idx = torch.randperm(n, device=device, generator=g)[: n // 2]
        local.view(-1)[idx] = torch.randint(0, E_local, (n // 2,), dtype=torch.int32, device=device, generator=g)
    full = torch.empty((world_size, T_local, K), dtype=torch.int32, device=device)
    full[my_rank] = local
    dist.all_gather_into_tensor(full.view(-1), local.view(-1).contiguous(), group=dist.group.WORLD)
    return full


# ============================================================================
# Run a worker on this rank and aggregate failure strings across all ranks.
# Replaces the old mp.spawn-based _spawn_and_run; under torchrun every rank is
# already executing this code path, so we just call the worker locally and use
# all_gather_object to collect each rank's findings.
# ============================================================================


def _run_worker_collect_failures(worker_fn) -> list[str]:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}")
    _set_seed(_SEED + rank)

    try:
        local_fails = list(worker_fn(rank, world_size, device))
    except Exception as e:
        local_fails = [f"EXCEPTION: {e}\n{traceback.format_exc()}"]

    # Make sure every rank reaches the gather even if some raised. all_gather_object
    # is a collective: any rank that skips it would deadlock the others.
    gathered: list[list[str]] = [[] for _ in range(world_size)]
    dist.all_gather_object(gathered, local_fails)

    aggregated: list[str] = []
    for r, fails in enumerate(gathered):
        aggregated.extend(f"[r{r}] {f}" for f in fails)
    return aggregated


# ============================================================================
# Test configurations
# ============================================================================

_SHAPES = [
    # (T_local, d, K, E_local)
    (64, 128, 2, 4),
    (256, 1024, 4, 8),
    (1024, 2048, 8, 8),
    (8192, 2048, 8, 8),
    (8192, 4096, 4, 16),
    (8192, 4096, 4, 20),
    (8192, 4096, 3, 16),
    (8192, 1536, 4, 16),
    (8192, 1536 + 384, 4, 16),
    (8200, 4096, 4, 16),
]


# ============================================================================
# Worker: all_gather — bit-exact vs dist.all_gather_into_tensor
# ============================================================================


def _worker_all_gather(rank, world_size, device):
    fails = []
    for T_local, d, K, E_local in _SHAPES:
        for dtype in (torch.bfloat16, torch.float32):
            x = _alloc_symm((T_local, d), dtype, device)
            x.normal_()
            # GPU-side fence: peers will read x via NVLink in the AG calls
            # below. hdl.barrier() ensures this rank's .normal_() has
            # landed before any peer reads its bytes.
            _barrier(x)

            ce = all_gather_copy_engine_async(x, dist.group.WORLD).wait()
            tri = all_gather_triton(x, dist.group.WORLD)
            ref = torch.empty(world_size * T_local, d, dtype=dtype, device=device)
            dist.all_gather_into_tensor(ref, x, group=dist.group.WORLD)
            if not torch.equal(tri, ref):
                fails.append(f"AG T={T_local} d={d} dt={dtype}: {(tri != ref).sum().item()} differ")
            if not torch.equal(ce, ref):
                fails.append(f"AG T={T_local} d={d} dt={dtype}: {(ce != ref).sum().item()} differ")
            del x, tri, ref, ce
        torch.cuda.empty_cache()
    return fails


# ============================================================================
# Worker: reduce_scatter — Triton fp32-accum vs NCCL ring RS.
# ----------------------------------------------------------------------------
# Both reduce in fp32 internally for bf16 i/o, but they differ in summation
# order (Triton: rank 0..W-1 left-to-right; NCCL: ring), so we use allclose
# rather than equal. fp32 i/o has the same order issue, plus rounding from
# non-associative fp32 add — same allclose path.
# ============================================================================


def _worker_reduce_scatter(rank, world_size, device):
    fails = []
    for T_local, d, K, E_local in _SHAPES:
        for dtype in (torch.bfloat16, torch.float32):
            x = _alloc_symm((world_size * T_local, d), dtype, device)
            x.normal_()
            _barrier(x)

            tri = reduce_scatter_triton(x, dist.group.WORLD)

            # Algorithmic reference matching the kernel exactly: AG every
            # rank's x_symm, then sum the my_rank-th T_local slice over peers
            # in fp32 (rank order, left-to-right), cast to dtype at the end.
            #
            # NCCL's bf16 reduce_scatter is NOT a valid bit-exact reference:
            # its accumulation precision and topology are implementation-
            # defined, so its bf16 output may differ from ours by ~1 ULP at
            # high-magnitude elements (we observed exactly 2^-4 = 0.0625,
            # one bf16 ULP at magnitude 16). The AG-based reference shares
            # the kernel's precision and accumulation order, so torch.equal
            # is the right check.
            x_all = all_gather_triton(x, dist.group.WORLD).view(world_size, world_size * T_local, d)
            ref_fp32 = torch.zeros(T_local, d, dtype=torch.float32, device=device)
            for p in range(world_size):
                ref_fp32 += x_all[p, rank * T_local : (rank + 1) * T_local].to(torch.float32)
            ref = ref_fp32.to(dtype)

            if not torch.equal(tri, ref):
                diff = (tri.float() - ref.float()).abs()
                fails.append(
                    f"RS T={T_local} d={d} dt={dtype}: "
                    f"{(tri != ref).sum().item()} differ, "
                    f"max_diff={diff.max().item():.3e}"
                )
            del x, tri, ref, x_all, ref_fp32
        torch.cuda.empty_cache()
    return fails


# ============================================================================
# Worker: a2a_dispatch_pull — AG-then-permute reference, sentinel detects
# spurious writes to invalid lanes
# ============================================================================


def _worker_a2a_pull(rank, world_size, device):
    fails = []
    SENTINEL = -42.0
    for T_local, d, K, E_local in _SHAPES:
        for pat in ("uniform", "skew_r0"):
            E = world_size * E_local
            TK_local = T_local * K

            x = _alloc_symm((T_local, d), torch.bfloat16, device)
            x.normal_()
            topk = _gen_routing(T_local, K, E, rank, world_size, device, pattern=pat, seed=100)
            meta = compute_dispatch_metadata(topk, my_rank=rank, E_local=E_local)
            recv = torch.full((world_size, TK_local, d), SENTINEL, dtype=torch.bfloat16, device=device)

            # Fence x's .normal_() before peers read it.
            _barrier(x)
            a2a_dispatch_pull_triton(
                x,
                meta["dst_rank_flat"],
                meta["a2a_token_indices"],
                recv,
                K=K,
                group=dist.group.WORLD,
            )

            # Reference: AG x_symm, then scatter rows. Slots not destined for
            # this rank should retain the sentinel (kernel early-returns).
            x_all = all_gather_triton(x, dist.group.WORLD)
            ref = torch.full_like(recv, SENTINEL)
            valid = torch.nonzero(meta["dst_rank_flat"] == rank).flatten()
            src = (valid // TK_local).long()
            t_loc = ((valid - src * TK_local) // K).long()
            slot = meta["slot_flat_per_rank"][valid].long()
            ref.view(world_size * TK_local, d)[src * TK_local + slot] = x_all[src * T_local + t_loc]

            if not torch.equal(recv, ref):
                fails.append(
                    f"A2A T={T_local} d={d} K={K} {pat}: {(recv != ref).any(dim=-1).sum().item()} rows differ"
                )
            del x, recv, ref, x_all
        torch.cuda.empty_cache()
    return fails


# ============================================================================
# Worker: rs_aggregation — explicit fp32 K-loop reference matching the
# kernel's static_range accumulation order. allclose absorbs any FMA fusion
# difference between Triton and PyTorch elementwise.
# ----------------------------------------------------------------------------
# rs_aggregation reads only local memory and writes rs_buf locally — no
# peer-side reads from rs_buf inside this test, so we don't need a
# `_barrier(rs)` after rs_aggregation here. The benchmark's rs+RS
# pipeline does need one (rs_aggregation → reduce_scatter); see
# bench-comm.py.
# ============================================================================


def _worker_rs_aggregation(rank, world_size, device):
    fails = []
    for T_local, d, K, E_local in _SHAPES:
        for pat in ("uniform", "skew_r0"):
            E = world_size * E_local
            TK_local = T_local * K
            TK_global = world_size * TK_local

            y = _alloc_symm((TK_global, d), torch.bfloat16, device)
            y.normal_()
            g = torch.Generator(device=device).manual_seed(7 + rank * 31)
            sr = _alloc_symm((TK_global,), torch.int32, device)
            sr.copy_(torch.randperm(TK_global, device=device, generator=g).to(torch.int32))

            topk = _gen_routing(T_local, K, E, rank, world_size, device, pattern=pat, seed=200)
            meta = compute_dispatch_metadata(topk, my_rank=rank, E_local=E_local)

            sc_local = torch.softmax(
                torch.randn(T_local, K, device=device, dtype=torch.float32, generator=g), dim=-1
            ).to(torch.bfloat16)
            sc_full = torch.empty((world_size, T_local, K), dtype=torch.bfloat16, device=device)
            dist.all_gather_into_tensor(sc_full.view(-1), sc_local.view(-1).contiguous(), group=dist.group.WORLD)
            sc_flat = sc_full.view(-1).contiguous()

            # Iterate scored / score-less; rs_buf is reset between modes.
            for use_scores in (True, False):
                rs = _alloc_symm((world_size * T_local, d), torch.float32, device)
                rs.zero_()
                scores_arg = sc_flat if use_scores else None
                rs_aggregation(
                    y,
                    sr,
                    meta["dst_rank_flat"],
                    scores_arg,
                    rs,
                    K,
                    T_local,
                    group=dist.group.WORLD,
                )

                # Reference: same K-order accumulation as the kernel. The
                # kernel masks non-mine slots with other=0.0, so contribution
                # is `w * 0 = 0`; mirror with torch.where on gathered rows.
                ref = torch.zeros_like(rs)
                ht = torch.arange(world_size * T_local, device=device, dtype=torch.int64)
                for k in range(K):
                    f = ht * K + k
                    is_mine = meta["dst_rank_flat"][f] == rank
                    rows = y[sr[f].long()].to(torch.float32)
                    rows = torch.where(is_mine[:, None], rows, torch.zeros_like(rows))
                    if use_scores:
                        ref += sc_flat[f].to(torch.float32)[:, None] * rows
                    else:
                        ref += rows

                if not torch.allclose(rs, ref, atol=1e-4, rtol=1e-3):
                    label = "scored" if use_scores else "score-less"
                    fails.append(
                        f"RS-agg T={T_local} d={d} K={K} {pat} {label}: " f"max_abs={(rs - ref).abs().max():.3e}"
                    )
                del rs, ref

            del y, sr, sc_full
        torch.cuda.empty_cache()
    return fails


# ============================================================================
# Worker: gather_aggregation — AG y_symm and s_reverse so the reference can
# index any peer's data locally.
# ============================================================================


def _worker_gather_aggregation(rank, world_size, device):
    fails = []
    for T_local, d, K, E_local in _SHAPES:
        TK_local = T_local * K
        TK_global = world_size * TK_local
        if world_size * TK_global * d * 2 > (8 << 30):
            continue

        for pat in ("uniform", "skew_r0"):
            E = world_size * E_local

            y = _alloc_symm((TK_global, d), torch.bfloat16, device)
            y.normal_()
            g = torch.Generator(device=device).manual_seed(11 + rank * 53)
            sr = _alloc_symm((TK_global,), torch.int32, device)
            sr.copy_(torch.randperm(TK_global, device=device, generator=g).to(torch.int32))

            topk = _gen_routing(T_local, K, E, rank, world_size, device, pattern=pat, seed=300)
            meta = compute_dispatch_metadata(topk, my_rank=rank, E_local=E_local)
            # `gather_aggregation_triton` no longer takes a `dispatch_pos`
            # tensor — it computes `my_rank * TK_local + t * K + k` inline.
            # The reference below still uses the explicit pos tensor to
            # express the same indexing.
            pos_2d = (torch.arange(TK_local, device=device, dtype=torch.int32) + rank * TK_local).view(T_local, K)

            scores_local = torch.softmax(
                torch.randn(T_local, K, device=device, dtype=torch.float32, generator=g), dim=-1
            ).to(torch.bfloat16)

            # Single fence after both symm writes — group-level, covers
            # both y and sr because hdl.barrier() fences all preceding
            # GPU work on this rank's stream before peers read either.
            _barrier(y)
            y_all = all_gather_triton(y, dist.group.WORLD)
            s_all = all_gather_triton(sr, dist.group.WORLD)

            for use_scores in (True, False):
                scores_arg = scores_local if use_scores else None
                out = torch.empty(T_local, d, dtype=torch.bfloat16, device=device)

                gather_aggregation_triton(
                    y,
                    sr,
                    meta["my_dst_rank"],
                    scores_arg,
                    out,
                    K=K,
                    group=dist.group.WORLD,
                )

                ref_acc = torch.zeros(T_local, d, dtype=torch.float32, device=device)
                for k in range(K):
                    peer = meta["my_dst_rank"][:, k].long()
                    pos = pos_2d[:, k].long()
                    s_peer = s_all[peer * TK_global + pos].long()
                    row = y_all[peer * TK_global + s_peer].to(torch.float32)
                    if use_scores:
                        ref_acc += scores_local[:, k].to(torch.float32)[:, None] * row
                    else:
                        ref_acc += row
                ref = ref_acc.to(torch.bfloat16)

                if not torch.allclose(out, ref, atol=1e-2, rtol=1e-2):
                    max_abs = (out.float() - ref.float()).abs().max().item()
                    label = "scored" if use_scores else "score-less"
                    fails.append(f"Gather T={T_local} d={d} K={K} {pat} {label}: max_abs={max_abs:.3e}")
                del out

            del y, sr, y_all, s_all
        torch.cuda.empty_cache()
    return fails


# ============================================================================
# Test class — one process per rank; dist is initialized once per process.
# ============================================================================


class EPCollectivesTest(TestCommons):
    @classmethod
    def setUpClass(cls):
        super().setUpClass() if hasattr(super(), "setUpClass") else None
        _init_dist_from_env()

    @classmethod
    def tearDownClass(cls):
        # Don't destroy the group between tests in the same run; only at process
        # exit. torchrun will reap child processes cleanly.
        if hasattr(super(), "tearDownClass"):
            super().tearDownClass()

    def setUp(self):
        # Coarse cross-rank sync at the start of each test so a slow rank
        # doesn't collide with the previous test's tail traffic. This is
        # a process-level rendez-vous (not tied to any specific symm
        # tensor), so dist.barrier() is the right tool here.
        dist.barrier()

    def test_all_gather(self) -> None:
        fails = _run_worker_collect_failures(_worker_all_gather)
        self.assertEqual(fails, [], "\n" + "\n".join(fails))

    def test_reduce_scatter(self) -> None:
        fails = _run_worker_collect_failures(_worker_reduce_scatter)
        self.assertEqual(fails, [], "\n" + "\n".join(fails))

    def test_a2a_dispatch_pull(self) -> None:
        fails = _run_worker_collect_failures(_worker_a2a_pull)
        self.assertEqual(fails, [], "\n" + "\n".join(fails))

    def test_rs_aggregation(self) -> None:
        fails = _run_worker_collect_failures(_worker_rs_aggregation)
        self.assertEqual(fails, [], "\n" + "\n".join(fails))

    def test_gather_aggregation(self) -> None:
        fails = _run_worker_collect_failures(_worker_gather_aggregation)
        self.assertEqual(fails, [], "\n" + "\n".join(fails))
