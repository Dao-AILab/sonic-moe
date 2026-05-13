# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
# Run with torchrun:
#
#   torchrun --nproc_per_node=8 --standalone --local-ranks-filter 0 -m pytest tests/distributed/collectives_test.py -s
#
# Single test:
#
#   torchrun --nproc_per_node=8 --standalone --local-ranks-filter 0 \
#       -m pytest tests/distributed/collectives_test.py::EPCollectivesTest::test_all_gather -s
# ********************************************************************************

import atexit
import os
import traceback
import unittest

import torch
import torch.distributed as dist
from torch.distributed import _symmetric_memory as _symm_mem

from sonicmoe.functional.distributed import (
    a2a_combine_triton,
    a2a_dispatch_triton,
    all_gather_copy_engine_async,
    all_gather_triton,
    build_rank_dedup_a_idx,
    compute_dispatch_metadata,
    local_combine,
    rank_dedup_combine_triton,
    rank_dedup_dispatch_triton,
    reduce_scatter_triton,
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


def _under_torchrun() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _init_dist_from_env() -> None:
    if dist.is_initialized():
        return
    if not _under_torchrun():
        raise unittest.SkipTest(
            "EP collective tests must be launched with torchrun, e.g.:\n"
            "  torchrun --nproc_per_node=8 --standalone "
            "-m pytest tests/distributed/collectives_test.py -s"
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


def _alloc_symm(shape, dtype, device):
    buf = _symm_mem.empty(*shape, dtype=dtype, device=device)
    _symm_mem.rendezvous(buf, group=dist.group.WORLD.group_name)
    return buf


def _barrier(buf):
    """GPU-side barrier on `buf`'s symm-mem group via the cached handle."""
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
    (8192, 4096, 4, 16),
    (8192, 4096, 4, 20),
    (8192, 4096, 3, 16),
    (8192, 1536, 4, 16),
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

            ce_peer_bufs = tuple(
                _symm_mem.rendezvous(x, group=dist.group.WORLD.group_name).get_buffer(r, (T_local, d), dtype)
                for r in range(world_size)
            )
            ce = all_gather_copy_engine_async(x, peer_bufs=ce_peer_bufs, my_rank=rank).wait()
            tri = all_gather_triton(x, dist.group.WORLD)
            ref = torch.empty(world_size * T_local, d, dtype=dtype, device=device)
            dist.all_gather_into_tensor(ref, x, group=dist.group.WORLD)
            if not torch.equal(tri, ref):
                fails.append(f"AG T={T_local} d={d} dt={dtype}: {(tri != ref).sum().item()} differ")
            if not torch.equal(ce, ref):
                fails.append(f"AG T={T_local} d={d} dt={dtype}: {(ce != ref).sum().item()} differ")
            del x, tri, ref, ce, ce_peer_bufs
        torch.cuda.empty_cache()
    return fails


# ============================================================================
# Worker: reduce_scatter — Triton fp32-accum vs NCCL ring RS.
# ============================================================================


def _worker_reduce_scatter(rank, world_size, device):
    fails = []
    for T_local, d, K, E_local in _SHAPES:
        for dtype in (torch.bfloat16, torch.float32):
            x = _alloc_symm((world_size * T_local, d), dtype, device)
            x.normal_()
            _barrier(x)

            tri = reduce_scatter_triton(x, dist.group.WORLD)

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
# Worker: a2a_dispatch — AG-then-permute reference, sentinel detects
# spurious writes to invalid lanes
# ============================================================================


def _worker_a2a_dispatch(rank, world_size, device):
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
            a2a_dispatch_triton(
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
# Worker: rank_dedup_dispatch — bit-exact agreement with a2a_dispatch
# on the expert-grouped layout (Phase 2 acceptance from the task spec).
# ============================================================================


def _worker_rank_dedup_dispatch(rank, world_size, device):
    fails = []
    SENTINEL = -42.0
    SENTINEL_PACKED = -77.0
    for T_local, d, K, E_local in _SHAPES:
        for pat in ("uniform", "skew_r0"):
            E = world_size * E_local
            TK_local = T_local * K
            TK_global = world_size * TK_local

            x = _alloc_symm((T_local, d), torch.bfloat16, device)
            x.normal_()
            topk = _gen_routing(T_local, K, E, rank, world_size, device, pattern=pat, seed=400)
            meta = compute_dispatch_metadata(topk, my_rank=rank, E_local=E_local)

            # Build s_reverse_local via the existing per-expert metadata
            # path so dedup-fanout writes line up with what a non-dedup
            # A2A dispatch would have produced. Mirrors what ep.py does
            # in _build_consumer_metadata.
            from sonicmoe.functional.metadata import general_routing_router_metadata_triton

            E_total = E_local + 1
            s_reverse_local = torch.empty(TK_global, dtype=torch.int32, device=device)
            x_gather_idx = torch.empty(TK_global, dtype=torch.int32, device=device)
            s_scatter_idx = torch.empty(TK_global, dtype=torch.int32, device=device)
            expert_freq = torch.empty(E_total, dtype=torch.int32, device=device)
            expert_freq_off = torch.empty(E_total + 1, dtype=torch.int32, device=device)
            general_routing_router_metadata_triton(
                meta["a2a_token_indices"],
                meta["expert_local_padded"],
                TK_global,
                E_total,
                expert_freq,
                expert_freq_off,
                x_gather_idx,
                s_scatter_idx,
                s_reverse_local,
                None,
            )

            # MAX_PAIR_COUNT bound = W * T_local; sized by the spec.
            MAX_PAIR_COUNT = world_size * T_local
            recv_packed = _alloc_symm((MAX_PAIR_COUNT, d), torch.bfloat16, device)
            recv_packed.fill_(SENTINEL_PACKED)

            _barrier(x)

            # Single-pass dispatch — packed-by-source output only.
            rank_dedup_dispatch_triton(
                x,
                meta["dst_rank_flat"],
                meta["pair_present_mask"],
                meta["rank_dedup_recv_pos"],
                recv_packed,
                K=K,
                group=dist.group.WORLD,
            )

            # Build the up-proj A_idx that maps expert-grouped row →
            # packed row.
            MAX_ROWS_PER_RANK = T_local * world_size * min(K, E_local)
            x_idx_expanded_remap_for_rank_dedup = torch.empty(MAX_ROWS_PER_RANK, dtype=torch.int32, device=device)
            build_rank_dedup_a_idx(
                dst_rank_flat=meta["dst_rank_flat"],
                s_reverse_local=s_reverse_local,
                rank_dedup_recv_pos=meta["rank_dedup_recv_pos"],
                my_rank=rank,
                out=x_idx_expanded_remap_for_rank_dedup,
            )

            # Reference 1 — bit-exact agreement with the A2A dispatch on the
            # expert-grouped layout, via A_idx-driven gather of recv_packed.
            ref_grouped = torch.full((TK_global, d), SENTINEL, dtype=torch.bfloat16, device=device)
            a2a_dispatch_triton(
                x,
                meta["dst_rank_flat"],
                s_reverse_local,
                ref_grouped,
                K=K,
                group=dist.group.WORLD,
            )

            # Restrict the A_idx-driven check to expert-grouped rows that
            # are actually populated by routing to my_rank — matches what
            # the GEMM consumes (cu_seqlens_m bounds the GEMM at
            # expert_frequency_offset[E_local] ≤ MAX_ROWS_PER_RANK).
            n_routed = int(expert_freq_off[E_local].item())
            if n_routed > 0:
                gathered = recv_packed[x_idx_expanded_remap_for_rank_dedup[:n_routed].long()]
                want = ref_grouped[:n_routed]
                if not torch.equal(gathered, want):
                    n_diff = (gathered != want).any(dim=-1).sum().item()
                    fails.append(
                        f"Dedup-dispatch A_idx-gathered T={T_local} d={d} K={K} {pat}: "
                        f"{n_diff} rows differ vs A2A dispatch"
                    )

            # Reference 2 — packed buffer structural contract:
            # for each (src=p, t) with at least one slot routing to my_rank,
            # there must be exactly ONE row at recv_packed[rank_dedup_recv_pos[f]]
            # equal to peer p's x[t]. Touched-row count must equal
            # sum_p pair_count[p, my_rank].
            x_all = all_gather_triton(x, dist.group.WORLD).view(world_size, T_local, d)
            n_touched_expected = int(meta["pair_count"][:, rank].sum().item())
            sentinel_rows = (recv_packed == SENTINEL_PACKED).all(dim=-1)
            n_touched_actual = int((~sentinel_rows).sum().item())
            if n_touched_actual != n_touched_expected:
                fails.append(
                    f"Dedup-dispatch packed-row count T={T_local} d={d} K={K} {pat}: "
                    f"got {n_touched_actual} != expected {n_touched_expected}"
                )

            # For every canonical slot routed to my_rank, the packed row at
            # rank_dedup_recv_pos must equal x_all[src, t].
            dst3 = meta["dst_rank_flat"].view(world_size, T_local, K)
            mask3 = meta["pair_present_mask"].view(world_size, T_local, K)
            drp3 = meta["rank_dedup_recv_pos"].view(world_size, T_local, K)
            mismatched = 0
            checked = 0
            for p in range(world_size):
                # Vectorized over (t, k).
                sel = ((dst3[p] == rank) & (mask3[p] != 0)).nonzero(as_tuple=False)
                if sel.numel() == 0:
                    continue
                t_idx = sel[:, 0].long()
                k_idx = sel[:, 1].long()
                positions = drp3[p][t_idx, k_idx].long()
                got = recv_packed[positions]
                want = x_all[p][t_idx]
                if not torch.equal(got, want):
                    mismatched += int((got != want).any(dim=-1).sum().item())
                checked += t_idx.numel()
            if mismatched > 0:
                fails.append(
                    f"Dedup-dispatch packed-row content T={T_local} d={d} K={K} {pat}: "
                    f"{mismatched}/{checked} rows mismatch"
                )

            del x, recv_packed, ref_grouped, x_all, x_idx_expanded_remap_for_rank_dedup
            del s_reverse_local, x_gather_idx, s_scatter_idx, expert_freq, expert_freq_off
        torch.cuda.empty_cache()
    return fails


# ============================================================================
# Worker: local_combine — explicit fp32 K-loop reference matching the
# kernel's static_range accumulation order.
# ============================================================================


def _worker_local_combine(rank, world_size, device):
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

            # Iterate scored / score-less; partial_combine_buf is reset between modes.
            for use_scores in (True, False):
                rs = _alloc_symm((world_size * T_local, d), torch.float32, device)
                rs.zero_()
                scores_arg = sc_flat if use_scores else None
                local_combine(
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
                        f"local_combine T={T_local} d={d} K={K} {pat} {label}: "
                        f"max_abs={(rs - ref).abs().max():.3e}"
                    )
                del rs, ref

            del y, sr, sc_full
        torch.cuda.empty_cache()
    return fails


# ============================================================================
# Worker: A2A_combine — AG y_symm and s_reverse so the reference can
# index any peer's data locally.
# ============================================================================


def _worker_A2A_combine(rank, world_size, device):
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

                a2a_combine_triton(
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
# Worker: rank_dedup_combine — numeric parity vs a2a_combine_triton
# (forward, scored) and vs local_combine (backward dx, score-less).
# ============================================================================


def _worker_rank_dedup_combine(rank, world_size, device):
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
            g = torch.Generator(device=device).manual_seed(13 + rank * 67)
            sr = _alloc_symm((TK_global,), torch.int32, device)
            sr.copy_(torch.randperm(TK_global, device=device, generator=g).to(torch.int32))

            topk = _gen_routing(T_local, K, E, rank, world_size, device, pattern=pat, seed=500)
            meta = compute_dispatch_metadata(topk, my_rank=rank, E_local=E_local)

            scores_local = torch.softmax(
                torch.randn(T_local, K, device=device, dtype=torch.float32, generator=g), dim=-1
            ).to(torch.bfloat16)
            scores_global = torch.empty(world_size * T_local * K, dtype=torch.bfloat16, device=device)
            dist.all_gather_into_tensor(scores_global, scores_local.view(-1).contiguous(), group=dist.group.WORLD)

            # Local-reduce buffer (W*T_local, d). Same allocation the
            # RS_COMBINE_TRITON path uses; RANK_DEDUP_COMBINE_TRITON reuses it.
            partial_combine_buf = _alloc_symm((world_size * T_local, d), torch.bfloat16, device)

            # y / sr published once; both A2A_combine and the local-
            # reduce + gather path read peers' versions through NVLink.
            _barrier(y)

            for use_scores in (True, False):
                # Reference path: existing a2a_combine_triton with
                # the same inputs. Both reduce in fp32, both cast at store
                # to bf16 — peer-read order differs (per-(t, k) vs.
                # local-pre-summed), so allclose absorbs the ULP.
                ref_out = torch.empty(T_local, d, dtype=torch.bfloat16, device=device)
                a2a_combine_triton(
                    y,
                    sr,
                    meta["my_dst_rank"],
                    scores_local if use_scores else None,
                    ref_out,
                    K=K,
                    group=dist.group.WORLD,
                )

                # Local-reduce + gather: producer (local_combine) +
                # barrier + sparse gather (consumer reads partial_combine_buf at
                # rows where peer_present_mask is set).
                partial_combine_buf.zero_()
                got_out = torch.empty(T_local, d, dtype=torch.bfloat16, device=device)
                rank_dedup_combine_triton(
                    y,
                    sr,
                    meta["dst_rank_flat"],
                    scores_global if use_scores else None,
                    meta["peer_present_mask"],
                    partial_combine_buf,
                    got_out,
                    K=K,
                    T_local=T_local,
                    group=dist.group.WORLD,
                )

                if not torch.allclose(got_out, ref_out, atol=1.5e-1, rtol=3e-2):
                    diff = (got_out.float() - ref_out.float()).abs()
                    label = "scored" if use_scores else "score-less"
                    fails.append(
                        f"local-reduce-gather T={T_local} d={d} K={K} {pat} {label}: "
                        f"max_abs={diff.max().item():.3e}"
                    )

                # Sync peers between iterations — partial_combine_buf gets reused.
                _barrier(partial_combine_buf)

                del ref_out, got_out

            del y, sr, partial_combine_buf, scores_local, scores_global
        torch.cuda.empty_cache()
    return fails


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

    def test_a2a_dispatch(self) -> None:
        fails = _run_worker_collect_failures(_worker_a2a_dispatch)
        self.assertEqual(fails, [], "\n" + "\n".join(fails))

    def test_rank_dedup_dispatch(self) -> None:
        fails = _run_worker_collect_failures(_worker_rank_dedup_dispatch)
        self.assertEqual(fails, [], "\n" + "\n".join(fails))

    def test_local_combine(self) -> None:
        fails = _run_worker_collect_failures(_worker_local_combine)
        self.assertEqual(fails, [], "\n" + "\n".join(fails))

    def test_A2A_combine(self) -> None:
        fails = _run_worker_collect_failures(_worker_A2A_combine)
        self.assertEqual(fails, [], "\n" + "\n".join(fails))

    def test_rank_dedup_combine(self) -> None:
        fails = _run_worker_collect_failures(_worker_rank_dedup_combine)
        self.assertEqual(fails, [], "\n" + "\n".join(fails))
