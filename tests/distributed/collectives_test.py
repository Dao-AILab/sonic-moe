# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
# Run: torchrun --nproc_per_node=8 --standalone --local-ranks-filter 0 -m pytest tests/distributed/collectives_test.py -s
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
    all_gather_multimem_triton,
    all_gather_triton,
    build_a2a_peer_base,
    build_rank_dedup_a_idx,
    compute_dispatch_metadata,
    local_combine,
    rank_dedup_combine_triton,
    rank_dedup_dispatch_triton,
    reduce_scatter_multimem_triton,
    reduce_scatter_triton,
)
from tests.test_commons import TestCommons

# Hierarchical inter-node EP compute kernels — validated here on real GPUs over NVLink; the cross-node
# GIN/RDMA transport that feeds them is exercised separately by the hier_ep_*_gin tests.
from sonicmoe.functional.distributed.ep_dispatch import expand_dispatch_triton, hier_gather_rt_triton
from sonicmoe.functional.distributed.ep_combine import (
    hier_combine_gateway_reduce_triton,
    hier_combine_origin_reduce_triton,
)


# CRITICAL: symm-mem tensors freed during interpreter shutdown trigger cuMemUnmap() from a C++
# destructor while the CUDA context tears down -> std::terminate()/SIGABRT; os._exit bypasses that.
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


# Run a worker on this rank and aggregate failures across ranks via all_gather_object; replaces the
# old mp.spawn-based _spawn_and_run now that torchrun already runs this code path on every rank.


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
            # GPU-side fence: peers read x via NVLink in the AG calls below, so hdl.barrier() must
            # ensure this rank's .normal_() has landed before any peer reads its bytes.
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


# Freeing a multicast-bound symm buffer mid-test aborts the process in the symm-mem destructor
# (driver multicast-unbind failure); production never frees these mid-run, so we keep them alive too.
_MULTIMEM_KEEPALIVE: list = []


# ============================================================================
# Worker: all_gather (multimem) — multimem.st is a raw bit copy so it must match dist.all_gather
# exactly; requires NVLink multicast (SHARP/MNNVL), skipped otherwise.
# ============================================================================


def _worker_all_gather_multimem(rank, world_size, device):
    fails = []
    for T_local, d, K, E_local in _SHAPES:
        for dtype in (torch.bfloat16, torch.float32):
            x = _alloc_symm((T_local, d), dtype, device)
            x.normal_()
            _barrier(x)

            # multimem AG allocates a multicast-backed symm output internally.
            if _symm_mem.rendezvous(x, group=dist.group.WORLD.group_name).multicast_ptr == 0:
                _MULTIMEM_KEEPALIVE.append(x)  # never freed mid-loop
                continue  # no multicast support on this fabric

            tri = all_gather_multimem_triton(x, dist.group.WORLD)
            ref = torch.empty(world_size * T_local, d, dtype=dtype, device=device)
            dist.all_gather_into_tensor(ref, x, group=dist.group.WORLD)
            if not torch.equal(tri, ref):
                fails.append(f"AG-mm T={T_local} d={d} dt={dtype}: {(tri != ref).sum().item()} differ")
            # x and tri are multicast symm buffers — keep alive, do not free.
            _MULTIMEM_KEEPALIVE.extend((x, tri))
            del ref
    return fails


# ============================================================================
# Worker: reduce_scatter (multimem) — ld_reduce reduces in the buffer dtype with hardware-defined
# order, so compare with a tolerance (not bit-exact). Requires NVLink multicast.
# ============================================================================


def _worker_reduce_scatter_multimem(rank, world_size, device):
    fails = []
    for T_local, d, K, E_local in _SHAPES:
        for dtype in (torch.bfloat16, torch.float32):
            x = _alloc_symm((world_size * T_local, d), dtype, device)
            x.normal_()
            _barrier(x)

            if _symm_mem.rendezvous(x, group=dist.group.WORLD.group_name).multicast_ptr == 0:
                _MULTIMEM_KEEPALIVE.append(x)  # never freed mid-loop
                continue  # no multicast support on this fabric

            tri = reduce_scatter_multimem_triton(x, dist.group.WORLD)

            # Reference via NCCL all-gather into a REGULAR tensor (no symm alloc,
            # no extra unicast peer-mapping of the multicast x), then fp32 reduce.
            x_all = torch.empty(world_size * world_size * T_local, d, dtype=dtype, device=device)
            dist.all_gather_into_tensor(x_all, x.contiguous(), group=dist.group.WORLD)
            x_all = x_all.view(world_size, world_size * T_local, d)
            ref_fp32 = torch.zeros(T_local, d, dtype=torch.float32, device=device)
            for p in range(world_size):
                ref_fp32 += x_all[p, rank * T_local : (rank + 1) * T_local].to(torch.float32)
            ref = ref_fp32.to(dtype)

            # bf16 hardware reduction accumulates in bf16 → looser tol than fp32.
            atol, rtol = (2e-1, 5e-2) if dtype == torch.bfloat16 else (1e-3, 1e-4)
            if not torch.allclose(tri, ref, atol=atol, rtol=rtol):
                diff = (tri.float() - ref.float()).abs()
                fails.append(
                    f"RS-mm T={T_local} d={d} dt={dtype}: "
                    f"{(~torch.isclose(tri, ref, atol=atol, rtol=rtol)).sum().item()} differ, "
                    f"max_diff={diff.max().item():.3e}"
                )
            # x is a multicast symm buffer — keep alive, do not free. x_all/tri/ref
            # are regular tensors and are safe to drop.
            _MULTIMEM_KEEPALIVE.append(x)
            del tri, ref, x_all, ref_fp32
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
# on the expert-grouped layout.
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

            # Build s_reverse_local via the existing per-expert metadata path (mirrors what ep.py's
            # _build_consumer_metadata does) so dedup-expand writes line up with a non-dedup A2A dispatch.
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

            # Restrict the A_idx-driven check to expert-grouped rows actually populated by routing to
            # my_rank — matches what the GEMM consumes (bounded by expert_frequency_offset[E_local]).
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

            # Reference 2 — structural contract: each (src,t) routing to my_rank has exactly ONE row at
            # recv_packed[rank_dedup_recv_pos[f]]; touched-row count must equal sum_p pair_count[p, my_rank].
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

                # Reference: same K-order accumulation as the kernel, which masks non-mine slots with
                # other=0.0 (contribution w*0=0); mirror that with torch.where on gathered rows.
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

            # Single fence after both symm writes: group-level hdl.barrier() fences all preceding
            # GPU work on this rank's stream, so it covers both y and sr before peers read either.
            _barrier(y)
            y_all = all_gather_triton(y, dist.group.WORLD)
            s_all = all_gather_triton(sr, dist.group.WORLD)

            # Runtime-peer-addressing A2A peer-base tensors (constant addresses;
            # built once per shape/pattern, scores-independent).
            peer_y_base, peer_s_base, a2a_rank = build_a2a_peer_base(y, sr, group=dist.group.WORLD)

            for use_scores in (True, False):
                scores_arg = scores_local if use_scores else None
                label = "scored" if use_scores else "score-less"
                out = torch.empty(T_local, d, dtype=torch.bfloat16, device=device)

                a2a_combine_triton(
                    y,
                    sr,
                    meta["my_dst_rank"],
                    scores_arg,
                    out,
                    K=K,
                    group=dist.group.WORLD,
                    peer_y_base=peer_y_base,
                    peer_s_base=peer_s_base,
                    my_rank=a2a_rank,
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
                    fails.append(f"Gather T={T_local} d={d} K={K} {pat} {label}: max_abs={max_abs:.3e}")
                del out

            del y, sr, y_all, s_all, peer_y_base, peer_s_base
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

            # Cached peer-buf tuples for the gather's single-contributor reads (matches production).
            # Resolving them per-call and dropping mid-flight is a use-after-free over NVLink.
            y_peer_bufs = tuple(
                _symm_mem.rendezvous(y, group=dist.group.WORLD.group_name).get_buffer(r, (TK_global, d), torch.bfloat16)
                for r in range(world_size)
            )
            s_reverse_peer_bufs = tuple(
                _symm_mem.rendezvous(sr, group=dist.group.WORLD.group_name).get_buffer(r, (TK_global,), torch.int32)
                for r in range(world_size)
            )

            for use_scores in (True, False):
                # Reference: a2a_combine_triton on the same inputs. Both reduce in fp32 and cast to bf16
                # at store, but peer-read order differs (per-(t,k) vs. pre-summed), so allclose absorbs the ULP.
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

                # Selective-dedup gather: single_row must equal peer q's s_reverse at that slot. Since
                # this test uses RANDOM sr (not real routing), derive single_row from sr, not _build_single_row.
                sr_all = torch.empty(world_size * TK_global, dtype=torch.int32, device=device)
                dist.all_gather_into_tensor(sr_all, sr, group=dist.group.WORLD)
                sr_all = sr_all.view(world_size, TK_global)
                _csk = meta["combine_single_k"].to(torch.int64)  # (W, T_local)
                _t = torch.arange(T_local, device=device, dtype=torch.int64)
                _pos = rank * TK_local + _t[None, :] * K + _csk.clamp(min=0)  # (W, T_local)
                _q = torch.arange(world_size, device=device, dtype=torch.int64)[:, None]
                single_row = torch.where(_csk >= 0, sr_all[_q, _pos], torch.zeros_like(_pos)).to(torch.int32)
                rank_dedup_kw = dict(
                    mine_slot_idx=meta["mine_slot_idx"],
                    mine_count=meta["mine_count"],
                    combine_contrib_C=meta["combine_contrib_C"],
                    combine_work_list=meta["combine_work_list_multi"],
                    combine_work_count=meta["combine_work_count_multi"],
                    combine_single_k=meta["combine_single_k"],
                    single_row=single_row,
                    y_peer_bufs=y_peer_bufs,
                    s_reverse_peer_bufs=s_reverse_peer_bufs,
                )
                modes = [("rank_dedup", rank_dedup_kw)]
                for prod, mode_kw in modes:
                    partial_combine_buf.zero_()
                    _barrier(partial_combine_buf)
                    got_out = torch.empty(T_local, d, dtype=torch.bfloat16, device=device)
                    rank_dedup_combine_triton(
                        y,
                        sr,
                        scores_global if use_scores else None,
                        meta["peer_present_mask"],
                        partial_combine_buf,
                        got_out,
                        K=K,
                        T_local=T_local,
                        group=dist.group.WORLD,
                        **mode_kw,
                    )

                    if not torch.allclose(got_out, ref_out, atol=1.5e-1, rtol=3e-2):
                        diff = (got_out.float() - ref_out.float()).abs()
                        label = "scored" if use_scores else "score-less"
                        fails.append(
                            f"local-reduce-gather[{prod}] T={T_local} d={d} K={K} {pat} {label}: "
                            f"max_abs={diff.max().item():.3e}"
                        )

                    # Sync peers between iterations — partial_combine_buf reused.
                    _barrier(partial_combine_buf)

                del ref_out, got_out

            # Peer-buf aliases first, then the symm tensors they alias — reversing this teardown order
            # cuMemUnmaps pages a peer alias still references (~AllocationRef "invalid peer access").
            del y_peer_bufs, s_reverse_peer_bufs
            del y, sr, partial_combine_buf, scores_local, scores_global
        torch.cuda.empty_cache()
    return fails


# ============================================================================
# Hierarchical inter-node EP COMPUTE, isolated from GIN/RDMA transport: the buffers a cross-node GIN
# put would land are filled locally over NVLink instead, so this tests the receiver/reduce kernels alone.
# ============================================================================
def _hier_global_routing(W, num_nodes, node_size, T_local, K, E_local, seed):
    """Deterministic global routing (identical on every rank), biased remote so cross-node slots exist."""
    g = torch.Generator().manual_seed(seed)
    E = W * E_local
    out = torch.empty(W, T_local, K, dtype=torch.int64)
    for r in range(W):
        my_node = r // node_size
        remote_ranks = [rk for rk in range(W) if rk // node_size != my_node]
        for t in range(T_local):
            picks = set()
            n_remote = K // 2 if num_nodes > 1 else 0
            while len(picks) < n_remote and remote_ranks:
                rk = remote_ranks[int(torch.randint(0, len(remote_ranks), (1,), generator=g))]
                picks.add(int(torch.randint(rk * E_local, (rk + 1) * E_local, (1,), generator=g)))
            while len(picks) < K:
                picks.add(int(torch.randint(0, E, (1,), generator=g)))
            out[r, t] = torch.tensor(sorted(picks)[:K], dtype=torch.int64)
    return out


def _hier_node_sizes(world_size):
    """node_size choices that divide the world into >= 2 simulated nodes (so remote slots exist)."""
    return [ns for ns in (1, 2, 4) if world_size % ns == 0 and world_size // ns >= 2]


def _worker_hier_dispatch_compute(rank, world_size, device):
    """HIER dispatch RECEIVER COMPUTE vs the oracle: dst_node_buffer is filled locally (standing in for
    the GIN put), exercising both receivers — the two-kernel pull+expand and the unified RT gather."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hier_ep_reference import compute_hier_dispatch_reference, recv_gpu

    fails = []
    T_local, K, E_local, d = 6, 4, 2, 128
    if world_size * T_local >= 256:
        T_local = max(1, 255 // world_size)  # keep the token id (rank*T_local+t) bf16-exact
    for node_size in _hier_node_sizes(world_size):
        num_nodes = world_size // node_size
        my_node, my_local = rank // node_size, rank % node_size
        topk = _hier_global_routing(world_size, num_nodes, node_size, T_local, K, E_local, seed=400)
        ref = compute_hier_dispatch_reference(topk, num_nodes, node_size, E_local)
        meta = compute_dispatch_metadata(topk.to(torch.int32).to(device), my_rank=rank, E_local=E_local,
                                         emit_combine=False, emit_hier=True, node_size=node_size)
        total = int(ref.recv_rows_per_rank[rank])

        # my x window: token id = rank*T_local + t, broadcast over d
        x = _alloc_symm((T_local, d), torch.bfloat16, device)
        for t in range(T_local):
            x[t].fill_(float(rank * T_local + t))
        # my dst_node_buffer: fill the rows the oracle says land at me (stands in for the GIN put), -1 else
        ROWS = max(ref.DST_NODE_BUF_ROWS, 1)
        dnb = _alloc_symm((ROWS, d), torch.bfloat16, device)
        dnb.fill_(-1.0)
        np_m = ref.node_present_mask.view(world_size, T_local, K)
        dn_m = ref.dst_node_flat.view(world_size, T_local, K)
        ds_m = ref.dst_slot.view(world_size, T_local, K)
        for src in range(world_size):
            for t in range(T_local):
                for k in range(K):
                    if int(np_m[src, t, k]) and recv_gpu(src, int(dn_m[src, t, k]), node_size) == rank:
                        dnb[int(ds_m[src, t, k])].fill_(float(src * T_local + t))
        _barrier(x)
        _barrier(dnb)

        hdl_x = _symm_mem.rendezvous(x, group=dist.group.WORLD.group_name)
        hdl_d = _symm_mem.rendezvous(dnb, group=dist.group.WORLD.group_name)
        x_peers = tuple(hdl_x.get_buffer(r, (T_local, d), torch.bfloat16) for r in range(world_size))
        dnb_peers = tuple(hdl_d.get_buffer(r, (ROWS, d), torch.bfloat16) for r in range(world_size))

        # oracle golden: each canonical slot routed to me -> its token id at recv_pos
        pp = ref.pair_present_mask.view(world_size, T_local, K)
        dr = ref.dst_rank_flat.view(world_size, T_local, K)
        pos = ref.rank_dedup_recv_pos.view(world_size, T_local, K)
        golden = torch.full((max(total, 1), d), -3.0, device=device, dtype=torch.bfloat16)
        for src in range(world_size):
            for t in range(T_local):
                for k in range(K):
                    if int(pp[src, t, k]) and int(dr[src, t, k]) == rank:
                        golden[int(pos[src, t, k])].fill_(float(src * T_local + t))

        if total > 0:
            # receiver 1: two-kernel same-node pull + remote expand
            recv1 = torch.full((total, d), -2.0, device=device, dtype=torch.bfloat16)
            rank_dedup_dispatch_triton(
                x, meta["dst_rank_flat"], meta["pair_present_mask"], meta["rank_dedup_recv_pos"],
                recv1, K=K, group=None, peer_bufs=x_peers, my_rank=rank, node_size=node_size)
            expand_dispatch_triton(
                dnb_peers, meta["pair_present_mask"], meta["is_local_slot"], meta["dst_rank_flat"],
                meta["dst_slot"], meta["rank_dedup_recv_pos"], recv1, K, my_rank=rank, node_size=node_size)
            torch.cuda.synchronize()
            if not torch.equal(recv1.cpu(), golden.cpu()):
                fails.append(f"hier dispatch pull+expand node_size={node_size} num_nodes={num_nodes}: "
                             "recv_packed mismatch vs oracle")

            # receiver 2: unified runtime-peer-addressed gather (same-node x + remote dst_node_buffer via LSA)
            x_lsa = torch.tensor([x_peers[my_node * node_size + l].data_ptr() for l in range(node_size)],
                                 dtype=torch.int64, device=device)
            dnb_lsa = torch.tensor([dnb_peers[my_node * node_size + l].data_ptr() for l in range(node_size)],
                                   dtype=torch.int64, device=device)
            recv2 = torch.full((total, d), -2.0, device=device, dtype=torch.bfloat16)
            hier_gather_rt_triton(
                x_lsa, dnb_lsa, meta["pair_present_mask"], meta["is_local_slot"], meta["dst_rank_flat"],
                meta["dst_slot"], meta["rank_dedup_recv_pos"], recv2, K, rank, world_size, node_size)
            torch.cuda.synchronize()
            if not torch.equal(recv2.cpu(), golden.cpu()):
                fails.append(f"hier dispatch unified-gather node_size={node_size} num_nodes={num_nodes}: "
                             "recv_packed mismatch vs oracle")

        _barrier(x)  # all peers done reading before teardown
        del x_peers, dnb_peers, hdl_x, hdl_d, x, dnb
        torch.cuda.empty_cache()
    return fails


def _worker_hier_combine_compute(rank, world_size, device):
    """HIER combine COMPUTE vs the oracle: gateway NVLink-reduce -> (GIN stripe-put stand-in) -> origin
    NVLink-reduce. Present peer q contributes 2^q (decodable bitmask); out[t] must equal the oracle sum."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hier_ep_reference import compute_hier_combine_reference, rank_of

    fails = []
    if world_size > 8:
        return fails  # 2^q fill needs q < W with 2^q bf16-exact (<= 256 => W <= 8)
    T_local, K, E_local, d = 6, 4, 2, 64
    POISON = -100.0
    for node_size in _hier_node_sizes(world_size):
        num_nodes = world_size // node_size
        my_node, my_local = rank // node_size, rank % node_size
        topk = _hier_global_routing(world_size, num_nodes, node_size, T_local, K, E_local, seed=500)
        cref = compute_hier_combine_reference(topk, num_nodes, node_size, E_local)
        present_all = cref.peer_present_mask.to(device)        # (R, q, t) int8 — read by both reduces
        contrib = cref.contrib_node_mask.to(device)            # (W, num_nodes) int8 — origin contrib gate
        present = cref.peer_present_mask                       # cpu copy for fill + golden
        RECV_ROWS = max(T_local * (num_nodes - 1), 1)

        # golden out[t] = Σ_q present[my_rank][q][t] * 2^q
        golden = torch.zeros(T_local, dtype=torch.float32)
        for q in range(world_size):
            for t in range(T_local):
                if int(present[rank, q, t]):
                    golden[t] += float(2 ** q)

        # my partial: row (R*T_local+t) = 2^my_rank when present[R][my_rank][t] else POISON
        partial = _alloc_symm((world_size * T_local, d), torch.bfloat16, device)
        partial.fill_(POISON)
        pv = float(2 ** rank)
        for R in range(world_size):
            for t in range(T_local):
                if int(present[R, rank, t]):
                    partial[R * T_local + t].fill_(pv)
        send_buf = _alloc_symm((RECV_ROWS, d), torch.bfloat16, device)
        send_buf.fill_(0.0)
        _barrier(partial)
        _barrier(send_buf)

        hdl_p = _symm_mem.rendezvous(partial, group=dist.group.WORLD.group_name)
        hdl_s = _symm_mem.rendezvous(send_buf, group=dist.group.WORLD.group_name)
        partial_peers = tuple(hdl_p.get_buffer(r, (world_size * T_local, d), torch.bfloat16)
                              for r in range(world_size))
        send_peers = tuple(hdl_s.get_buffer(r, (RECV_ROWS, d), torch.bfloat16) for r in range(world_size))
        node_partials = tuple(partial_peers[my_node * node_size + j] for j in range(node_size))

        # 1) gateway reduce: my node-local peers' partials -> my send_buf
        hier_combine_gateway_reduce_triton(
            node_partials, present_all, send_buf, T_local=T_local, node_size=node_size,
            num_nodes=num_nodes, W=world_size, my_rank=rank, d=d)
        torch.cuda.synchronize()
        _barrier(send_buf)

        # 2) stand in for the GIN stripe-put: recv stripes <- serving gateways' send stripes (NVLink).
        #    Gateway for source-node g is rank_of(g, my_local); s_send/s_recv are the closed-form stripe indices.
        recv_buf = torch.zeros(RECV_ROWS, d, device=device, dtype=torch.bfloat16)
        for g in range(num_nodes):
            if g == my_node or not int(contrib[rank, g]):
                continue
            C = rank_of(g, my_local, node_size)
            s_send = my_node if my_node < g else my_node - 1
            s_recv = g if g < my_node else g - 1
            recv_buf[s_recv * T_local:(s_recv + 1) * T_local].copy_(
                send_peers[C][s_send * T_local:(s_send + 1) * T_local])
        torch.cuda.synchronize()

        # 3) origin reduce: node-local peers (present-gated) + remote stripes (contrib-gated) -> out
        out = torch.full((T_local, d), -1.0, device=device, dtype=torch.bfloat16)
        hier_combine_origin_reduce_triton(
            node_partials, present_all, contrib, recv_buf, out, T_local=T_local, node_size=node_size,
            num_nodes=num_nodes, W=world_size, my_rank=rank, d=d)
        torch.cuda.synchronize()

        want = golden[:, None].expand(T_local, d)
        got = out.to(torch.float32).cpu()
        if not torch.equal(got, want):
            n_bad = int((got != want).any(dim=1).sum())
            fails.append(f"hier combine node_size={node_size} num_nodes={num_nodes}: "
                         f"{n_bad}/{T_local} tokens mismatch vs oracle")

        _barrier(partial)
        del partial_peers, send_peers, node_partials, hdl_p, hdl_s, partial, send_buf
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
        # Coarse cross-rank sync so a slow rank doesn't collide with the previous test's tail traffic;
        # this is a process-level rendezvous (not tied to any symm tensor), so dist.barrier() is right here.
        dist.barrier()

    def test_all_gather(self) -> None:
        fails = _run_worker_collect_failures(_worker_all_gather)
        self.assertEqual(fails, [], "\n" + "\n".join(fails))

    def test_reduce_scatter(self) -> None:
        fails = _run_worker_collect_failures(_worker_reduce_scatter)
        self.assertEqual(fails, [], "\n" + "\n".join(fails))

    def test_all_gather_multimem(self) -> None:
        fails = _run_worker_collect_failures(_worker_all_gather_multimem)
        self.assertEqual(fails, [], "\n" + "\n".join(fails))

    def test_reduce_scatter_multimem(self) -> None:
        fails = _run_worker_collect_failures(_worker_reduce_scatter_multimem)
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

    def test_hier_dispatch_compute(self) -> None:
        fails = _run_worker_collect_failures(_worker_hier_dispatch_compute)
        self.assertEqual(fails, [], "\n" + "\n".join(fails))

    def test_hier_combine_compute(self) -> None:
        fails = _run_worker_collect_failures(_worker_hier_combine_compute)
        self.assertEqual(fails, [], "\n" + "\n".join(fails))
