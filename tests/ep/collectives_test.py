# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import _symmetric_memory as _symm_mem

from sonicmoe.functional.ep import a2a_dispatch_pull
from sonicmoe.functional.ep import all_gather as triton_all_gather
from sonicmoe.functional.ep import compute_dispatch_metadata, gather_aggregation, rs_aggregation
from tests.test_commons import TestCommons


_SEED = 0


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_world_size() -> int:
    n = int(os.environ.get("EP_TEST_WORLD_SIZE", str(min(torch.cuda.device_count(), 8))))
    if n < 2:
        raise unittest.SkipTest(f"EP collective tests require ≥2 GPUs (have {n})")
    return n


def _setup_dist(rank: int, world_size: int, port: str) -> None:
    os.environ.update(MASTER_ADDR="localhost", MASTER_PORT=port, RANK=str(rank), WORLD_SIZE=str(world_size))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=torch.device(f"cuda:{rank}"))
    _symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)


def _alloc_symm(shape, dtype, device):
    buf = _symm_mem.empty(*shape, dtype=dtype, device=device)
    _symm_mem.rendezvous(buf, group=dist.group.WORLD.group_name)
    return buf


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
# Subprocess driver: spawn N workers, collect failures via SimpleQueue.
# Worker functions must be module-level (picklable for spawn context).
# ============================================================================


def _subprocess_entry(rank, world_size, worker_fn, port, error_queue):
    try:
        _setup_dist(rank, world_size, port=port)
        device = torch.device(f"cuda:{rank}")
        _set_seed(_SEED + rank)
        failures = worker_fn(rank, world_size, device)
        error_queue.put((rank, list(failures)))
    except Exception as e:
        import traceback

        error_queue.put((rank, [f"EXCEPTION: {e}\n{traceback.format_exc()}"]))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _spawn_and_run(world_size, worker_fn, port, timeout=600):
    """Spawn `world_size` processes running worker_fn(rank, world_size, device).
    Returns aggregated list of failure strings from all ranks."""
    ctx = mp.get_context("spawn")
    error_queue = ctx.SimpleQueue()
    procs = []
    for rank in range(world_size):
        p = ctx.Process(target=_subprocess_entry, args=(rank, world_size, worker_fn, port, error_queue))
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=timeout)

    failures = []
    for i, p in enumerate(procs):
        if p.is_alive():
            p.terminate()
            p.join()
            failures.append(f"[r{i}] subprocess timed out after {timeout}s")
        elif p.exitcode not in (0, None):
            failures.append(f"[r{i}] subprocess exited with code {p.exitcode}")

    while not error_queue.empty():
        rank, fails = error_queue.get()
        failures.extend(f"[r{rank}] {f}" for f in fails)
    return failures


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
            tri = triton_all_gather(x, dist.group.WORLD)
            ref = torch.empty(world_size * T_local, d, dtype=dtype, device=device)
            dist.all_gather_into_tensor(ref, x, group=dist.group.WORLD)
            if not torch.equal(tri, ref):
                fails.append(f"AG T={T_local} d={d} dt={dtype}: " f"{(tri != ref).sum().item()} differ")
            del x, tri, ref
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

            dist.barrier()
            a2a_dispatch_pull(x, meta["dst_rank_flat"], meta["slot_flat_per_rank"], recv, K=K, group=dist.group.WORLD)

            # Reference: AG x_symm, then scatter rows. Slots not destined for
            # this rank should retain the sentinel (kernel early-returns).
            x_all = triton_all_gather(x, dist.group.WORLD)
            ref = torch.full_like(recv, SENTINEL)
            valid = torch.nonzero(meta["dst_rank_flat"] == rank).flatten()
            src = (valid // TK_local).long()
            t_loc = ((valid - src * TK_local) // K).long()
            slot = meta["slot_flat_per_rank"][valid].long()
            ref.view(world_size * TK_local, d)[src * TK_local + slot] = x_all[src * T_local + t_loc]

            if not torch.equal(recv, ref):
                fails.append(
                    f"A2A T={T_local} d={d} K={K} {pat}: " f"{(recv != ref).any(dim=-1).sum().item()} rows differ"
                )
            del x, recv, ref, x_all
        torch.cuda.empty_cache()
    return fails


# ============================================================================
# Worker: rs_aggregation — explicit fp32 K-loop reference matching the
# kernel's static_range accumulation order. allclose absorbs any FMA fusion
# difference between Triton and PyTorch elementwise.
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

            # AG per-rank (T_local, K) scores into a flat (TK_global,) tensor.
            sc_local = torch.softmax(
                torch.randn(T_local, K, device=device, dtype=torch.float32, generator=g), dim=-1
            ).to(torch.bfloat16)
            sc_full = torch.empty((world_size, T_local, K), dtype=torch.bfloat16, device=device)
            dist.all_gather_into_tensor(sc_full.view(-1), sc_local.view(-1).contiguous(), group=dist.group.WORLD)
            sc_flat = sc_full.view(-1).contiguous()

            rs = _alloc_symm((world_size * T_local, d), torch.float32, device)
            rs.zero_()
            rs_aggregation(y, sr, meta["dst_rank_flat"], sc_flat, rs, K, T_local)

            # Reference: same K-order accumulation. The kernel masks non-mine
            # slots with other=0.0, so contribution is score * 0 = 0; mirror
            # that with torch.where on the gathered rows.
            ref = torch.zeros_like(rs)
            ht = torch.arange(world_size * T_local, device=device, dtype=torch.int64)
            for k in range(K):
                f = ht * K + k
                is_mine = meta["dst_rank_flat"][f] == rank
                rows = y[sr[f].long()].to(torch.float32)
                rows = torch.where(is_mine[:, None], rows, torch.zeros_like(rows))
                ref += sc_flat[f].to(torch.float32)[:, None] * rows

            if not torch.allclose(rs, ref, atol=1e-4, rtol=1e-3):
                fails.append(f"RS T={T_local} d={d} K={K} {pat}: " f"max_abs={(rs - ref).abs().max():.3e}")
            del y, sr, rs, sc_full
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
        # y_all reference is (W * TK_global, d). Skip if it would exceed 8 GB.
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
            scores = torch.softmax(
                torch.randn(T_local, K, device=device, dtype=torch.float32, generator=g), dim=-1
            ).to(torch.bfloat16)
            out = torch.empty(T_local, d, dtype=torch.bfloat16, device=device)

            dist.barrier()
            gather_aggregation(y, sr, meta["my_dst_rank"], pos_2d, scores, out, K=K, group=dist.group.WORLD)

            y_all = triton_all_gather(y, dist.group.WORLD)
            s_all = triton_all_gather(sr, dist.group.WORLD)
            ref_acc = torch.zeros(T_local, d, dtype=torch.float32, device=device)
            for k in range(K):
                peer = meta["my_dst_rank"][:, k].long()
                pos = pos_2d[:, k].long()
                s_peer = s_all[peer * TK_global + pos].long()
                row = y_all[peer * TK_global + s_peer].to(torch.float32)
                ref_acc += scores[:, k].to(torch.float32)[:, None] * row
            ref = ref_acc.to(torch.bfloat16)

            if not torch.allclose(out, ref, atol=1e-2, rtol=1e-2):
                max_abs = (out.float() - ref.float()).abs().max().item()
                fails.append(f"Gather T={T_local} d={d} K={K} {pat}: " f"max_abs={max_abs:.3e}")
            del y, sr, out, y_all, s_all
        torch.cuda.empty_cache()
    return fails


# ============================================================================
# Test class
# ============================================================================


class EPCollectivesTest(TestCommons):
    def test_all_gather(self) -> None:
        fails = _spawn_and_run(_get_world_size(), _worker_all_gather, port="29556")
        self.assertEqual(fails, [], "\n" + "\n".join(fails))

    def test_a2a_dispatch_pull(self) -> None:
        fails = _spawn_and_run(_get_world_size(), _worker_a2a_pull, port="29557")
        self.assertEqual(fails, [], "\n" + "\n".join(fails))

    def test_rs_aggregation(self) -> None:
        fails = _spawn_and_run(_get_world_size(), _worker_rs_aggregation, port="29558")
        self.assertEqual(fails, [], "\n" + "\n".join(fails))

    def test_gather_aggregation(self) -> None:
        fails = _spawn_and_run(_get_world_size(), _worker_gather_aggregation, port="29559")
        self.assertEqual(fails, [], "\n" + "\n".join(fails))
