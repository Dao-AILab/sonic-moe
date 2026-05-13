# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
#
# Triton + symm-mem collectives shared by the dispatch and combine
# primitives in sibling submodules. Provides the AG / RS primitives, copy-
# engine fast path, autotune helpers, and symm-mem rendezvous wrappers.
#
# Stateless: no global caches, no autograd. Every function takes tensors in,
# launches kernels, returns tensors out. The caller owns all buffer
# allocation and barrier placement.
#
# Autotuning notes:
#   - All non-trivial kernels are autotuned over BLOCK_SIZE / num_warps /
#     num_stages. Configs that would produce grid_y > 65535 for the given
#     problem size are pruned at autotune time via prune_configs_by.
#   - EVEN_K is computed via @triton.heuristics, so problem sizes that happen
#     to be block-aligned automatically take the unmasked fast path. We do
#     NOT pad source buffers up to a block multiple — that would require
#     out-of-bounds loads from peer symm-mem buffers, which is unsafe.
#
# Naming convention used throughout:
#   T_local     tokens per rank
#   K           top-K experts per token
#   TK_local    T_local * K
#   W           EP world size
#   TK_global   W * TK_local
#   E_local     experts per rank
#
# Dispatch / combine primitives provided here:
#   AG dispatch              all_gather_triton, all_gather_copy_engine_async
#   A2A dispatch             a2a_dispatch_triton
#   RANK_DEDUP dispatch       rank_dedup_dispatch_triton
#                            (provably minimum row count: one row per
#                            (token, peer) pair that has ≥ 1 routed expert)
#   A2A combine              a2a_combine_triton
#   RS combine               local_combine + reduce_scatter_triton
#   RANK_DEDUP combine    rank_dedup_combine_triton
#                            (local_combine local pre-sum + sparse
#                            per-token gather guided by peer_present_mask;
#                            same minimum row count as RANK_DEDUP dispatch
#                            in expectation)
# ********************************************************************************

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from cuda.bindings import driver
from torch.distributed import _symmetric_memory as symm_mem


_CUDA_MAX_GRID_Y = 65535

# AG and RS use different best warp choices, so keep their autotune
# spaces separate instead of sharing one _IO_BLOCK_CONFIGS list.
#
# AG is mostly peer memcpy: 4/8 warps are often enough, with 16 kept only
# as a large-tile fallback.
_AG_BLOCK_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE": 16384}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE": 16384}, num_warps=16, num_stages=3),
    triton.Config({"BLOCK_SIZE": 32768}, num_warps=16, num_stages=3),
]


def _prune_by_grid_y(numel_key: str):
    """Return a dict suitable for triton.autotune's `prune_configs_by` arg.
    Drops configs whose grid_y along the tile axis would exceed CUDA's 65535
    limit for the given problem size.

    Note: Triton's autotuner passes positional kernel args via `named_args`
    (a dict mapping arg name → value) and keyword kernel args via `**kwargs`.
    Wrappers in this file pass `numel_per_rank=...` etc. as kwargs, so the
    callback must consult both."""

    def _prune(configs, named_args, **kwargs):
        if numel_key in kwargs:
            numel = kwargs[numel_key]
        elif numel_key in named_args:
            numel = named_args[numel_key]
        else:
            # Don't know the value — keep all configs.
            return list(configs)
        kept = []
        for cfg in configs:
            bs = cfg.kwargs["BLOCK_SIZE"]
            if triton.cdiv(numel, bs) <= _CUDA_MAX_GRID_Y:
                kept.append(cfg)
        if not kept:
            kept = [max(configs, key=lambda c: c.kwargs["BLOCK_SIZE"])]
        return kept

    return {"early_config_prune": _prune}


def _prune_block_d_vs_d(configs, named_args, **kwargs):
    d = kwargs.get("d", named_args.get("d"))
    if d is None:
        return list(configs)
    kept = []
    for cfg in configs:
        bd = cfg.kwargs["BLOCK_D"]
        if bd > max(d, 1):
            continue  # BLOCK_D > d wastes lanes
        if triton.cdiv(d, bd) > _CUDA_MAX_GRID_Y:
            continue  # CUDA grid_y limit
        kept.append(cfg)
    if not kept:
        # Fall back to the largest BLOCK_D ≤ d if everything got dropped.
        valid_for_d = [c for c in configs if c.kwargs["BLOCK_D"] <= max(d, 1)]
        kept = [max(valid_for_d or configs, key=lambda c: c.kwargs["BLOCK_D"])]
    return kept


@triton.autotune(
    configs=_AG_BLOCK_CONFIGS,
    key=["numel_per_rank", "world_size"],
    prune_configs_by=_prune_by_grid_y("numel_per_rank"),
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["numel_per_rank"] % args["BLOCK_SIZE"] == 0,
    }
)
@triton.jit
def _all_gather_kernel(
    buf_tuple,
    output_ptr,
    numel_per_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """output[r*N:(r+1)*N] ← buf_tuple[r][0:N]"""
    pid_rank = tl.program_id(0)
    pid_tile = tl.program_id(1)
    offs = pid_tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    for i in tl.static_range(world_size):
        if pid_rank == i:
            if EVEN_K:
                data = tl.load(buf_tuple[i] + offs)
                tl.store(output_ptr + i * numel_per_rank + offs, data)
            else:
                mask = offs < numel_per_rank
                data = tl.load(buf_tuple[i] + offs, mask=mask)
                tl.store(output_ptr + i * numel_per_rank + offs, data, mask=mask)


# ============================================================================
# Reduce-scatter (sum) — symmetric to all_gather. Each rank reads peers'
# my_rank-th chunk via NVLink and accumulates locally. fp32 accumulation,
# implicit cast on store. No NCCL.
#
# Shape contract: x_symm is (W*T_local, ...) — i.e. the same total-size
# input that NCCL's reduce_scatter_tensor expects. Output is (T_local, ...).
#
# Determinism: summation order is rank 0 → W-1, fixed at compile time via
# tl.static_range. Bitwise reproducible across runs for the same inputs;
# NCCL ring RS is not, because the algorithm reorders depending on topology.
# ============================================================================

_RS_BLOCK_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE": 16384}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE": 16384}, num_warps=16, num_stages=3),
    triton.Config({"BLOCK_SIZE": 32768}, num_warps=16, num_stages=3),
]


@triton.autotune(
    configs=_RS_BLOCK_CONFIGS,
    key=["numel_per_rank", "world_size"],
    prune_configs_by=_prune_by_grid_y("numel_per_rank"),
)
@triton.heuristics({"EVEN_K": lambda args: args["numel_per_rank"] % args["BLOCK_SIZE"] == 0})
@triton.jit
def _reduce_scatter_kernel(
    buf_tuple,
    output_ptr,
    numel_per_rank: tl.constexpr,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """output[0:N] ← Σ_{r in 0..W} buf_tuple[r][my_rank*N : (my_rank+1)*N]

    AG fans out: 1 program reads 1 peer chunk, writes 1 output chunk.
    RS fans in:  1 program reads W peer chunks, writes 1 local chunk.
    fp32 accumulation; tl.store implicitly casts to output dtype.
    """
    pid_tile = tl.program_id(0)
    offs = pid_tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    base = my_rank * numel_per_rank
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    if EVEN_K:
        for i in tl.static_range(world_size):
            data = tl.load(buf_tuple[i] + base + offs)
            acc += data.to(tl.float32)
        tl.store(output_ptr + offs, acc)
    else:
        mask = offs < numel_per_rank
        for i in tl.static_range(world_size):
            data = tl.load(buf_tuple[i] + base + offs, mask=mask, other=0.0)
            acc += data.to(tl.float32)
        tl.store(output_ptr + offs, acc, mask=mask)


def reduce_scatter_triton(x_symm, group, out=None, hdl=None, peer_bufs=None, my_rank=None):
    """Sum-reduce-scatter via Triton + symm-mem (no NCCL).

    Equivalent to:
        dist.reduce_scatter_tensor(out, x_symm, op=ReduceOp.SUM, group=group)
    up to fp32-accumulation order. Output dtype matches x_symm.

    Caller contract: x_symm has been written and a barrier has been issued
    before the call (peers read this rank's bytes via NVLink).
    """
    if peer_bufs is None:
        if hdl is None:
            hdl = rendezvous(x_symm, group)
        W = hdl.world_size
        my_rank = hdl.rank if my_rank is None else my_rank
        peer_bufs = tuple(hdl.get_buffer(r, tuple(x_symm.shape), x_symm.dtype) for r in range(W))
    else:
        W = len(peer_bufs)
        if my_rank is None:
            my_rank = hdl.rank if hdl is not None else dist.get_rank(group)
    assert x_symm.shape[0] % W == 0, f"reduce_scatter: x_symm.shape[0]={x_symm.shape[0]} not divisible by W={W}"
    T_local = x_symm.shape[0] // W
    out_shape = (T_local,) + tuple(x_symm.shape[1:])
    if out is None:
        out = torch.empty(out_shape, dtype=x_symm.dtype, device=x_symm.device)

    numel_per_rank = out.numel()
    grid = lambda META: (triton.cdiv(numel_per_rank, META["BLOCK_SIZE"]),)
    _reduce_scatter_kernel[grid](
        peer_bufs, out, numel_per_rank=numel_per_rank, my_rank=my_rank, world_size=W,
    )
    return out


# ============================================================================
# Utilities
# ============================================================================


def rendezvous(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    return symm_mem.rendezvous(tensor, group=group)


# ============================================================================
# Python wrappers
# ============================================================================


def all_gather_triton(x_symm, group, out=None, hdl=None, peer_bufs=None):
    if peer_bufs is None:
        if hdl is None:
            hdl = rendezvous(x_symm, group)
        W = hdl.world_size
        peer_bufs = tuple(hdl.get_buffer(r, tuple(x_symm.shape), x_symm.dtype) for r in range(W))
    else:
        W = len(peer_bufs)
    numel = x_symm.numel()
    if out is None:
        out = torch.empty(
            (W * x_symm.shape[0],) + tuple(x_symm.shape[1:]),
            dtype=x_symm.dtype,
            device=x_symm.device,
        )
    grid = lambda META: (W, triton.cdiv(numel, META["BLOCK_SIZE"]))
    _all_gather_kernel[grid](
        peer_bufs, out, numel_per_rank=numel, world_size=W,
    )
    return out


# ============================================================================
# AllGather via Copy Engine (cuMemcpyAsync direct).
# ----------------------------------------------------------------------------
# Routes peer copies through the GPU's Copy Engines instead of a SM-side
# copy kernel. The driver dispatches intra-device P2P cuMemcpyAsync to a
# CE; torch.Tensor.copy_ falls through to a SM kernel even on P2P-mapped
# tensors, which is why the prior implementation of this function failed
# to actually use the CE despite the name. We bypass torch.copy_ by issuing
# the copies through cuda-python driver bindings directly.
#
# Caller contract: x_symm has been written AND a barrier (NCCL or symm-mem)
# has been issued before the call, so peer reads observe valid data.
# Same contract as all_gather_triton.
# ============================================================================


def _check(result):
    """Unwrap cuda-python driver returns; raise on error."""
    if not isinstance(result, tuple):
        return result
    err, *rest = result
    if err != driver.CUresult.CUDA_SUCCESS:
        _, name = driver.cuGetErrorName(err)
        _, msg = driver.cuGetErrorString(err)
        decode = lambda b: b.decode() if isinstance(b, bytes) else b
        raise RuntimeError(f"CUDA error {decode(name)}: {decode(msg)}")
    if len(rest) == 0:
        return None
    if len(rest) == 1:
        return rest[0]
    return tuple(rest)


class CEHandle:
    __slots__ = ("out", "_streams")

    def __init__(self, out, streams):
        self.out = out
        self._streams = streams

    def wait(self):
        main = torch.cuda.current_stream()
        for s in self._streams:
            main.wait_stream(s)
        return self.out

    def __call__(self):
        return self.wait()


_CE_STREAM_POOL: dict[tuple, list[torch.cuda.Stream]] = {}


def _get_ce_streams(device, n):
    """Return n cached high-priority CUDA streams for CE copies."""
    key = (str(device), n)
    pool = _CE_STREAM_POOL.get(key)
    if pool is not None:
        return pool
    pool = [torch.cuda.Stream(device=device, priority=-1) for _ in range(n)]
    _CE_STREAM_POOL[key] = pool
    return pool


def all_gather_copy_engine_async(x, peer_bufs, my_rank, out=None, num_streams=2):
    """All-gather routed through the Copy Engines via cuMemcpyAsync.

    For each remote rank r, issues
        cuMemcpyAsync(out[r], peer_bufs[r], ...)
    on a dedicated CUDA stream. The local chunk is filled via cuMemcpyAsync
    on the torch stream from this rank's ``x`` (intra-device, no NVLink).
    Going through the driver bindings rather than torch.Tensor.copy_
    ensures the driver picks the Copy Engine path; torch.copy_ launches
    a SM copy kernel even on P2P-mapped tensors. To verify CE usage in
    practice:
        ncu --metrics lts__t_sectors_srcunit_l1ces_op_read.sum,\\
                     lts__t_sectors_srcunit_tex_op_read.sum
    The L1ces counter should account for the (W-1) cross-rank chunks;
    tex should be near-zero for those copies.

    Caller contract:
        ``x`` is this rank's data — any tensor with a valid CUDA pointer.
        ``peer_bufs[r]`` is any tensor that points at peer ``r``'s data
        in a way the CE can read through (P2P-enabled, IPC-mapped,
        symm-mem rendezvous'd — all are fine; the function does not
        care which mechanism produced the mapping).

        The caller is responsible for fencing peer-visible writes
        before this call (NCCL barrier, symm-mem barrier, etc.) so
        that peer reads observe valid data. The CE streams take a
        ``wait_stream`` dependency on the torch stream, so anything
        enqueued on the main stream (e.g. the caller's barrier)
        serializes correctly.

    Args:
        x:           this rank's tensor, shape (T_local, ...). Must be
                     contiguous.
        peer_bufs:   tuple of W tensors — one per rank — each pointing
                     at that peer's published data. Length determines W.
        my_rank:     index in ``peer_bufs`` corresponding to this rank.
        out:         optional pre-allocated output (W*T_local, ...) with
                     matching dtype/device. Allocated here if None.
        num_streams: number of CE streams. Default 2. More streams =
                     more concurrent peer copies; past 4 the host
                     overhead usually outweighs the parallelism on a
                     single NVLink island.

    Returns:
        CEHandle. Use .wait() to sync, then access .out (or call() it).

    Example:
        h = all_gather_copy_engine_async(
            x_local, peer_bufs=peer_bufs, my_rank=rank,
        )
        y = some_gemm(...)              # overlaps with CE copies
        x_global = h.wait()
    """
    W = len(peer_bufs)

    if out is None:
        out = torch.empty(
            (W * x.shape[0],) + tuple(x.shape[1:]),
            dtype=x.dtype,
            device=x.device,
        )

    num_streams = max(1, min(num_streams, W))
    streams = _get_ce_streams(x.device, num_streams)
    chunks = out.chunk(W)
    bytes_per_chunk = x.numel() * x.element_size()

    main = torch.cuda.current_stream()
    # CE streams wait on whatever was last enqueued on the torch stream
    # (typically the caller's barrier on peer_bufs), so peer reads
    # observe valid data.
    for s in streams:
        s.wait_stream(main)

    # Self-copy on the torch stream — intra-device, no NVLink. Reads
    # straight from raw x (no symm-mem hop). The cross-stream
    # parallelism is enough that this isn't on the critical path.
    _check(
        driver.cuMemcpyAsync(
            chunks[my_rank].data_ptr(),
            x.data_ptr(),
            bytes_per_chunk,
            main.cuda_stream,
        )
    )

    # Cross-rank pulls via cuMemcpyAsync on dedicated CE streams. Round-
    # robin across the pool. Self-skipped (handled above). Iteration
    # order starts at (rank+1) to spread the initial requests across
    # peers — at large W this avoids transient hot-spotting on rank 0.
    peers = [(my_rank + i) % W for i in range(1, W)]
    for idx, r in enumerate(peers):
        s = streams[idx % num_streams]
        _check(
            driver.cuMemcpyAsync(
                chunks[r].data_ptr(),
                peer_bufs[r].data_ptr(),
                bytes_per_chunk,
                s.cuda_stream,
            )
        )

    return CEHandle(out, streams)
