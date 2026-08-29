# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
#
# Triton + symm-mem AG/RS collectives for dispatch/combine. Stateless: caller owns buffer
# allocation and barriers. Naming: T_local=tokens/rank, TK_local=T_local*K, W=world size, TK_global=W*TK_local.
# ********************************************************************************

from __future__ import annotations

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from cuda.bindings import driver
from torch.distributed import _symmetric_memory as symm_mem


_CUDA_MAX_GRID_Y = 65535

# Separate autotune spaces: AG/RS want different best warp counts.
# AG is peer-memcpy-bound, so 4-8 warps suffice; 16 is a large-tile fallback.
_AG_BLOCK_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE": 16384}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE": 16384}, num_warps=16, num_stages=3),
    triton.Config({"BLOCK_SIZE": 32768}, num_warps=16, num_stages=3),
]


def _prune_by_grid_y(numel_key: str):
    """Drops configs whose grid_y would exceed CUDA's 65535 limit; checks both
    named_args and kwargs since Triton's autotuner splits kernel args across both."""

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


# Reduce-scatter via NVLink reads + fp32 accum (no NCCL); shape matches
# reduce_scatter_tensor. Deterministic (static_range order), unlike NCCL ring RS.

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
    """RS fans in (1 program reads W peer chunks -> 1 local chunk), unlike AG's fan-out.
    fp32 accumulation; tl.store casts to output dtype."""
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
    """Equivalent to dist.reduce_scatter_tensor(SUM), up to accumulation order.
    Caller must barrier after writing x_symm before calling (NVLink peer reads)."""
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
        peer_bufs,
        out,
        numel_per_rank=numel_per_rank,
        my_rank=my_rank,
        world_size=W,
    )
    return out


def rendezvous(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    return symm_mem.rendezvous(tensor, group=group)


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
        peer_bufs,
        out,
        numel_per_rank=numel,
        world_size=W,
    )
    return out


# torch.Tensor.copy_ uses a SM kernel even on P2P tensors, not the Copy Engine —
# bypass via raw cuMemcpyAsync driver calls instead. Caller must barrier before calling.


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
    """peer_bufs may be any P2P/IPC/symm-mem-mapped tensor; caller must fence writes before calling.
    num_streams>4 rarely helps — host overhead outweighs parallelism on one NVLink island."""
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
    # CE streams wait on the torch stream's last op (typically the caller's
    # barrier), so peer reads observe valid data.
    for s in streams:
        s.wait_stream(main)

    # Self-copy on the torch stream (intra-device, no NVLink) — reads raw x
    # directly; cross-stream parallelism keeps this off the critical path.
    _check(
        driver.cuMemcpyAsync(
            chunks[my_rank].data_ptr(),
            x.data_ptr(),
            bytes_per_chunk,
            main.cuda_stream,
        )
    )

    # Round-robin across CE streams; iteration starts at rank+1 (not 0) to
    # avoid transient hot-spotting on rank 0 at large W.
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


# Adapted from https://github.com/yifuwang/symm-mem-recipes/tree/main

# NVLink-SHARP multicast: multimem.st fans out / ld_reduce sums in one instruction (switch does the
# reduce). 16-byte-aligned packs only. AG is bit-exact; bf16 RS is NOT (HW-defined summation order) — compare with tolerance.

# NVLink-BW-bound: only num_warps/BLOCK_SIZE/num_stages matter for keeping ops in flight.
# Grid size isn't tuned — a one-pass grid already fills every SM.
_MULTIMEM_CONFIGS = [
    triton.Config({"BLOCK_SIZE": bs}, num_warps=nw, num_stages=ns)
    for bs in [2048, 4096, 8192]
    for nw in [4, 8, 16, 32]
    for ns in [2, 3]
    if 1 <= bs // (nw * 32) <= 16
]


@triton.jit
def _multimem_st_v4(mc_ptr, x0, x1, x2, x3, mask):
    """Broadcasts one 128-bit pack to every peer via the multicast address.
    Raw bit copy (dtype-agnostic); no-op where mask == 0."""
    tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $6, 1;
            @!%p0 bra end;
            multimem.st.relaxed.sys.global.v4.f32 [$1], {$2, $3, $4, $5};
            end:
        }
        """,
        "=r,l,r,r,r,r,r",
        args=[mc_ptr, x0, x1, x2, x3, mask.to(tl.int32)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _multimem_ld_reduce_v4_bf16(mc_ptr, mask):
    """Load + bf16 sum-reduce one 128-bit pack (8 bf16) across all peers."""
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $5, 1;
            @!%p0 bra end;
            multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {$0, $1, $2, $3}, [$4];
            end:
        }
        """,
        "=r,=r,=r,=r,l,r",
        args=[mc_ptr, mask.to(tl.int32)],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _multimem_ld_reduce_v4_f32(mc_ptr, mask):
    """Load + fp32 sum-reduce one 128-bit pack (4 fp32) across all peers."""
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $5, 1;
            @!%p0 bra end;
            multimem.ld_reduce.relaxed.sys.global.add.v4.f32 {$0, $1, $2, $3}, [$4];
            end:
        }
        """,
        "=r,=r,=r,=r,l,r",
        args=[mc_ptr, mask.to(tl.int32)],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.autotune(configs=_MULTIMEM_CONFIGS, key=["n_packs"])
@triton.jit
def _all_gather_multimem_kernel(
    x_u32_ptr,  # (numel_local // 2 or // 1) uint32 — this rank's local source
    out_mc_ptr,  # multicast base address (int) of the (W*T_local, d) output
    n_packs,  # 128-bit packs in this rank's chunk
    my_rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """out_symm[my_rank*chunk : (my_rank+1)*chunk] ← x, fanned out to all peers
    via multimem.st. One pack per lane, one coalesced 128-bit load of the source."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_packs
    u32 = offs * 4
    x0 = tl.load(x_u32_ptr + u32 + 0, mask=mask)
    x1 = tl.load(x_u32_ptr + u32 + 1, mask=mask)
    x2 = tl.load(x_u32_ptr + u32 + 2, mask=mask)
    x3 = tl.load(x_u32_ptr + u32 + 3, mask=mask)
    dst_pack = my_rank * n_packs + offs
    mc_u64 = out_mc_ptr.to(tl.pointer_type(tl.uint64))
    _multimem_st_v4(mc_u64 + dst_pack * 2, x0, x1, x2, x3, mask)


@triton.autotune(configs=_MULTIMEM_CONFIGS, key=["n_packs"])
@triton.jit
def _reduce_scatter_multimem_kernel(
    x_mc_ptr,  # multicast base address (int) of the (W*T_local, d) input
    out_u32_ptr,  # (T_local*d ... ) uint32 — this rank's local output chunk
    n_packs,  # 128-bit packs in this rank's chunk
    my_rank: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """out ← Σ_peers x_symm[my_rank*chunk : (my_rank+1)*chunk] via
    multimem.ld_reduce. One pack per lane, one coalesced 128-bit store of the result."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_packs
    src_pack = my_rank * n_packs + offs
    mc = x_mc_ptr.to(tl.pointer_type(tl.uint64)) + src_pack * 2
    if IS_BF16:
        x0, x1, x2, x3 = _multimem_ld_reduce_v4_bf16(mc, mask)
    else:
        x0, x1, x2, x3 = _multimem_ld_reduce_v4_f32(mc, mask)
    u32 = offs * 4
    tl.store(out_u32_ptr + u32 + 0, x0, mask=mask)
    tl.store(out_u32_ptr + u32 + 1, x1, mask=mask)
    tl.store(out_u32_ptr + u32 + 2, x2, mask=mask)
    tl.store(out_u32_ptr + u32 + 3, x3, mask=mask)


def _n_packs(numel: int, elem_size: int) -> int:
    nbytes = numel * elem_size
    assert nbytes % 16 == 0, f"multimem requires 16-byte-aligned size, got {nbytes} bytes"
    return nbytes // 16


def all_gather_multimem_triton(x_symm, group, out=None, hdl=None, out_hdl=None, fence=True):
    """AG is a PUSH — needs a post-barrier after the write (opposite of the pull-based
    all_gather_triton's pre-barrier). fence=False lets the caller barrier externally."""
    if hdl is None:
        hdl = rendezvous(x_symm, group)
    W = hdl.world_size
    my_rank = hdl.rank
    T_local = x_symm.shape[0]
    tail = tuple(x_symm.shape[1:])
    if out is None:
        out = symm_mem.empty((W * T_local,) + tail, dtype=x_symm.dtype, device=x_symm.device)
    if out_hdl is None:
        out_hdl = rendezvous(out, group)
    assert out_hdl.multicast_ptr != 0, "multimem AG needs multicast support (NVLink SHARP / MNNVL)"

    n_packs = _n_packs(x_symm.numel(), x_symm.element_size())
    x_u32 = x_symm.reshape(-1).view(torch.int32)
    grid = lambda META: (triton.cdiv(n_packs, META["BLOCK_SIZE"]),)
    _all_gather_multimem_kernel[grid](
        x_u32,
        out_hdl.multicast_ptr,
        n_packs=n_packs,
        my_rank=my_rank,
    )
    # Fence: every peer's multicast write into `out` must land before any
    # rank reads the gathered result. Skippable when the caller fences itself.
    if fence:
        out_hdl.barrier()
    return out


def reduce_scatter_multimem_triton(x_symm, group, out=None, hdl=None, my_rank=None):
    """Reduces in buffer dtype (bf16!=fp32-exact vs reduce_scatter_triton).
    Caller must barrier BEFORE calling (pull-based multicast load)."""
    if hdl is None:
        hdl = rendezvous(x_symm, group)
    W = hdl.world_size
    my_rank = hdl.rank if my_rank is None else my_rank
    assert hdl.multicast_ptr != 0, "multimem RS needs multicast support (NVLink SHARP / MNNVL)"
    assert x_symm.shape[0] % W == 0, f"reduce_scatter: x_symm.shape[0]={x_symm.shape[0]} not divisible by W={W}"
    assert x_symm.dtype in (torch.bfloat16, torch.float32), "multimem RS supports bf16 / fp32 only"

    T_local = x_symm.shape[0] // W
    out_shape = (T_local,) + tuple(x_symm.shape[1:])
    if out is None:
        out = torch.empty(out_shape, dtype=x_symm.dtype, device=x_symm.device)

    chunk_numel = out.numel()  # elements in this rank's chunk
    n_packs = _n_packs(chunk_numel, x_symm.element_size())
    out_u32 = out.reshape(-1).view(torch.int32)
    grid = lambda META: (triton.cdiv(n_packs, META["BLOCK_SIZE"]),)
    _reduce_scatter_multimem_kernel[grid](
        hdl.multicast_ptr,
        out_u32,
        n_packs=n_packs,
        my_rank=my_rank,
        IS_BF16=(x_symm.dtype == torch.bfloat16),
    )
    return out
