# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
# Bump-allocated activation stash for SonicMoE EP.
#
# ``bump_pack`` copies the first ``count`` rows of a static-max buffer into a
# shared pool at the current head, then advances the head on-device.
# ``bump_unpack`` reverses it (head -= count, copy back). ``count`` lives on the
# GPU and is read INSIDE the kernel, so neither op syncs with the host — the
# launch grid is bounded by the static ``max_rows`` (a Python int), so the path
# is CUDA-graph-capturable.
#
# The copy is fully CONTIGUOUS (``src[:count]`` and ``pool[head:head+count]`` are
# both contiguous blocks of ``count*HIDDEN`` elements), so the kernels treat it
# as a FLAT 1D copy with a grid-strided, vectorized loop to saturate HBM. These
# are pure bitwise copies — pool dtype must equal src/dst dtype, so the
# round-trip is exact (the callers enforce this).
#
# Ordering contract (enforced by the caller, see _MoeEPFunction): packs/unpacks
# form a strict FILO stack — forward pushes layer activations in order, backward
# pops them in exact reverse. The single bump head is correct only under that
# invariant; the autograd wiring guards it with a host-side depth/level check.
#
# Overflow: the pool is sized for the SUM of live layers' real rows. If a pack
# would exceed the pool capacity the kernel writes nothing, sets an overflow
# flag, and does NOT advance the head (never an out-of-bounds write); unpack then
# preserves the head (never drives it negative). The flag is surfaced via
# ``_EPWorkspace.stash_overflow`` / ``stash_overflowed``; the documented fallback
# for a too-small pool is ``CPU_sync_on_runtime=False`` (no stash).
import torch
import triton
import triton.language as tl


# Fixed launch config (NOT autotuned): triton.autotune benchmarks configs with
# event syncs on the first call, which cannot run inside a CUDA-graph capture
# region — and graph-safety is the whole point of this stash. A fixed config is
# directly capturable and saturates HBM (~6 TB/s, R+W) at production transfer
# sizes (hundreds of MB); tiny test-shape transfers are launch-overhead-bound.
STASH_BLOCK = 4096  # elements copied per program per grid-stride step
STASH_MAX_PROGRAMS = 8192  # static grid cap; the grid-stride loop covers the rest
STASH_NUM_WARPS = 8

__all__ = ["bump_pack", "bump_unpack", "STASH_BLOCK", "STASH_MAX_PROGRAMS"]


@triton.jit
def _bump_pack_kernel(
    src_ptr,
    pool_ptr,
    count_ptr,
    head_ptr,
    new_head_ptr,
    overflow_ptr,
    CAP: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    count = tl.load(count_ptr)
    head = tl.load(head_ptr)

    # Capacity guard (in rows): never write out of bounds. All programs see the
    # same (head, count, CAP), so they all return together — no partial write.
    if head + count > CAP:
        if pid == 0:
            tl.store(overflow_ptr, 1)
            tl.store(new_head_ptr, head)
        return

    # Only program 0 publishes the advanced head, to a SEPARATE tensor — the
    # caller copies it into ``head`` after the kernel, so concurrent programs
    # never read a half-updated head.
    if pid == 0:
        tl.store(new_head_ptr, head + count)

    # Flat contiguous copy of [0, count*HIDDEN) elements: src[:count] rows ->
    # pool rows [head, head+count). int64 element offsets for large pools.
    n = count.to(tl.int64) * HIDDEN
    dst_base = head.to(tl.int64) * HIDDEN
    i = pid.to(tl.int64) * BLOCK
    step = nprog.to(tl.int64) * BLOCK
    while i < n:
        offs = i + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(pool_ptr + dst_base + offs, tl.load(src_ptr + offs, mask=mask), mask=mask)
        i += step


@triton.jit
def _bump_unpack_kernel(
    pool_ptr,
    dst_ptr,
    count_ptr,
    head_ptr,
    new_head_ptr,
    overflow_ptr,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    count = tl.load(count_ptr)
    head = tl.load(head_ptr)
    overflow = tl.load(overflow_ptr)

    if overflow == 1:
        # Pool overflowed earlier this pass (flag set, results known-bad). Leave
        # the head UNCHANGED (mirror the pack overflow path — never negative) and
        # skip the copy so we don't read never-written rows.
        if pid == 0:
            tl.store(new_head_ptr, head)
        return

    base = head - count  # this tensor's rows live at [head-count, head)
    if pid == 0:
        tl.store(new_head_ptr, base)

    n = count.to(tl.int64) * HIDDEN
    src_base = base.to(tl.int64) * HIDDEN
    i = pid.to(tl.int64) * BLOCK
    step = nprog.to(tl.int64) * BLOCK
    while i < n:
        offs = i + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(dst_ptr + offs, tl.load(pool_ptr + src_base + offs, mask=mask), mask=mask)
        i += step


def _grid(max_rows, hidden):
    # Static grid (Python ints only → no host sync, CUDA-graph-safe), capped; the
    # kernel grid-strides over the device-bounded [0, count*HIDDEN) region.
    return (min(triton.cdiv(int(max_rows) * int(hidden), STASH_BLOCK), STASH_MAX_PROGRAMS),)


def bump_pack(src, pool, head, count, overflow, max_rows):
    """Copy ``src[:count]`` -> ``pool[head:head+count]``; ``head += count`` — all on device.

    ``count`` is a ``(1,)`` int32 GPU tensor (read inside the kernel, never
    ``.item()``'d). ``max_rows`` is the static-max row count (a Python int), used
    only for the launch-grid bound → no host sync. ``count <= max_rows`` is
    guaranteed by the structural row ceiling.
    """
    new_head = torch.empty_like(head)
    hidden = src.shape[1]
    _bump_pack_kernel[_grid(max_rows, hidden)](
        src,
        pool,
        count,
        head,
        new_head,
        overflow,
        CAP=pool.shape[0],
        HIDDEN=hidden,
        BLOCK=STASH_BLOCK,
        num_warps=STASH_NUM_WARPS,
    )
    head.copy_(new_head)


def bump_unpack(pool, dst, head, count, overflow, max_rows):
    """``head -= count``; copy ``pool[head:head+count]`` -> ``dst[:count]`` — all on device.

    Mirror of :func:`bump_pack`. ``dst`` must have the same dtype as ``pool`` so
    the round-trip is bitwise-exact.
    """
    new_head = torch.empty_like(head)
    hidden = dst.shape[1]
    _bump_unpack_kernel[_grid(max_rows, hidden)](
        pool,
        dst,
        count,
        head,
        new_head,
        overflow,
        HIDDEN=hidden,
        BLOCK=STASH_BLOCK,
        num_warps=STASH_NUM_WARPS,
    )
    head.copy_(new_head)
