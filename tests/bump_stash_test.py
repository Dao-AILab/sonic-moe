# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
# Single-GPU unit tests for the bump-stash kernels. Run with:
#   pytest tests/bump_stash_test.py -v
import pytest
import torch

from sonicmoe.functional.distributed.bump_stash import bump_pack, bump_unpack


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
@pytest.mark.parametrize("hidden", [3072, 4096, 3000])  # 3000 exercises the HIDDEN mask branch
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("real", [0, 1, 1337, 4096])  # 0 = empty rank, 4096 = full (real == max_rows)
def test_pack_unpack_roundtrip(hidden, dtype, real):
    dev = "cuda"
    max_rows, cap = 4096, 8192
    h = torch.randn(max_rows, hidden, device=dev, dtype=dtype)
    pool = torch.empty(cap, hidden, device=dev, dtype=dtype)
    head = torch.zeros(1, dtype=torch.int32, device=dev)
    over = torch.zeros(1, dtype=torch.int32, device=dev)
    count = torch.tensor([real], dtype=torch.int32, device=dev)

    bump_pack(h, pool, head, count, over, max_rows)
    assert int(head.item()) == real
    assert int(over.item()) == 0
    if real > 0:
        assert torch.equal(pool[:real], h[:real])

    dst = torch.empty(max_rows, hidden, device=dev, dtype=dtype)
    bump_unpack(pool, dst, head, count, over, max_rows)
    assert int(head.item()) == 0  # head fully unwound (FILO)
    if real > 0:
        assert torch.equal(dst[:real], h[:real])  # exact: pure copy round-trip


@cuda
def test_count_zero_is_noop():
    # A rank with zero local rows: pack/unpack must be clean no-ops (head
    # unchanged, no overflow), and a downstream consumer reads 0 rows.
    dev, hidden, dtype = "cuda", 3072, torch.bfloat16
    max_rows, cap = 4096, 8192
    h = torch.randn(max_rows, hidden, device=dev, dtype=dtype)
    pool = torch.empty(cap, hidden, device=dev, dtype=dtype)
    head = torch.zeros(1, dtype=torch.int32, device=dev)
    over = torch.zeros(1, dtype=torch.int32, device=dev)
    count = torch.tensor([0], dtype=torch.int32, device=dev)

    bump_pack(h, pool, head, count, over, max_rows)
    assert int(head.item()) == 0
    assert int(over.item()) == 0
    dst = torch.empty(max_rows, hidden, device=dev, dtype=dtype)
    bump_unpack(pool, dst, head, count, over, max_rows)
    assert int(head.item()) == 0
    assert int(over.item()) == 0


@cuda
def test_filo_multi_layer():
    # Push several layers in forward order, pop in exact reverse (FILO).
    dev, hidden, dtype = "cuda", 3072, torch.bfloat16
    max_rows, cap = 4096, 16384
    counts = [1000, 1500, 800]
    hs = [torch.randn(max_rows, hidden, device=dev, dtype=dtype) for _ in counts]
    pool = torch.empty(cap, hidden, device=dev, dtype=dtype)
    head = torch.zeros(1, dtype=torch.int32, device=dev)
    over = torch.zeros(1, dtype=torch.int32, device=dev)
    cts = [torch.tensor([c], dtype=torch.int32, device=dev) for c in counts]

    for h, c in zip(hs, cts):  # forward order: push
        bump_pack(h, pool, head, c, over, max_rows)
    assert int(head.item()) == sum(counts)
    assert int(over.item()) == 0

    for h, c in zip(reversed(hs), reversed(cts)):  # backward order: pop (FILO)
        dst = torch.empty(max_rows, hidden, device=dev, dtype=dtype)
        bump_unpack(pool, dst, head, c, over, max_rows)
        assert torch.equal(dst[: int(c.item())], h[: int(c.item())])
    assert int(head.item()) == 0


@cuda
def test_overflow_guard_no_oob():
    # A pack that exceeds the pool must set the flag, write nothing, and NOT
    # advance the head — no out-of-bounds access.
    dev, hidden, dtype = "cuda", 3072, torch.bfloat16
    max_rows, cap = 4096, 1000
    h = torch.randn(max_rows, hidden, device=dev, dtype=dtype)
    pool = torch.empty(cap, hidden, device=dev, dtype=dtype)
    head = torch.zeros(1, dtype=torch.int32, device=dev)
    over = torch.zeros(1, dtype=torch.int32, device=dev)
    count = torch.tensor([2000], dtype=torch.int32, device=dev)  # > cap

    bump_pack(h, pool, head, count, over, max_rows)
    assert int(over.item()) == 1
    assert int(head.item()) == 0  # not advanced; no out-of-bounds write


@cuda
def test_overflow_unpack_skips_copy_and_preserves_head():
    # Once overflow is flagged, unpack must skip the copy (avoid reading
    # never-written rows) rather than fault, AND must leave the head unchanged
    # — mirroring the pack overflow path. A too-small pool must never drive the
    # head negative (which repeated unpacks would compound).
    dev, hidden, dtype = "cuda", 3072, torch.bfloat16
    max_rows, cap = 4096, 1000
    h = torch.randn(max_rows, hidden, device=dev, dtype=dtype)
    pool = torch.empty(cap, hidden, device=dev, dtype=dtype)
    head = torch.zeros(1, dtype=torch.int32, device=dev)
    over = torch.zeros(1, dtype=torch.int32, device=dev)
    count = torch.tensor([2000], dtype=torch.int32, device=dev)

    bump_pack(h, pool, head, count, over, max_rows)  # overflow -> flag set, head preserved
    assert int(over.item()) == 1
    assert int(head.item()) == 0  # pack did not advance
    dst = torch.empty(max_rows, hidden, device=dev, dtype=dtype)
    bump_unpack(pool, dst, head, count, over, max_rows)  # must not fault
    torch.cuda.synchronize()
    assert int(over.item()) == 1
    assert int(head.item()) == 0  # unpack preserved head (NOT head-count = -2000)


@cuda
def test_overflow_multilayer_head_never_negative():
    # Two layers fit; a third overflows. Unpacking all three (sticky flag) must
    # never drive the head negative — it stays put once overflow trips.
    dev, hidden, dtype = "cuda", 3072, torch.bfloat16
    max_rows, cap = 4096, 1000
    pool = torch.empty(cap, hidden, device=dev, dtype=dtype)
    head = torch.zeros(1, dtype=torch.int32, device=dev)
    over = torch.zeros(1, dtype=torch.int32, device=dev)
    hs = [torch.randn(max_rows, hidden, device=dev, dtype=dtype) for _ in range(3)]
    cts = [torch.tensor([400], dtype=torch.int32, device=dev) for _ in range(3)]

    for h, c in zip(hs, cts):  # 0..400, 400..800, then 800+400>1000 -> overflow
        bump_pack(h, pool, head, c, over, max_rows)
    assert int(over.item()) == 1
    dst = torch.empty(max_rows, hidden, device=dev, dtype=dtype)
    for c in reversed(cts):
        bump_unpack(pool, dst, head, c, over, max_rows)
        assert int(head.item()) >= 0  # never negative


@cuda
def test_bump_ops_are_sync_free():
    # The whole point of the stash is to right-size memory WITHOUT a host sync.
    # Assert the kernels issue no implicit device->host sync (no .item(), the
    # grid bound is a Python int). Isolated from NCCL/symm-mem so it is not flaky.
    dev, hidden, dtype = "cuda", 3072, torch.bfloat16
    max_rows, cap = 4096, 8192
    h = torch.randn(max_rows, hidden, device=dev, dtype=dtype)
    pool = torch.empty(cap, hidden, device=dev, dtype=dtype)
    head = torch.zeros(1, dtype=torch.int32, device=dev)
    over = torch.zeros(1, dtype=torch.int32, device=dev)
    count = torch.tensor([1337], dtype=torch.int32, device=dev)
    dst = torch.empty(max_rows, hidden, device=dev, dtype=dtype)

    # Warm up (Triton JIT compile) OUTSIDE the sync-debug window.
    bump_pack(h, pool, head, count, over, max_rows)
    bump_unpack(pool, dst, head, count, over, max_rows)
    torch.cuda.synchronize()

    prev = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")  # raise on any host sync
        bump_pack(h, pool, head, count, over, max_rows)
        bump_unpack(pool, dst, head, count, over, max_rows)
    finally:
        torch.cuda.set_sync_debug_mode(prev)
    torch.cuda.synchronize()
