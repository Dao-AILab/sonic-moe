# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
"""Detects the fastest ACTIVE RDMA NICs (IB or RoCE, by link RATE not fabric type) for NCCL_IB_HCA.
Pure/side-effect-free — LAUNCHER must export NCCL_IB_HCA BEFORE the first NCCL comm init (device list is cached then).
"""
from __future__ import annotations

import glob
import os

_IB_SYSFS = "/sys/class/infiniband"


def _parse_rate_gbps(rate_str: str) -> "float | None":
    """Parse a sysfs port ``rate`` line (e.g. ``'400 Gb/sec (4X NDR)'``) → ``400.0`` Gb/s, or None."""
    try:
        return float(rate_str.split()[0])
    except (ValueError, IndexError):
        return None


def detect_rdma_hca() -> list[str]:
    """Returns all ACTIVE RDMA NICs (IB or RoCE) at the node's TOP link rate, by device name — fabric-
    agnostic, rate-driven. Empty list on a host with no ACTIVE RDMA device."""
    cands: list[tuple[str, float]] = []
    for d in sorted(glob.glob(os.path.join(_IB_SYSFS, "*"))):
        try:
            with open(os.path.join(d, "ports/1/state")) as f:
                state = f.read().strip()
            with open(os.path.join(d, "ports/1/rate")) as f:
                rate = f.read().strip()
        except OSError:
            continue
        gbps = _parse_rate_gbps(rate)
        if "ACTIVE" in state and gbps is not None:
            cands.append((os.path.basename(d), gbps))
    if not cands:
        return []
    top = max(g for _, g in cands)
    return [name for name, g in cands if g >= top]


if __name__ == "__main__":
    # Emit the comma-separated NIC list for a launcher to assign to NCCL_IB_HCA.
    print(",".join(detect_rdma_hca()))
