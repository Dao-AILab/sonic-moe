# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
"""Detects whether N co-allocated nodes form ONE MNNVL/LSA domain or separate per-node NVLink domains
(sets HIER's ``node_size``). Needs a live multi-rank GIN comm — run under srun, not standalone.
"""
import os
import socket
import sys

import torch
import torch.distributed as dist
import nccl.core as nccl
import nccl.core.interop.torch as nccl_torch


def main():
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
    world = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1)))
    local = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0)))
    if "MASTER_ADDR" not in os.environ:
        import subprocess
        nl = os.environ.get("SLURM_NODELIST", "127.0.0.1")
        if "[" in nl or "," in nl:
            os.environ["MASTER_ADDR"] = subprocess.check_output(
                ["scontrol", "show", "hostnames", nl]).decode().split()[0]
        else:
            os.environ["MASTER_ADDR"] = nl
    os.environ.setdefault("MASTER_PORT", "29573")
    torch.cuda.set_device(local)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)

    uid = nccl.get_unique_id() if rank == 0 else None
    o = [uid]
    dist.broadcast_object_list(o, src=0)
    uid = o[0]
    comm = nccl.Communicator.init(nranks=world, rank=rank, unique_id=uid)
    buf = nccl_torch.empty(8, dtype=torch.bfloat16, device=local)
    buf.zero_()
    torch.cuda.synchronize()
    win = comm.register_window(buf)
    dev = comm.create_dev_comm(requirements=nccl.NCCLDevCommRequirements(
        gin_connection_type=nccl.NcclGinConnectionType.FULL, gin_signal_count=1))
    try:
        ls = int(dev.dev_comm.lsa_size)
        lr = int(dev.dev_comm.lsa_rank)
    except Exception as e:  # noqa: BLE001
        ls, lr = -1, -1
        print(f"[rank {rank}] lsa_size read failed: {e!r}", flush=True)

    gathered = [None] * world
    dist.all_gather_object(gathered, (rank, socket.gethostname(), local, ls, lr))

    if rank == 0:
        gathered.sort()
        print(f"[nvlink-domain] world={world} per-rank (rank, host, localGPU, lsa_size, lsa_rank):", flush=True)
        for g in gathered:
            print(f"  {g}", flush=True)
        # NCCL assigns contiguous global ranks to a team, so domain d owns ranks [d*L, (d+1)*L).
        L = gathered[0][3]
        if L and L > 0:
            for d in range((world + L - 1) // L):
                members = [g for g in gathered if d * L <= g[0] < (d + 1) * L]
                hosts = sorted({g[1] for g in members})
                print(f"[nvlink-domain] domain {d}: lsa_size={L} ranks={[m[0] for m in members]} "
                      f"-> {len(hosts)} host(s): {hosts}", flush=True)
            d0_hosts = len({g[1] for g in gathered if 0 <= g[0] < L})
            verdict = "YES" if d0_hosts > 1 else "NO (per-node)"
            print(f"[nvlink-domain] VERDICT: each NVLink domain spans {d0_hosts} node(s) (lsa_size={L}); "
                  f"MNNVL-multinode={verdict}; HIER node_size={L}", flush=True)
        else:
            print(f"[nvlink-domain] VERDICT: lsa_size={L} (unexpected)", flush=True)

    dev.close()
    win.close()
    comm.destroy()
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
