<!-- ********************************************************************************
Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
******************************************************************************** -->

# SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations
[![arXiv](https://img.shields.io/badge/arXiv-2512.14080-b31b1b.svg)](https://arxiv.org/abs/2512.14080) [![PyPI](https://img.shields.io/pypi/v/sonic-moe?cache=no)](https://pypi.org/project/sonic-moe/)

**SonicMoE** is a simple but blazing-fast Mixture-of-Experts (MoE) implementation optimized for NVIDIA Hopper (SM90), Blackwell datacenter (SM100, e.g. B200/B300), and Blackwell consumer (SM120, e.g. RTX 5090) GPUs. It mainly leverages [CuTeDSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html) and [Triton](https://triton-lang.org/main/getting-started/tutorials/index.html) to deliver state-of-the-art performance through IO-aware optimizations. These 2 figures provide an overview of activation memory usage and training throughput on Hopper GPUs (H100) and Blackwell GPUs (B300). The current version of SonicMoE builds on the Grouped GEMM kernels from the [QuACK](https://github.com/Dao-AILab/quack/tree/main) library which is itself built on [CUTLASS](https://github.com/NVIDIA/cutlass). SonicMoE also ships intra-node Expert Parallelism (EP) built from [PyTorch Symmetric Memory](https://docs.pytorch.org/docs/2.11/symmetric_memory.html) with a runtime profiler that picks the fastest primitive for the local cluster configurations.

![Activation Memory](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/mem.png)
![Training Throughput](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/tput.png)

## News

- 05/13/2026: We add a basic intra-node Expert Parallelism (EP) support.

- 04/22/2026: We release a [blogpost](./assets/2026-04-22-sonicmoe-blackwell.md) on SonicMoE's activation memory-efficient and IO-aware design, and how we extend it to Blackwell GPUs through [QuACK](https://github.com/Dao-AILab/quack)'s software abstraction.

- 04/19/2026: we release SonicMoE with Blackwell (SM100) support, built on [QuACK](https://github.com/Dao-AILab/quack)'s Grouped GEMM kernels. 

## 📦 Installation

### Prerequisites

- NVIDIA Hopper GPUs (H100, H200, etc.), Blackwell datacenter GPUs (GB200, B200, B300, etc.), or Blackwell consumer GPUs (e.g. RTX 5090, SM120)
- CUDA 12.9+ (13.0+ for B300 GPUs)
- Python 3.12+ recommended
- PyTorch 2.7+ (2.9.1 recommended)

> **B300 users:** please manually upgrade Triton to 3.6.0 after installing PyTorch.


### Install from pip
```bash
pip install sonic-moe
```

### Install from Source

```bash
# Clone the repository
git clone https://github.com/Dao-AILab/sonic-moe.git
cd sonic-moe

# Install dependencies
pip install -r requirements.txt

# Install SonicMoE
pip install -e .
```

## 🎯 Quick Start

### Basic Usage

```python
import torch
from sonicmoe import MoE, KernelBackendMoE
from sonicmoe.enums import ActivationType

# Create MoE layer
moe = MoE(
    num_experts=128,                           # Number of experts
    num_experts_per_tok=8,                     # Top-k experts per token
    hidden_size=4096,                          # Hidden dimension
    intermediate_size=1536,                    # Expert intermediate size
    activation_function=ActivationType.SWIGLU, # SwiGLU activation
    add_bias=False,                            # Add bias to linear layers
    std=0.02,                                  # Weight initialization std
).to(device="cuda", dtype=torch.bfloat16)

# Forward pass
x = torch.randn(32768, 4096, device="cuda", dtype=torch.bfloat16)
output, aux_loss = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe)
```

#### Intra-node Expert Parallelism

`moe_ep_TC_softmax_topk_forward` runs the same MoE across an NVLink-connected EP group. Each rank holds `E // W` experts and processes `T_local = T // W` tokens; the forward dispatches tokens to their owning ranks, runs the grouped GEMMs locally, and combines the per-expert outputs back. `NetworkProfiler` benchmarks the dispatch/combine primitives on the local hardware and returns the fastest pair. Launch with `torchrun --nproc_per_node=<W> --standalone your_script.py`.

```python
import os
import torch
import torch.distributed as dist
from sonicmoe import MoE
from sonicmoe.distributed_utils import NetworkProfiler
from sonicmoe.enums import ActivationType
from sonicmoe.functional.ep import moe_ep_TC_softmax_topk_forward

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(local_rank)

device = torch.device(f"cuda:{local_rank}")
dist.init_process_group("nccl", device_id=device)

T, H, I, E, K = 131072, 4096, 1536, 128, 8   # T is the global token count
T_local, E_local = T // world_size, E // world_size

# Build the global MoE once, then slice each rank's E_local expert shard.
moe = MoE(
    num_experts=E,
    num_experts_per_tok=K,
    hidden_size=H,
    intermediate_size=I,
    activation_function=ActivationType.SWIGLU,
    add_bias=False,
    std=0.02,
).to(device=device, dtype=torch.bfloat16)
for p in moe.parameters():
    dist.broadcast(p.data, src=0)

# QuACK's grouped GEMM requires the original (E, *, *) strides, preserved via
# empty_strided + copy_ after permuting to the EP layout.

# EP: shard expert weights evenly across all ranks
w1_sharded = moe.c_fc.weight[rank * E_local : (rank + 1) * E_local].permute(1, 2, 0)    # (2I, H, E_local) view
w2_sharded = moe.c_proj.weight[rank * E_local : (rank + 1) * E_local].permute(0, 2, 1)  # (E_local, I, H) view
w1_sharded_contiguous = torch.empty_strided(w1_sharded.shape, w1_sharded.stride(), dtype=w1_sharded.dtype, device=device).copy_(w1_sharded)
w2_sharded_contiguous = torch.empty_strided(w2_sharded.shape, w2_sharded.stride(), dtype=w2_sharded.dtype, device=device).copy_(w2_sharded)

# !!!!! We assume the router weights are replicated across ranks !!!!!
router_w = moe.router.weight

# Pick the fastest dispatch and combine primitives for this GPU cluster once per (T_local, H, K, dtype).
# We have also construct a `sonicmoe.distributed_utils.RuntimeEPConfig` from scratch by overwriting the Dispatch and Combine mode.
ep_config = NetworkProfiler(T_local=T_local, H=H, K=K, dtype=torch.bfloat16).profile()

# we always assume DP -> EP -> DP !!!
x_local = torch.randn(T_local, H, device=device, dtype=torch.bfloat16)
output_local = moe_ep_TC_softmax_topk_forward(
    x_local,
    router_w,
    w1_sharded_contiguous, None,
    w2_sharded_contiguous, None,
    K=K, E=E,
    ep_config=ep_config,
    activation_type=ActivationType.SWIGLU,
)
```

## 🧪 Tests

- Run the single-GPU test suite to verify correctness:

```bash
make test
```

- Multi-GPU EP correctness (1 dispatch + grouped GEMM + 1 combine end-to-end, plus per-primitive parity vs NCCL):

```bash
torchrun --nproc_per_node=8 --standalone tests/moe_ep_test.py
```

## 📊 Benchmarks

Single-GPU MoE throughput:

```bash
python benchmarks/moe-cute.py --thiek 32768,4096,1024,128,8 --activation swiglu
python benchmarks/moe-token-rounding.py --routing nr --thiekq 16384,4096,1024,256,8,128
```

Intra-node EP:

```bash
torchrun --nproc_per_node=8 --standalone benchmarks/distributed/moe-ep.py --thiek 131072,4096,1536,128,8
```

Intra-node EP communication primitives (Triton vs NCCL baselines on the same byte volume):

```bash
torchrun --nproc_per_node=8 --standalone benchmarks/distributed/bench-ep-comm.py
```

### Example usage

#### Single GPU

- SonicMoE with TC top-K routing (softmax-over-topk, or `softmax(topk(logits))`) and interleaved weight layout format for up-proj weights
    ```bash
    python benchmarks/moe-cute.py --thiek 32768,4096,1024,128,8 --activation swiglu
    ```

- SonicMoE with Qwen3-style routing (topk-over-softmax, or `topk(softmax(logits))`) with topk probabilities renormalization and interleaved weight layout format for up-proj weights
    ```bash
    python benchmarks/moe-cute.py --thiek 32768,4096,1024,128,8 --topk_over_softmax --norm_topk_probs
    ```

- SonicMoE with token rounding routing (SwiGLU activation) and interleaved weight layout format for up-proj weights
    ```bash
    python benchmarks/moe-token-rounding.py --routing nr --thiekq 16384,4096,1024,256,8,128
    ```

- SonicMoE with concatenated weight layout format for up-proj weights

    By default, SonicMoE expects `w1` (the gated up-projection weights) in **interleaved** format: `[gate_0, up_0, gate_1, up_1, ...]`. HuggingFace models (Qwen3, Mixtral, DeepSeek, etc.) store `gate_up_proj` in **concatenated** format: `[gate_0, gate_1, ..., gate_{I-1}, up_0, up_1, ..., up_{I-1}]`.

    ```bash
    # Concatenated weight layout format with TC top-K routing
    python benchmarks/moe-cute.py --thiek 32768,4096,1024,128,8 --concat_layout
    ```

#### Intra-node Expert Parallelism

SonicMoE supports intra-node EP via `moe_ep_TC_softmax_topk_forward`. The forward dispatches each rank's `T_local` tokens to the experts that hold them via NVLink symmetric memory, runs the grouped GEMMs locally, and combines back across NVLink. A runtime `NetworkProfiler` benchmarks the three dispatch and three combine primitives on the local hardware and picks the fastest pair per workload.

EP world size 8:
```bash
torchrun --nproc_per_node=8 --standalone benchmarks/distributed/moe-ep.py --thiek 131072,4096,1536,128,8
```

Override `--dispatch_mode` / `--combine_mode` to lock a specific dispatch / combine primitive (useful for reproducing prior runs):
```bash
torchrun --nproc_per_node=8 --standalone benchmarks/distributed/moe-ep.py \
    --thiek 131072,4096,1536,128,8 \
    --dispatch_mode RANK_DEDUP_DISPATCH_TRITON --combine_mode A2A_COMBINE_TRITON
```

The EP forward exposes an optional flag that trades off activation memory for a host-stall on the forward.

**`--CPU_sync_on_runtime`** (default to False): initiate D2H sync to run *before* the dispatch step to shrink the saved activation cache. The trade-off is a single host stall per forward. Inference mode skips this since no cache is saved.

```bash
torchrun --nproc_per_node=8 --standalone benchmarks/distributed/moe-ep.py --thiek 131072,4096,1536,128,8 --CPU_sync_on_runtime
```

The bench's activation-cache audit prints the actual bytes saved, so you can verify the memory savings on your own workload before turning the flag on in training.


## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use SonicMoE in your research, please cite:

```bibtex
@misc{guo2025sonicmoeacceleratingmoeio,
      title={SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations}, 
      author={Wentao Guo and Mayank Mishra and Xinle Cheng and Ion Stoica and Tri Dao},
      year={2025},
      eprint={2512.14080},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.14080}, 
}
```
