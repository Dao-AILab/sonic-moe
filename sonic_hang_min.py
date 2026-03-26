"""
Minimal FSDP + sonic-moe race condition reproducer.

Dependencies tested with:

uv venv --python 3.12
. .venv/bin/activate
uv pip install torch==2.12.0.dev20260324+cu126 --index-url https://download.pytorch.org/whl/nightly/cu126
uv pip install --no-deps -e .
uv pip install nvidia-cutlass-dsl==4.4.0 quack-kernels==0.2.5

Usage:
  torchrun --nproc_per_node=2 <this-file.py>
"""

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from sonicmoe.enums import ActivationType
from sonicmoe.functional import moe_general_routing_inputs
from torch.distributed._composable.fsdp import fully_shard

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=100_000)
parser.add_argument("--num-experts", type=int, default=128)
parser.add_argument("--num-layers", type=int, default=4)
parser.add_argument("--top-k", type=int, default=16)
parser.add_argument("--hidden-dim", type=int, default=128)
parser.add_argument("--model-dim", type=int, default=2048)
parser.add_argument("--seq-len", type=int, default=2048)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--timeout", type=int, default=60)


class SonicMoELayer(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts, top_k):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        nn.init.trunc_normal_(self.gate.weight, std=0.02)
        nn.init.trunc_normal_(self.w1, std=0.02)
        nn.init.trunc_normal_(self.w2, std=0.02)
        nn.init.trunc_normal_(self.w3, std=0.02)

    def _route(self, x_flat):
        scores = F.softmax(self.gate(x_flat).float(), dim=-1)
        top_scores, selected = torch.topk(scores, k=self.top_k, dim=-1)
        return top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-20), selected

    def forward(self, x):
        bs, slen, dim = x.shape
        x_flat = x.view(-1, dim)
        T = x_flat.shape[0]

        top_scores, selected_experts = self._route(x_flat)
        token_indices = (
            torch.arange(T, device=x.device)
            .unsqueeze(1)
            .expand(-1, self.top_k)
            .reshape(-1)
        )
        w1_w3 = (
            torch.stack([self.w1, self.w3], dim=2)
            .reshape(self.num_experts, 2 * self.hidden_dim, dim)
            .permute(1, 2, 0)
        )
        w2_sonic = self.w2.permute(1, 2, 0)

        out, _ = moe_general_routing_inputs(
            x=x_flat,
            router_scores=top_scores.view(-1),
            token_indices=token_indices.int(),
            expert_indices=selected_experts.view(-1).int(),
            w1=w1_w3,
            b1=None,
            w2=w2_sonic,
            b2=None,
            E=self.num_experts,
            stream_id=0,
            activation_type=ActivationType.SWIGLU,
            is_inference_mode_enabled=False,
        )
        return out.view(bs, slen, dim)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SonicMoELayer(
                    args.model_dim,
                    args.hidden_dim,
                    args.num_experts,
                    args.top_k,
                )
                for _ in range(args.num_layers)
            ]
        )
        self.input = nn.Linear(args.model_dim, args.model_dim, bias=False)
        self.output = nn.Linear(args.model_dim, args.model_dim, bias=False)

    def forward(self, x):
        x = self.input(x)
        for layer in self.layers:
            x = x + layer(x)
        return self.output(x)


if __name__ == "__main__":
    args = parser.parse_args()
    try:
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=args.timeout))
        rank = dist.get_rank()
        device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
        torch.cuda.set_device(device)

        def log(msg):
            if rank == 0:
                print(msg, flush=True)

        log(f"{args=}")

        with device:
            model = Model().bfloat16()
        x = torch.randn(
            args.batch_size,
            args.seq_len,
            args.model_dim,
            device=device,
            dtype=torch.bfloat16,
        )

        # Run once, in case we need to compile the kernels
        model(x).sum().backward()

        for layer in model.layers:
            fully_shard(layer)
        fully_shard(model)

        log(f"Running {args.steps} steps\n{model=}\n")
        for step in range(args.steps):
            model(x).sum().backward()
            model.zero_grad()
            if step % 10 == 0:
                log(f"  Step {step}/{args.steps}")

        log(f"Completed {args.steps} steps without error!")
    finally:
        dist.destroy_process_group()
