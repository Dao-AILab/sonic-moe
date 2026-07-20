"""Profile forward + forward/backward of ydt_core's fp8 MoE expert path, mirroring
`fp8_trace.py` (same warmup / torch.profiler / chrome-trace flow) so the two fp8 MoE
implementations can be compared.

Replicates the full ydt MoE block: TopKRouter.forward (gate matmul -> softmax -> top-k ->
histc, ydt_core/nn/modules/moe/router.py) then FP8ForwardImpl.__call__
(ydt_core/nn/modules/moe/experts.py): quantize activations -> `aligned_permute` (dispatch
tokens into expert-major order) -> `moe_swiglu` (grouped fp8 expert GEMMs) ->
`aligned_unpermute` (gather back to token order). The router is replicated inline with the
same ops (importing TopKRouter would pull in module/distributed machinery). Sized to match
fp8_trace.py: T=32768, H=4096, I=1024, E=512, K=10.

Two env shims (isolated to this script; the ydt source is untouched):
  - stub `ydt_core.distributed.activation_checkpointing` — ydt_core.fp8.__init__ eagerly
    imports the yccl distributed stack, which is ABI-broken in this env.
  - alias `F.grouped_mm = torch._grouped_mm` — the ydt backward calls the former; this
    torch (2.9.1) exposes the same aten op only as the latter.
"""

import sys
import types

_dist = types.ModuleType("ydt_core.distributed")
_dist.__path__ = []
_ac = types.ModuleType("ydt_core.distributed.activation_checkpointing")
_ac.unwrap_module = lambda m: m
sys.modules["ydt_core.distributed"] = _dist
sys.modules["ydt_core.distributed.activation_checkpointing"] = _ac

import torch
from torch.profiler import ProfilerActivity, profile

if not hasattr(torch.nn.functional, "grouped_mm"):
    torch.nn.functional.grouped_mm = torch._grouped_mm

from ydt_core.fp8.permute import aligned_permute, aligned_unpermute
from ydt_core.fp8.swiglu import TMA_ALIGN, moe_swiglu
from ydt_core.fp8.tensor import FP8BlockwiseTensor

# Match fp8_trace.py: T, H, I, E, K = 32768, 4096, 1024, 512, 10
T, H, I, E, K = 8192, 4096, 1024, 512 // 8, 10
torch.manual_seed(0)
device = "cuda"

x = (0.2 * torch.randn(T, H, device=device, dtype=torch.bfloat16)).requires_grad_()
gate_w = (torch.randn(E, H, device=device, dtype=torch.bfloat16) / H**0.5).requires_grad_()  # router
up_proj = (torch.randn(E, I, H, device=device, dtype=torch.bfloat16) / H**0.5).requires_grad_()
gate_proj = (torch.randn(E, I, H, device=device, dtype=torch.bfloat16) / H**0.5).requires_grad_()
down_proj = (torch.randn(E, H, I, device=device, dtype=torch.bfloat16) / I**0.5).requires_grad_()
dout = torch.randn(T, H, device=device, dtype=torch.bfloat16)


def _router(x_in):
    # TopKRouter.forward: gate matmul -> float32 softmax -> top-k -> per-expert histogram.
    expert_scores = torch.nn.functional.linear(x_in, gate_w)  # (T, E)
    expert_scores = torch.softmax(expert_scores, dim=-1, dtype=torch.float32)
    top_k_scores, top_k_indices = expert_scores.topk(K, dim=-1)  # (T, K)
    top_k_indices = top_k_indices.to(torch.int32)
    num_tokens_per_expert = torch.histc(
        top_k_indices.view(-1).float(), bins=E, min=0, max=E - 1
    ).to(torch.int64)  # (E,)
    return top_k_scores, top_k_indices, num_tokens_per_expert


def _experts(x_in):
    top_k_scores, top_k_indices, num_tokens_per_expert = _router(x_in)
    # quantize activations -> permute into expert-major order -> grouped fp8 SwiGLU
    # -> unpermute back to token order  (FP8ForwardImpl.__call__)
    xq = FP8BlockwiseTensor.to_activations(x_in)
    xp, inverse_offsets, moe_indices, probs = aligned_permute(
        xq,
        top_k_indices,
        num_tokens_per_expert,
        TMA_ALIGN,
        None,  # num_out_tokens / token_drop_threshold
        top_k_expert_scores=top_k_scores,
    )
    y = moe_swiglu(
        xp,
        up_proj,
        gate_proj,
        down_proj,
        probs=probs,
        batch_sizes=num_tokens_per_expert,
        moe_indices=moe_indices,
    )
    return aligned_unpermute(y, inverse_offsets, probs=None)


def fwd():
    with torch.no_grad():
        return _experts(x)


def fwd_bwd():
    o = _experts(x)
    o.backward(dout)
    x.grad = gate_w.grad = up_proj.grad = gate_proj.grad = down_proj.grad = None


for _ in range(20):  # warmup: compile + autotune
    fwd()
    fwd_bwd()
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        fwd()
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
trace = "scratchpad/ydt_moe_fwd_trace.json"
prof.export_chrome_trace(trace)
print("\nchrome trace:", trace)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        fwd_bwd()
    torch.cuda.synchronize()

print("\n" + prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
trace = "scratchpad/ydt_moe_fwd_bwd_trace.json"
prof.export_chrome_trace(trace)
print("\nchrome trace:", trace)
