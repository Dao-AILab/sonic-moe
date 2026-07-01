# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
"""Forward-only MXFP8 blockscaled MoE (SM90).

The kernel is validated against a *dequant reference*: the same 1x128 activation /
128x128 weight quantization the kernel uses, dequantized and run in float32
(including re-quantizing the SwiGLU intermediate before the down GEMM). This
isolates kernel/layout correctness from the inherent fp8 rounding loss — a
comparison against true bf16 would conflate the two (fp8 loss alone is several %).
"""

import pytest
import torch
import torch.nn.functional as F

from sonicmoe import KernelBackendMoE, MoE
from sonicmoe.enums import ActivationType

_SF = 128
_IS_SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] == 9
_skip_not_sm90 = pytest.mark.skipif(not _IS_SM90, reason="MXFP8 MoE forward is SM90-only")


def _cos_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return (1 - F.cosine_similarity(a, b, dim=0)).abs().item()


def _dequant_ref(x, router_w, w1, w2, K, activation):
    """Full MoE in float32 using tensors quantized+dequantized exactly like the kernel."""
    from quack.gemm_blockscaled_interface import quantize_act_sm90, quantize_weight_sm90

    T, H = x.shape
    E = router_w.shape[0]

    def deq_act(t):
        q, sc = quantize_act_sm90(t)
        return q.float() * sc.float().repeat_interleave(_SF, dim=-1)

    def deq_w(w):
        q, sc = quantize_weight_sm90(w)
        return q.float() * sc.float().repeat_interleave(_SF, dim=-2).repeat_interleave(_SF, dim=-1)

    x_dq = deq_act(x)
    w1_dq = deq_w(w1)
    w2_dq = deq_w(w2)

    logits = F.linear(x, router_w).float()
    top_v, top_i = logits.topk(K, dim=-1)
    probs = torch.softmax(top_v, dim=-1)  # softmax-over-topk
    out = torch.zeros(T, H, device=x.device, dtype=torch.float32)
    for e in range(E):
        tok, slot = (top_i == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        pre = x_dq[tok] @ w1_dq[e].T  # (n, 2I)
        g, u = pre[:, 0::2], pre[:, 1::2]  # interleaved [gate, up]
        if activation == ActivationType.SWIGLU:
            act = u * F.silu(g)
        else:  # GEGLU
            act = u * F.gelu(g)
        act_dq = deq_act(act.to(torch.bfloat16))  # kernel re-quantizes before down GEMM
        down = act_dq @ w2_dq[e].T
        out[tok] += probs[tok, slot].unsqueeze(-1) * down
    return out


@_skip_not_sm90
@pytest.mark.parametrize("activation", [ActivationType.SWIGLU, ActivationType.GEGLU])
@pytest.mark.parametrize(
    "T, H, I, E, K",
    [
        (512, 512, 512, 8, 2),
        (1024, 768, 512, 16, 4),
        (2048, 1536, 1024, 8, 2),
        (333, 512, 256, 8, 2),  # non-multiple-of-128 token count (varlen padding stress)
    ],
)
def test_moe_fp8_forward(T, H, I, E, K, activation):
    torch.manual_seed(42)
    with torch.device("cuda"):
        moe = MoE(
            num_experts=E,
            num_experts_per_tok=K,
            hidden_size=H,
            intermediate_size=I,
            activation_function=activation,
            add_bias=False,
            std=0.02,
        ).to(dtype=torch.bfloat16)
    moe.eval()

    x = 0.2 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        y_fp8 = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe_fp8)[0]

    y_ref = _dequant_ref(x, moe.router.weight, moe.c_fc.weight, moe.c_proj.weight, K, activation)

    assert y_fp8.shape == (T, H)
    cos = _cos_diff(y_fp8, y_ref)
    assert cos < 1e-2, f"fp8 MoE vs dequant ref cos_diff={cos:.6f} >= 1e-2"
