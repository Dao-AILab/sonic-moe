# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from quack import rmsnorm
from quack.rmsnorm import rmsnorm_fwd

from .enums import ActivationType, KernelBackendMoE, is_glu
from .functional import FP8BlockwiseTensor, moe_TC_softmax_topk_layer, moe_TC_softmax_topk_layer_fp8

# SM90 K-block / dQaccum row-pad granularity, must match forward_fp8._SF
_FP8_QUANT_BLOCK_SIZE = 128


try:
    from xma.modules.moe import scattered_experts

    _IS_XMA_AVAILABLE = True
except ImportError:
    _IS_XMA_AVAILABLE = False


def _swiglu(x: torch.Tensor) -> torch.Tensor:
    u = x[..., 1::2]
    g = x[..., ::2]
    return u * F.silu(g)


def _geglu(x: torch.Tensor) -> torch.Tensor:
    u = x[..., 1::2]
    g = x[..., ::2]
    return (F.gelu(g.to(dtype=torch.float32)) * u).to(dtype=g.dtype)


def _gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x.to(dtype=torch.float32)).to(dtype=x.dtype)


def _reglu(x: torch.Tensor) -> torch.Tensor:
    u = x[..., 1::2]
    g = x[..., ::2]
    return (F.relu(g) * u).to(dtype=g.dtype)


def _relu(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x)


def _relu_sq(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x) ** 2


def _silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


class Experts(nn.Module):
    def __init__(
        self, num_experts: int, in_features: int, out_features: int, add_bias: bool = True, std: float | None = None
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.empty(num_experts, out_features, in_features))

        self.bias = None
        if add_bias:
            self.bias = nn.Parameter(torch.empty(num_experts, out_features))

        self.std = std

        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features

        self.reset_parameters()

    def up_projection_scattermoe_forward(
        self,
        input: torch.Tensor,
        num_experts_per_token: int | None = None,
        sorted_expert_idxs: torch.Tensor | None = None,
        sorted_scattered_idxs: torch.Tensor | None = None,
        expert_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.bias is None

        if not _IS_XMA_AVAILABLE:
            raise ImportError(
                "install accelerated-model-architectures from https://github.com/open-lm-engine/accelerated-model-architectures"
            )

        input = scattered_experts(
            inputs=input,
            expert_weights=self.weight.permute(0, 2, 1),
            k=num_experts_per_token,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            expert_offsets=expert_offsets,
            gates=None,
            grouped_in=False,
            grouped_out=True,
        )

        return input

    def down_projection_scattermoe_forward(
        self,
        input: torch.Tensor,
        num_experts_per_token: int | None = None,
        sorted_expert_idxs: torch.Tensor | None = None,
        sorted_scattered_idxs: torch.Tensor | None = None,
        expert_offsets: torch.Tensor | None = None,
        gates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.bias is None

        if not _IS_XMA_AVAILABLE:
            raise ImportError(
                "install accelerated-model-architectures from https://github.com/open-lm-engine/accelerated-model-architectures"
            )

        input = scattered_experts(
            inputs=input,
            expert_weights=self.weight.permute(0, 2, 1),
            k=num_experts_per_token,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            expert_offsets=expert_offsets,
            gates=gates,
            grouped_in=True,
            grouped_out=False,
        )

        return input

    def torch_forward(
        self, input: torch.Tensor, expert_frequency: torch.Tensor | None, return_list: bool = False
    ) -> list[torch.Tensor] | torch.Tensor:
        if isinstance(input, torch.Tensor):
            input = input.split(expert_frequency.tolist(), dim=0)
        else:
            assert expert_frequency is None

        input = [
            F.linear(input[i], self.weight[i], None if self.bias is None else self.bias[i])
            for i in range(self.num_experts)
        ]

        if not return_list:
            input = torch.cat(input, dim=0)

        return input

    def fp8_weight(self) -> FP8BlockwiseTensor:
        """Blockwise-FP8 (128x128) view of ``self.weight`` for the pre-quantized MoE path.

        In eval / ``no_grad`` (weights static) the quantized payload is cached and reused,
        invalidated on any in-place weight update via ``Tensor._version``. In training a
        fresh straight-through wrapper is built each call so the weight gradient routes
        back into ``self.weight``.
        """
        if torch.is_grad_enabled() and self.weight.requires_grad:
            return FP8BlockwiseTensor.to_weights(self.weight)
        version = self.weight._version
        if getattr(self, "_fp8_weight", None) is None or self._fp8_weight_version != version:
            with torch.no_grad():
                self._fp8_weight = FP8BlockwiseTensor.to_weights(self.weight.detach())
            self._fp8_weight_version = version
        return self._fp8_weight

    def extra_repr(self):
        return "num_experts={}, in_features={}, out_features={}".format(
            self.num_experts, self.in_features, self.out_features
        )

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0, std=self.std)
        if hasattr(self, "bias") and self.bias is not None:
            self.bias.zero_()


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.hidden_size = hidden_size

    def forward(
        self, input: torch.Tensor, quant: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not quant:
            return rmsnorm(input, self.weight, eps=self.eps)

        # Inference-only fused kernel: the norm and its blockwise FP8 quant are
        # computed in one kernel launch (no autograd graph, no extra read/write
        # pass over the normed activation for a separate blockwise_quant call).
        output, _, _, output_q, output_sc = rmsnorm_fwd(
            input, self.weight, eps=self.eps, quant_block_size=_FP8_QUANT_BLOCK_SIZE
        )
        return output, output_q, output_sc

    def extra_repr(self) -> str:
        return "hidden_size={}, eps={}".format(self.hidden_size, self.eps)


class MoE(nn.Module):
    def __init__(
        self,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        intermediate_size: int,
        activation_function: ActivationType,
        add_bias: bool,
        std: float,
    ) -> None:
        super().__init__()

        self.num_experts = num_experts
        self.top_k = num_experts_per_tok

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.router = nn.Linear(in_features=self.hidden_size, out_features=num_experts, bias=False)

        self.activation_function = activation_function

        self.c_fc = Experts(
            num_experts=num_experts,
            in_features=self.hidden_size,
            out_features=2 * self.intermediate_size if is_glu(activation_function) else self.intermediate_size,
            add_bias=add_bias,
            std=std,
        )

        self.c_proj = Experts(
            num_experts=num_experts,
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            add_bias=add_bias,
            std=std,
        )

        self.stream_id = torch.cuda.current_stream().cuda_stream

    def forward(
        self,
        hidden_states: torch.Tensor,
        kernel_backend_moe: KernelBackendMoE = KernelBackendMoE.sonicmoe,
        is_inference_mode: bool = False,
        prequant_fp8_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = hidden_states.shape

        # hidden_states -> (batch_size, query_length, hidden_size)
        hidden_states = hidden_states.view(-1, self.hidden_size)

        is_fp8 = kernel_backend_moe == KernelBackendMoE.sonicmoe_fp8
        if is_fp8:
            # MXFP8 blockscaled path (SM90). `is_inference_mode` skips storing the
            # up-projection's preact (no backward possible then, but saves
            # memory/bandwidth in eval mode). With `prequant_fp8_weights`, the expert
            # weights are handed in already 128x128-quantized (cached across eval calls),
            # so the kernel skips its internal `quantize_weight_sm90`.
            assert self.c_fc.bias is None and self.c_proj.bias is None, (
                "sonicmoe_fp8 does not support bias yet"
            )
            w1 = self.c_fc.fp8_weight() if prequant_fp8_weights else self.c_fc.weight
            w2 = self.c_proj.fp8_weight() if prequant_fp8_weights else self.c_proj.weight
            hidden_states, router_logits, expert_frequency = moe_TC_softmax_topk_layer_fp8(
                hidden_states,
                self.router.weight,
                w1,
                w2,
                self.top_k,
                self.activation_function,
                is_inference_mode=is_inference_mode or not self.training,
            )
        elif kernel_backend_moe == KernelBackendMoE.sonicmoe and self.num_experts <= 32768:
            hidden_states, router_logits, expert_frequency = moe_TC_softmax_topk_layer(
                hidden_states,
                self.router.weight,
                self.c_fc.weight.permute(1, 2, 0),
                self.c_fc.bias,
                self.c_proj.weight.permute(1, 2, 0),
                self.c_proj.bias,
                self.top_k,
                self.stream_id,
                self.activation_function,
                is_inference_mode or not self.training,
            )
        else:
            # hidden_states -> (total_q, hidden_size)
            router_logits, router_weights, selected_experts = self._compute_routing_weights(hidden_states)

            # router_logits -> (total_q, num_experts)
            # router_weights -> (total_q, top_k)
            # selected_experts -> (total_q, top_k)

            hidden_states, expert_frequency = self._compute_experts(
                hidden_states,
                router_weights,
                selected_experts,
                kernel_backend_moe=kernel_backend_moe,
            )

        hidden_states = hidden_states.view(original_shape)

        # hidden_states -> (batch_size, query_length, hidden_size)

        if is_inference_mode or is_fp8:
            aux_loss = None
        else:
            aux_loss = self._compute_switch_loss(
                logits=router_logits,
                probs=F.softmax(router_logits, dim=-1, dtype=torch.float32),
                expert_frequency=expert_frequency,
            )

        return hidden_states, aux_loss

    # copied from https://github.com/open-lm-engine/lm-engine/blob/1447883df709727839bbbb367ce727fa56962a6a/lm_engine/hf_models/modeling_utils/mlp_blocks/moe.py#L432-L455
    # NOTE we don't do all_reduce here for expert frequency for simplicity across data parallel workers
    def _compute_switch_loss(
        self, logits: torch.Tensor, probs: torch.Tensor, expert_frequency: torch.Tensor
    ) -> torch.Tensor:
        logits = logits.view(-1, logits.size(-1))
        probs = probs.view(-1, probs.size(-1))

        num_experts = logits.size(1)
        acc_probs = probs.sum(0)

        expert_frequency = expert_frequency.float()

        aux_loss = num_experts * (F.normalize(acc_probs, p=1, dim=0) * F.normalize(expert_frequency, p=1, dim=0)).sum()

        return aux_loss

    def _compute_routing_weights(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        # hidden_states -> (total_q, hidden_size)
        router_logits = self.router(hidden_states)
        # router_logits -> (total_q, num_experts)

        router_weights, selected_experts = self._get_topk(router_logits)

        # router_weights -> (total_q, top_k)
        # selected_experts -> (total_q, top_k)

        router_weights = F.softmax(router_weights.float(), dim=-1)
        router_weights = router_weights.type_as(hidden_states)

        return router_logits, router_weights, selected_experts

    def _compute_experts(
        self,
        hidden_states: torch.Tensor,
        router_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        kernel_backend_moe: KernelBackendMoE,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected_experts = selected_experts.flatten()

        with torch.no_grad():
            sorted_expert_idxs, sorted_scattered_idxs = selected_experts.sort()

        expert_frequency = selected_experts.bincount(minlength=self.num_experts).to(torch.int32)
        expert_offsets = expert_frequency.cumsum(-1).to(torch.int32)

        act_func = {
            ActivationType.SWIGLU: _swiglu,
            ActivationType.GEGLU: _geglu,
            ActivationType.REGLU: _reglu,
            ActivationType.GELU: _gelu,
            ActivationType.RELU: _relu,
            ActivationType.SILU: _silu,
            ActivationType.RELU_SQ: _relu_sq,
        }[self.activation_function]

        T = hidden_states.size(0)

        if kernel_backend_moe == KernelBackendMoE.scattermoe:
            hidden_states = self.c_fc.up_projection_scattermoe_forward(
                input=hidden_states,
                num_experts_per_token=self.top_k,
                sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                expert_offsets=expert_offsets,
            )
            hidden_states = act_func(hidden_states)
            hidden_states = self.c_proj.down_projection_scattermoe_forward(
                input=hidden_states,
                num_experts_per_token=1,
                sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                expert_offsets=expert_offsets,
                gates=router_weights,
            )
        elif kernel_backend_moe == KernelBackendMoE.torch:
            # sort and group input tokens according to expert assignment
            fan_in_index = sorted_scattered_idxs // self.top_k

            # gather the gate values for grouped input tokens
            router_weights = router_weights.flatten()
            batch_gates = router_weights[sorted_scattered_idxs]

            hidden_states = hidden_states[fan_in_index]

            hidden_states = self.c_fc.torch_forward(
                input=hidden_states, expert_frequency=expert_frequency, return_list=True
            )

            hidden_states = [act_func(i) for i in hidden_states]
            hidden_states = self.c_proj.torch_forward(input=hidden_states, expert_frequency=None, return_list=False)

            hidden_states = hidden_states * batch_gates.unsqueeze(-1)
            zeros = torch.zeros((T, self.hidden_size), dtype=torch.float32, device=hidden_states.device)
            hidden_states = zeros.index_add(0, fan_in_index, hidden_states)
        else:
            raise ValueError(f"unexpected kernel_backend_moe ({kernel_backend_moe})")

        return hidden_states, expert_frequency

    def _get_topk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.top_k == 1:
            x, indices = x.max(dim=-1, keepdim=True)
        else:
            x, indices = x.topk(self.top_k, dim=-1)

        return x, indices
