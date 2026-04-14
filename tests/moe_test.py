# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import torch
from parameterized import parameterized

from sonicmoe import KernelBackendMoE, MoE, enable_quack_gemm, moe_general_routing_inputs
from sonicmoe.enums import ActivationType

from .test_commons import TestCommons


_SEED = 42
torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.accumulated_cache_size_limit = 1024
torch._functorch.config.donated_buffer = False


class MoETest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")],
            [torch.bfloat16],
            [
                ((16384 + 512) * 16, 512, 512, 128, 8)(8192, 768, 256, 128, 8),
                (8192, 768, 512, 64, 4),
                (8192, 768, 1024, 32, 2),
                (8192, 1536, 256, 128, 8),
                (8192, 1536, 512, 64, 4),
                (8192, 1536, 1024, 32, 2),
                (8192, 4096, 256, 256, 16),
                (8192, 4096, 512, 128, 8),
                (8192, 4096, 1024, 64, 4),
                (8192, 4096, 512, 256, 16),
                (8192, 4096, 1024, 128, 8),
                (8192, 4096, 2048, 64, 4),
            ],
            [KernelBackendMoE.sonicmoe],  # kernel_backend_moe
            [False, True],  # is_compiling
            [False, True],  # add_bias
            [False, True],  # use_quack_gemm
        )
    )
    def test_moe(
        self,
        device: torch.device,
        dtype: torch.dtype,
        problem_shape: tuple[int, int, int, int, int],
        kernel_backend_moe: KernelBackendMoE,
        is_compiling: bool,
        add_bias: bool,
        use_quack_gemm: bool,
    ) -> None:
        if use_quack_gemm and (is_compiling or add_bias):
            self.skipTest("unsupported test")

        self.set_seed(_SEED)

        T, H, I, E, K = problem_shape
        with torch.device(device):
            moe = MoE(
                num_experts=E,
                num_experts_per_tok=K,
                hidden_size=H,
                intermediate_size=I,
                activation_function=ActivationType.SWIGLU,
                add_bias=add_bias,
                std=0.02,
            ).to(dtype=dtype)

        if add_bias:
            b1, b2 = moe.c_fc.bias, moe.c_proj.bias
            torch.nn.init.normal_(b1, 0, 0.01)
            torch.nn.init.normal_(b2, 0, 0.01)

        moe_kernel = moe
        moe_torch = moe

        if is_compiling:
            moe_kernel = torch.compile(moe_kernel, fullgraph=True)

        torch.cuda.empty_cache()
        x_torch = 0.02 * torch.randn(T, H, device=device, dtype=dtype, requires_grad=True)
        x_kernel = x_torch.clone().detach().requires_grad_()

        with torch.autocast(x_torch.device.type, torch.float32):
            with enable_quack_gemm(use_quack_gemm):
                y_kernel = moe_kernel(x_kernel, kernel_backend_moe=kernel_backend_moe)[0]

            y_torch = moe_torch(x_torch, kernel_backend_moe=KernelBackendMoE.torch)[0]
            self.assert_equal_tensors(
                y_kernel.float(),
                y_torch.float(),
                False,
                atol_bfloat16=1.4e-2,
                rtol_bfloat16=2e-2,
                dtype=dtype,
            )

        dy_torch = 0.02 * torch.randn(T, H, device=device, dtype=dtype, requires_grad=True)
        dy_kernel = dy_torch.clone().detach().requires_grad_()

        W = list(moe.parameters())

        with torch.autocast(x_torch.device.type, torch.float32):
            kernel_grads = torch.autograd.grad(y_kernel, [x_kernel] + W, grad_outputs=dy_kernel, retain_graph=True)
            torch_grads = torch.autograd.grad(y_torch, [x_torch] + W, grad_outputs=dy_torch, retain_graph=True)

            for _torch_grad, _kernel_grad in zip(torch_grads, kernel_grads):
                self.assert_equal_tensors(
                    _kernel_grad.float(),
                    _torch_grad.float(),
                    False,
                    atol_bfloat16=2e-2,
                    rtol_bfloat16=2e-2,
                    dtype=dtype,
                )

            for w in W:
                w.grad = None

        torch_grads = kernel_grads = None
        torch.cuda.empty_cache()


class ConcatenatedGateUpTest(TestCommons):
    """Tests for is_concatenated_gate_up: verify concatenated layout matches interleaved (used as reference)."""

    @staticmethod
    def _make_weights(E, I, H, device, dtype):
        """Create interleaved (reference) and concatenated weight pairs."""
        w1_inter = torch.randn(E, 2 * I, H, device=device, dtype=dtype).permute(1, 2, 0)
        gate = w1_inter[0::2].permute(2, 0, 1)
        up = w1_inter[1::2].permute(2, 0, 1)
        w1_concat = torch.cat([gate, up], dim=1).contiguous().permute(1, 2, 0)
        w2 = torch.randn(E, H, I, device=device, dtype=dtype).permute(1, 2, 0)
        return w1_inter, w1_concat, w2

    @staticmethod
    def _make_routing(T, E, K, device):
        topk_indices = torch.randint(0, E, (T, K), device=device)
        topk_weights = torch.randn(T, K, device=device, dtype=torch.bfloat16).softmax(dim=-1)
        token_idx = torch.arange(T, device=device, dtype=torch.int32).unsqueeze(1).expand(-1, K).reshape(-1)
        expert_ids = topk_indices.reshape(-1).to(torch.int32)
        router_scores = topk_weights.reshape(-1).to(torch.bfloat16)
        return token_idx, expert_ids, router_scores

    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")],
            [torch.bfloat16],
            [
                (8192, 768, 256, 128, 8),
                (8192, 768, 512, 64, 4),
                (8192, 4096, 512, 128, 8),
                (8192, 4096, 1024, 64, 4),
            ],
            [ActivationType.SWIGLU, ActivationType.GEGLU, ActivationType.REGLU],
            [False, True],  # add_bias
        )
    )
    def test_concatenated_gate_up(
        self,
        device: torch.device,
        dtype: torch.dtype,
        problem_shape: tuple[int, int, int, int, int],
        activation_type: ActivationType,
        add_bias: bool,
    ) -> None:
        self.set_seed(_SEED)

        T, H, I, E, K = problem_shape
        w1_inter, w1_concat, w2 = self._make_weights(E, I, H, device, dtype)
        b1 = torch.randn(E, 2 * I, device=device, dtype=dtype) if add_bias else None
        b2 = torch.randn(E, H, device=device, dtype=dtype) if add_bias else None

        x_ref = 0.02 * torch.randn(T, H, device=device, dtype=dtype, requires_grad=True)
        x_test = x_ref.clone().detach().requires_grad_()
        w1_inter_p = w1_inter.clone().detach().requires_grad_()
        w1_concat_p = w1_concat.clone().detach().requires_grad_()

        token_idx, expert_ids, router_scores = self._make_routing(T, E, K, device)
        stream_id = torch.cuda.current_stream(device).cuda_stream

        with torch.autocast(device.type, torch.float32):
            out_ref, _ = moe_general_routing_inputs(
                x_ref, router_scores, token_idx, expert_ids,
                w1_inter_p, b1, w2, b2,
                E, stream_id, activation_type,
                is_inference_mode_enabled=False, is_concatenated_gate_up=False,
            )
            out_test, _ = moe_general_routing_inputs(
                x_test, router_scores, token_idx, expert_ids,
                w1_concat_p, b1, w2, b2,
                E, stream_id, activation_type,
                is_inference_mode_enabled=False, is_concatenated_gate_up=True,
            )

            self.assertTrue(torch.equal(out_ref, out_test),
                            f"fwd max_diff={(out_ref - out_test).abs().max().item()}")

        # Backward
        dy = 0.02 * torch.randn(T, H, device=device, dtype=dtype)
        grads_ref = torch.autograd.grad(out_ref, [x_ref, w1_inter_p], grad_outputs=dy)
        grads_test = torch.autograd.grad(out_test, [x_test, w1_concat_p], grad_outputs=dy)

        # dx
        self.assertTrue(torch.equal(grads_ref[0], grads_test[0]),
                        f"dx max_diff={(grads_ref[0] - grads_test[0]).abs().max().item()}")

        # dw1: convert interleaved grad to concatenated layout for comparison
        dw1_i = grads_ref[1]
        gate_grad = dw1_i[0::2].permute(2, 0, 1)
        up_grad = dw1_i[1::2].permute(2, 0, 1)
        dw1_as_concat = torch.cat([gate_grad, up_grad], dim=1).contiguous().permute(1, 2, 0)
        self.assertTrue(torch.equal(dw1_as_concat, grads_test[1]),
                        f"dw1 max_diff={(dw1_as_concat - grads_test[1]).abs().max().item()}")

        torch.cuda.empty_cache()
