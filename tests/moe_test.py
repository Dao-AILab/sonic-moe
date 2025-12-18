# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import torch
from parameterized import parameterized

from sonicmoe import KernelBackendMoE, MoE, enable_quack_gemm

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
                (8192, 768, 256, 128, 8),
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
                is_glu=True,
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
