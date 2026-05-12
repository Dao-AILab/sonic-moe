# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import torch
from parameterized import parameterized

from sonicmoe import KernelBackendMoE, MoE
from sonicmoe.enums import ActivationType
from sonicmoe.functional import moe_TC_softmax_topk_layer

from .test_commons import TestCommons


_SEED = 42
torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.accumulated_cache_size_limit = 1024
torch._functorch.config.donated_buffer = False


# Scale shape matrix to GPU memory: 80GB+ (H100/B200) runs everything; 40-80GB
# (A100-40) shrinks the largest-T shape; <40GB (RTX 5090/4090, A6000) further
# shrinks T0 and drops H=4096 shapes whose expert weights alone won't fit.
_GPU_MEM_GB = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0.0
if _GPU_MEM_GB >= 80:
    _T0 = (16384 + 512) * 16
    _DROP_LARGE_H4096 = False
elif _GPU_MEM_GB >= 40:
    _T0 = 65536
    _DROP_LARGE_H4096 = False
else:
    _T0 = 8192
    _DROP_LARGE_H4096 = True

_SHAPES = [
    (_T0, 512, 512, 128, 8),
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
]
if _DROP_LARGE_H4096:
    # Weight footprint = E * 2I * H * 4 bytes (transient f32 init); skip cases
    # where E*I > 64K so MoE construction fits alongside the previous test's
    # allocator carry-over on a 32 GB GPU.
    _SHAPES = [s for s in _SHAPES if not (s[1] == 4096 and s[3] * s[2] > 65536)]


class MoETest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")],
            [torch.bfloat16],
            _SHAPES,
            [KernelBackendMoE.sonicmoe],  # kernel_backend_moe
            [
                False,
            ],  # is_compiling
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
        torch.cuda.empty_cache()
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

        x_torch = 0.02 * torch.randn(T, H, device=device, dtype=dtype, requires_grad=True)
        x_kernel = x_torch.clone().detach().requires_grad_()

        with torch.autocast(x_torch.device.type, torch.float32):
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


# Small shapes where H and I are divisible by 32 (MXFP8 block-size requirement)
_FP8_SHAPES = [
    (512, 512, 256, 32, 2),
    (1024, 1024, 512, 64, 4),
]


class MixedFp8TrainingTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")],
            _FP8_SHAPES,
        )
    )
    def test_mixed_fp8_fwd_bf16_bwd(
        self,
        device: torch.device,
        problem_shape: tuple[int, int, int, int, int],
    ) -> None:
        # Require SM100 (Blackwell) for MXFP8 block-scaled GEMMs
        cap = torch.cuda.get_device_capability()
        if cap < (10, 0):
            self.skipTest(f"mixed fp8 requires SM100+, got SM{cap[0]}{cap[1]}")

        self.set_seed(_SEED)
        T, H, I, E, K = problem_shape
        dtype = torch.bfloat16

        with torch.device(device):
            moe = MoE(
                num_experts=E,
                num_experts_per_tok=K,
                hidden_size=H,
                intermediate_size=I,
                activation_function=ActivationType.SWIGLU,
                add_bias=False,
                std=0.02,
            ).to(dtype=dtype)

        w1 = moe.c_fc.weight.permute(1, 2, 0)    # (2*I, H, E)
        w2 = moe.c_proj.weight.permute(1, 2, 0)  # (H, I, E)
        router_w = moe.router.weight

        x_ref = 0.02 * torch.randn(T, H, device=device, dtype=dtype, requires_grad=True)
        x_fp8 = x_ref.clone().detach().requires_grad_()
        dout = 0.02 * torch.randn(T, H, device=device, dtype=dtype)

        # bf16 reference forward + backward
        o_ref, _, _ = moe_TC_softmax_topk_layer(
            x_ref, router_w, w1, None, w2, None, K, None,
            ActivationType.SWIGLU, is_inference_mode_enabled=False, use_fp8=False,
        )
        o_ref.backward(dout)
        dx_ref = x_ref.grad.clone()

        # mixed fp8 forward + bf16 backward
        o_fp8, _, _ = moe_TC_softmax_topk_layer(
            x_fp8, router_w, w1, None, w2, None, K, None,
            ActivationType.SWIGLU, is_inference_mode_enabled=False, use_fp8=True,
        )
        o_fp8.backward(dout)
        dx_fp8 = x_fp8.grad.clone()

        # Loose tolerance — mixed fp8 fwd, bf16 bwd
        torch.testing.assert_close(o_fp8.float(), o_ref.float(), rtol=0.2, atol=0.2)
        torch.testing.assert_close(dx_fp8.float(), dx_ref.float(), rtol=0.2, atol=0.2)
