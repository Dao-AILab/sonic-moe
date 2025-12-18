# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

# You'd need to install transformer_engine >= 2.8, which isn't available on PyPI yet.
# So you'd need to install from source (see their README)

import itertools
import os
import time

import torch

# from test_utilities import Utils
import torch.nn.functional as F
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec  # TE spec provider
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version

# from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


def construct_moe_layer(
    tp_size, ep_size, num_moe_experts, hidden_size, intermediate_size, K, moe_token_dispatcher_type, grouped_gemm
):
    """
    Tests the forward and backward pass of a MoELayer.
    """
    if not torch.cuda.is_available():
        print("Skipping test, CUDA not available.")
        return

    if grouped_gemm and not is_te_min_version("1.7.0.dev0"):
        print("Skipping test: Grouped GEMM for MoE requires TE >= 1.7.0")
        return

    # _set_random_seed(seed_=123, data_parallel_random_init=False)

    transformer_config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        moe_ffn_hidden_size=intermediate_size,
        num_attention_heads=4,
        num_moe_experts=num_moe_experts,
        use_cpu_initialization=False,
        # moe_token_dispatcher_type=moe_token_dispatcher_type,
        moe_router_topk=K,
        # moe_aux_loss_coeff=0.01,
        moe_grouped_gemm=grouped_gemm,
        add_bias_linear=False,  # ？
        bf16=True,
        params_dtype=torch.bfloat16,
        # tensor_model_parallel_size=tp_size,
        # expert_model_parallel_size=ep_size,
        # sequence_parallel=tp_size > 1,
        activation_func=F.silu,
        gated_linear_unit=True,
    )

    # transformer_layer_spec = get_gpt_layer_local_spec(
    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=num_moe_experts, moe_grouped_gemm=grouped_gemm
    )

    moe_layer = MoELayer(transformer_config, transformer_layer_spec.submodules.mlp.submodules).cuda()

    return moe_layer, transformer_config


#     hidden_states = torch.rand(
#         (10, 4096, transformer_config.hidden_size), dtype=torch.bfloat16, device="cuda", requires_grad=True
#     )

#     output, _ = moe_layer(hidden_states)
#     output.mean().backward()

#     assert hidden_states.grad is not None
#     assert moe_layer.router.weight.grad is not None
# print("... PASSED")
# finally:
#     Utils.destroy_model_parallel()


def test_forward(moe_layer, input_tensor, dout):
    # Utils.initialize_model_parallel(1,1)
    output, aux_loss = moe_layer(input_tensor)


def test_forward_backward(moe_layer, input_tensor, dout):
    # Utils.initialize_model_parallel(1,1)
    output, aux_loss = moe_layer(input_tensor)
    output.backward(dout)

    assert input_tensor.grad is not None
    assert moe_layer.router.weight.grad is not None


if __name__ == "__main__":
    os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM"] = "1"
    # os.environ["NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK"] = "1"
    # os.environ["GLOO_DEBUG_LEVEL"] = "0"
    # os.environ["TORCH_CPP_LOG_LEVEL"] = "FATAL"
    Utils.initialize_model_parallel(1, 1)

    from triton.testing import do_bench

    T, I, H, K, E = 40960, 256, 1536, 16, 256
    for T, H, I, E, K in [
        (40960, 768, 64, 512, 32),
        (40960, 768, 256, 128, 8),
        (40960, 768, 1024, 32, 2),
        (24576, 1536, 64, 512, 32),
        (24576, 1536, 256, 128, 8),
        (24576, 1536, 1024, 32, 2),
        (32768, 4096, 256, 256, 16),
        (32768, 4096, 512, 128, 8),
        (32768, 4096, 1024, 64, 4),
        (32768, 4096, 512, 256, 16),
        (32768, 4096, 1024, 128, 8),
        (32768, 4096, 2048, 64, 4),
    ]:
        print(f"\nT={T}, H={H}, I={I}, E={E}, K={K}")

        # bs, sl = 10, 4096
        sl = 2048
        bs = T // sl
        # assert T == bs * sl
        parallel_sizes = [(1, 1)]
        expert_counts = [E]
        # dispatchers = ["allgather", "alltoall"]
        dispatchers = ["alltoall"]
        # gemm_modes = [True, False]
        gemm_modes = [True]

        param_combinations = itertools.product(parallel_sizes, expert_counts, dispatchers, gemm_modes)
        input_tensor = torch.randn(bs, sl, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        dout = torch.randn_like(input_tensor) * 0.2
        for (tp_size, ep_size), num_moe_experts, moe_token_dispatcher_type, grouped_gemm in param_combinations:
            moe_layer, transformer_cfg = construct_moe_layer(
                tp_size, ep_size, num_moe_experts, H, I, K, moe_token_dispatcher_type, grouped_gemm
            )
            # print(f"\nTesting MoE Layer: tp={tp_size}, ep={ep_size}, experts={num_moe_experts}, dispatcher={moe_token_dispatcher_type}, grouped_gemm={grouped_gemm}")
            time.sleep(1.0)
            # fn = lambda: test_forward_backward(moe_layer, input_tensor, dout)
            fn = lambda: test_forward(moe_layer, input_tensor, dout)
            e2e_time = do_bench(fn, warmup=10, rep=100)
            # print(f"Average time per iteration: {e2e_time:.3f} ms")
            # flops = 18 * T * I * H * K
            flops = 6 * T * I * H * K
            print(f"TFLOPS:{flops / (e2e_time / 1e3) / 1e12:.2f}")

            # test_forward_backward(tp_size, ep_size, num_moe_experts, moe_token_dispatcher_type, grouped_gemm)

    print("\nAll tests finished successfully.")
