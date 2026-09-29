# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from collections import Counter

import quack.autotuner
import torch

from sonicmoe import KernelBackendMoE, MoE
from sonicmoe.enums import ActivationType

from .test_commons import TestCommons


class AutotuneKeyTest(TestCommons):
    def test_new_token_count_in_bucket_does_not_retune(self) -> None:
        # Packed variable-length batches give a new token count almost every step; once a bucket is
        # tuned, none of the GEMMs (including the varlen-K weight-gradient ones) may sweep again.
        self.set_seed(42)
        with torch.device("cuda"):
            moe = MoE(
                num_experts=64,
                num_experts_per_tok=4,
                hidden_size=768,
                intermediate_size=512,
                activation_function=ActivationType.SWIGLU,
                add_bias=False,
                std=0.02,
            ).to(dtype=torch.bfloat16)

        sweeps = Counter()
        prune_configs = quack.autotuner.Autotuner.prune_configs

        def counting_prune_configs(self, kwargs):  # QuACK prunes only when it is about to sweep
            sweeps[self.fn.__name__] += 1
            return prune_configs(self, kwargs)

        quack.autotuner.Autotuner.prune_configs = counting_prune_configs
        try:
            for i, T in enumerate([5000, 6143, 7001]):  # one M bucket, three token counts
                x = 0.02 * torch.randn(T, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                y = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe)[0]
                y.backward(torch.randn_like(y))
                torch.cuda.synchronize()
                if i == 0:
                    sweeps.clear()
            self.assertEqual(dict(sweeps), {})
        finally:
            quack.autotuner.Autotuner.prune_configs = prune_configs
