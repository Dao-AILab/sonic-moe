# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import random
from typing import Callable

import torch
from parameterized import parameterized

from sonicmoe import count_cumsum

from .test_commons import TestCommons


def get_1d_tensor_sizes() -> list[tuple[int]]:
    sizes = set()
    # powers of 2
    for i in range(15):
        start = 2**i
        for j in range(10):
            sizes.add(start + j)
    # not powers of 2
    for _ in range(50):
        sizes.add(3000 + random.randint(-1000, 1000))
    return sizes


class CountCumsumTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            list(get_1d_tensor_sizes()) + [2097152],  # size
            [4, 8, 72, 256, 1920, 2048, 16384, 50000],  # num_experts
            [False, True],  # do_cumsum
            [torch.device("cuda")],  # device
            [torch.long, torch.int],  # dtype
            [count_cumsum, torch.compile(count_cumsum, fullgraph=True)],  # function
        )
    )
    def test_count_cumsum(
        self,
        size: int,
        num_experts: int,
        do_cumsum: bool,
        device: torch.device,
        dtype: torch.dtype,
        function: Callable,
    ) -> None:
        torch._dynamo.config.cache_size_limit = 1024
        torch._dynamo.config.accumulated_cache_size_limit = 1024

        x = torch.randint(0, num_experts, (size,), device=device, dtype=dtype)

        z_kernel_cumsum = None
        z_kernel_indices = None

        if do_cumsum:
            z_kernel_count, z_kernel_cumsum = function(x=x, E=num_experts, do_cumsum=do_cumsum)
        else:
            z_kernel_count = function(x=x, E=num_experts, do_cumsum=do_cumsum)

        z_expected_count = x.view(-1).bincount(minlength=num_experts)
        self.assert_equal_tensors(z_kernel_count, z_expected_count, True)

        if z_kernel_cumsum is not None:
            z_expected_cumsum = z_expected_count.cumsum(-1)
            self.assert_equal_tensors(z_kernel_cumsum, z_expected_cumsum, True)
