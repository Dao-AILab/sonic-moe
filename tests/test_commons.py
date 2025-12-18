# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import random
from itertools import product
from typing import Any
from unittest import TestCase

import numpy as np
import torch
import torch.nn as nn
from torch.testing import assert_close


class TestCommons(TestCase):
    @staticmethod
    def get_dtypes() -> list[torch.dtype]:
        return [torch.float32, torch.float16, torch.bfloat16]

    @staticmethod
    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def make_args_matrix(*args_lists) -> list[Any]:
        return [p for p in product(*args_lists)]

    def assert_equal_tensors(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        exact_match: bool,
        rtol_float32: float = None,
        atol_float32: float = None,
        rtol_float16: float = None,
        atol_float16: float = None,
        rtol_bfloat16: float = None,
        atol_bfloat16: float = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if exact_match:
            assert x.equal(y)
        else:
            assert x.dtype == y.dtype

            if dtype == torch.float32:
                assert_close(x, y, rtol=rtol_float32, atol=atol_float32)
            elif dtype == torch.float16:
                assert_close(x, y, rtol=rtol_float16, atol=atol_float16)
            elif dtype == torch.bfloat16:
                assert_close(x, y, rtol=rtol_bfloat16, atol=atol_bfloat16)
            else:
                raise ValueError(f"unexpected dtype ({dtype})")

    def get_activation_function(self, is_glu: bool) -> nn.Module:
        return nn.GLU() if is_glu else nn.GELU(approximate="tanh")

    def collect_gradients_from_module_and_zero_grads(self, model: nn.Module) -> dict[str, torch.Tensor]:
        grads = {}
        for weight_name, weight in model.named_parameters():
            grads[weight_name] = weight.grad

        model.zero_grad()

        return grads
