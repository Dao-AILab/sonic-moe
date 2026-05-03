# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
import math

import torch
import triton
import triton.language as tl

from .triton_comm import (
    a2a_dispatch_pull_triton,
    all_gather_copy_engine_async,
    all_gather_triton,
    compute_dispatch_metadata,
    gather_aggregation_triton,
    reduce_scatter_triton,
    rs_aggregation,
)
