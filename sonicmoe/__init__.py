# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

__version__ = "0.1.1"

from .enums import KernelBackendMoE
from .functional import enable_quack_gemm, moe_general_routing_inputs, moe_TC_softmax_topk_layer
from .moe import MoE
