# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import torch

from ..enums import LIBRARY_NAME
from ..jit import cpp_jit


@torch.library.custom_op(f"{LIBRARY_NAME}::count_cumsum_cuda", mutates_args={"count_output", "cumsum_output"})
@cpp_jit()
def count_cumsum_cuda(x: torch.Tensor, count_output: torch.Tensor, cumsum_output: torch.Tensor | None, stream: int) -> None: ...


@torch.no_grad()
def count_cumsum(x: torch.Tensor, E: int, do_cumsum: bool = True) -> torch.Tensor:
    assert x.dim() == 1, "x should be 1-dimensional"
    assert x.dtype in [torch.int32, torch.long]

    count_output = torch.empty(E, dtype=torch.int32, device=x.device)
    cumsum_output = torch.empty(E, dtype=torch.int32, device=x.device) if do_cumsum else None
    stream = torch.cuda.current_stream(x.device).cuda_stream

    count_cumsum_cuda(x=x, count_output=count_output, cumsum_output=cumsum_output, stream=stream)

    if do_cumsum:
        return count_output, cumsum_output
    else:
        return count_output
