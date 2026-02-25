import torch
import torch.nn.functional as F
import argparse
from triton.testing import do_bench
from rich import print
from triton_kernels.swiglu import swiglu_fn
from triton_kernels.matmul import matmul, PrecisionConfig, FnSpecs, FusedActivation
from triton_kernels.topk import topk
from triton_kernels.tensor import make_ragged_tensor_metadata
from triton_kernels.reduce import reduce





def run_mlp_single_gpu(x_bf16, x_fp8,
                       wg, bg,
                       w1, act1,
                       w2,
                       n_expts_act, pc):
    # gate matrix multiplication
    l = matmul(x_bf16, wg, bg, precision_config=pc)
    # topk (no all_gather needed for single GPU)
    l_active = topk(l, n_expts_act, apply_softmax=True, all_gather=False, symm_mem_pool=None)
    # expert histogram, dispatch/combine indices
    expt_sizes = l_active.mask_metadata.col_sum
    dispatch_indx = l_active.mask_metadata.row_sorted_indx
    combine_indx = l_active.mask_metadata.col_sorted_indx
    # ragged tensor metadata
    x_metadata = make_ragged_tensor_metadata(expt_sizes, dispatch_indx.shape[0])
    # first matmul + swiglu (fused gather via gather_indx)
    gather_indx = combine_indx // n_expts_act
    y = matmul(x_fp8, w1, None, a_ragged_metadata=x_metadata,
               gather_indx=gather_indx, precision_config=pc, fused_activation=act1)
    # second matmul
    y = matmul(y, w2, None, a_ragged_metadata=x_metadata, scatter_indx=combine_indx, precision_config=pc)
    # scatter_indx=combine_indx maps expert-sorted results back to original [T*K] order
    z = y
    # build softmax scale in original (token-grouped) order
    # Must use .expand() so the H-dim stride is 0, which tells the
    # triton reduce kernel to broadcast (it checks stride == 0).
    z = z.view(-1, n_expts_act, z.shape[-1])
    scale = l_active.vals.unsqueeze(-1).expand_as(z)  # stride: (K,1,0)
    # weighted average of expert outputs
    z, _ = reduce(z, dim=1, scale=scale)
    return z

def run(args):
    n = args.n
    T, K, E, H, I = args.T, args.K, args.E, args.H, args.I
    dev = torch.cuda.current_device()

    # -- init parameters (bf16, column-major layout expected by matmul kernel) --
    def make_weight(*shape):
        w = torch.randn(shape, device=dev, dtype=torch.bfloat16)
        return w.transpose(-1, -2).contiguous().transpose(-1, -2)
    wg = make_weight(H, E) 
    w1 = make_weight(E, H, 2 * I) 
    w2 = make_weight(E, I, H) 
    bg = torch.randn((E,), device=dev) 
    pc = PrecisionConfig()

    # -- init activation --
    x_bf16 = torch.randn((T, H), device=dev, dtype=torch.bfloat16)
    x_fp8 = x_bf16.clone()

    # -- matmul fusion --
    act1 = FusedActivation(FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2), (1.0, 1.0))

    print(f"[bold]Config: T={T}, H={H}, I={I}, E={E}, K={K}[/bold]")


    # -- benchmark (uncomment to enable) --
    def test_forward():
       run_mlp_single_gpu(x_bf16, x_fp8,
                           wg, bg,
                           w1, act1,
                           w2,
                           K, pc)
    
    forward_time = do_bench(test_forward, warmup=5, rep=n)
    flops = 6 * T * I * H * K
    tflops = flops / (forward_time / 1e3) / 1e12
    print(f"[bold green]\[Forward][/bold green] Average time: {forward_time:.3f} ms, TFLOPS: {tflops:.1f}")


if __name__ == "__main__":
    # python bench_triton_moe_v1.py
    torch.cuda.set_device(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--T", type=int, default=40960)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--E", type=int, default=64)
    parser.add_argument("--H", type=int, default=2880)
    parser.add_argument("--I", type=int, default=2880)
    args = parser.parse_args()
    run(args)
