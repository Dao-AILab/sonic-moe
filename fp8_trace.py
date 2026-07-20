import torch
from torch.profiler import profile, ProfilerActivity
from sonicmoe import KernelBackendMoE, MoE
from sonicmoe.enums import ActivationType

# T, H, I, E, K = 32768, 4096, 1024, 128, 8
T, H, I, E, K = 8192, 4096, 1024, 512 // 8, 10
torch.manual_seed(0)
moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H, intermediate_size=I,
          activation_function=ActivationType.SWIGLU, add_bias=False, std=0.02).to(torch.bfloat16).cuda()
x = 0.2 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
w1, w2, router_w = moe.c_fc.weight, moe.c_proj.weight, moe.router.weight
dout = torch.randn_like(x)

def fwd():
    with torch.no_grad():
        return moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe_fp8, is_inference_mode=True)[0]

def fwd_bwd():
    o = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe_fp8, is_inference_mode=False)[0]
    o.backward(dout, retain_graph=True)
    x.grad = w1.grad = w2.grad = router_w.grad = None

for _ in range(20):  # warmup: compile + autotune
    fwd()
    fwd_bwd()
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        fwd()
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
trace = "scratchpad/fp8_fwd_trace.json"
prof.export_chrome_trace(trace)
print("\nchrome trace:", trace)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        fwd_bwd()
    torch.cuda.synchronize()

print("\n" + prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
trace = "scratchpad/fp8_fwd_bwd_trace.json"
prof.export_chrome_trace(trace)
print("\nchrome trace:", trace)
