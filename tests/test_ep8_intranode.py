import torch
from sonicmoe import MoE, KernelBackendMoE
from sonicmoe.enums import ActivationType

import subprocess

if __name__ == "__main__":
    result = subprocess.run(['python', 'gen_ep8_res.py'], stdout=subprocess.PIPE, text=True)

    rank = torch.device("cuda:0")
    hidden_size = 512
    intermediate_size = 384
    moe = MoE(
        num_experts=32,
        num_experts_per_tok=8,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation_function=ActivationType.SWIGLU,
        add_bias=False,
        std=0.02,
    ).to(device=rank, dtype=torch.bfloat16)

    moe.router = moe.router.to(torch.float32)

    saved_tensor = torch.load("moe_ep8_tensor_res.pth")
    moe.router.weight.data.copy_(saved_tensor["router_w"])
    moe.c_fc.weight.data.copy_(saved_tensor["c_fc_w"])
    moe.c_proj.weight.data.copy_(saved_tensor["c_proj_w"])

    x = saved_tensor["input"]
    ep1_output, ep1_auxloss = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe)
    ep8_output = saved_tensor["output"]
    # print("ep1 output : ", ep1_output.shape)
    # print("ep8 output : ", ep8_output.shape)

    ep8_auxloss = saved_tensor["aux_loss"]

    ep8_router_grad = saved_tensor["router_grad"]
    
    torch.testing.assert_close(ep1_output, ep8_output, rtol=1.4e-2, atol=2e-2)
    print("auxloss : ", ep1_auxloss, ep8_auxloss)
    torch.testing.assert_close(ep1_auxloss, ep8_auxloss, rtol=1.4e-2, atol=2e-2)

    eg = torch.ones_like(ep1_output)
    loss = (ep1_output * eg).sum()
    loss.backward()
    ep1_router_grad = moe.router.weight.grad
    print("ep1 grad : ", ep1_router_grad)
    print("ep8 grad : ", ep8_router_grad)

    # print("ep1 grad : ", ep1_router_grad.shape, ep1_router_grad.sum())
    # print("ep1 weight : ", moe.router.weight.shape, moe.router.weight.sum())

    torch.testing.assert_close(ep1_router_grad, ep8_router_grad, rtol=1.4e-1, atol=2e-1)

    ep1_c_fc_grad = moe.c_fc.weight.grad
    ep8_c_fc_grad = saved_tensor["c_fc_g"]
    # print("ep1 grad : ", ep1_c_fc_grad)
    # print("ep8 grad : ", ep8_c_fc_grad)


    torch.testing.assert_close(ep1_c_fc_grad, ep8_c_fc_grad, rtol=1.4e-2, atol=2e-2)


    