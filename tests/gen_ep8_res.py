
import deep_ep
import torch
import os
import torch.distributed as dist

from sonicmoe import MoE, KernelBackendMoE
from sonicmoe.enums import ActivationType

def test_loop(local_rank: int, num_local_ranks: int):

    torch.cuda.manual_seed(0)
    num_nodes = int(os.getenv('MLP_WORKER_NUM', 1))

    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '8361'))
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    node_rank = int(os.getenv('RANK', 0))
    assert (num_local_ranks < 8 and num_nodes == 1) or num_local_ranks == 8

    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{ip}:{port}',
        world_size=num_nodes * num_local_ranks,
        rank=node_rank * num_local_ranks + local_rank
    )
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('cuda')
    torch.cuda.set_device(local_rank)

    rank, num_ranks, group = dist.get_rank(), dist.get_world_size(), dist.new_group(list(range(num_local_ranks * num_nodes)))

    hidden_size = 512
    intermediate_size = 384


    test_ll_compatibility, num_rdma_bytes = False, 0
    ep_buffer = deep_ep.Buffer(group, int(1e9), num_rdma_bytes, low_latency_mode=False,
                            num_qps_per_rank=(1))

    nvl_buffer_size = 256
    num_sms = 24
    ep_config = deep_ep.Config(num_sms, 8, nvl_buffer_size)

    moe = MoE(
        num_experts=32,
        num_experts_per_tok=8,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation_function=ActivationType.SWIGLU,
        add_bias=False,
        std=0.02,
        rank=rank,
        ep_size=num_ranks,
        ep_group=group,
        ep_buffer=ep_buffer,
        ep_config=ep_config
    ).to(device=rank, dtype=torch.bfloat16)

    moe.router = moe.router.to(torch.float32)

    x = torch.randn(16, hidden_size, device=rank, dtype=torch.bfloat16)
    output, aux_loss = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe)
    if rank == 0:
        print("forward output : ", output)

    eg = torch.ones_like(output)
    loss = (output * eg).sum()
    loss.backward()
    moe.sync_router_grad()

    def torch_allgather(x, size, group, dim=0):
        gathered_x = [torch.zeros_like(x) for _ in range(size)]
        dist.all_gather(gathered_x, x, group=group)
        x = torch.cat(gathered_x, dim=dim)
        return x
    
    router_weight = moe.router.weight
    c_fc_weight = moe.c_fc.weight
    c_fc_grad = moe.c_fc.weight.grad
    c_proj_weight = moe.c_proj.weight

    # router_weight = torch_allgather(router_weight, num_ranks, group)
    c_fc_weight = torch_allgather(c_fc_weight, num_ranks, group)
    c_proj_weight = torch_allgather(c_proj_weight, num_ranks, group)
    c_fc_grad = torch_allgather(c_fc_grad, num_ranks, group)

    x = torch_allgather(x, num_ranks, group)
    output = torch_allgather(output, num_ranks, group)

    if rank == 0:
        # print("gatherd router weight : ", router_weight.shape)
        # print("gatherd c_fc weight : ", c_fc_weight.shape)
        # print("gatherd c_proj weight : ", c_proj_weight.shape)
        # print("gatherd input : ", x.shape)
        # print("gatherd output : ", output.shape)
        # print("aux loss : ", aux_loss)

        saved_tensor = {}
        saved_tensor["router_w"] = router_weight
        saved_tensor["c_fc_w"] = c_fc_weight
        saved_tensor["c_fc_g"] = c_fc_grad
        saved_tensor["c_proj_w"] = c_proj_weight
        saved_tensor["input"] = x
        saved_tensor["output"] = output
        saved_tensor["aux_loss"] = aux_loss

        saved_tensor["router_grad"] = moe.router.weight.grad

        torch.save(saved_tensor, "moe_ep8_tensor_res.pth")

if __name__ == '__main__':
    num_processes = 8
    torch.multiprocessing.spawn(test_loop, args=(num_processes, ), nprocs=num_processes)
