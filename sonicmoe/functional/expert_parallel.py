# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Portions of this code are from DeepSeek DeepEP project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE


try:
    import deep_ep

    DEEPEP_IS_INSTALLED = True
except ImportError:
    DEEPEP_IS_INSTALLED = False

import torch


class DeepEPDispatch(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x,
        buffer,
        config,
        token_indices,
        token_probs,
        num_experts,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):

        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())

        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = buffer.get_dispatch_layout(
            token_indices,
            num_experts,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            num_recv_tokens_per_expert_list,
            handle,
            after_event_overlap,
        ) = buffer.dispatch(
            x,
            config=config,
            topk_idx=token_indices,
            topk_weights=token_probs,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            num_tokens_per_rank=num_tokens_per_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        if async_finish:
            after_event_overlap.current_stream_wait()

        ctx.buffer = buffer
        ctx.config = config
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        tokens_per_expert = torch.tensor(num_recv_tokens_per_expert_list)

        return (recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle)

    @staticmethod
    def backward(
        ctx,
        grad_output,
        grad_token_indices,
        grad_token_probs,
        grad_tokens_per_expert,
        grad_handle,
    ):

        buffer = ctx.buffer
        config = ctx.config
        handle = ctx.handle
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        grad_x, grad_token_probs, after_event = buffer.combine(
            grad_output.contiguous(),
            handle,
            config=config,
            topk_weights=grad_token_probs.float(),
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )

        if ctx.async_finish:
            after_event.current_stream_wait()
        
        return grad_x, None, None, None, grad_token_probs, None, None, None


class DeepEPCombine(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x,
        buffer,
        config,
        handle,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())

        combined_x, _, after_event = buffer.combine(
            x,
            config=config,
            handle=handle,
            async_finish=async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        if async_finish:
            after_event.current_stream_wait()

        ctx.buffer = buffer
        ctx.config = config
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        return combined_x, None

    @staticmethod
    def backward(ctx, grad_output, previous_event=None):
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = ctx.buffer
        config = ctx.config
        grad_x, _, _, _, _, after_event = buffer.dispatch(
            grad_output.contiguous(),
            config=config,
            handle=ctx.handle,
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )

        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, None, None, None, None


if DEEPEP_IS_INSTALLED:

    def ep_dispatch(
        x,
        buffer,
        config,
        token_indices,
        token_probs,
        num_experts,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        return DeepEPDispatch.apply(
            x.contiguous(),
            buffer,
            config,
            token_indices,
            token_probs,
            num_experts,
            async_finish,
            allocate_on_comm_stream,
        )


    def ep_combine(
        x, buffer, config, handle, async_finish=False, allocate_on_comm_stream=False
    ):
        return DeepEPCombine.apply(
            x, buffer, config, handle, async_finish, allocate_on_comm_stream
        )

    DeepEPBuffer = deep_ep.Buffer
    DeepEPConfig = deep_ep.Config

else:

    def DeepEPImportErr():
        raise ImportError("Deepep is required by for expert parallel!")

    ep_dispatch = DeepEPImportErr()
    ep_combine = DeepEPImportErr()

    DeepEPBuffer = None
    DeepEPConfig = None
