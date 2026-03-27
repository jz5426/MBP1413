import torch
import torch.distributed as dist


def DistAutogradAllGatherFunction(partial=False):
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            """
            same as the default all_gather function from pytorch
            input: shape of the tensor on the current rank (B, D), (B, N, D), and etc.
            return (tensor_from_rank0, tensor_from_rank1, ..., tensor_from_rankN)
            """
            ctx.save_for_backward(input)
            output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]

            # every rank gets the exact same gathered result.
            # needed for CL where sensitive to batch size.
            dist.all_gather(output, input)
            return tuple(output)

        @staticmethod
        def backward(ctx, *grads):
            """
            if forward function returns (output_rank0, output_rank1, output_rank2, output_rank3)
            then backward function receive grads as (grad_output_rank0, grad_output_rank1, grad_output_rank2, grad_output_rank3)
            """
            # grads[i] = gradient wrt the i-th output tensor from forward.
            # Because each rank sees all gathered tensors in forward, each rank computes gradients wrt all of those tensors.

            (input,) = ctx.saved_tensors
            grad_out = torch.zeros_like(input)

            if partial:
                grad_out[:] = grads[dist.get_rank()]
                assert False, 'Should not be here, not best practice for each node only see its own gradient'
            else:
                # list(grads) is the list of gradients, one per rank.
                # reduce_scatter performs two operations:
                #  - Reduce (sum): add up corresponding entries across all ranks （contribution of total loss for each gpu/rank.
                        # - most likely the sum of partial derivative for rank output with respect to each rank tensor.
                #  - Scatter: distribute slices of the reduced result back to the ranks.
                # compute all partial derivatives and sum up the partial gradients.
                dist.reduce_scatter(grad_out, list(grads), dist.ReduceOp.SUM)

            return grad_out

    return F
