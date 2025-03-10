# Copyright The FMS HF Tuning Authors
# Copyright 2023 MegaBlocks authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Third Party
import numpy as np
import torch

try:
    # if megablocks is installed, import the kernels, distributed
    # and kernel functions

    # - mixture of triton and cuda kernels
    # Third Party
    from megablocks import ops

    # - distributed autograd
    from megablocks.layers.all_to_all import all_to_all
    from megablocks.ops import gather, histogram, inclusive_cumsum, scatter

    # this is a radix sort for integral indices 0 .. num_bins-1
    def sort(indices: torch.Tensor, num_bins: int):
        bits = max(int(np.ceil(np.log2(num_bins))), 1)
        # TODO: figure out why we need this upcast
        bins, inds = ops.sort(indices, bits)
        return bins, inds.to(torch.int64)

    # replicate indices with bins
    def replicate(indices: torch.Tensor, bins: torch.Tensor):
        replicate_bins = inclusive_cumsum(bins.flatten(), 0)
        # pylint: disable=use-implicit-booleaness-not-len
        replicate_bins = (
            replicate_bins.view(1) if not len(replicate_bins.size()) else replicate_bins
        )

        return ops.replicate(
            indices.unsqueeze(dim=0),
            replicate_bins,
            replicate_bins[-1],
        ).flatten()

except ImportError:

    # - distributed autograd
    # Local
    from .megablocks import all_to_all, gather, scatter

    # take the histogram of integral indices from 0 .. num_bins-1
    def histogram(indices: torch.Tensor, num_bins: int):
        # - this has an Aten for the GPU backend
        return torch.histc(indices, bins=num_bins, min=0, max=num_bins - 1)

    def inclusive_cumsum(x: torch.Tensor, dim: int):
        # - convert to int332 type as that is what is expected by the
        #   megablocks gather and scatter kernels
        return x.cumsum(axis=dim, dtype=torch.int32)

    # this is a radix sort for integral indices 0 .. num_bins-1
    def sort(indices: torch.Tensor, num_bins: int):
        return torch.sort(indices)

    # replicate, this replicates an integral indices according to bin times
    def replicate(indices: torch.Tensor, bins: torch.Tensor):
        return torch.repeat_interleave(indices, bins)


# from megablocks
def no_indices_just_bins(top_expert, num_experts):
    # Sort the expert ids to produce the scatter/gather
    # indices for the permutation.

    # Histogram the expert ids to identify the number of
    # tokens routed to each expert.
    #
    tokens_per_expert = histogram(top_expert, num_experts)

    # Calculate the bin bounds for the sorted tokens.
    bins = inclusive_cumsum(tokens_per_expert, 0)
    # pylint: disable=use-implicit-booleaness-not-len
    bins = bins.view(1) if not len(bins.size()) else bins
    return bins, tokens_per_expert


# modified from https://github.com/databricks/megablocks/blob/main/megablocks/layers/mlp.py
# - credit to trevor-gale
def all_to_all_gather_inputs(
    x: torch.Tensor,
    top_experts: torch.Tensor,
    bin_ids: torch.Tensor,
    indices: torch.Tensor,
    expert_parallel_group: torch.distributed.ProcessGroup,
    top_k: int,
    experts_per_rank: int,
):
    """
    Extracted from megablocks. This function performs all-to-all input
    gathering for expert parallel.
    """

    # Compute the mapping of local tokens to experts.
    # expert_weights = expert_weights.flatten()
    top_experts = top_experts.flatten()
    world_size = expert_parallel_group.size()
    with torch.no_grad():
        bins, tokens_per_expert = no_indices_just_bins(
            top_experts, experts_per_rank * world_size
        )

        # Pass token count information to the device on which the
        # target expert resides.
        parallel_tokens_per_expert = torch.empty_like(
            tokens_per_expert,
        )
        tpe_handle = torch.distributed.all_to_all_single(
            parallel_tokens_per_expert,
            tokens_per_expert,
            group=expert_parallel_group,
            async_op=True,
        )

    # Permute locally and without any padding so that tokens for each
    # parallel device are stored contiguously.
    #
    # This view updates the shape of the tensor from [sl, bs, hs] to
    # [sl * bs, hs] prior to the permutation.
    x = gather(x, indices, bin_ids, bins, top_k)

    # Compute the number of tokens that will be received from each
    # device and permute the input data across the devices.
    with torch.no_grad():
        tpe_handle.wait()

        # Reshape to [world_size, num_experts_per_rank].
        tokens_per_expert = tokens_per_expert.view(world_size, experts_per_rank)
        parallel_tokens_per_expert = parallel_tokens_per_expert.view(
            world_size, experts_per_rank
        )

        # TODO(tgale): It might be faster to do this on the GPU and
        # then communicate the results back to the host.
        send_counts = tokens_per_expert.cpu().sum(dim=-1)
        parallel_tokens_per_expert_cpu = parallel_tokens_per_expert.cpu()
        recv_counts = parallel_tokens_per_expert_cpu.sum(dim=-1)

        # Convert the send/recv counts to lists.
        send_counts = send_counts.tolist()
        recv_counts = recv_counts.tolist()

    # Start the cross-device permutation asynchronously so we can
    # overlap communication with computation.
    parallel_x, parallel_x_handle = all_to_all(
        x,
        recv_counts,
        send_counts,
        expert_parallel_group,
        async_op=True,
    )

    with torch.no_grad():
        # After we do the cross-device permutation we have the tokens on the
        # correct device but not yet grouped by expert because we received
        # tokens from each device as contiguous chunks. To group the tokens
        # for expert computation we'll do one more local permutation. The
        # rest of this torch.no_grad() scope sets up the indices and bins
        # for this permutation.

        # Construct the expert indices for the permuted tokens.
        parallel_top_expert = torch.remainder(
            torch.arange(
                experts_per_rank * world_size,
                dtype=torch.int32,
                device=indices.device,
            ),
            experts_per_rank,
        )

        parallel_top_expert = replicate(
            parallel_top_expert,
            parallel_tokens_per_expert.flatten(),
        )

        parallel_bin_ids, parallel_indices = sort(parallel_top_expert, experts_per_rank)

    parallel_x_handle.wait()

    return (
        parallel_x,
        parallel_bin_ids,
        parallel_indices,
        send_counts,
        recv_counts,  # for all to all
        bins,  # local
    )


def scatter_with_routing_weights(
    x: torch.Tensor,
    expert_weights: torch.Tensor,
    send_counts: torch.Tensor,
    recv_counts: torch.Tensor,
    bins: torch.Tensor,
    bin_ids: torch.Tensor,
    indices: torch.Tensor,
    expert_parallel_group: torch.distributed.ProcessGroup,
    top_k: int,
):
    """
    Extracted from megablocks. This function undoes the all-to-all
    gathering for expert parallel.
    """

    # Un-permute the tokens across the devices.
    x, _ = all_to_all(
        x,
        send_counts,
        recv_counts,
        expert_parallel_group,
    )

    # Un-permute locally to setup for the next series of operations.
    return scatter(x, indices, bin_ids, expert_weights, bins, top_k)
