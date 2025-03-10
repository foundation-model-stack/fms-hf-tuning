# Copyright The FMS HF Tuning Authors
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

# Standard
from typing import Tuple

# Third Party
from peft import LoraConfig
from peft.utils import INCLUDE_LINEAR_LAYERS_SHORTHAND
from torch.distributed._tensor import DTensor

# pylint: disable=import-error
from torch.distributed._tensor.device_mesh import DeviceMesh
from transformers.activations import ACT2FN
import torch
import torch.nn.functional as F

# Local
from .scattermoe_constants import SCATTERMOE_SPEC_HAS_GATE
from .scattermoe_utils import all_to_all_gather_inputs, scatter_with_routing_weights
from .scattermoe_utils.khd.kernels.ops import (
    padded_block_indices,
    scattered_experts,
)


# helper function to fetch the local tensor if its a dtensor
def _maybe_get_local_tensor(weight: torch.Tensor):
    if isinstance(weight, DTensor):
        return weight.to_local()
    return weight


class ScatteredExperts(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,
        fan_out: int,
        grouped_in: bool = False,
        grouped_out: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cpu"),
        lora_config: LoraConfig = None,
    ):
        """
        ScatteredExperts is the module that implements a group of experts. The
        forward function will call scattermoe triton kernels.

        NOTE: in the current implementation, we do not initialize the parameters.
        We assume this will be done outside.

        Paramters:
            in_features (int): num of input features per expert.
            out_features (int): num of output features per expert.
            num_experts (int): the number of experts.
            fan_out (int): if the number of embedding inputs are expected to be
                a factor less than the bind_ids and indices at the forward.
            grouped_in (bool): if the embedding inputs are expected to be already
                grouped in at the forward.
            grouped_out (bool): if the outputs are expected to be grouped
                when they are returned from the forward.
            dtype (torch.dtype): the dtype of the parameter tensors. Note, for now the
                adapter takes the same dtype as base layer if LoRA is enabled.
            device (torch.device): the cuda device in which the model should be loaded.
                Only cuda is supported since only triton kernels are supported.
            lora_config (peft.LoraConfig): Optional, to be passed if lora is to be used.
        """
        super().__init__()

        # parameters the experts (not initialized).
        self.weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                in_features,
                out_features,
                dtype=dtype,
                device=device,
            ),
            requires_grad=True,
        )

        # handle lora embeddings
        self.lora_A, self.lora_B = None, None
        self.lora_r = 0
        if lora_config is not None:
            # if LoRA, then disable gradient for the base layer.
            self.weight.requires_grad = False

            # NOTE : - for now adapter takes same dtype as base
            self.lora_A = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    in_features,
                    lora_config.r,
                    dtype=dtype,
                    device=device,
                ),
                requires_grad=True,
            )
            self.lora_B = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    lora_config.r,
                    out_features,
                    dtype=dtype,
                    device=device,
                ),
                requires_grad=True,
            )
            self.lora_r = lora_config.r

        # store these settings
        self.fan_out = fan_out
        self.grouped_in = grouped_in
        self.grouped_out = grouped_out

    def forward(
        self,
        x: torch.Tensor,
        bin_ids: torch.Tensor,
        indices: torch.Tensor,
        padded_block_idxs: torch.Tensor,
        expert_offsets: torch.Tensor,
        gates: torch.Tensor = None,
    ):
        """
        ScatteredExperts executes grouped forwards where each group is a single expert.

        Parameters:
            x (tensor): the emebeddings fed as input.
            bin_ids (tensor): the expert index where each embedding is to be sent.
                Expect that these indices are sorted.
            indices (tensor): the sorting index that brings the input embeddings to the
                sorted order corresponding to bin_ids.
            padded_block_idxs (tensor): the indices for passing triton block info to the
                scattermoe kernels.
            expert_offsets (tensor): the offsets for passing triton block info to the
                scattermoe kernels.
            gates (tensor): Optional. the weighting coefficients that should be applied
                at the output of the scattermoe kernels.
        """
        weight = _maybe_get_local_tensor(self.weight)
        lora_A, lora_B = None, None
        if self.lora_r > 0:
            lora_A, lora_B = (
                _maybe_get_local_tensor(self.lora_A),
                _maybe_get_local_tensor(self.lora_B),
            )

        # NOTE: x is of shape (seqlen, in_features)
        # bin_ids is of shape (seqlen,)
        # padded_block_idxs is a 1-dim tensor, whose length depends on
        # triton kernel block size and input.
        # expert_offsets is of shape (num_experts, )
        return scattered_experts(
            x,
            weight,
            self.fan_out,
            bin_ids,  # sorted_expert_idxs,
            indices,  # sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            gates=gates,  # we dont have router weights
            grouped_in=self.grouped_in,
            grouped_out=self.grouped_out,
            expert_lora_A=lora_A,
            expert_lora_B=lora_B,
            lora_alp=self.lora_r,
        )


# NOTE: this name should match scattermoe_constants.CLASS_NAME_SCATTERMOE
# similar to of MoE_Triton from https://github.com/mayank31398/kernel-hyperdrive
# and ParameterizedScatteredExperts from
# https://github.com/IBM/dolomite-engine/blob/main/dolomite_engine/hf_models/models/moe_dolomite/moe/scatter.py
# - support expert parallel where the data is communicated via all_to_all
# pylint: disable=too-many-instance-attributes
class ScatterMoE(torch.nn.Module):

    def __init__(
        self,
        hidden_size: int,
        hidden_act: str,
        intermediate_size: int,
        num_experts: int,
        has_bias: bool = False,
        mlp_arch: str = None,
        top_k: int = 2,
        dtype: torch.dtype = torch.bfloat16,
        device: str = torch.device("cpu"),
        ep_device_mesh: DeviceMesh = None,
        lora_config: LoraConfig = None,
    ):
        """
        ScatterMoE is the module swap that replaces a sparse mixture-of-experts module
        in order to run the scatter moe kernels and the all_to_all expert parallel routines.

        The submodules are a i) router (nn.Linear) and ii) w1, w2, ... (ScatteredExperts);
        the latter hold the expert weights and run the triton kernels.

        Parameters:

            hidden_size (int): the hidden dimension.
            hidden_act (str): the activation fucntion.
            intermediate_size (int): the intermediate dimension.
            num_experts (int): the number of experts.
            has_bias (bool): if the router and experts have bias.
            mlp_arch (str): unique key that specifies the MLP architecture,
                e.g., if there is a gate forward.
            top_k (int): the number of experts each token gets routed to.
            dtype (torch.dtype): the dtype of the parameter tensors.
            device (torch.device): the cuda device in which the model should be loaded.
                Only cuda is supported since only triton kernels are supported.
            ep_device_mesh (torch.distributed.DeviceMesh): Optional, to be passed if there is
                sharding. Only pass the mesh for the experts.
            lora_config (peft.LoraConfig): Optional, to be passed if lora is to be used.
        """
        assert (
            not has_bias
        ), "ScatterMoE currently unable to handle bias in both gates and experts."

        if lora_config is not None:
            # since this is self implemented, we really only support basic lora funcs
            assert (
                lora_config.bias == "none"
            ), "ScatterMoE currently unable to handle bias in the lora adapters"
            assert (
                lora_config.target_modules == INCLUDE_LINEAR_LAYERS_SHORTHAND
                or INCLUDE_LINEAR_LAYERS_SHORTHAND in lora_config.target_modules
            ), "ScatterMoe currently only handles lora adapters on all linears."

            assert lora_config.init_lora_weights in {
                True,
                "gaussian",
            }, "ScatterMoe currently only handles gaussian initialization."

        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.hidden_act = hidden_act
        self.activation = ACT2FN[hidden_act]
        self.top_k = top_k
        self.all_to_all = (
            ep_device_mesh.size() > 1 if ep_device_mesh is not None else False
        )

        # NOTE: we should then use this to distribute inside
        # and not do the distribution outside
        self.expert_parallel_group = (
            ep_device_mesh.get_group(0) if ep_device_mesh is not None else None
        )

        # build the router
        self.router = torch.nn.Linear(
            in_features=hidden_size,
            out_features=num_experts,
            bias=has_bias,
            dtype=dtype,
            device=device,
        )

        # the experts. The architecture may depend on the model
        # - w1: the up_projection.
        # - w2: the down_projection.
        # - w3 (optional): the gate projection.
        self.w1 = ScatteredExperts(
            in_features=self.hidden_size,
            out_features=self.intermediate_size,
            num_experts=self.num_experts,
            fan_out=self.top_k if not self.all_to_all else 1,
            grouped_out=True,
            dtype=dtype,
            device=device,
            lora_config=lora_config,
        )
        self.w2 = ScatteredExperts(
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            num_experts=self.num_experts,
            fan_out=1,
            grouped_in=True,
            dtype=dtype,
            device=device,
            lora_config=lora_config,
        )
        if mlp_arch == SCATTERMOE_SPEC_HAS_GATE:
            self.w3 = ScatteredExperts(
                in_features=self.hidden_size,
                out_features=self.intermediate_size,
                num_experts=self.num_experts,
                fan_out=self.top_k if not self.all_to_all else 1,
                grouped_out=True,
                dtype=dtype,
                device=device,
                lora_config=lora_config,
            )

    # referenced from dolomite-engine
    def _compute_routing_weights(self, hidden_states: torch.Tensor):

        # router_logits: (batch * sequence_length, n_experts)
        weight = _maybe_get_local_tensor(self.router.weight)
        bias = self.router.bias
        if bias:
            bias = _maybe_get_local_tensor(bias)
        # pylint: disable=not-callable
        router_logits = F.linear(hidden_states, weight, bias)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        return router_logits, routing_weights, selected_experts

    def _get_expert_idxs_and_maybe_gather(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
    ):
        """
        gets the expert indices, and also gather the hidden_states if
        all-to-all processing is required.

        Parameters:
            hidden_states (tensor): 2D batch-flattened hidden states.
            selected_experts (tensor): indices of experts selected for each
                hidden state.
        """

        # megablocks has a cuda kernel for computing a radix sort, but
        # just use the torch version
        sorted_expert_idxs, sorted_scattered_idxs = torch.sort(
            selected_experts.flatten()
        )
        if not self.all_to_all:
            # in this case, no gathering required for hidden states
            return hidden_states, sorted_expert_idxs, sorted_scattered_idxs

        # outputs will:
        # - parallel_x: gathered version of hidden_states
        # - parallel_bin_ids: gathered version of sorted_expert_idxs,
        # - parallel_ind: gathered version of sorted_scattered_idxs.
        #
        # followed by some counting metrics:
        # - send_counts, recv_counts, bins (local)
        outputs = all_to_all_gather_inputs(
            hidden_states,
            selected_experts,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            self.expert_parallel_group,
            self.top_k,
            self.num_experts,
        )

        return outputs + (sorted_expert_idxs, sorted_scattered_idxs)

    def _maybe_scatter(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor = None,
        gather_products: Tuple[torch.Tensor, ...] = None,
    ):
        """
        maybe undo the earlier scatter operation during all-to-all.

        Parameters:
            hidden_states (tensor): batch-flattened hidden states.
            routing_weights (tensor): Optional, routing weights for each expert.
            gather_products (tensor): Optional, tuple of tensors that would have been
                produced by the earlier gather call.
        """

        if not self.all_to_all:
            # in this case scattering is already handled by
            # scattermoe when computing w2
            # - then there is nothing to do
            return hidden_states

        # expect these products to be produced by an earlier
        # all-to-all gather call
        (send_counts, recv_counts, bins, sorted_expert_idxs, sorted_scattered_idxs) = (
            gather_products
        )

        # perform the scattering with the gather products,
        hidden_states = scatter_with_routing_weights(
            hidden_states,
            routing_weights.flatten(),
            send_counts,
            recv_counts,
            bins,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            self.expert_parallel_group,
            self.top_k,
        )

        return hidden_states

    def forward(self, hidden_states: torch.Tensor):
        """
        ScatterMoe.forward replaces the forward of the sparse
        mixture-of-expert module.
        """

        # flatten the batch dimension
        original_shape = hidden_states.shape  # take a record
        hidden_states = hidden_states.view(-1, self.hidden_size)

        # compute the routing logits, weights, and expert assigments
        # - router_logits: will be passed out of forward, used for computing
        #   routing loss.
        (router_logits, routing_weights, selected_experts) = (
            self._compute_routing_weights(hidden_states)
        )

        # get the sorted expert idxs and scattered idxs.
        # - if a gather is required, then the hidden-states will be
        #   communicated from other ranks, and will change.
        # - in gather is required, then some _gather_products will be
        #   required for the scattering later, so return these out also.
        (
            hidden_states,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            *_gather_products,
        ) = self._get_expert_idxs_and_maybe_gather(
            hidden_states,
            selected_experts,
        )

        # scattemoe specific computation.
        # - padded indicies need to be computed for the scattermoe
        #   triton kernels.
        with torch.no_grad():
            padded_block_idxs, expert_offsets = padded_block_indices(
                sorted_expert_idxs, self.num_experts
            )

        # compute the up projection
        out = self.w1(
            hidden_states,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
        )
        out = self.activation(out)

        # - if the arch has a seperate gate projection
        if self.w3:
            out *= self.w3(
                hidden_states,
                sorted_expert_idxs,
                sorted_scattered_idxs,
                padded_block_idxs,
                expert_offsets,
            )

        # compute the down projection
        # - if no all-to-all processing, then depend on
        # scattermoe kernel to perform the final scattering
        hidden_states = self.w2(
            out,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            gates=(None if self.all_to_all else routing_weights),
        )

        # maybe scatter
        hidden_states = self._maybe_scatter(
            hidden_states,
            routing_weights,
            _gather_products,
        )

        # return hidden states and router logits
        return (hidden_states.view(original_shape), router_logits)
