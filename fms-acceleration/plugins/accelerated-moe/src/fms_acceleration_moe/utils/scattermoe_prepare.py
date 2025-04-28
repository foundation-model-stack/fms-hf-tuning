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
from collections import OrderedDict
from contextlib import nullcontext

# Third Party
from accelerate import init_empty_weights
from peft import LoraConfig
from torch.distributed._tensor import DTensor, Replicate, Shard, distribute_tensor

# pylint: disable=import-error
from torch.distributed._tensor.device_mesh import DeviceMesh, init_device_mesh
from tqdm import tqdm
from transformers.modeling_utils import is_fsdp_enabled, is_local_dist_rank_0
import torch

# Local
from .checkpoint_utils import get_resolved_checkpoint_location, load_weight_map
from .scattermoe_constants import (
    FILE_SAFETENSOR_INDEX,
    KEY_EXPERT_PARALLEL,
    KEY_REPLICATE,
    KEY_SCATTERMOE_ROUTER,
    get_scattermoe_conv_spec_from_archs,
    get_shared_expert_cls_from_archs,
)
from .scattermoe_state_dict import (
    convert_state_dict,
    get_checkpoint_meta_from_sharded_safetensor,
    get_state_dict_from_checkpoint_metadata,
)


# this function will load the sharded experts onto the device.
# - this assumes that the "dmoe" module is the megablocks.layers.dmoe.dMoE distributed
#   implementation of the mixture of experts.
def load_experts_onto_device(
    module: torch.nn.Module,
    state_dict: OrderedDict,
    device_mesh: DeviceMesh,
    num_experts_per_device: int,
):

    # hook for scaling the gradient
    scaling = device_mesh[KEY_EXPERT_PARALLEL].size()

    def _hook(grad):
        if grad is not None:
            grad.div_(scaling)
        return grad

    # required replication placements
    reps = [Replicate() for _ in range(device_mesh.ndim - 1)]

    for weight_name, param in state_dict.items():

        if KEY_SCATTERMOE_ROUTER in weight_name:
            # if its the router, replicate
            param = distribute_tensor(param, device_mesh, reps + [Replicate()])
        elif param.shape[0] > num_experts_per_device:
            # if its a weight param and the number of experts exceed that of
            # the device, shard
            param = distribute_tensor(param, device_mesh, reps + [Shard(0)])
        else:
            # if its a weight and the already sharded by number of experts
            param = DTensor.from_local(
                param, device_mesh=device_mesh, placements=reps + [Shard(0)]
            )

        # get the module we want to shard
        name = weight_name.split(".")
        path, name = ".".join(name[:-1]), name[-1]
        mod = module.get_submodule(path)
        requires_grad = getattr(mod, name).requires_grad

        param = torch.nn.Parameter(
            param,
            requires_grad=requires_grad,
        )

        # install gradient scaling hook
        if KEY_SCATTERMOE_ROUTER not in weight_name:
            if param.requires_grad:
                param.register_hook(_hook)

        # register the sharded parameter onto the megablocks.dmoe
        mod.register_parameter(name, param)


def prepare_scattermoe(
    model: torch.nn.Module,
    checkpoint_name_or_path: str = None,
    rank: int = None,
    world_size: int = None,
    ep_degree: int = 1,
    disable_distributed: bool = False,
    key_rep: str = KEY_REPLICATE,
    key_ep: str = KEY_EXPERT_PARALLEL,
    device_type: str = "cuda",
    mixed_precision: bool = False,
    lora_config: LoraConfig = None,
):

    # guarded because may have third party package deps
    # Local
    # pylint: disable=import-outside-toplevel
    from .scattermoe import ScatterMoE

    if disable_distributed and ep_degree > 1:
        raise ValueError(
            "expert sharding can not be deferred to top level sharding"
            "protocol (e.g. FSDP) when ep_degree > 1"
        )

    assert world_size % ep_degree == 0, (
        f"world size ({world_size}) " f"not divisible by ep_size ({ep_degree})."
    )

    # NOTE:
    # mulitple breaking changes are included which would be massaged later
    # llama4's config is wrapped inside text_config so we had to choose it that way
    # in production use we ideally should be able to infer or by specifying this
    # in scattermoe_constants.py

    moe_num_experts: int = model.config.text_config.num_local_experts
    num_experts_per_device = moe_num_experts // ep_degree
    assert (
        moe_num_experts % ep_degree == 0
    ), f"moe num experts ({moe_num_experts}) not divisible by ep_shard_factor ({ep_degree})."

    # current rank of the device
    device = torch.device(f"{device_type}:{rank}")

    # NOTE:
    # this change was needed since cpu ram efficient loading is faulty with FSDPv2
    # therefore for now we will load the models to cpu so that we at least dont
    # end up with cuda OOM.
    # large models can easily be trained now by bumping up your cpu memory
    # once fsdpv2 fix falls in place we will revert this change back to its original form

    if ep_degree == 1 and disable_distributed:
        device = torch.device("cpu")

    # get the scattermoe conversion spec
    (
        moe_cls,
        router_name,
        expert_name,
        expert_mlp_spec,
        sharded_expert_ckpt,
    ) = get_scattermoe_conv_spec_from_archs(model.config.architectures)

    # split the names first
    expert_name = expert_name.split("|")

    rep_size = world_size // ep_degree

    if ep_degree == 1:
        # in this case no need for sharding
        device_mesh = None
    elif rep_size == 1:
        # in this case a 1D device mesh suffices
        device_mesh = init_device_mesh(
            device_type,
            (ep_degree,),
            mesh_dim_names=(key_ep,),
        )
    else:
        # in this case it will distribute experts on a different dim
        # - this will achieve the effect that the expert sharding can be
        #   hierachical (e.g., can be over a slower network plane since
        #   the communication overhead is less
        device_mesh = init_device_mesh(
            device_type,
            (rep_size, ep_degree),
            mesh_dim_names=(key_rep, key_ep),
        )

    # - compute the shard indices for current expert, if sharding is
    #   indeed taking place
    expert_shards = None
    if device_mesh is not None:
        _index = device_mesh[KEY_EXPERT_PARALLEL].get_local_rank()
        expert_shards = list(
            range(
                _index * num_experts_per_device, (_index + 1) * num_experts_per_device
            )
        )

    # - if mixed precision is specified then we upcast
    dtype = model.dtype if not mixed_precision else torch.float32

    # for all the MoE related params, e.g., gate, experts
    # get a dictionary
    # parent_mod: (child_instance_name, [list of fqdn keys])
    found = {}
    for name, mod in model.named_modules():
        name = name.split(".")
        parent, child = ".".join(name[:-1]), name[-1]

        # check the module depending if moe_cls is a str or class
        # pylint: disable=isinstance-second-argument-not-valid-type
        if (
            mod.__class__.__name__ == moe_cls
            if isinstance(moe_cls, str)
            else isinstance(mod, moe_cls)
        ):
            fqdn_keys = [  # all params, including childs'
                f"{parent}.{child}.{n}" for n, _ in mod.named_parameters()
            ]

            # check if there are any biases in any of the experts
            # if there are biases
            # Assumption: assume that if one expert has bias,then the others
            # will have it to
            has_bias = any(
                expert_name[0] in k and k.endswith("bias") for k in fqdn_keys
            )

            found[parent] = (child, fqdn_keys, has_bias)

    assert len(found) > 0, "cannot find scattermoe modules to replace"

    moe_module_names = set()

    # pylint: disable=too-many-nested-blocks
    # NOTE: for now we only support sharded safetensors
    # - most MOE models should be used using this checkpoint format
    try:
        loc = get_resolved_checkpoint_location(checkpoint_name_or_path)

        weight_map = load_weight_map(loc, "model.safetensors", FILE_SAFETENSOR_INDEX)

        # e.g., prefix: 'model.layers.0',
        #       module_name: 'block_sparse_moe'
        for prefix, (module_name, _, has_bias) in tqdm(
            found.items(), disable=(rank > 0), desc="Converting ScatterMoE layers"
        ):
            checkpoint_metadata = get_checkpoint_meta_from_sharded_safetensor(
                weight_map,
                prefix,
                module_name,
                router_name,
                "|".join(expert_name),
                "shared_expert",
            )

            # the parent module
            parent = model.get_submodule(prefix)

            # - handle state dict loading
            # - NOTE: convert_state_dict does not have logic to concat sharded
            #   experts so cannot handle the case where sharded_expert_ckpt=True
            if (
                ep_degree == 1
                and (not is_fsdp_enabled() or is_local_dist_rank_0())
                and not sharded_expert_ckpt  # cannot be a sharded checkpoint
            ):
                # - if there is no sharding, and model is not loaded on the
                #   meta device, we can simply convert the state dict
                sd = convert_state_dict(
                    prefix + "." + module_name + ".",
                    checkpoint_metadata,
                    getattr(parent, module_name).state_dict(),
                    model.config.text_config.num_local_experts,
                    model.config.text_config.intermediate_size,
                    dtype,
                )
            else:
                # if there is sharding, then we want the model to be loaded
                # on meta in general, since the actual model may be alot smaller
                sd = get_state_dict_from_checkpoint_metadata(
                    loc,
                    checkpoint_metadata,
                    num_experts_per_device,
                    model.config.text_config.intermediate_size,
                    expert_shards,
                    dtype,
                )

            if device_mesh is None:
                if not is_fsdp_enabled() or is_local_dist_rank_0():
                    _init_scattermoe_context = nullcontext
                else:
                    _init_scattermoe_context = init_empty_weights
            else:
                # in this case we need to distribute parameters, so just initialize
                # the scattermoe module swap with empty weights,
                # since they are going to replaced.
                _init_scattermoe_context = init_empty_weights

            # - conver to a scatter moe
            # - very hard to do patching, settle for module swap
            with _init_scattermoe_context():
                moe = ScatterMoE(
                    hidden_size=model.config.text_config.hidden_size,
                    hidden_act=model.config.text_config.hidden_act,
                    intermediate_size=model.config.text_config.intermediate_size,
                    num_experts=num_experts_per_device,
                    has_bias=has_bias,
                    mlp_arch=expert_mlp_spec,
                    top_k=model.config.text_config.num_experts_per_tok,
                    dtype=model.dtype,
                    device=device,
                    ep_device_mesh=(
                        device_mesh[key_ep] if device_mesh is not None else None
                    ),
                    lora_config=lora_config,
                    shared_expert_cls=get_shared_expert_cls_from_archs(
                        model.config.architectures
                    ),
                    shared_expert_args={"config": model.config.text_config},
                )  #

            # the state dict logic below will not have lora adapters
            # - so we need to initialize them
            # - initialize them
            if lora_config is not None:

                # update the state_dict
                for name, param in moe.named_parameters():
                    # NOTE: is his reliable?
                    if "lora_" in name:
                        if device_mesh is not None:
                            # this means it has been loaded with empty context above
                            # - so materialize the tensor
                            param = torch.empty(
                                *param.size(), dtype=dtype, requires_grad=True
                            )

                        sd[name] = param  # set the param in state dict

                        # initialize the loras here
                        if "lora_A" in name:
                            torch.nn.init.zeros_(sd[name])
                        elif "lora_B" in name:
                            torch.nn.init.normal_(sd[name])

            if device_mesh is None:
                # - if not on meta, just load the state dict
                # - and then put on the device
                if not is_fsdp_enabled() or is_local_dist_rank_0():
                    moe.load_state_dict(sd)
                    moe = moe.to(device)
            else:
                # - otherwise, we need to distribtue and will
                #   replace the parameters
                load_experts_onto_device(moe, sd, device_mesh, num_experts_per_device)
            # module swap
            setattr(parent, module_name, moe)

            # - keep track of the name for returning
            moe_module_names.add(module_name)

    except ValueError as e:
        raise ValueError(
            f"Unable to load checkpoint_path '{checkpoint_name_or_path}'. "
            "Currently only support non-GGUF safetensor checkpoints. "
        ) from e

    return moe_module_names
