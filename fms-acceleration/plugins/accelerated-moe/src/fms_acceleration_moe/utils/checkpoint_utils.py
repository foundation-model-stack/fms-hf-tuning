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
from collections import defaultdict
from typing import Dict, List, Union
import json
import os
import re
import shutil

# Third Party
from accelerate.logging import get_logger
from accelerate.utils.constants import FSDP_MODEL_NAME, OPTIMIZER_NAME
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import load_file, safe_open, save_file
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from transformers import PretrainedConfig
from transformers.utils import CONFIG_NAME, SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME
import torch
import torch.distributed.checkpoint as dcp

# Local
from .scattermoe_constants import (
    FILE_SAFETENSOR_INDEX,
    PARAM_NAME_ROUTER_SCATTERMOE,
    PARAM_NAME_WEIGHT_SCATTERMOE,
    get_scattermoe_conv_spec_from_archs,
)
from .scattermoe_state_dict import get_checkpoint_meta_from_sharded_safetensor

logger = get_logger(__name__)

# - variable to capture the model variable
#   in the save/load model calls
MODEL_INDEX = None
KEY_MODEL = "model"
KEY_OPTIMIZER = "optimizer"

# Below are rewrite of HF FSDP model saving functions to be able to handle
# that the parameters are now a mixture of regular and Dtensors.
# - these functions are found in accelerate.utils.fsdp_utils.py
# - save_fsdp_model, save_fsdp_optimizer, load_fsdp_model, load_fsdp_optimizer
# NOTE: we will observe warnings such as
# /torch/distributed/checkpoint/state_dict.py:520:
# FutureWarning: Please use DTensor instead and we are deprecating ShardedTensor.


# Load weight map either with index file or manually in single-shard state
def load_weight_map(loc, file_safetensor, file_safetensor_index):
    index_path = os.path.join(loc, file_safetensor_index)
    safetensor_path = os.path.join(loc, file_safetensor)

    try:
        if os.path.exists(index_path):
            # Load weight map from index file
            with open(index_path, encoding="utf-8") as f:
                index = json.load(f)
            weight_map = index["weight_map"]
        else:
            # If no index file, assume single shard
            weight_map = {}
            with safe_open(safetensor_path, framework="pt") as f:
                weight_map = {key: file_safetensor for key in f.keys()}
    except (FileNotFoundError, json.JSONDecodeError, KeyError, IOError) as e:
        raise ValueError(
            f"Failed to load weight map from {file_safetensor} or {file_safetensor_index}: {e}"
        ) from e

    return weight_map


# rewrite of func from accelerate.utils.fsdp_utils.py
# - empty function, the main logic will be in save_fsdp_optimizer (see below).
def save_fsdp_model(
    fsdp_plugin, accelerator, model, output_dir, model_index=0, adapter_only=False
):
    # pylint: disable=global-statement
    global MODEL_INDEX
    MODEL_INDEX = model_index


# rewrite of func from accelerate.utils.fsdp_utils.py
# - saves both model and optimizer at the same time
def save_fsdp_optimizer(
    fsdp_plugin, accelerator, optimizer, model, output_dir, optimizer_index=0
):

    if fsdp_plugin.state_dict_type != StateDictType.SHARDED_STATE_DICT:
        raise NotImplementedError(
            "Checkpointing for megablocks only enabled for sharded state dict."
        )

    # get the state dicts for model and optimize
    (model_state_dict, optimizer_state_dict) = get_state_dict(model, optimizer)

    # - save model
    ckpt_model = os.path.join(output_dir, f"{FSDP_MODEL_NAME}_{MODEL_INDEX}")
    os.makedirs(ckpt_model, exist_ok=True)
    logger.info(f"Saving model to {ckpt_model}")
    dcp.save(
        state_dict={KEY_MODEL: model_state_dict},
        storage_writer=dcp.FileSystemWriter(ckpt_model),
        planner=DefaultSavePlanner(),
    )
    logger.info(f"Model saved to {ckpt_model}")

    # - save optimizer
    ckpt_opt = os.path.join(output_dir, f"{OPTIMIZER_NAME}_{optimizer_index}")
    os.makedirs(ckpt_opt, exist_ok=True)
    logger.info(f"Saving Optimizer state to {ckpt_opt}")
    dcp.save(
        state_dict={KEY_OPTIMIZER: optimizer_state_dict},
        storage_writer=dcp.FileSystemWriter(ckpt_opt),
        planner=DefaultSavePlanner(),
    )
    logger.info(f"Optimizer state saved in {ckpt_opt}")


# rewrite of func from accelerate.utils.fsdp_utils.py
# - empty function, main logic in load_fsdp_optimizer (see below).
def load_fsdp_model(
    fsdp_plugin, accelerator, model, input_dir, model_index=0, adapter_only=False
):
    # pylint: disable=global-statement
    global MODEL_INDEX
    MODEL_INDEX = model_index


# rewrite of func from accelerate.utils.fsdp_utils.py
# - loads both model and optimizer
def load_fsdp_optimizer(
    fsdp_plugin,
    accelerator,
    optimizer,
    model,
    input_dir,
    optimizer_index=0,
    adapter_only=False,
):

    accelerator.wait_for_everyone()
    if fsdp_plugin.state_dict_type != StateDictType.SHARDED_STATE_DICT:
        raise NotImplementedError(
            "Checkpointing for megablocks only enabled for sharded state dict."
        )

    # - get the state dicts
    model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)

    # - load the model state dict
    ckpt_model = os.path.join(input_dir, f"{FSDP_MODEL_NAME}_{MODEL_INDEX}")
    dcp.load(
        state_dict={KEY_MODEL: model_state_dict},
        storage_reader=dcp.FileSystemReader(ckpt_model),
        planner=DefaultLoadPlanner(),
    )

    # - load the optimizer state dict
    ckpt_opt = os.path.join(input_dir, f"{OPTIMIZER_NAME}_{optimizer_index}")
    dcp.load(
        state_dict={KEY_OPTIMIZER: optimizer_state_dict},
        storage_reader=dcp.FileSystemReader(ckpt_opt),
        planner=DefaultLoadPlanner(),
    )

    # - set the state dicts
    set_state_dict(
        model,
        optimizer,
        model_state_dict=model_state_dict,
        optim_state_dict=optimizer_state_dict,
    )

    # FIXME:
    # - We see errors that occur in optimizer.step()
    # - torch/optim/optimizer.py", line 89, in _use_grad
    # - torch/optim/adamw.py", line 214, in step beta1,
    #   beta2 = cast(Tuple[float, float], group["betas"])
    # - KeyError: 'betas'
    # - Fortunately, this seems to be limited to the empty groups case, where
    #   it seems that it is just the params are not initialized. Since we suppose
    #   these groups are never used, we simply initialize the empty groups with
    #   random values so the errors do not throw.
    for group in optimizer.param_groups:
        if len(group["params"]) == 0:
            group["betas"] = (0.9, 0.999)
            group["lr"] = 0.0
            group["initial_lr"] = 0.0
            group["eps"] = 1e-8
            group["weight_decay"] = 0.0


# function to replace various trainer functions in HF with the ones
# above
def patch_huggingface_save_and_load_for_dtensors():
    # Third Party
    # NOTE: this is really a global replacement, which we use the patcher
    # to do
    # pylint: disable=import-outside-toplevel
    from fms_acceleration.model_patcher import patch_target_module

    patch_target_module("transformers.trainer.save_fsdp_model", save_fsdp_model)
    patch_target_module("transformers.trainer.save_fsdp_optimizer", save_fsdp_optimizer)
    patch_target_module("transformers.trainer.load_fsdp_model", load_fsdp_model)
    patch_target_module("transformers.trainer.load_fsdp_optimizer", load_fsdp_optimizer)


# this function implements a trick to get the resolved cache file to acccess the safetensor
# - NOTE: does not work if _dict_from_json_file is not called, such as in the case of GGUF files.
def get_resolved_checkpoint_location(model_name_or_path: str):

    result = None
    _old_func = PretrainedConfig._dict_from_json_file

    def _dict_from_json_file(resolved_config_file):
        nonlocal result
        result = resolved_config_file
        return _old_func(resolved_config_file)

    # make a hook and restrive
    PretrainedConfig._dict_from_json_file = _dict_from_json_file
    PretrainedConfig.from_pretrained(model_name_or_path)
    PretrainedConfig._dict_from_json_file = _old_func
    return os.path.dirname(result)


# function to get the state dict from dcp_checkpoint
def get_state_dict_from_dcp_checkpoint(
    dcp_checkpoint_dir: str,
):
    # guarded, load some internal functions
    # pylint: disable=import-outside-toplevel
    # Third Party
    from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
    from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

    sd: STATE_DICT_TYPE = {}
    _load_state_dict(
        sd,
        storage_reader=dcp.FileSystemReader(dcp_checkpoint_dir),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    return sd[KEY_MODEL]


# function to get state dict from regular checkpoint
def get_state_dict_from_safe_checkpoint(safe_checkpoint_dir: str):
    safe_index_file = os.path.join(safe_checkpoint_dir, SAFE_WEIGHTS_INDEX_NAME)
    sd = {}
    if os.path.exists(safe_index_file):
        # Load the index for sharded checkpoints
        with open(safe_index_file, "r", encoding="utf-8") as f:
            index = json.load(f)
        shard_files = list(set(index["weight_map"].values()))
        for shard_file in shard_files:
            for key, v in load_file(
                os.path.join(safe_checkpoint_dir, shard_file)
            ).items():
                sd[key] = v

        return sd
    # No index file found, so assume the checkpoint is not sharded.
    checkpoint_file = os.path.join(safe_checkpoint_dir, "model.safetensors")
    if os.path.exists(checkpoint_file):
        for key, v in load_file(checkpoint_file).items():
            sd[key] = v

        return sd
    files = [
        f for f in os.listdir(safe_checkpoint_dir) if f.endswith("model.safetensors")
    ]
    if len(files) == 1:
        checkpoint_file = os.path.join(safe_checkpoint_dir, files[0])
        for key, v in load_file(checkpoint_file).items():
            sd[key] = v

        return sd
    raise FileNotFoundError("No valid safetensors checkpoint found in directory.")


# function to get the ScatterMoE state dict from its DCP checkpoint
# - if the original pretrained_model_name_or_path is specified, will use the checkpoint as hints
#   to map the ScatterMoE checkpoint to that of the original model. This is useful so that we
#   can restore the checkpoint to be loaded by the original architecture.
def recover_original_state_dict_from_checkpoint(
    sd: Dict,
    pretrained_model_name_or_path: str = None,
):
    """
    Parameters:
        dcp_checkpoint_dir (str): the DCP to be converted.
        pretrained_model_name_or_path (str): Optional, if provided we will
            use the hints to remap the
    """

    # reference dcp_to_torch_save from torch.distributed.checkpoint.format_utils.py
    # - strategy is to use _EmptyStateDictLoadPlanner to populate the state dict, then we remap

    # now do the remap
    loc = get_resolved_checkpoint_location(pretrained_model_name_or_path)

    weight_map = load_weight_map(loc, "model.safetensors", FILE_SAFETENSOR_INDEX)

    # config
    config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path)

    (
        _,
        router_name,
        expert_name,
        __,
        sharded_expert_ckpt,
    ) = get_scattermoe_conv_spec_from_archs(config.architectures)

    # the sd from the module swap must have keys like
    # 'model.layers.0.block_sparse_moe.w1.weight'
    # 'model.layers.0.block_sparse_moe.w2.weight'
    # 'model.layers.0.block_sparse_moe.router.weight'
    # so we use this fact to infer that
    # prefix = model.layers.0 and module_name = block_sparse_moe

    def _infer_prefixes_and_module_names(
        sd_keys: List[str],
        min_count: int = 3,
    ):
        _name = "|".join([PARAM_NAME_ROUTER_SCATTERMOE, *PARAM_NAME_WEIGHT_SCATTERMOE])
        # pylint: disable=anomalous-backslash-in-string
        _reg = re.compile(f"(.*)\.({_name})\.weight")
        found = {}

        for k in sd_keys:
            m = _reg.match(k)
            if m is None:
                continue

            prefix, _ = m.groups()
            found[prefix] = 1 + found.get(prefix, 0)

        results = []
        for prefix, cnt in found.items():
            # if at least router, w1 and w2 are found, take it
            # otherwise we delete
            if cnt >= min_count:
                results.append(prefix)

        return results

    for prefix in _infer_prefixes_and_module_names(sd.keys()):
        prefix = prefix.split(".")
        prefix, module_name = ".".join(prefix[:-1]), prefix[-1]

        # checkpoint metadata is will be a  map
        # key -> list of tuples
        # where each in the list is (param_name, stfile)
        # - if the list is larger than one, it means that the
        #   actual model has a sharded checkpoint

        # defaultdict(list,
        #     {'w1.weight': [('model.layers.0.block_sparse_moe.input_linear.weight',
        #        'model-00001-of-00002.safetensors')],
        #      'w3.weight': [('model.layers.0.block_sparse_moe.input_linear.weight',
        #        'model-00001-of-00002.safetensors')],
        #      'w2.weight': [('model.layers.0.block_sparse_moe.output_linear.weight',
        #        'model-00001-of-00002.safetensors')],
        #      'router.weight': [('model.layers.0.block_sparse_moe.router.layer.weight',
        #        'model-00001-of-00002.safetensors')]})

        checkpoint_metadata = get_checkpoint_meta_from_sharded_safetensor(
            weight_map,
            prefix,
            module_name,
            router_name,
            expert_name,
        )

        model2scatter = defaultdict(dict)
        # construct a map of model_key -> {scatter_key: [params, ...]}
        # - if the param list > 1, that means many scatter keys map to 1
        #   model param and they need to be cat
        for scatter_key, list_of_params in checkpoint_metadata.items():
            scatter_key_fqdn = ".".join([prefix, module_name, scatter_key])
            scatter_param = sd[scatter_key_fqdn]

            # remove from state dict
            del sd[scatter_key_fqdn]

            n = len(list_of_params)
            if scatter_key.startswith(PARAM_NAME_ROUTER_SCATTERMOE):
                assert n == 1, "Router parameters should not be sharded."
            elif not sharded_expert_ckpt:
                assert n == 1, "Expert weights expected to be non-sharded."
            else:
                # if sharded, we just assume that there should be 1 expert
                # per shard
                assert (
                    n == scatter_param.shape[0]
                ), "Sharded expert weights should be 1 expert per shard."

            if any(scatter_key.startswith(k) for k in PARAM_NAME_WEIGHT_SCATTERMOE):
                scatter_param = scatter_param.permute(0, 2, 1)

            # go through all the model keys

            for i, (model_key, _) in enumerate(list_of_params):
                if n == 1:
                    # handles routers and non-sharded experts case
                    _param = scatter_param
                else:
                    # then it needs to be sharded
                    _param = scatter_param[i]

                model2scatter[model_key][scatter_key] = _param

        # replace them back in the sd
        for model_key in list(model2scatter.keys()):

            scatter_params = model2scatter[model_key]

            # - there is an assumption that the ifthere is a cat, then
            #  it will go by order of scatter keys
            scatter_keys = sorted(scatter_params.keys())

            assert (
                len(scatter_keys) > 0
            ), f"Obtained zero scatter keys for model_key '{model_key}'"

            if len(scatter_keys) == 1:
                sd[model_key] = scatter_params[scatter_keys[0]]
            else:
                # unfortunately, there this is a in
                # scattermoe_state_dict._maybe_reshape_scattermoe_expert_weights
                # that we split on the dim=1, so we cat back on that
                sd[model_key] = torch.cat(
                    [scatter_params[k] for k in scatter_keys], dim=1
                )

            # remove from this intemediate mapping
            del model2scatter[model_key]

        rem_keys = ",".join(list(model2scatter))
        assert len(rem_keys) == 0, f"Did not handle model parameters '{rem_keys}'"

    return sd


def save_sharded_safetensors(
    input_state_dict: Dict,
    save_directory: str,
    metadata: Dict,
    max_shard_size: Union[int, str] = "5GB",
):
    filename_pattern = SAFE_WEIGHTS_NAME.replace(".bin", "{suffix}.bin").replace(
        ".safetensors", "{suffix}.safetensors"
    )
    state_dict_split = split_torch_state_dict_into_shards(
        input_state_dict,
        filename_pattern=filename_pattern,
        max_shard_size=max_shard_size,
    )
    index = {
        "metadata": state_dict_split.metadata,
        "weight_map": state_dict_split.tensor_to_filename,
    }
    # Save the index
    with open(
        os.path.join(save_directory, SAFE_WEIGHTS_INDEX_NAME), "w", encoding="utf-8"
    ) as f:
        content = json.dumps(index, indent=2, sort_keys=True) + "\n"
        f.write(content)

    filename_to_tensors = state_dict_split.filename_to_tensors.items()
    for shard_file, tensors in filename_to_tensors:
        shard = {tensor: input_state_dict[tensor].contiguous() for tensor in tensors}
        save_file(shard, os.path.join(save_directory, shard_file), metadata=metadata)


# --------------------------- SCRIPT -------------------------


def recover_safetensors_from_dcp(
    checkpoint_dir, pretrained_model_name_or_path, output_dir
):
    if checkpoint_dir.startswith(FSDP_MODEL_NAME):
        loader = get_state_dict_from_dcp_checkpoint
    else:
        fsdp_checkpoint_dirs = [
            x
            for x in os.listdir(checkpoint_dir)
            if os.path.isdir(os.path.join(checkpoint_dir, x))
            and x.startswith(FSDP_MODEL_NAME)
        ]
        if len(fsdp_checkpoint_dirs) == 1:
            checkpoint_dir = os.path.join(checkpoint_dir, fsdp_checkpoint_dirs[0])
            loader = get_state_dict_from_dcp_checkpoint
        elif len(fsdp_checkpoint_dirs) > 1:
            raise ValueError(
                f"Found > 1 dirs in dcp checkpoint dir {checkpoint_dir} "
                f"that starts with {FSDP_MODEL_NAME}. Please spectify the exact dir."
            )
        else:
            # then take it as a safetensors checkpoint
            # - do not support .bin checkpoints
            loader = get_state_dict_from_safe_checkpoint

    # - pretrained model name
    _name_or_path = pretrained_model_name_or_path

    # assume output directory exists, we do not create it
    # - copy the config file if exists
    config_file = os.path.join(checkpoint_dir, CONFIG_NAME)
    target_config_file = os.path.join(output_dir, CONFIG_NAME)
    if os.path.exists(config_file):
        shutil.copyfile(config_file, target_config_file)

        # try to populate pretrained_model_name_or_path from the config path
        # if it was None
        if not _name_or_path:
            with open(target_config_file, "r", encoding="utf-8") as file:
                _name_or_path = json.load(file).get("_name_or_path")

    # get the state_dict
    state_dict = loader(checkpoint_dir)

    # recover the original state dict
    state_dict = recover_original_state_dict_from_checkpoint(state_dict, _name_or_path)

    # save it as a safetensors file
    save_sharded_safetensors(
        {k: v.contiguous() for k, v in state_dict.items()},
        output_dir,
        metadata={"format": "pt"},
    )


# have it serve as a conversion script
if __name__ == "__main__":
    # Standard
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Utility for converting ScatterMoE checkpoint back to the "
            "orginal state dict format. "
            "The ScatterMoE checkpoint was saved after the pretrained model "
            "had been converted by a module swap, hence the state dict will "
            "no longer resemble the original. This utility creaes"
        )
    )

    parser.add_argument(
        "checkpoint_dir",
        help="Path to the checkpoint.",
    )

    parser.add_argument(
        "output_dir", help="Path to the location to write the converted checkpoint."
    )

    parser.add_argument(
        "pretrained_model_name_or_path",
        help=(
            "In order to reconstruct the state dict, we requre hints from "
            "the original pretrained model checkpoint (from which this "
            "checkpoint is obtained)."
        ),
        default=None,
    )

    args = parser.parse_args()
    recover_safetensors_from_dcp(
        args.checkpoint_dir, args.pretrained_model_name_or_path, args.output_dir
    )
