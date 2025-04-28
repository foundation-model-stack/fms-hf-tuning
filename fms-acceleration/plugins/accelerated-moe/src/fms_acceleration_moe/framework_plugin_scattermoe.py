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
from typing import Dict, Tuple

# Third Party
from fms_acceleration import AccelerationPlugin
from peft import LoraConfig
from transformers import TrainingArguments
import torch

# Local
from .utils import (
    patch_huggingface_save_and_load_for_dtensors,
    patch_torch_optim_foreach_to_not_apply_to_dtensors,
    prepare_scattermoe,
)


# pylint: disable=too-many-instance-attributes
class ScatterMoEAccelerationPlugin(AccelerationPlugin):

    restricted_model_archs = [
        "GraniteMoeForCausalLM",
        "MixtralForCausalLM",
        "GraniteMoeSharedForCausalLM",
    ]

    def __init__(self, configurations: Dict[str, Dict]):
        super().__init__(configurations)

        # ep_degree determines the expert parallel sharding
        # If disable_distributed==False, the moe plugin handles sharding / replication,
        # otherwise user will need handle this manually (e.g., using FSDP)
        #
        # ep_degree=1 (default):
        # - disable_distributed=False (default) means
        # experts are replicated while using ScatterMoE kernels.
        # - disable_distributed=True means no replication (please use
        #    own training framework)
        #
        # ep_degree > 1:
        # - disabled_distributed=False (default) means expert sharding with
        # Scatter MoE Kernels.
        # disable_distributed=True cannot be set in this case; errors out.

        self._ep_degree = self._check_config_and_maybe_check_values(
            key="training.moe.scattermoe.ep_degree",
            default=1,
        )

        self._disable_distributed = self._check_config_and_maybe_check_values(
            key="training.moe.scattermoe.disable_distributed",
            default=False,
        )

    @property
    def requires_augmentation(self):
        return True

    def augmentation(
        self,
        model,
        train_args: TrainingArguments,
        modifiable_args: Tuple[LoraConfig],
    ):
        rank, world_size = 0, 1
        (peft_config,) = modifiable_args
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            # we do not need to use the fallback as this is wrapped in an `is_initialized` block
            rank = torch.distributed.get_node_local_rank()

        if not hasattr(model.config, "name_or_path") or not model.config.name_or_path:
            raise ValueError(
                "The model configuration is missing the 'name_or_path' attribute."
            )

        model_name = model.config.name_or_path

        self._moe_component_module_names = prepare_scattermoe(
            model,
            checkpoint_name_or_path=model_name,
            rank=rank,
            world_size=world_size,
            ep_degree=self._ep_degree,
            disable_distributed=self._disable_distributed,
            mixed_precision=False,  # Currently this is hardcoded to OFF
            lora_config=peft_config,
        )
        return model, modifiable_args

    def get_callbacks_and_ready_for_train(
        self, model: torch.nn.Module = None, accelerator=None
    ):

        callbacks = []
        if (
            accelerator is not None
            and getattr(accelerator.state, "fsdp_plugin", None) is not None
        ):

            if not self._disable_distributed:
                # - use an internal function call to get the no split
                # module names, which are typically layers
                _layers = model._get_no_split_modules("")
                accelerator.state.fsdp_plugin.ignored_modules = [
                    getattr(layer, name)
                    for name in self._moe_component_module_names
                    for layer in model.modules()
                    if layer.__class__.__name__ in _layers
                ]

                # call this to patch the HF save and load functions to be able
                # to save DTensors propery
                patch_huggingface_save_and_load_for_dtensors()

                # call this to patch torch optim to not use
                # foreach for dtensors
                patch_torch_optim_foreach_to_not_apply_to_dtensors()

        return callbacks


# register
AccelerationPlugin.register_plugin(
    ScatterMoEAccelerationPlugin,
    configuration_and_paths=[
        "training.moe.scattermoe",
    ],
)
