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
from .utils import patch_activation_checkpointing_fsdp


# pylint: disable=too-many-instance-attributes
class CheckpointingAccelerationPlugin(AccelerationPlugin):


    def __init__(self, configurations: Dict[str, Dict]):
        super().__init__(configurations)
        
        # level 1: decoder layer # default huggingface gradient_checkpointing
        # level 2: decoder layer -> self_attn
        # level 3: decoder layer -> self_attn -> qkvo
        self._ac_level = self._check_config_and_maybe_check_values(
            key="training.ac.lac.level",
            default=1,
        )

    @property
    def requires_augmentation(self):
        return False

    def augmentation(
        self,
        model,
        train_args: TrainingArguments,
        modifiable_args: Tuple[LoraConfig],
    ):
        pass

    def get_callbacks_and_ready_for_train(
        self, model: torch.nn.Module = None, accelerator=None
    ):
        model._ac_level = self._ac_level

        callbacks = []
        if (
            accelerator is not None
            and getattr(accelerator.state, "fsdp_plugin", None) is not None
        ):
            accelerator.state.fsdp_plugin.activation_checkpointing = True
            patch_activation_checkpointing_fsdp()
        return callbacks


# register
AccelerationPlugin.register_plugin(
    CheckpointingAccelerationPlugin,
    configuration_and_paths=[
        "training.ac.lac",
    ],
)
