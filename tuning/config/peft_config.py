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
from dataclasses import dataclass, field
from typing import List


@dataclass
class LoraConfig:
    """
    This is the configuration class to store the configuration of a [`LoraModel`].

    Args:
        r (`int`):
            Lora attention dimension (the "rank").
        target_modules (List[str]]):
            The names of the modules to apply the adapter to. \
            If this is specified, only the modules with the specified \
            names will be replaced. Please specify modules as per model architecture. \
            If the value is ["all-linear"], \
            then LORA selects all linear and Conv1D modules as per model architecture, \
            except for the output layer.
        lora_alpha (`int`):
            The alpha parameter for Lora scaling.
        lora_dropout (`float`):
            The dropout probability for Lora layers.
        bias (`str`):
            Bias type for LoRA. Can be 'none', 'all' or 'lora_only'. \
            If 'all' or 'lora_only', the corresponding biases will be updated during training. \
            Be aware that this means that, even when disabling the adapters, the model \
            will not produce the same output as the base model would have without adaptation.
    """

    r: int = 8
    lora_alpha: int = 32
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"],
        metadata={
            "help": "The names of the modules to apply LORA to. LORA selects modules which either \
            completely match or "
            'end with one of the strings. If the value is ["all-linear"], \
            then LORA selects all linear and Conv1D '
            "modules except for the output layer."
        },
    )
    bias = "none"
    lora_dropout: float = 0.05


@dataclass
class PromptTuningConfig:
    """
    This is the configuration class for Prompt Tuning.

    Args:
        prompt_tuning_init : str: The initialization of the prompt embedding. \
            Allowed values "TEXT" or "RANDOM".
        prompt_tuning_init_text (`str`, *optional*):
            The text to initialize the prompt embedding. \
            Only used if `prompt_tuning_init` is `TEXT`.
        num_virtual_tokens (`int`): The number of virtual tokens to use.
    """

    prompt_tuning_init: str = "TEXT"
    num_virtual_tokens: int = 8
    prompt_tuning_init_text: str = "Classify if the tweet is a complaint or not:"
