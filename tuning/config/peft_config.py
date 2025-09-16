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
from enum import Enum
from typing import List

# Third Party
from peft import LoraConfig as _LoraConfig
from transformers.utils.quantization_config import Mxfp4Config as HfMxfp4Config


class QUANT_METHOD(Enum):
    MXFP4 = "mxfp4"


class PEFT_METHOD(Enum):
    PT = "pt"
    LORA = "lora"
    ALORA = "alora"


@dataclass
class Mxfp4Config:
    dequantize: bool = True

    def to_hf_config(self):
        return HfMxfp4Config(dequantize=self.dequantize)


@dataclass
class LoraConfig(_LoraConfig):
    """
    This is the configuration class that extends peft.LoraConfig with a few defaults.

    Args:
        lora_alpha (`int`):
            The alpha parameter for Lora scaling.
        lora_dropout (`float`):
            The dropout probability for Lora layers.
    """
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    def __post_init__(self):
        # If target_modules is a single-element list, convert it into a plain string
        if self.target_modules == ["all-linear"]:
            self.target_modules = "all-linear"

        super().__post_init__()


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
