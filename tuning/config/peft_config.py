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
from typing import List, Optional

# Third Party
from peft import LoraConfig as HFLoraConfig
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
class LoraConfig(HFLoraConfig):
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

    # HACK: The following list of arguments listed below
    # is a fix which reduces the field annotation from
    # Optional[List[str], str] type to Optional[List[str]] type
    # This is done for compatibility with HFArgumentParser
    # Please see: https://github.com/huggingface/peft/issues/2798 for further explanation!
    target_modules: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with LoRA."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. "
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D "
                "(if the model is a PreTrainedModel, the output layer excluded). "
                "If not specified, modules will be chosen according to the model architecture, "
                "If the architecture is not known, an error will be raised -- "
                "in this case, you should specify the target modules manually. "
                "To avoid targeting any modules (because you want to apply `target_parameters`) "
                ", set `target_modules=[]`."
            ),
        },
    )
    exclude_modules: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to exclude from Lora."
            )
        },
    )
    init_lora_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "How to initialize the weights of the LoRA layers. "
                "Passing True (default) results in the default initialization from "
                "the reference implementation from "
                "Microsoft, with the LoRA B weight being set to 0. "
                "This means that without further training, "
                "the LoRA adapter will be a no-op. "
                "Setting the initialization to False leads to random initialization of "
                "LoRA A and B, meaning that LoRA is not a no-op before training; "
                "this setting is intended for debugging purposes."
            ),
        },
    )
    layers_to_transform: Optional[list[int]] = field(
        default=None,
        metadata={
            "help": (
                "The layer indexes to transform, is this argument is specified, "
                "PEFT will transform only the layers indexes that are specified inside this list. "
                "If a single integer is passed, PEFT will transform only the layer at this index. "
                "This only works when target_modules is a list of str."
            )
        },
    )
    layers_pattern: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "The layer pattern name, used only if `layers_to_transform` is different to None "
                "and if the layer pattern is not in the common layers pattern. "
                "This only works when target_modules is a list of str. "
                "This should target the `nn.ModuleList` of the "
                "model, which is often called `'layers'` or `'h'`."
            )
        },
    )
    trainable_token_indices: Optional[list[int]] = field(
        default=None,
        metadata={
            "help": (
                "Lets you specify which token indices to selectively fine-tune "
                "without requiring to re-train the "
                "whole embedding matrix using the `peft.TrainableTokensModel` method. "
                "You can specify token indices in two ways. "
                "Either you specify a list of indices which will then target the model's input "
                "embedding layer (or, if not found, `embed_tokens`). "
                "(Not supported yet) Alternatively, you can specify a dictionary "
                "where the key is the name of the embedding module "
                "and the values are the list of token indices, e.g. "
                "`{'embed_tokens': [0, 1, ...]}`. Note that training "
                "with FSDP requires `use_orig_params=True` to "
                "avoid issues with non-uniform `requires_grad`."
            )
        },
    )
    loftq_config: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The configuration of LoftQ. If this is passed, "
                "then LoftQ will be used to quantize the backbone "
                "weights and initialize Lora layers. Also set `init_lora_weights='loftq'` "
                "in this case."
            )
        },
    )

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
