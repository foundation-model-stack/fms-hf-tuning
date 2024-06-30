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
from dataclasses import dataclass
from typing import List

# Local
from .utils import ensure_nested_dataclasses_initialized, parsable_dataclass


@parsable_dataclass
@dataclass
class AutoGPTQLoraConfig:

    # auto_gptq supports various kernels, to select the kernel to use.
    kernel: str = "triton_v2"

    # allow auto_gptq to quantize a model before training commences.
    # NOTE: currently this is not allowed.
    from_quantized: bool = True

    def __post_init__(self):

        if self.kernel != "triton_v2":
            raise ValueError("only 'triton_v2' kernel currently supported.")

        if not self.from_quantized:
            raise ValueError("only 'from_quantized' == True currently supported.")


@parsable_dataclass
@dataclass
class BNBQLoraConfig(List):

    # type of quantization applied
    quant_type: str = "nf4"

    # if we only want to quantize the base layer, and defer to the
    # huggingface to prepare the peft (i.e. lora) model
    no_peft_model: bool = False

    def __post_init__(self):

        if self.quant_type not in ["nf4", "fp4"]:
            raise ValueError("quant_type can only be either 'nf4' or 'fp4.")


@dataclass
class QuantizedLoraConfig:

    # to use auto_gptq 4bit lora base layers
    auto_gptq: AutoGPTQLoraConfig = None

    # to use auto_gptq 4bit lora base layers
    bnb_qlora: BNBQLoraConfig = None

    def __post_init__(self):
        # ensure nested dataclasses initialized
        ensure_nested_dataclasses_initialized(self)
