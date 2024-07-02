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
class FusedLoraConfig(List):

    # load unsloth optimizations for these 4bit base layer weights.
    # currently only support "auto_gptq" and "bitsandbytes"
    base_layer: str = None

    # fused kernels for lora linear layers
    fused_lora: bool = False

    def __post_init__(self):

        if self.base_layer is not None and self.base_layer not in {
            "auto_gptq",
            "bitsandbytes",
        }:
            raise ValueError(f"base_layer set to invalid value '{self.base_layer}'")

        if self.base_layer is not None and not self.fused_lora:
            raise ValueError(
                f"base_layer set to '{self.base_layer}' so fused_lora must be set to True"
            )


@parsable_dataclass
@dataclass
class FastKernelsConfig(List):

    # fast loss triton kernels
    fast_loss: bool = False

    # fast rms norm triton kernels
    fast_rsm_layernorm: bool = False

    # fast RoPE embedding triton kernels
    fast_rope_embeddings: bool = False

    def __post_init__(self):

        if not self.fast_loss == self.fast_rsm_layernorm == self.fast_rope_embeddings:
            raise ValueError(
                "fast_loss, fast_rms_layernorm and fast_rope_embedding must be enabled "
                "together. This restriction may be relaxed in the future."
            )


@dataclass
class FusedOpsAndKernelsConfig:

    # fused lora ops
    fused_lora: FusedLoraConfig = None

    # fast kernels
    fast_kernels: FastKernelsConfig = None

    def __post_init__(self):
        if (self.fused_lora is not None and self.fast_kernels is None) or (
            self.fused_lora is None and self.fast_kernels is not None
        ):
            raise ValueError(
                "fused lora and fast_kernels must be used together. "
                "This restriction may be relaxed in the future."
            )

        # ensure nested dataclasses initialized
        ensure_nested_dataclasses_initialized(self)
