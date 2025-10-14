###############################################################################
# Adapted from https://github.com/ModelCloud/GPTQModel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################
# Standard
from typing import Dict, List, Optional, Union

# Third Party
import torch

# Local
from ..utils import Backend
from ..utils.model import check_and_get_model_type
from .base import BaseGPTQModel, QuantizeConfig
from .dbrx import DbrxGPTQ
from .dbrx_converted import DbrxConvertedGPTQ
from .gemma import GemmaGPTQ
from .gpt_bigcode import GPTBigCodeGPTQ
from .gpt_neox import GPTNeoXGPTQ
from .granite import GraniteGPTQ
from .llama import LlamaGPTQ
from .mistral import MistralGPTQ
from .mixtral import MixtralGPTQ

MODEL_MAP = {
    "gpt_neox": GPTNeoXGPTQ,
    "llama": LlamaGPTQ,
    "gpt_bigcode": GPTBigCodeGPTQ,
    "mistral": MistralGPTQ,
    "mixtral": MixtralGPTQ,
    "gemma": GemmaGPTQ,
    "granite": GraniteGPTQ,
    "dbrx": DbrxGPTQ,
    "dbrx_converted": DbrxConvertedGPTQ,
}

at_least_one_cuda_v6 = any(
    torch.cuda.get_device_capability(i)[0] >= 6
    for i in range(torch.cuda.device_count())
)

if not at_least_one_cuda_v6:
    raise EnvironmentError(
        "GPTQModel requires at least one GPU device with CUDA compute capability >= `6.0`."
    )


class GPTQModel:
    def __init__(self):
        raise EnvironmentError(
            "ModelGPTQ is not designed to be instantiated\n"
            "use `ModelGPTQ.from_pretrained` to load pretrained model and prepare for quantization via `.quantize()`.\n"
            "use `ModelGPTQ.from_quantized` to inference with post-quantized model."
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        quantize_config: QuantizeConfig,
        max_memory: Optional[dict] = None,
        trust_remote_code: bool = False,
        **model_init_kwargs,
    ) -> BaseGPTQModel:
        model_type = check_and_get_model_type(
            pretrained_model_name_or_path, trust_remote_code
        )
        return MODEL_MAP[model_type].from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            quantize_config=quantize_config,
            max_memory=max_memory,
            trust_remote_code=trust_remote_code,
            **model_init_kwargs,
        )

    @classmethod
    def from_quantized(
        cls,
        model_name_or_path: Optional[str],
        device_map: Optional[Union[str, Dict[str, Union[str, int]]]] = None,
        max_memory: Optional[dict] = None,
        device: Optional[Union[str, int]] = None,
        backend: Backend = Backend.AUTO,
        use_cuda_fp16: bool = True,
        quantize_config: Optional[Union[QuantizeConfig, Dict]] = None,
        model_basename: Optional[str] = None,
        use_safetensors: bool = True,
        trust_remote_code: bool = False,
        warmup_triton: bool = False,
        # verify weight files matches predefined hash during loading
        # usage: hash_format:hash_value, example: md5:ugkdh232
        # supports all hashlib hash methods
        verify_hash: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> BaseGPTQModel:
        model_type = check_and_get_model_type(model_name_or_path, trust_remote_code)
        quant_func = MODEL_MAP[model_type].from_quantized

        return quant_func(
            model_name_or_path=model_name_or_path,
            device_map=device_map,
            max_memory=max_memory,
            device=device,
            backend=backend,
            use_cuda_fp16=use_cuda_fp16,
            quantize_config=quantize_config,
            model_basename=model_basename,
            use_safetensors=use_safetensors,
            trust_remote_code=trust_remote_code,
            warmup_triton=warmup_triton,
            verify_hash=verify_hash,
            **kwargs,
        )
