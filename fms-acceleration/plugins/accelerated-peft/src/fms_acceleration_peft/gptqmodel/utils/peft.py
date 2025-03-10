###############################################################################
# Adapted from https://github.com/AutoGPTQ/AutoGPTQ
# MIT License
# Copyright (c) 2024

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# Standard
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union

# Third Party
from peft import PeftConfig, PeftModel, PeftType, get_peft_model
from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from peft.tuners.lora import LoraConfig, LoraModel
from peft.tuners.lora.gptq import QuantLinear as LoraLinearGPTQ
import torch

# Local
from ..models.base import BaseGPTQModel
from ..nn_modules.qlinear import BaseQuantLinear


class GPTQLoraConfig(LoraConfig):
    injected_fused_attention: bool = False
    injected_fused_mlp: bool = False


class GPTQLoraModel(LoraModel):
    def _replace_module(self, parent_module, child_name, new_module, old_module):
        # replace the lora linear
        setattr(parent_module, child_name, new_module)
        # dispatch to correct device
        # FIXME: refactor
        for name, module in new_module.named_modules():
            if "lora_" in name:
                device = (list(old_module.parameters()) + list(old_module.buffers()))[
                    0
                ].device
                module.to(device)

    @staticmethod
    def _create_new_module(
        lora_config: LoraConfig,
        adapter_name: str,
        target: torch.nn.Module,
        target_cls: torch.nn.Module = BaseQuantLinear,
        **kwargs,
    ):
        # if the base layer module matches a supported class, dispatch the lora linear
        # to be installed
        new_module = None
        if isinstance(target, target_cls):
            new_module = LoraLinearGPTQ(
                target, adapter_name, lora_config=lora_config, **kwargs
            )

        # if module cannot be found, return None which results in a raise in the call-stack
        return new_module

    def merge_adapter(self):
        raise NotImplementedError("gptq model not support merge ada lora adapter")

    def unmerge_adapter(self):
        raise NotImplementedError("gptq model not support unmerge ada lora adapter")

    def merge_and_unload(self):
        raise NotImplementedError("gptq model not support merge and unload")


def find_all_linear_names(
    model: BaseGPTQModel,
    ignore: Optional[List[str]] = None,
    ignore_lm_head: bool = True,
):
    if not ignore:
        ignore = []
    lm_head_name = model.lm_head
    if ignore_lm_head and lm_head_name not in ignore:
        ignore.append(lm_head_name)
    results = set()
    for n, m in model.named_modules():
        if isinstance(m, BaseQuantLinear):
            res = n.split(".")[-1]
            if res not in ignore:
                results.add(res)
    return list(results)


@contextmanager
def hijack_peft_mappings():
    PEFT_TYPE_TO_CONFIG_MAPPING[PeftType.LORA] = GPTQLoraConfig
    PEFT_TYPE_TO_MODEL_MAPPING[PeftType.LORA] = GPTQLoraModel

    try:
        yield
    except:
        PEFT_TYPE_TO_CONFIG_MAPPING[PeftType.LORA] = GPTQLoraConfig
        PEFT_TYPE_TO_MODEL_MAPPING[PeftType.LORA] = GPTQLoraModel
        raise
    finally:
        PEFT_TYPE_TO_CONFIG_MAPPING[PeftType.LORA] = GPTQLoraConfig
        PEFT_TYPE_TO_MODEL_MAPPING[PeftType.LORA] = GPTQLoraModel


def get_gptq_peft_model(
    model: BaseGPTQModel,
    peft_config: PeftConfig = None,
    model_id: str = None,
    adapter_name: str = "default",
    auto_find_all_linears: bool = True,
    train_mode: bool = False,
    ignore_lm_head=True,
):
    if train_mode and not peft_config:
        raise ValueError("peft_config not specified when in train mode.")
    if not train_mode and not model_id:
        raise ValueError(
            "model_id(where to load adapters) not specified when in inference mode."
        )

    if train_mode:
        peft_type = peft_config.peft_type
        if not isinstance(peft_type, str):
            peft_type = peft_type.value
        if peft_type in [PeftType.LORA.value]:
            if auto_find_all_linears:
                peft_config.target_modules = find_all_linear_names(
                    model, ignore_lm_head=ignore_lm_head
                )
            if peft_type == PeftType.LORA.value and not isinstance(
                peft_config, GPTQLoraConfig
            ):
                peft_config = GPTQLoraConfig(**peft_config.to_dict())

    # this hijack is needed as `get_peft_model` uses PEFTModelForCausalLM which inherits from
    # PEFTModel and it in turn relies on PEFT_TYPE_TO_MODEL_MAPPING to initialize its base LoraModel
    with hijack_peft_mappings():
        try:
            if train_mode:
                peft_model = get_peft_model(
                    model.model, peft_config, adapter_name=adapter_name
                )
            else:
                peft_model = PeftModel.from_pretrained(
                    model.model, model_id, adapter_name
                )
        except Exception as exc:
            raise NotImplementedError(
                f"{model.__class__.__name__} not support \
                    {peft_config.peft_type.value} peft type yet."
            ) from exc

    return peft_model


__all__ = [
    "GPTQLoraConfig",
    "GPTQLoraModel",
    "find_all_linear_names",
    "get_gptq_peft_model",
]
