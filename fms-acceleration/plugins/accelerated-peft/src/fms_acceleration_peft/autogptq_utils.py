# Copyright The IBM Tuning Team
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

# SPDX-License-Identifier: Apache-2.0
# https://spdx.dev/learn/handling-license-info/

# Standard
from functools import partial
from typing import Callable, List

# Third Party
from fms_acceleration.model_patcher import (
    ModelPatcher,
    ModelPatcherRule,
    ModelPatcherTrigger,
)
from peft import LoraConfig
from peft.tuners.lora.gptq import GPTQLoraLinear
import torch

# these parameters are to be patched for triton v2
# consider making a map if patching more kernels
PATCH_FOR_FSDP_TRITON_V2 = ["qweight", "qzeros"]
PEFT_ALL_LINEAR = "all-linear"


def requires_installation_on_all_linears(peft_config, model_type: str = None):
    tm = peft_config.target_modules

    if tm is None:
        try:
            # Third Party
            # pylint: disable=import-outside-toplevel
            from peft.utils.constants import (
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
            )

            tm = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_type]
        except (ImportError, KeyError) as e:
            raise ValueError(
                "target modules not specified and unable to determine default "
                f"for given model type {model_type}."
            ) from e

        # replace with some defaults
    assert isinstance(
        tm, (list, set, str)
    ), "if provided, target modules can only be list, set or string"
    if isinstance(tm, (list, set)):
        if PEFT_ALL_LINEAR not in tm:
            return False
        assert len(tm) == 1, f"`{PEFT_ALL_LINEAR}` must exist alone in target modules"
        return True
    return tm == PEFT_ALL_LINEAR


def build_patch_to_view_tensor_to_parameter_for_fsdp_gptq(
    module,
    torch_dtype,
):
    # convert all patched attributes to Parameters of torch_dtype
    # so FSDP can shard them
    for attr_name in PATCH_FOR_FSDP_TRITON_V2:
        attr = getattr(module, attr_name)
        attr = torch.nn.Parameter(attr.view(torch_dtype), requires_grad=False)
        setattr(module, attr_name, attr)

    # this patches the forward to convert them back to original
    # type (i.e. int32) before the function call into the kernels
    return patch_forward_to_view_attributes_before_call(
        module.forward,
        attribute_names=PATCH_FOR_FSDP_TRITON_V2,
        torch_dtype=torch.int32,  # patch it back to
    )


def register_tensors_as_parameters_patch_rule(target_module, torch_dtype):
    # Register patch
    ModelPatcher.register(
        ModelPatcherRule(
            rule_id="autogptq_patch_tensors_as_float_parameters",
            trigger=ModelPatcherTrigger(check=target_module),
            forward_builder=partial(
                build_patch_to_view_tensor_to_parameter_for_fsdp_gptq,
                torch_dtype=torch_dtype,
            ),
        )
    )


def make_sure_no_tensor_in_meta_device(
    model,
    use_triton: bool,
    desc_act: bool,
    group_size: int,
    bits: int,
    disable_exllama: bool,
    disable_exllamav2: bool,
    use_marlin: bool = False,
    use_tritonv2: bool = False,
):
    # Third Party
    # guarded import
    from auto_gptq.utils.import_utils import (  # pylint: disable=import-outside-toplevel,import-error
        dynamically_import_QuantLinear,
    )

    QuantLinear = dynamically_import_QuantLinear(
        use_triton,
        desc_act,
        group_size,
        bits=bits,
        disable_exllama=disable_exllama,
        disable_exllamav2=disable_exllamav2,
        use_marlin=use_marlin,
        use_tritonv2=use_tritonv2,
    )
    for _, m in model.named_modules():
        bias = getattr(m, "bias", None)
        if bias:
            if isinstance(m, QuantLinear) and bias.device == torch.device("meta"):
                m.register_buffer(
                    "bias",
                    torch.zeros((m.outfeatures), dtype=torch.float16, device="cpu"),
                )


def replace_module_peft(self, parent_module, child_name, new_module, old_module):

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


def create_new_module_peft(
    lora_config: LoraConfig,
    adapter_name: str,
    target: torch.nn.Module,
    target_cls,
    **kwargs,
):
    # if the base layer module matches a supported class, dispatch the lora linear
    # to be installed
    new_module = None
    if isinstance(target, target_cls):
        new_module = GPTQLoraLinear(
            target, adapter_name, lora_config=lora_config, **kwargs
        )

    # if module cannot be found, return None which results in a raise in the call-stack
    return new_module


# consider to move this somewhere more general
def patch_forward_to_view_attributes_before_call(
    old_forward: Callable,
    attribute_names: List[str],
    torch_dtype: torch.dtype,
    submodule_names: str = None,
    is_method_forward: bool = True,
):
    # patch old_forward to view attribtues to torch_dype
    # before call

    if submodule_names is None:
        submodule_names = ""
    if isinstance(submodule_names, str):
        submodule_names = [submodule_names]

    def _forward(self, *args, **kwargs):

        for sub_name in submodule_names:
            mod = self.get_submodule(sub_name)

            # perform a view on all these attributes
            for attr_name in attribute_names:

                # the view should be a passthrough
                # if attr.dtype == torch_dtype
                attr = getattr(mod, attr_name)

                # perform view
                attr = attr.view(torch_dtype)

                try:
                    setattr(mod, attr_name, attr)
                except TypeError:
                    # this means already have attr_name as a parameter, then
                    # just assign this way
                    mod.__dict__[attr_name] = attr

        if is_method_forward:
            # in this case, the self is already bound
            return old_forward(*args, **kwargs)
        return old_forward(self, *args, **kwargs)

    return _forward
