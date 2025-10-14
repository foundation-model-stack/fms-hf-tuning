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
from types import MethodType
from typing import Dict, Tuple
import os

# Third Party
from fms_acceleration import AccelerationPlugin
from fms_acceleration.model_patcher import patch_target_module
from peft import LoraConfig, prepare_model_for_kbit_training
from peft.tuners.lora.model import LoraModel
from transformers import AutoModelForCausalLM, TrainingArguments
from transformers.modeling_utils import is_fsdp_enabled
from transformers.utils.import_utils import _is_package_available
import torch
import torch.distributed

# Local
from .autogptq_utils import (
    register_tensors_as_parameters_patch_rule,
    requires_installation_on_all_linears,
)
from .fsdp_utils import put_selected_meta_tensors_on_cpu


class AutoGPTQAccelerationPlugin(AccelerationPlugin):

    require_packages = []

    def __init__(self, configurations: Dict[str, Dict]):
        super().__init__(configurations)

        # just do checking, nothing must to configure at this point
        # if need to configure then do something like this:
        self._check_config_equal(
            key="peft.quantization.auto_gptq.kernel", value="triton_v2"
        )
        self._check_config_equal(
            key="peft.quantization.auto_gptq.from_quantized", value=True
        )
        self.use_external_lib = self._check_config_and_maybe_check_values(
            key="peft.quantization.auto_gptq.use_external_lib",
            values=[True, False],
            default=False,
        )

        if self.use_external_lib:

            assert _is_package_available("auto_gptq") is True, (
                "Unable to use external library, auto_gptq module not found. "
                "Refer to README for installation instructions  "
                "as a specific version might be required."
            )

    def model_loader(self, model_name: str, **kwargs):
        # guarded imports
        # Third Party
        if self.use_external_lib:
            # Third Party
            from auto_gptq.nn_modules.qlinear.qlinear_tritonv2 import (  # pylint: disable=import-outside-toplevel,import-error
                QuantLinear,
            )

            from auto_gptq import (  # isort:skip pylint: disable=import-outside-toplevel,import-error
                AutoGPTQForCausalLM as GPTQModel,
            )
            from auto_gptq import (  # isort:skip pylint: disable=import-outside-toplevel,import-error
                BaseQuantizeConfig as QuantizeConfig,
            )
        else:
            # Local
            from .gptqmodel import (  # pylint: disable=import-outside-toplevel,import-error
                GPTQModel,
                QuantizeConfig,
            )
            from .gptqmodel.nn_modules.qlinear.qlinear_tritonv2 import (  # pylint: disable=import-outside-toplevel,import-error
                QuantLinear,
            )
            from .gptqmodel.utils import (  # pylint: disable=import-outside-toplevel,import-error
                Backend,
            )

        # Currently we allow only a quantized checkpoint to be loaded, we do not
        # implement the quantization process here.
        #
        # The quantization process is used to convert a non-quantized checkpoint
        # (provided in model_name) into a quantized one. This entails
        # 1. providing a BaseQuantizeConfig with the appropriate quantization settings
        # 2. calling BaseGPTQForCausalLM.quantize to run the quantization algorithm
        # (may take time, e.g. hours)
        # 3. calling BaseGPTQForCausalLM.save_pretrained to save a quantized checkpoint
        #
        # The reasons for not implementing the flow at this point are.
        # 1. The quantization can take very long for large models. As such, it is more appropriate
        # to run it once outside of training, and save the checkpoint to be used for multiple runs.
        # 2. Requires some API changes to point to where the quantized checkpoint should be saved.
        #    Can be confusing to the user since it will be different from model_name
        # NOTE: there will be a warning that can be ignored
        # "WARNING - QuantLinear with the exllama backend not does support the trainable mode yet,
        # switching to cuda/cuda_old/triton backend."
        # assume model_name points to a quantized checkpoint. Thus we load the quantization
        # config directly from the checkpoint.
        quantize_config = QuantizeConfig.from_pretrained(model_name)

        # get additional parameters
        torch_dtype = kwargs.get("torch_dtype", torch.float32)
        low_cpu_mem_usage = kwargs.get("low_cpu_mem_usage", False)
        attn_implementation = kwargs.get("attn_implementation")

        # there are some kwargs that we wont be passed to AutoModel, so we need
        # to patch them in
        _old_from_config = AutoModelForCausalLM.from_config

        _from_config = partial(
            _old_from_config, attn_implementation=attn_implementation
        )
        AutoModelForCausalLM.from_config = _from_config  # patch

        if self.use_external_lib:
            kwargs = {
                "low_cpu_mem_usage": low_cpu_mem_usage,
                "use_marlin": False,  # disable, cannot be used for training (no forward+backward)
                "disable_exllama": True,  # disable, cannot be used for training (no backward)
                "use_tritonv2": True,
                "trainable": True,  # only support trainable mode
            }
        else:
            kwargs = {
                "low_cpu_mem_usage": low_cpu_mem_usage,  # this is only used for device map
                "backend": Backend.TRITON,
            }

        # this is a HF method that checks if the low_cpu_mem mode is enabled
        # via HF accelerate
        if is_fsdp_enabled():
            kwargs["low_cpu_mem_usage"] = True
            if self.use_external_lib:
                # Local
                from .autogptq_utils import (  # pylint: disable=import-outside-toplevel
                    make_sure_no_tensor_in_meta_device,
                )

                # We patch `make_sure_no_tensor_in_meta_device`
                # from autogptq to avoid errors on models without bias
                patch_target_module(
                    to_patch="auto_gptq.modeling._utils.make_sure_no_tensor_in_meta_device",
                    replace_with=make_sure_no_tensor_in_meta_device,
                    target_module="auto_gptq.modeling._base",
                )

                # NOTE: need to set the device map as below as we want to use AutoGPTQ for training.
                # For low_cpu_mem_usage = True, we have to set the device map to load checkpoints
                # to "cpu" to avoid gpu consumption before train
                # This approach will divert consumption to cpu memory,
                # a better approach would be to load the checkpoints to meta device
                # QLoRA is currently implemented by the former approach and
                # will encounter the same issue.
                # see https://github.com/huggingface/transformers/pull/25107#issuecomment-2134833262

                kwargs["device_map"] = {
                    "": (
                        (
                            torch.cuda.current_device()
                            if not kwargs["low_cpu_mem_usage"]
                            else "cpu"
                        )
                        if torch.cuda.is_available()
                        else None
                    )
                }

        # currently only enable triton_v2, because the triton kernels are the only ones
        # that have backwards
        model = GPTQModel.from_quantized(
            model_name,
            quantize_config=quantize_config,
            torch_dtype=torch_dtype,
            warmup_triton=False,  # disable for now as it will try to run the warmup while on CPU
            **kwargs,
        )

        # https://github.com/foundation-model-stack/fms-acceleration/pull/15
        # if FSDP distributed need to convert the AutoGPTQ model's
        # parameters (in tensors) to parameters. Also need to
        # store the int32 tensors in a float type

        try:
            world_size = torch.distributed.get_world_size()
        except ValueError:
            world_size = 1  # pg not init

        if (
            world_size > 1
            and os.environ.get("ACCELERATE_USE_FSDP", "false").lower() == "true"
        ):
            # register FSDP patch
            register_tensors_as_parameters_patch_rule(
                target_module=QuantLinear,
                torch_dtype=torch_dtype,
            )

            # replace
            AutoModelForCausalLM.from_config = _old_from_config

            # in low_cpu_mem_mode, if certain tensors like embeddings
            # are in the meta device, then certain operations like
            # embedding resizing will fail
            put_selected_meta_tensors_on_cpu(model)

        # AutoGPTQ does not set the torch_dtype of the model carefully
        model.config.torch_dtype = torch_dtype

        # these will be properly set since it is not loaded using from_pretrained
        # - so, set them here.
        # - in particular "is_loaded_in_4bit" will be checked in prepare_model_for_kbit_training
        #   and there is a section of code that will be skipped if not set.
        setattr(model, "is_loaded_in_4bit", True)
        setattr(model, "quantization_method", "gptq")
        return model

    @property
    def requires_custom_loading(self):
        return True

    @property
    def requires_augmentation(self):
        return True

    def augmentation(
        self,
        model,
        train_args: TrainingArguments,
        modifiable_args: Tuple[LoraConfig],
    ):
        # guarded imports
        # Third Party
        if self.use_external_lib:
            # Third Party
            from auto_gptq.nn_modules.qlinear.qlinear_tritonv2 import (  # pylint: disable=import-outside-toplevel,import-error
                QuantLinear,
            )
            from auto_gptq.utils.peft_utils import (  # pylint: disable=import-outside-toplevel,import-error
                GPTQLoraModel,
                get_gptq_peft_model,
            )

            # Local
            from .autogptq_utils import (  # pylint: disable=import-outside-toplevel
                create_new_module_peft,
                replace_module_peft,
            )
        else:
            # Local
            from .gptqmodel.utils.peft import (  # pylint: disable=import-outside-toplevel,import-error
                get_gptq_peft_model,
            )

        (peft_config,) = modifiable_args  # unpack modifiable args

        # some assertions
        assert peft_config is not None, "need peft_config to install PEFT adapters"
        # running this plugin in float16 is the most performant
        # https://github.com/foundation-model-stack/fms-acceleration/issues/84
        assert (
            model.dtype == torch.float16 or train_args.fp16
        ), "need to run in fp16 mixed precision or load model in fp16"

        # call the prepare_model_for_kbit_training. This will no longer be called
        # inside SFTTrainer, because we eventually return None for the peft_config.
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=train_args.gradient_checkpointing,
            gradient_checkpointing_kwargs=train_args.gradient_checkpointing_kwargs,
        )

        if self.use_external_lib:
            # These functions need to replaced due to some incompatibliites
            # with newer PEFT packages.
            # - on augmentation we call auto_gptq.utils.peft_utils.get_gptq_peft_model
            # - this internally calls peft.utils.other.get_peft_model
            # - however the problem is that peft API moves very fast, and there are incompatiblities
            #
            # During peft wrapping there are two key operations
            # 1. LoraModel._create_new_module is called to create a LoraLinear layer that is
            #    compatible with the base layer. For quantized base layers, the LoraLinear
            #    may be different.
            # 2. GPTQLoraModel._replace_module to replace the existing Linear with the LoraLinear.
            #    Also move to device (which may depend on how base layer is implemented)

            # NOTE: GPTQLoraModel inherits from LoraModel,
            # and the _create_new_module method is called
            # on the parent. Hence _create_new_module is patched on the parent

            # FIXME:
            # 1. investigate using BaseGPTQForCausalLM.make_sure_compatible_with_peft
            #    to see if we can get around the patching

            _old_create_new_module = LoraModel._create_new_module
            _old_replace_module = GPTQLoraModel._replace_module
            _create_new_module = partial(create_new_module_peft, target_cls=QuantLinear)
            LoraModel._create_new_module = staticmethod(_create_new_module)
            GPTQLoraModel._replace_module = MethodType(
                replace_module_peft, GPTQLoraModel
            )

        # Install GPTQ adapters using the AutoGPTQ package (with the above patches)
        # - try to get a model config to auto determine the layers that can be helpful
        #
        model_type = (
            model.config.model_type if hasattr(model.config, "model_type") else None
        )
        model = get_gptq_peft_model(
            model,
            peft_config=peft_config,
            auto_find_all_linears=requires_installation_on_all_linears(
                peft_config,
                model_type=model_type,
            ),
            train_mode=True,  # install adapaters for training
        )

        # We do not set `is_loaded_in_4bit`` at this point because otherwise
        # `accelerate.prepare_model` will think the device placement is finalized
        # for the quantized model, and will raise
        # Reassign `quantization_method` after PEFT installation replaces the top-level class
        setattr(model, "quantization_method", "gptq")

        modifiable_args = (None,)  # return a None for peft_config

        if self.use_external_lib:
            # undo the patching for hygine
            LoraModel._create_new_module = staticmethod(_old_create_new_module)
            GPTQLoraModel._replace_module = MethodType(
                _old_replace_module, GPTQLoraModel
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

            # for autogptq we will install the fix regardless of transformers or
            # trl version, because those fixes were only for BNB. Here we control
            # our own model loading
            # NOTE: guard this later with a more specific accelerate version check
            try:
                # Third Party
                # pylint: disable=import-outside-toplevel
                from torch.distributed.utils import ensure_weights_retied

                # then its handled internally and there is nothing to do
            except ImportError:
                # need to use our internal version
                # Local
                from .fsdp_utils import (  # pylint: disable=import-outside-toplevel
                    ensure_weights_retied,
                )

                accelerator.state.fsdp_plugin.param_init_fn = ensure_weights_retied(
                    accelerator.state.fsdp_plugin.param_init_fn,
                    model.get_base_model(),
                    accelerator.device,
                )
        return callbacks


# register
AccelerationPlugin.register_plugin(
    AutoGPTQAccelerationPlugin,
    configuration_and_paths=["peft.quantization.auto_gptq"],
)
