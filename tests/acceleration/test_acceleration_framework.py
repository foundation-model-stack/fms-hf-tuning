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
from unittest.mock import patch
import tempfile

# Third Party
import pytest
import yaml

# First Party
from tests.helpers import causal_lm_train_kwargs
from tests.test_sft_trainer import BASE_LORA_KWARGS

# Local
from tuning import sft_trainer
from tuning.utils.import_utils import is_fms_accelerate_available
import tuning.config.configs as config
from tuning.config.acceleration_configs import (
    AccelerationFrameworkConfig, QuantizedLoraConfig
)
from tuning.config.acceleration_configs.quantized_lora_config import (
    AutoGPTQLoraConfig, BNBQLoraConfig
)

# pylint: disable=import-error
if is_fms_accelerate_available():

    # Third Party
    from fms_acceleration.framework import KEY_PLUGINS, AccelerationFramework
    from fms_acceleration.utils.test_utils import build_framework_and_maybe_instantiate

    if is_fms_accelerate_available(plugins="peft"):
        # Third Party
        from fms_acceleration_peft import AutoGPTQAccelerationPlugin


# There are more extensive unit tests in the
# https://github.com/foundation-model-stack/fms-acceleration
# repository.
# - see plugins/framework/tests/test_framework.py

# helper function
def create_mock_plugin_class(plugin_cls):
    "Create a mocked acceleration framework class that can be used to spy"

    # mocked plugin class
    class MockPlugin(plugin_cls):

        # counters used for spying
        model_loader_calls: int
        augmentation_calls: int
        get_callback_calls: int

        @classmethod
        def reset_calls(cls):
            # reset the counters
            cls.model_loader_calls = cls.augmentation_calls = cls.get_callback_calls = 0

        def model_loader(self, *args, **kwargs):
            MockPlugin.model_loader_calls += 1
            return super().model_loader(*args, **kwargs)

        def augmentation(
            self,
            *args,
            **kwargs,
        ):
            MockPlugin.augmentation_calls += 1
            return super().augmentation(*args, **kwargs)

        def get_callbacks_and_ready_for_train(self, *args, **kwargs):
            MockPlugin.get_callback_calls += 1
            return super().get_callbacks_and_ready_for_train(*args, **kwargs)

    return MockPlugin


def test_construct_framework_config_with_incorrect_configurations():
    "Ensure that framework configuration cannot have empty body"

    with pytest.raises(
        ValueError, match="AccelerationFrameworkConfig construction requires at least one dataclass"
    ):
        AccelerationFrameworkConfig.from_dataclasses()

    # test a currently not supported config
    with pytest.raises(
        ValueError, match="only 'from_quantized' == True currently supported."
    ):
        AutoGPTQLoraConfig(from_quantized=False)

    # test an invalid activation of two standalone configs. 
    quantized_lora_config = QuantizedLoraConfig(
        auto_gptq=AutoGPTQLoraConfig(), bnb_qlora=BNBQLoraConfig()
    )
    with pytest.raises(
        ValueError, match="Configuration path 'peft.quantization' already has one standalone config."
    ):
        AccelerationFrameworkConfig.from_dataclasses(quantized_lora_config).get_framework()

@pytest.mark.skipif(
    not is_fms_accelerate_available(plugins="peft"),
    reason="Only runs if fms-accelerate is installed along with accelerated-peft plugin",
)
def test_construct_framework_with_auto_gptq_peft():
    "Ensure that framework object is correctly configured."

    quantized_lora_config = QuantizedLoraConfig(auto_gptq=AutoGPTQLoraConfig())
    acceleration_config = AccelerationFrameworkConfig.from_dataclasses(quantized_lora_config)

    # for this test we skip the require package check as second order package
    # dependencies of accelerated_peft is not required
    with build_framework_and_maybe_instantiate(
        [],
        acceleration_config.to_dict(),
        reset_registrations=False,
        require_packages_check=False,
    ) as framework:

        # the configuration file should successfully activate the plugin
        assert len(framework.active_plugins) == 1

@pytest.mark.skipif(
    not is_fms_accelerate_available(),
    reason="Only runs if fms-accelerate is installed",
)
def test_framework_not_installed_or_initalized_properly():
    """Ensure that specifying an framework config without installing fms_acceleration
    results in raise.
    """
    with tempfile.TemporaryDirectory() as tempdir:
        TRAIN_KWARGS = {
            **BASE_LORA_KWARGS,
            **{
                "output_dir": tempdir,
            },
        }
        model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
            TRAIN_KWARGS
        )
        quantized_lora_config = QuantizedLoraConfig(auto_gptq=AutoGPTQLoraConfig())

        # patch is_fms_accelerate_available to return False inside sft_trainer
        # to simulate fms_acceleration not installed
        with patch(
            "tuning.config.acceleration_configs.acceleration_framework_config.is_fms_accelerate_available", return_value=False
        ):
            with pytest.raises(
                ValueError,
                match="No acceleration framework package found."
            ):
                sft_trainer.train(
                    model_args,
                    data_args,
                    training_args,
                    tune_config,
                    quantized_lora_config=quantized_lora_config
                )

@pytest.mark.skipif(
    not is_fms_accelerate_available(plugins="peft"),
    reason="Only runs if fms-accelerate is installed along with accelerated-peft plugin",
)
def test_framework_intialized_properly():
    """Ensure that specifying an framework config without installing fms_acceleration
    results in raise.
    """
    with tempfile.TemporaryDirectory() as tempdir:
        TRAIN_KWARGS = {
            **BASE_LORA_KWARGS,
            **{"fp16": True},
            **{"model_name_or_path": "TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ"},
            **{"output_dir": tempdir},
        }
        model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
            TRAIN_KWARGS
        )
        quantized_lora_config = QuantizedLoraConfig(auto_gptq=AutoGPTQLoraConfig())

        # create mocked plugin class for spying
        MockedClass = create_mock_plugin_class(AutoGPTQAccelerationPlugin)
        MockedClass.reset_calls()

        # test utils to register the mocked pluigin class and call
        # sft_trainer
        with build_framework_and_maybe_instantiate(
            [(["peft.quantization.auto_gptq"], MockedClass)],
            instantiate=False,
        ):
            sft_trainer.train(
                model_args,
                data_args,
                training_args,
                tune_config,
                # acceleration_framework_args=framework_args,
                quantized_lora_config=quantized_lora_config
            )

        # spy to ensure that the plugin functions were called.
        # as expected given the configuration pointed to by CONFIG_PATH_AUTO_GPTQ
        assert MockedClass.model_loader_calls == 1
        assert MockedClass.augmentation_calls == 1
        assert MockedClass.get_callback_calls == 1
