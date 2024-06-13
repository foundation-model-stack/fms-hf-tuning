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

# First Party
from tests.helpers import causal_lm_train_kwargs
from tests.test_sft_trainer import BASE_LORA_KWARGS

# Local
from .spying_utils import create_mock_plugin_class
from tuning import sft_trainer
from tuning.config.acceleration_configs import (
    AccelerationFrameworkConfig,
    QuantizedLoraConfig,
)
from tuning.config.acceleration_configs.quantized_lora_config import (
    AutoGPTQLoraConfig,
    BNBQLoraConfig,
)
from tuning.utils.import_utils import is_fms_accelerate_available

# pylint: disable=import-error
if is_fms_accelerate_available():

    # Third Party
    from fms_acceleration.utils.test_utils import build_framework_and_maybe_instantiate

    if is_fms_accelerate_available(plugins="peft"):
        # Third Party
        from fms_acceleration_peft import AutoGPTQAccelerationPlugin


# There are more extensive unit tests in the
# https://github.com/foundation-model-stack/fms-acceleration
# repository.
# - see plugins/framework/tests/test_framework.py


def test_acceleration_framework_fail_construction():
    """Ensure that construct of the framework will fail if rules regarding
    the dataclasess are violated.
    """

    # 1. Rule 1: No two standalone dataclasses can exist at the same path
    # - Test that the framework will fail to construct if there are multiple
    #    standalone plugins under the same path that are simultaneously requested.
    quantized_lora_config = QuantizedLoraConfig(
        auto_gptq=AutoGPTQLoraConfig(), bnb_qlora=BNBQLoraConfig()
    )
    with pytest.raises(
        ValueError,
        match="Configuration path 'peft.quantization' already has one standalone config.",
    ):
        AccelerationFrameworkConfig.from_dataclasses(
            quantized_lora_config
        ).get_framework()

    # 2. Rule 2: Dataclass cannot request a plugin that is not yet installed.
    # - Test that framework will fail to construct if trying to activate a plugin
    #   that is not yet installed
    # with patch(
    #     "tuning.config.acceleration_configs.acceleration_framework_config."
    #     "is_fms_accelerate_available",
    #     return_value=False,
    # ):


@pytest.mark.skip(
    """ NOTE: this scenario will actually never happen, since in the code we always
    provide at least one dataclass (can consider to remove this test).
    """
)
def test_construct_framework_config_raise_if_constructing_with_no_dataclassess():
    """Ensure that framework configuration config will refused to construct
    if no dataclasses are provided.
    """

    with pytest.raises(
        ValueError,
        match="AccelerationFrameworkConfig construction requires at least one dataclass",
    ):
        AccelerationFrameworkConfig.from_dataclasses()


@pytest.mark.skipif(
    not is_fms_accelerate_available(plugins="peft"),
    reason="Only runs if fms-accelerate is installed along with accelerated-peft plugin",
)
def test_construct_framework_with_auto_gptq_peft_successfully():
    "Ensure that framework object is correctly configured."

    # 1. correctly initialize a set of quantized lora config dataclass
    #    with auto-gptq
    quantized_lora_config = QuantizedLoraConfig(auto_gptq=AutoGPTQLoraConfig())

    # - instantiate the acceleration config
    acceleration_config = AccelerationFrameworkConfig.from_dataclasses(
        quantized_lora_config
    )

    # build the framework by
    # - passing acceleration configuration contents (via .to_dict()).
    # - NOTE: we skip the required packages check in the framework since it is
    #         not necessary for this test (e.g., we do not need auto_gptq installed)
    # - check that the plugin is correctly activated
    with build_framework_and_maybe_instantiate(
        [],
        acceleration_config.to_dict(),  # pass in contents
        reset_registrations=False,
        require_packages_check=False,  # not required
    ) as framework:

        # plugin activated!
        assert len(framework.active_plugins) == 1


@pytest.mark.skipif(
    not is_fms_accelerate_available(),
    reason="Only runs if fms-accelerate is installed",
)
def test_framework_raises_if_used_with_missing_package():
    """Ensure that trying the use the framework, without first installing fms_acceleration
    will raise.
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
            "tuning.config.acceleration_configs.acceleration_framework_config."
            "is_fms_accelerate_available",
            return_value=False,
        ):
            with pytest.raises(
                ValueError, match="No acceleration framework package found."
            ):
                sft_trainer.train(
                    model_args,
                    data_args,
                    training_args,
                    tune_config,
                    quantized_lora_config=quantized_lora_config,
                )


@pytest.mark.skipif(
    not is_fms_accelerate_available(plugins="peft"),
    reason="Only runs if fms-accelerate is installed along with accelerated-peft plugin",
)
def test_framework_intialized_properly():
    """Ensure that specifying a properly configured acceleration dataclass
    properly activates the framework plugin and runs the train sucessfully.
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

        # setup default quantized lora args dataclass
        # - with auth gptq as the quantized method
        quantized_lora_config = QuantizedLoraConfig(auto_gptq=AutoGPTQLoraConfig())

        # create mocked plugin class for spying
        MockedPlugin = create_mock_plugin_class(AutoGPTQAccelerationPlugin)
        MockedPlugin.reset_calls()

        # 1. mock a plugin class
        # 2. register the mocked plugin
        # 3. call sft_trainer.train
        with build_framework_and_maybe_instantiate(
            [(["peft.quantization.auto_gptq"], MockedPlugin)],
            instantiate=False,
        ):
            sft_trainer.train(
                model_args,
                data_args,
                training_args,
                tune_config,
                quantized_lora_config=quantized_lora_config,
            )

        # spy inside the train to ensure that the acceleration plugin
        # was called. In the context of the AutoGPTQ plugin
        # 1. Will sucessfully load the model (to load AutoGPTQ 4-bit linear)
        # 2. Will successfully agument the model (to install PEFT)
        # 3. Will succesfully call get_ready_for_train
        assert MockedPlugin.model_loader_calls == 1
        assert MockedPlugin.augmentation_calls == 1
        assert MockedPlugin.get_ready_for_train_calls == 1
