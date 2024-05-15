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
import os
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

if is_fms_accelerate_available():

    # Third Party
    from fms_acceleration.framework import KEY_PLUGINS, AccelerationFramework
    from fms_acceleration.utils.test_utils import build_framework_and_maybe_instantiate

    if is_fms_accelerate_available(plugins="peft"):
        # Third Party
        from fms_acceleration_peft import AutoGPTQAccelerationPlugin

# see https://github.com/foundation-model-stack/fms-acceleration/blob/main/plugins/framework/tests/test_framework.py
# for more extensive unit tests

CONFIG_PATH_AUTO_GPTQ = os.path.join(
    os.path.dirname(__file__),
    "../../fixtures/accelerated-peft-autogptq-sample-configuration.yaml",
)

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

        def __init__(self, *args):
            super().__init__(*args)

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


@pytest.mark.skipif(
    not is_fms_accelerate_available(),
    reason="Only runs if fms-accelerate is installed",
)
def test_construct_framework_with_empty_file():
    "Ensure that framework configuration cannot have empty body"

    with pytest.raises(ValueError) as e:
        with tempfile.NamedTemporaryFile("w") as f:
            yaml.dump({KEY_PLUGINS: None}, f)
            AccelerationFramework(f.name)

    e.match(f"Configuration file must contain a '{KEY_PLUGINS}' body")


@pytest.mark.skipif(
    not is_fms_accelerate_available(plugins="peft"),
    reason="Only runs if fms-accelerate is installed along with accelerated-peft plugin",
)
def test_construct_framework_with_auto_gptq_peft():
    "Ensure that framework object initializes correctly with the sample config"

    # the test util below requires to read the file first
    with open(CONFIG_PATH_AUTO_GPTQ) as f:
        configuration = yaml.safe_load(f)

    # for this test we skip the require package check as second order package
    # dependencies of accelerated_peft is not required
    with build_framework_and_maybe_instantiate(
        [],
        configuration["plugins"],
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
        framework_args = config.AccelerationFrameworkArguments(
            acceleration_framework_config_file=CONFIG_PATH_AUTO_GPTQ,
        )

        # patch is_fms_accelerate_available to return False inside sft_trainer
        # to simulate fms_acceleration not installed
        with patch(
            "tuning.sft_trainer.is_fms_accelerate_available", return_value=False
        ):
            with pytest.raises(ValueError) as e:
                sft_trainer.train(
                    model_args,
                    data_args,
                    training_args,
                    tune_config,
                    acceleration_framework_args=framework_args,
                )
        e.match("Specified acceleration framework config")

        # test with a dummy configuration file that will fail to activate any
        # framework plugin
        with tempfile.NamedTemporaryFile("w") as f:
            yaml.dump({KEY_PLUGINS: {"dummy": 1}}, f)

            framework_args_dummy_file = config.AccelerationFrameworkArguments(
                acceleration_framework_config_file=f.name,
            )

            # patch is_fms_accelerate_available to return False inside sft_trainer
            # to simulate fms_acceleration not installed
            with pytest.raises(ValueError) as e:
                sft_trainer.train(
                    model_args,
                    data_args,
                    training_args,
                    tune_config,
                    acceleration_framework_args=framework_args_dummy_file,
                )
            e.match("No plugins could be configured.")


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
        framework_args = config.AccelerationFrameworkArguments(
            acceleration_framework_config_file=CONFIG_PATH_AUTO_GPTQ,
        )

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
                acceleration_framework_args=framework_args,
            )

        # spy to ensure that the plugin functions were called.
        # as expected given the configuration pointed to by CONFIG_PATH_AUTO_GPTQ
        assert MockedClass.model_loader_calls == 1
        assert MockedClass.augmentation_calls == 1
        assert MockedClass.get_callback_calls == 1
