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
from unittest.mock import patch
import os

# Third Party
from fms_acceleration import AccelerationPluginConfigError
from fms_acceleration.utils import (
    instantiate_framework,
    read_configuration,
    update_configuration_contents,
)
import pytest

MODEL_NAME_AUTO_GPTQ = "TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ"

# instantiate_fromwork will handle registering and activating AutoGPTQAccelerationPlugin

# configuration
DIRNAME = os.path.dirname(__file__)
CONFIG_PATH_AUTO_GPTQ = os.path.join(DIRNAME, "../configs/autogptq.yaml")
CONFIG_PATH_BNB = os.path.join(DIRNAME, "../configs/bnb.yaml")


# We do not enable the skip since this test does not actually require the packages
# installed
# @pytest.mark.skipif(
#     not check_plugin_packages(AutoGPTQAccelerationPlugin),
#     reason="missing package requirements of AutoGPTQAccelerationPlugin",
# )
def test_configure_gptq_plugin():
    "test auto_gptq plugin loads correctly"

    # test that provided configuration correct correct instantiates plugin
    with instantiate_framework(
        read_configuration(CONFIG_PATH_AUTO_GPTQ), require_packages_check=False
    ) as framework:

        # check flags and callbacks
        assert framework.requires_custom_loading
        assert framework.requires_augmentation
        assert len(framework.get_callbacks_and_ready_for_train()) == 0

    # attempt to activate plugin with configuration pointing to wrong path
    # - raise with message that no plugins can be configured
    with pytest.raises(ValueError) as e:
        with instantiate_framework(
            update_configuration_contents(
                read_configuration(CONFIG_PATH_AUTO_GPTQ),
                "peft.quantization.auto_gptq",
                "something",
            ),
            require_packages_check=False,
        ):
            pass

    e.match("No plugins could be configured")

    # attempt to actiavte plugin with unsupported settings
    # - raise with appropriate message complaining about wrong setting
    for key, wrong_value in [
        ("peft.quantization.auto_gptq.kernel", "triton"),
        ("peft.quantization.auto_gptq.from_quantized", False),
    ]:
        with pytest.raises(AccelerationPluginConfigError) as e:
            with instantiate_framework(
                update_configuration_contents(
                    read_configuration(CONFIG_PATH_AUTO_GPTQ), key, wrong_value
                ),
                require_packages_check=False,
            ):
                pass

        e.match(f"AutoGPTQAccelerationPlugin: Value at '{key}'")


def test_autogptq_loading():
    "Test for correctness of autogptq loading logic"

    def autogptq_unavailable(package_name: str):
        return False

    # - Test that error is thrown when use_external_lib is True but no package found.
    # 1. mock import function `_is_package_available` to return autogptq not available
    # 2. instantiate the framework with the plugin
    # 3. check when using external package and it is not available, an AssertionError is thrown
    with pytest.raises(
        AssertionError,
        match="Unable to use external library, auto_gptq module not found. "
        "Refer to README for installation instructions  as a specific version might be required.",
    ):
        with patch(
            "transformers.utils.import_utils._is_package_available",
            autogptq_unavailable,
        ):
            with instantiate_framework(
                update_configuration_contents(
                    read_configuration(CONFIG_PATH_AUTO_GPTQ),
                    "peft.quantization.auto_gptq.use_external_lib",
                    True,
                ),
                require_packages_check=False,
            ) as framework:
                pass

    # First Party
    from fms_acceleration_peft.framework_plugin_autogptq import (  # pylint: disable=import-outside-toplevel
        AutoGPTQAccelerationPlugin,
    )

    # - Test that plugin attribute is set when config field `use_external_lib` is False
    # When plugin attribute is set correctly, it will route to correct package on model loading
    with instantiate_framework(
        update_configuration_contents(
            read_configuration(CONFIG_PATH_AUTO_GPTQ),
            "peft.quantization.auto_gptq.use_external_lib",
            False,
        ),
        require_packages_check=False,
    ) as framework:
        for _, plugin in framework.active_plugins:
            if isinstance(plugin, AutoGPTQAccelerationPlugin):
                assert (
                    plugin.use_external_lib is False
                ), "Plugin attribute not correctly set from config field"

    # - Test that plugin attribute is set when config field `use_external_lib` is None
    # When plugin attribute is set correctly, it will route to correct package on model loading
    default_config = read_configuration(CONFIG_PATH_AUTO_GPTQ)
    default_config["peft"]["quantization"]["auto_gptq"].pop("use_external_lib")
    with instantiate_framework(
        default_config,
        require_packages_check=False,
    ) as framework:
        for _, plugin in framework.active_plugins:
            if isinstance(plugin, AutoGPTQAccelerationPlugin):
                assert (
                    plugin.use_external_lib is False
                ), "Plugin attribute not correctly set from config field"


# We do not enable the skip since this test does not actually require the packages
# installed
# @pytest.mark.skipif(
#     not check_plugin_packages(BNBAccelerationPlugin),
#     reason="missing package requirements of BNBAccelerationPlugin",
# )
def test_configure_bnb_plugin():
    "test bnb plugin loads correctly"

    # test that provided configuration correct correct instantiates plugin
    with instantiate_framework(
        read_configuration(CONFIG_PATH_BNB), require_packages_check=False
    ) as framework:

        # check flags and callbacks
        assert framework.requires_custom_loading
        assert framework.requires_augmentation
        assert len(framework.get_callbacks_and_ready_for_train()) == 0

    # test valid combinatinos
    for key, correct_value in [
        ("peft.quantization.bitsandbytes.quant_type", "nf4"),
        ("peft.quantization.bitsandbytes.quant_type", "fp4"),
    ]:
        with instantiate_framework(
            update_configuration_contents(
                read_configuration(CONFIG_PATH_BNB), key, correct_value
            ),
            require_packages_check=False,
        ):
            # check flags and callbacks
            assert framework.requires_custom_loading
            assert framework.requires_augmentation
            assert len(framework.get_callbacks_and_ready_for_train()) == 0

    # test no_peft_model is true skips plugin.augmentation
    for key, correct_value in [
        ("peft.quantization.bitsandbytes.no_peft_model", True),
        ("peft.quantization.bitsandbytes.no_peft_model", False),
    ]:
        with instantiate_framework(
            update_configuration_contents(
                read_configuration(CONFIG_PATH_BNB), key, correct_value
            ),
            require_packages_check=False,
        ):
            # check flags and callbacks
            assert (not correct_value) == framework.requires_augmentation

    # attempt to activate plugin with configuration pointing to wrong path
    # - raise with message that no plugins can be configured
    with pytest.raises(ValueError) as e:
        with instantiate_framework(
            update_configuration_contents(
                read_configuration(CONFIG_PATH_BNB),
                "peft.quantization.bitsandbytes",
                "something",
            ),
            require_packages_check=False,
        ):
            pass

    e.match("No plugins could be configured")

    # attempt to actiavte plugin with unsupported settings
    # - raise with appropriate message complaining about wrong setting
    for key, correct_value in [
        ("peft.quantization.bitsandbytes.quant_type", "wrong_type"),
    ]:
        with pytest.raises(AccelerationPluginConfigError) as e:
            with instantiate_framework(
                update_configuration_contents(
                    read_configuration(CONFIG_PATH_BNB), key, correct_value
                ),
                require_packages_check=False,
            ):
                pass

        e.match(f"BNBAccelerationPlugin: Value at '{key}'")
