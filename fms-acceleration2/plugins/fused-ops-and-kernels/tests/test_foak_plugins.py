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
import os

# Third Party
from fms_acceleration import AccelerationPluginConfigError
from fms_acceleration.utils import (
    instantiate_framework,
    read_configuration,
    update_configuration_contents,
)
import pytest  # pylint: disable=import-error

# instantiate_fromwork will handle registering and activating AutoGPTQAccelerationPlugin

# configuration
DIRNAME = os.path.dirname(__file__)
CONFIG_PATH_AUTO_GPTQ_FOAK = os.path.join(
    DIRNAME, "../configs/fast_quantized_peft.yaml"
)


@pytest.mark.skip(reason="Installation logic has changed - test to be fixed in future.")
def test_configure_gptq_foak_plugin():
    "test foak plugin loads correctly"

    # test that provided configuration correct correct instantiates plugin
    with instantiate_framework(
        read_configuration(CONFIG_PATH_AUTO_GPTQ_FOAK), require_packages_check=False
    ) as framework:

        # check flags and callbacks
        assert framework.requires_custom_loading is False
        assert framework.requires_augmentation
        assert len(framework.get_callbacks_and_ready_for_train()) == 0

    # attempt to activate plugin with configuration pointing to wrong path
    # - raise with message that no plugins can be configured
    with pytest.raises(ValueError) as e:
        with instantiate_framework(
            update_configuration_contents(
                read_configuration(CONFIG_PATH_AUTO_GPTQ_FOAK),
                "peft.quantization.fused_ops_and_kernels",
                "something",
            ),
        ):
            pass

    e.match("No plugins could be configured")

    # NOTE: currently only have all-or-one until address the generic patching
    # rules
    # attempt to actiavte plugin with unsupported settings
    # - raise with appropriate message complaining about wrong setting
    for key, wrong_value in [
        ("peft.quantization.fused_ops_and_kernels.fused_lora", False),
        ("peft.quantization.fused_ops_and_kernels.fast_loss", False),
        ("peft.quantization.fused_ops_and_kernels.fast_rsm_layernorm", False),
        ("peft.quantization.fused_ops_and_kernels.fast_rope_embeddings", False),
    ]:
        with pytest.raises(AccelerationPluginConfigError) as e:
            with instantiate_framework(
                update_configuration_contents(
                    read_configuration(CONFIG_PATH_AUTO_GPTQ_FOAK), key, wrong_value
                ),
            ):
                pass

        e.match(f"FastQuantizedPeftAccelerationPlugin: Value at '{key}'")
