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
from tempfile import NamedTemporaryFile
from typing import Dict
import os

# Third Party
import pytest
import yaml

# Local
from tuning.utils.import_utils import is_fms_accelerate_available
import tuning.config.configs as config

if is_fms_accelerate_available():

    # Third Party
    from fms_acceleration.framework import KEY_PLUGINS, AccelerationFramework

# see https://github.com/foundation-model-stack/fms-acceleration/blob/main/plugins/framework/tests/test_framework.py
# for more extensive unit tests


CONFIG_PATH_AUTO_GPTQ = os.path.join(
    os.path.dirname(__file__),
    "../../fixtures/accelerated-peft-autogptq-sample-configuration.yaml",
)


@pytest.mark.skipif(
    not is_fms_accelerate_available(),
    reason="Only runs if fms-accelerate is installed",
)
def test_construct_framework_with_empty_file():

    with pytest.raises(ValueError) as e:
        with NamedTemporaryFile("w") as f:
            yaml.dump({KEY_PLUGINS: None}, f)
            AccelerationFramework(f.name)

    e.match(f"Configuration file must contain a '{KEY_PLUGINS}' body")


@pytest.mark.skipif(
    not is_fms_accelerate_available(plugins="peft"),
    reason="Only runs if fms-accelerate is installed along with accelerated-peft plugin",
)
def test_construct_framework_with_auto_gptq_peft():

    args = config.AccelerationFrameworkArguments(
        acceleration_framework_config_file=CONFIG_PATH_AUTO_GPTQ,
    )

    # for this test we skip the require package check
    framework = AccelerationFramework(
        args.acceleration_framework_config_file, require_packages_check=False
    )

    # the configuration file should successfully activate the plugin
    assert len(framework.active_plugins) == 1
