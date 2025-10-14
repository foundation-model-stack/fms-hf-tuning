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
import os

# Third Party
from fms_acceleration.utils import instantiate_framework, read_configuration

# First Party
from fms_acceleration_moe import ScatterMoEAccelerationPlugin

# configuration
DIRNAME = os.path.dirname(__file__)
CONFIG_PATH_SCATTERMOE = os.path.join(DIRNAME, "../configs/scattermoe.yaml")


def test_framework_installs_scattermoe_plugin():
    with instantiate_framework(
        read_configuration(CONFIG_PATH_SCATTERMOE), require_packages_check=False
    ) as framework:
        for plugin in framework.active_plugins:
            assert isinstance(plugin[1], ScatterMoEAccelerationPlugin)
