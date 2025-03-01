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
# use importlib to load the packages, if they are installed
import importlib

# Local
from .constants import PLUGIN_PREFIX, PLUGINS
from .framework import AccelerationFramework
from .framework_plugin import (
    AccelerationPlugin,
    AccelerationPluginConfigError,
    get_relevant_configuration_sections,
)

for postfix in PLUGINS:
    plugin_name = f"{PLUGIN_PREFIX}{postfix}"
    if importlib.util.find_spec(plugin_name):
        importlib.import_module(plugin_name)
