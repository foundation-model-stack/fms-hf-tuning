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

KEY_PLUGINS = "plugins"
PLUGIN_PREFIX = "fms_acceleration_"

ACCELERATION_FRAMEWORK_ENV_KEY = "ACCELERATION_FRAMEWORK_CONFIG_FILE"

# the order below is a linear precedence in which the plugins will be registered
# and activated.
# - hence the plugins that have model loaders should be on top of this list

PLUGINS = ["peft", "foak", "aadp", "moe"]
