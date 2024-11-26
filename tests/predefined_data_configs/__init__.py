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

"""Helpful datasets for configuring individual unit tests.
"""
# Standard
import os

### Constants used for data
PREDEFINED_DATA_CONFIGS = os.path.join(os.path.dirname(__file__))
APPLY_CUSTOM_TEMPLATE_YAML = os.path.join(
    PREDEFINED_DATA_CONFIGS, "apply_custom_template.yaml"
)
PRETOKENIZE_JSON_DATA_YAML = os.path.join(
    PREDEFINED_DATA_CONFIGS, "pretokenized_json_data.yaml"
)
TOKENIZE_AND_INSTRUCTION_MASKING_YAML = os.path.join(
    PREDEFINED_DATA_CONFIGS, "tokenize_and_instruction_masking.yaml"
)
