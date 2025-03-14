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
DATA_CONFIG_APPLY_CUSTOM_TEMPLATE_YAML = os.path.join(
    PREDEFINED_DATA_CONFIGS, "apply_custom_template.yaml"
)
DATA_CONFIG_APPLY_CUSTOM_JINJA_TEMPLATE_YAML = os.path.join(
    PREDEFINED_DATA_CONFIGS, "apply_custom_jinja_template.yaml"
)
DATA_CONFIG_PRETOKENIZE_JSON_DATA_YAML = os.path.join(
    PREDEFINED_DATA_CONFIGS, "pretokenized_json_data.yaml"
)
DATA_CONFIG_TOKENIZE_AND_APPLY_INPUT_MASKING_YAML = os.path.join(
    PREDEFINED_DATA_CONFIGS, "tokenize_and_apply_input_masking.yaml"
)
DATA_CONFIG_MULTIPLE_DATASETS_SAMPLING_YAML = os.path.join(
    PREDEFINED_DATA_CONFIGS, "multiple_datasets_with_sampling.yaml"
)
DATA_CONFIG_MULTITURN_DATA_YAML = os.path.join(
    PREDEFINED_DATA_CONFIGS, "multi_turn_data_with_chat_template.yaml"
)
DATA_CONFIG_MULTITURN_GRANITE_3_1B_DATA_YAML = os.path.join(
    PREDEFINED_DATA_CONFIGS, "multi_turn_data_with_chat_template_granite_3_1B.yaml"
)
DATA_CONFIG_YAML_STREAMING_INPUT_OUTPUT = os.path.join(
    PREDEFINED_DATA_CONFIGS, "tokenize_and_apply_input_masking_streaming.yaml"
)
DATA_CONFIG_YAML_STREAMING_PRETOKENIZED = os.path.join(
    PREDEFINED_DATA_CONFIGS, "pretokenized_json_data_streaming.yaml"
)
DATA_CONFIG_DUPLICATE_COLUMNS = os.path.join(
    PREDEFINED_DATA_CONFIGS, "duplicate_columns.yaml"
)
DATA_CONFIG_RENAME_SELECT_COLUMNS = os.path.join(
    PREDEFINED_DATA_CONFIGS, "rename_select_columns.yaml"
)
DATA_CONFIG_TOKENIZE_AND_TRAIN_WITH_HANDLER = os.path.join(
    PREDEFINED_DATA_CONFIGS, "tokenize_using_handler_and_train.yaml"
)
DATA_CONFIG_SKIP_LARGE_COLUMNS_HANDLER = os.path.join(
    PREDEFINED_DATA_CONFIGS, "skip_large_columns_data_handler_template.yaml"
)
