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

"""Helpful datasets for configuring individual unit tests.
"""
# Standard
import os

### Constants used for data
_DATA_DIR = os.path.join(os.path.dirname(__file__))
TRAINER_CONFIG_TEST_LOSS_ON_THRESHOLD_YAML = os.path.join(
    _DATA_DIR, "loss_on_threshold.yaml"
)
TRAINER_CONFIG_TEST_LOSS_ON_THRESHOLD_WITH_TRAINER_STATE_YAML = os.path.join(
    _DATA_DIR, "loss_on_threshold_with_trainer_state.yaml"
)
TRAINER_CONFIG_EXPOSED_METRICS_YAML = os.path.join(_DATA_DIR, "exposed_metrics.yaml")
TRAINER_CONFIG_INCORRECT_SOURCE_EVENT_EXPOSED_METRICS_YAML = os.path.join(
    _DATA_DIR, "incorrect_source_event_exposed_metrics.yaml"
)
TRAINER_CONFIG_TEST_INVALID_TYPE_RULE_YAML = os.path.join(
    _DATA_DIR, "loss_with_invalid_type_rule.yaml"
)
TRAINER_CONFIG_TEST_MALICIOUS_OS_RULE_YAML = os.path.join(
    _DATA_DIR, "loss_with_malicious_os_rule.yaml"
)
TRAINER_CONFIG_TEST_MALICIOUS_INPUT_RULE_YAML = os.path.join(
    _DATA_DIR, "loss_with_malicious_input_rule.yaml"
)
TRAINER_CONFIG_TEST_INVALID_TRIGGER_YAML = os.path.join(
    _DATA_DIR, "loss_invalid_trigger.yaml"
)
TRAINER_CONFIG_TEST_INVALID_OPERATION_YAML = os.path.join(
    _DATA_DIR, "loss_invalid_operation.yaml"
)
TRAINER_CONFIG_TEST_INVALID_OPERATION_ACTION_YAML = os.path.join(
    _DATA_DIR, "loss_invalid_operation_action.yaml"
)
TRAINER_CONFIG_TEST_INVALID_METRIC_YAML = os.path.join(
    _DATA_DIR, "loss_invalid_metric.yaml"
)
TRAINER_CONFIG_TEST_CUSTOM_METRIC_YAML = os.path.join(
    _DATA_DIR, "loss_custom_metric.yaml"
)
TRAINER_CONFIG_TEST_CUSTOM_OPERATION_YAML = os.path.join(
    _DATA_DIR, "loss_custom_operation.yaml"
)
TRAINER_CONFIG_TEST_CUSTOM_OPERATION_INVALID_ACTION_YAML = os.path.join(
    _DATA_DIR, "loss_custom_operation_invalid_action.yaml"
)
