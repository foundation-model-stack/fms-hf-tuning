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
from dataclasses import dataclass
from typing import List

# Third Party
from simpleeval import FunctionNotDefined
from transformers import IntervalStrategy, TrainerControl, TrainerState
import pytest

# First Party
from tests.trainercontroller.custom_metric import CustomMetric
from tests.trainercontroller.custom_operation import CustomOperation
from tests.trainercontroller.custom_operation_invalid_action import (
    CustomOperationInvalidAction,
)
import tests.artifacts.testdata.trainercontroller as td

# Local
import tuning.config.configs as config
import tuning.trainercontroller as tc


@dataclass
class InputData:
    """Stores the operation handler instance and corresponding action"""

    args: config.TrainingArguments
    states: List[TrainerState]
    metrics: dict


def _setup_data() -> InputData:
    """
    Sets up the test data for the test cases.
    This includes the logs, arguments for training and state
    of the training.

    Returns:
        InputData.
    """
    # Test data to mimic the fields of trainer loop log-lines
    # trainer arguments and the initial state
    return InputData(
        args=config.TrainingArguments(
            output_dir="",
            logging_strategy=IntervalStrategy.STEPS,
            logging_steps=1,
        ),
        states=[
            TrainerState(
                log_history=[
                    {"loss": 2.0, "epoch": 0.1},
                    {"loss": 2.1, "epoch": 0.25},
                ],
                epoch=0.6,
                global_step=1,
            ),
            TrainerState(
                log_history=[
                    {"loss": 2.0, "epoch": 0.1},
                    {"loss": 2.1, "epoch": 0.25},
                    {"loss": 1.3, "epoch": 0.5},
                ],
                epoch=1.0,
                global_step=2,
            ),
            TrainerState(
                log_history=[
                    {"loss": 2.0, "epoch": 0.1},
                    {"loss": 2.1, "epoch": 0.25},
                    {"loss": 1.3, "epoch": 0.5},
                    {"loss": 0.9, "epoch": 0.6},
                ],
                epoch=1.6,
                global_step=3,
            ),
        ],
        metrics=[
            {"eval_loss": 2.2},
            {"eval_loss": 2.1},
            {"eval_loss": 2.3},
            {"eval_loss": 2.4},
            {"eval_loss": 2.5},
        ],
    )


def test_loss_on_threshold():
    """Tests the loss threshold example in
    `examples/trainer-controller-configs/loss_on_threshold.yaml`
    """
    test_data = _setup_data()
    tc_callback = tc.TrainerControllerCallback(
        td.TRAINER_CONFIG_TEST_LOSS_ON_THRESHOLD_YAML
    )
    control = TrainerControl(should_training_stop=False)
    # Trigger on_init_end to perform registration of handlers to events
    tc_callback.on_init_end(
        args=test_data.args, state=test_data.states[2], control=control
    )
    # Trigger rule and test the condition
    tc_callback.on_log(args=test_data.args, state=test_data.states[2], control=control)
    assert control.should_training_stop is True


def test_thresholded_training_loss():
    """Tests the thresholded training loss example in
    `examples/trainer-controller-configs/thresholded-training-loss.yaml`
    """
    test_data = _setup_data()
    tc_callback = tc.TrainerControllerCallback(
        td.TRAINER_CONFIG_TEST_THRESHOLDED_TRAINING_LOSS_YAML
    )
    control = TrainerControl(should_training_stop=False)
    # Trigger on_init_end to perform registration of handlers to events
    tc_callback.on_init_end(
        args=test_data.args, state=test_data.states[2], control=control
    )
    # Trigger rule and test the condition
    tc_callback.on_log(args=test_data.args, state=test_data.states[2], control=control)
    assert control.should_training_stop is True


def test_thresholded_training_loss_on_save():
    """Tests the thresholded training loss example in
    `examples/trainer-controller-configs/on-save.yaml`
    """
    test_data = _setup_data()
    tc_callback = tc.TrainerControllerCallback(td.TRAINER_CONFIG_TEST_ON_SAVE_YAML)
    control = TrainerControl(should_training_stop=False)
    # Trigger on_init_end to perform registration of handlers to events
    tc_callback.on_init_end(
        args=test_data.args, state=test_data.states[2], control=control
    )
    # Trigger rule and test the condition
    tc_callback.on_save(args=test_data.args, state=test_data.states[2], control=control)
    assert control.should_training_stop is True


def test_log_controller(caplog):
    """Tests the expose metric scenario example in
    `examples/trainer-controller-configs/log_controller.yaml`
    """
    test_data = _setup_data()
    tc_callback = tc.TrainerControllerCallback(td.TRAINER_CONFIG_LOG_CONTROLLER_YAML)
    control = TrainerControl(should_log=False)
    # Trigger on_init_end to perform registration of handlers to events
    tc_callback.on_init_end(
        args=test_data.args, state=test_data.states[2], control=control
    )
    tc_callback.on_step_end(
        args=test_data.args, state=test_data.states[2], control=control
    )
    assert "This is a test log format" in caplog.text


def test_non_decreasing_training_loss():
    """Tests the non-decreasing training loss example in
    `examples/trainer-controller-configs/non-decreasing-training-loss.yaml`
    """
    test_data = _setup_data()
    tc_callback = tc.TrainerControllerCallback(
        td.TRAINER_CONFIG_TEST_NON_DECREASING_TRAINING_LOSS_YAML
    )
    control = TrainerControl(should_training_stop=False)
    # Trigger on_init_end to perform registration of handlers to events
    tc_callback.on_init_end(
        args=test_data.args, state=test_data.states[2], control=control
    )
    # Trigger rule and test the condition
    incremental_history = []
    original_history = test_data.states[2].log_history
    for log in original_history:
        incremental_history.append(log)
        test_data.states[2].log_history = incremental_history
        tc_callback.on_log(
            args=test_data.args, state=test_data.states[2], control=control
        )
        if control.should_training_stop:
            assert True


def test_epoch_level_training_loss():
    """Tests the epoch level training loss example in
    `examples/trainer-controller-configs/epoch-level-training-loss.yaml`
    """
    test_data = _setup_data()
    tc_callback = tc.TrainerControllerCallback(
        td.TRAINER_CONFIG_TEST_EPOCH_LEVEL_TRAINING_LOSS_YAML
    )
    control = TrainerControl(should_training_stop=False)
    # Trigger on_init_end to perform registration of handlers to events
    tc_callback.on_init_end(
        args=test_data.args, state=test_data.states[2], control=control
    )
    # Trigger rule and test the condition
    incremental_history = []
    original_history = test_data.states[2].log_history
    test_passes = False
    for log in original_history:
        incremental_history.append(log)
        test_data.states[2].log_history = incremental_history
        tc_callback.on_log(
            args=test_data.args, state=test_data.states[2], control=control
        )
        tc_callback.on_epoch_end(
            args=test_data.args, state=test_data.states[2], control=control
        )
        if control.should_training_stop is True:
            test_passes = True
    assert test_passes is True


def test_epoch_level_eval_loss():
    """Tests the epoch level eval loss example in
    `examples/trainer-controller-configs/epoch-level-eval-loss.yaml`
    """
    test_data = _setup_data()
    tc_callback = tc.TrainerControllerCallback(
        td.TRAINER_CONFIG_TEST_EPOCH_LEVEL_EVAL_LOSS_YAML
    )
    control = TrainerControl(should_training_stop=False)
    # Trigger on_init_end to perform registration of handlers to events
    tc_callback.on_init_end(
        args=test_data.args, state=test_data.states[2], control=control
    )
    # Trigger rule and test the condition
    tc_callback.on_evaluate(
        args=test_data.args,
        state=test_data.states[2],
        control=control,
        metrics=test_data.metrics[0],
    )
    tc_callback.on_epoch_end(
        args=test_data.args, state=test_data.states[2], control=control
    )
    assert control.should_training_stop is True


def test_epoch_level_eval_loss_patience():
    """Tests the epoch level eval loss with patience threshold example in
    `examples/trainer-controller-configs/epoch-level-eval-loss-patience.yaml`
    """
    test_data = _setup_data()
    tc_callback = tc.TrainerControllerCallback(
        td.TRAINER_CONFIG_TEST_EPOCH_LEVEL_EVAL_LOSS_PATIENCE_YAML
    )
    control = TrainerControl(should_training_stop=False)
    # Trigger on_init_end to perform registration of handlers to events
    tc_callback.on_init_end(
        args=test_data.args, state=test_data.states[2], control=control
    )
    # Trigger rule and test the condition
    for metrics in test_data.metrics:
        control = TrainerControl(should_training_stop=False)
        tc_callback.on_evaluate(
            args=test_data.args,
            state=test_data.states[2],
            control=control,
            metrics=metrics,
        )
        tc_callback.on_epoch_end(
            args=test_data.args, state=test_data.states[2], control=control
        )
        if control.should_training_stop:
            break
    assert control.should_training_stop is True


def test_loss_on_threshold_with_trainer_state():
    """Tests the loss threshold with trainer state example in
    `examples/trainer-controller-configs/loss_on_threshold_with_trainer_state.yaml`
    """
    test_data = _setup_data()
    tc_callback = tc.TrainerControllerCallback(
        td.TRAINER_CONFIG_TEST_LOSS_ON_THRESHOLD_WITH_TRAINER_STATE_YAML
    )
    control = TrainerControl(should_training_stop=False)
    # Trigger on_init_end to perform registration of handlers to events
    tc_callback.on_init_end(
        args=test_data.args, state=test_data.states[2], control=control
    )
    # Trigger rule and test the condition
    tc_callback.on_log(args=test_data.args, state=test_data.states[2], control=control)


def test_exposed_metrics():
    """Tests the expose metric scenario example in
    `examples/trainer-controller-configs/exposed_metrics.yaml`
    """
    test_data = _setup_data()
    tc_callback = tc.TrainerControllerCallback(td.TRAINER_CONFIG_EXPOSED_METRICS_YAML)
    control = TrainerControl(should_training_stop=False)
    metrics = {"eval_loss": 2.2}
    # Trigger on_init_end to perform registration of handlers to events
    tc_callback.on_init_end(
        args=test_data.args, state=test_data.states[2], control=control
    )
    # Trigger rule and test the condition
    tc_callback.on_evaluate(
        args=test_data.args, state=test_data.states[2], control=control, metrics=metrics
    )
    assert control.should_training_stop is True


def test_incorrect_source_event_exposed_metrics():
    """Tests the expose metric scenario example in
    `examples/trainer-controller-configs/incorrect_source_event_exposed_metrics.yaml`
    """
    with pytest.raises(ValueError) as exception_handler:
        test_data = _setup_data()
        tc_callback = tc.TrainerControllerCallback(
            td.TRAINER_CONFIG_INCORRECT_SOURCE_EVENT_EXPOSED_METRICS_YAML
        )
        control = TrainerControl(should_training_stop=False)
        metrics = {"eval_loss": 2.2}
        # Trigger on_init_end to perform registration of handlers to events
        tc_callback.on_init_end(
            args=test_data.args, state=test_data.states[2], control=control
        )
        # Trigger rule and test the condition
        tc_callback.on_evaluate(
            args=test_data.args,
            state=test_data.states[2],
            control=control,
            metrics=metrics,
        )
        assert (
            str(exception_handler.value).strip("'")
            == "Specified source event [on_incorrect_event] is invalid for EvalMetrics"
        )
        assert control.should_training_stop is True


def test_custom_metric_handler():
    """Tests the custom metric registration
    `examples/trainer-controller-configs/loss_custom_metric.yaml`
    """
    test_data = _setup_data()
    tc_callback = tc.TrainerControllerCallback(
        td.TRAINER_CONFIG_TEST_CUSTOM_METRIC_YAML
    )
    tc_callback.register_metric_handlers([CustomMetric])
    control = TrainerControl()
    # Trigger on_init_end to perform registration of handlers to events
    tc_callback.on_init_end(
        args=test_data.args, state=test_data.states[2], control=control
    )
    # Trigger rule and test the condition
    tc_callback.on_log(args=test_data.args, state=test_data.states[2], control=control)
    assert control.should_training_stop is True


def test_custom_operation_handler():
    """Tests the custom operation registration
    `examples/trainer-controller-configs/loss_custom_operation.yaml`
    """
    test_data = _setup_data()
    tc_callback = tc.TrainerControllerCallback(
        td.TRAINER_CONFIG_TEST_CUSTOM_OPERATION_YAML
    )
    tc_callback.register_operation_handlers([CustomOperation])
    control = TrainerControl(should_training_stop=False)
    # Trigger on_init_end to perform registration of handlers to events
    tc_callback.on_init_end(
        args=test_data.args, state=test_data.states[2], control=control
    )
    # Trigger rule and test the condition
    tc_callback.on_log(args=test_data.args, state=test_data.states[2], control=control)
    assert control.should_training_stop is True


def test_custom_operation_invalid_action_handler():
    """Tests the registration of custom operation with an invalid action. Uses:
    `examples/trainer-controller-configs/loss_custom_operation_invalid_action.yaml`
    """
    test_data = _setup_data()
    with pytest.raises(KeyError) as exception_handler:
        tc_callback = tc.TrainerControllerCallback(
            td.TRAINER_CONFIG_TEST_CUSTOM_OPERATION_INVALID_ACTION_YAML
        )
        tc_callback.register_operation_handlers([CustomOperationInvalidAction])
        control = TrainerControl(should_training_stop=False)
        # Trigger on_init_end to perform registration of handlers to events
        tc_callback.on_init_end(
            args=test_data.args, state=test_data.states[2], control=control
        )
        # Trigger rule and test the condition
        tc_callback.on_log(
            args=test_data.args, state=test_data.states[2], control=control
        )
    assert str(exception_handler.value).strip("'") == (
        "Invalid operation custom_operation.should_ for control"
        + " loss_controller_custom_operation_invalid_action"
    )


def test_invalid_type_rule():
    """Tests the invalid type rule using configuration
    `examples/trainer-controller-configs/loss_with_invalid_type_rule.yaml`
    """
    test_data = _setup_data()
    with pytest.raises(TypeError) as exception_handler:
        tc_callback = tc.TrainerControllerCallback(
            td.TRAINER_CONFIG_TEST_INVALID_TYPE_RULE_YAML
        )
        control = TrainerControl(should_training_stop=False)
        # Trigger on_init_end to perform registration of handlers to events
        tc_callback.on_init_end(
            args=test_data.args, state=test_data.states[2], control=control
        )
        # Trigger rule and test the condition
        tc_callback.on_log(
            args=test_data.args, state=test_data.states[2], control=control
        )
    assert str(exception_handler.value) == "Rule failed due to incorrect type usage"


def test_malicious_os_rule():
    """Tests the malicious rule using configuration
    `examples/trainer-controller-configs/loss_with_malicious_os_rule.yaml`
    """
    test_data = _setup_data()
    with pytest.raises(ValueError) as exception_handler:
        tc_callback = tc.TrainerControllerCallback(
            td.TRAINER_CONFIG_TEST_MALICIOUS_OS_RULE_YAML
        )
        control = TrainerControl(should_training_stop=False)
        # Trigger on_init_end to perform registration of handlers to events
        tc_callback.on_init_end(
            args=test_data.args, state=test_data.states[2], control=control
        )
        # Trigger rule and test the condition
        tc_callback.on_log(
            args=test_data.args, state=test_data.states[2], control=control
        )
    assert (
        str(exception_handler.value)
        == "Rule for control loss_controller_wrong_os_rule is invalid"
    )


def test_malicious_input_rule():
    """Tests the malicious rule using configuration
    `examples/trainer-controller-configs/loss_with_malicious_input_rule.yaml`
    """
    test_data = _setup_data()
    tc_callback = tc.TrainerControllerCallback(
        td.TRAINER_CONFIG_TEST_MALICIOUS_INPUT_RULE_YAML
    )
    control = TrainerControl(should_training_stop=False)
    with pytest.raises(FunctionNotDefined) as exception_handler:
        # Trigger on_init_end to perform registration of handlers to events
        tc_callback.on_init_end(
            args=test_data.args, state=test_data.states[2], control=control
        )
        # Trigger rule and test the condition
        tc_callback.on_log(
            args=test_data.args, state=test_data.states[2], control=control
        )
    assert (
        str(exception_handler.value)
        == "Function 'input' not defined, for expression 'input('Please enter your password:')'."
    )


def test_invalid_trigger():
    """Tests the invalid trigger scenario in the controller. Uses:
    `examples/trainer-controller-configs/loss_invalid_trigger.yaml`
    """
    test_data = _setup_data()
    with pytest.raises(KeyError) as exception_handler:
        tc_callback = tc.TrainerControllerCallback(
            td.TRAINER_CONFIG_TEST_INVALID_TRIGGER_YAML
        )
        control = TrainerControl(should_training_stop=False)
        # Trigger on_init_end to perform registration of handlers to events
        tc_callback.on_init_end(
            args=test_data.args, state=test_data.states[2], control=control
        )
        # Trigger rule and test the condition
        tc_callback.on_log(
            args=test_data.args, state=test_data.states[2], control=control
        )
    assert str(exception_handler.value).strip("'") == (
        "Controller loss_controller_invalid_trigger has"
        + " an invalid event (log_it_all_incorrect_trigger_name)"
    )


def test_invalid_operation():
    """Tests the invalid operation scenario in the controller. Uses:
    `examples/trainer-controller-configs/loss_invalid_operation.yaml`
    """
    test_data = _setup_data()
    with pytest.raises(KeyError) as exception_handler:
        tc_callback = tc.TrainerControllerCallback(
            td.TRAINER_CONFIG_TEST_INVALID_OPERATION_YAML
        )
        control = TrainerControl(should_training_stop=False)
        # Trigger on_init_end to perform registration of handlers to events
        tc_callback.on_init_end(
            args=test_data.args, state=test_data.states[2], control=control
        )
        # Trigger rule and test the condition
        tc_callback.on_log(
            args=test_data.args, state=test_data.states[2], control=control
        )
    assert str(exception_handler.value).strip("'") == (
        "Invalid operation missingop.should_training_stop"
        + " for control loss_controller_invalid_operation"
    )


def test_invalid_operation_action():
    """Tests the invalid operation action scenario in the controller. Uses:
    `examples/trainer-controller-configs/loss_invalid_operation_action.yaml`
    """
    test_data = _setup_data()
    with pytest.raises(KeyError) as exception_handler:
        tc_callback = tc.TrainerControllerCallback(
            td.TRAINER_CONFIG_TEST_INVALID_OPERATION_ACTION_YAML
        )
        control = TrainerControl(should_training_stop=False)
        # Trigger on_init_end to perform registration of handlers to events
        tc_callback.on_init_end(
            args=test_data.args, state=test_data.states[2], control=control
        )
        # Trigger rule and test the condition
        tc_callback.on_log(
            args=test_data.args, state=test_data.states[2], control=control
        )
    assert str(exception_handler.value).strip("'") == (
        "Invalid operation hfcontrols.missingaction"
        + " for control loss_controller_invalid_operation_action"
    )


def test_invalid_metric():
    """Tests the invalid metric scenario in the controller. Uses:
    `examples/trainer-controller-configs/loss_invalid_metric.yaml`
    """
    test_data = _setup_data()
    with pytest.raises(KeyError) as exception_handler:
        tc_callback = tc.TrainerControllerCallback(
            td.TRAINER_CONFIG_TEST_INVALID_METRIC_YAML
        )
        control = TrainerControl(should_training_stop=False)
        # Trigger on_init_end to perform registration of handlers to events
        tc_callback.on_init_end(
            args=test_data.args, state=test_data.states[2], control=control
        )
        # Trigger rule and test the condition
        tc_callback.on_log(
            args=test_data.args, state=test_data.states[2], control=control
        )
    assert (
        str(exception_handler.value).strip("'")
        == "Undefined metric handler MissingMetricClass"
    )


def test_unavailable_metric():
    """Tests the invalid metric scenario in the controller. Uses:
    `examples/trainer-controller-configs/loss_invalid_metric.yaml`
    """
    test_data = _setup_data()
    tc_callback = tc.TrainerControllerCallback(
        td.TRAINER_CONFIG_TEST_UNAVAILABLE_METRIC_YAML
    )
    control = TrainerControl(should_training_stop=False)
    # Trigger on_init_end to perform registration of handlers to events
    tc_callback.on_init_end(
        args=test_data.args, state=test_data.states[2], control=control
    )
    # Trigger rule and test the condition
    tc_callback.on_step_end(
        args=test_data.args, state=test_data.states[2], control=control
    )
