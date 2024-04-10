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
from importlib import resources as impresources
from typing import List, Union
import inspect
import os
import re

# Third Party
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.utils import logging
import yaml

# Local
from tuning.trainercontroller import controllermetrics, operations
from tuning.trainercontroller.control import Control, OperationAction
from tuning.trainercontroller.controllermetrics import (
    handlers as default_metric_handlers,
)
from tuning.trainercontroller.controllermetrics.metricshandler import MetricHandler
from tuning.trainercontroller.operations import Operation
from tuning.trainercontroller.operations import (
    operation_handlers as default_operation_handlers,
)

logger = logging.get_logger(__name__)

# Configuration keys
CONTROLLER_METRICS_KEY = "controller-metrics"
OPERATIONS_KEY = "operations"
CONTROLLERS_KEY = "controllers"

CONTROLLER_NAME_KEY = "name"
CONTROLLER_TRIGGERS_KEY = "triggers"
CONTROLLER_RULE_KEY = "rule"
CONTROLLER_OPERATIONS_KEY = "operations"


# pylint: disable=too-many-instance-attributes
class TrainerControllerCallback(TrainerCallback):
    """Implements the trainer loop control based
    on trainer controller configuration file and metrics"""

    def __init__(self, trainer_controller_config: Union[dict, str]):
        """Initializes the callback for trainer controller.

        Args:
            trainer_controller_config: dict. Trainer controller configuration
        """
        # Checks if the trainer control config is of string type, in which case, it \
        # is a file path for the configuration yaml. On the other hand, if it is a \
        # dictionary, then it the yaml directly passed as such.
        if isinstance(trainer_controller_config, str):
            if os.path.exists(trainer_controller_config):
                with open(trainer_controller_config, "r", encoding="utf-8") as f:
                    self.trainer_controller_config = yaml.safe_load(f)
            else:
                raise FileNotFoundError(
                    f"Trainer controller configuration \
                                        [{trainer_controller_config}] does NOT exist"
                )
        else:
            self.trainer_controller_config = trainer_controller_config

        # Initialize the list of metrics from default `metrics.yaml` in the \
        # controllermetric package. In addition, any metrics mentioned in \
        # the trainer controller config are added to this list.
        default_metrics_config_yaml = (
            impresources.files(controllermetrics) / "metrics.yaml"
        )
        with default_metrics_config_yaml.open("r") as f:
            default_metrics_config = yaml.safe_load(f)
        if (
            default_metrics_config is not None
            and CONTROLLER_METRICS_KEY in default_metrics_config
            and len(default_metrics_config[CONTROLLER_METRICS_KEY]) > 0
        ):
            for metric_name in default_metrics_config[CONTROLLER_METRICS_KEY].keys():
                if (
                    metric_name
                    not in self.trainer_controller_config[CONTROLLER_METRICS_KEY]
                ):
                    self.trainer_controller_config[CONTROLLER_METRICS_KEY][
                        metric_name
                    ] = default_metrics_config[CONTROLLER_METRICS_KEY][metric_name]

        # Initialize the list of operations from default `operations.yaml` \
        # in the operations package. In addition, any operations mentioned \
        # in the trainer controller config are added to this list.
        default_operations_config_yaml = (
            impresources.files(operations) / "operations.yaml"
        )
        with default_operations_config_yaml.open("r") as f:
            default_operations_config = yaml.safe_load(f)
        if (
            default_operations_config is not None
            and OPERATIONS_KEY in default_operations_config
            and len(default_operations_config[OPERATIONS_KEY]) > 0
        ):
            for operation_name in default_operations_config[OPERATIONS_KEY].keys():
                if OPERATIONS_KEY not in self.trainer_controller_config:
                    self.trainer_controller_config[OPERATIONS_KEY] = {}
                if operation_name not in self.trainer_controller_config[OPERATIONS_KEY]:
                    self.trainer_controller_config[OPERATIONS_KEY][
                        operation_name
                    ] = default_operations_config[OPERATIONS_KEY][operation_name]

        # Load list of valid events for the trainercontroller callback
        # These events are assumed to start with "on_" prefix (on_epoch_end(), on_step_end() etc)
        self.valid_events = set()
        for callback_method_name, _ in inspect.getmembers(
            self, predicate=inspect.ismethod
        ):
            if re.search(r"^on_", callback_method_name) is not None:
                self.valid_events.add(callback_method_name)
        logger.debug("List of valid events %s", repr(self.valid_events))

        # Handlers to trigger on each metric
        self.metric_handlers = {}
        self.metrics_on_event = {}
        self.register_metric_handlers(default_metric_handlers)

        # Supported operations
        self.operation_handlers = {}
        self.operation_actions = {}
        self.register_operation_handlers(default_operation_handlers)

        # controls
        self.control_actions_on_event = {}

        # List of fields produced by the metrics
        self.metrics = {}

    def register_metric_handlers(self, handlers: List[MetricHandler]):
        """Registers the metric handlers

        Args:
            handlers: List[MetricHandler]. List of handlers.
        """
        for handler in handlers:
            self.metric_handlers[handler.__name__] = handler

    def register_operation_handlers(self, operation_handlers: List[Operation]):
        """Registers the operation handlers

        Args:
            operation_handlers: List[Operation]. List of operation handlers.
        """
        for operation_handler in operation_handlers:
            self.operation_handlers[operation_handler.__name__] = operation_handler

    def _compute_metrics(self, event_name: str, **kwargs):
        """Invokes the compute() for all the metrics registered for a given event.

        Args:
            event_name: str. Event name.
        """
        if event_name in self.metrics_on_event:
            for m in self.metrics_on_event[event_name]:
                self.metrics[m.get_name()] = m.compute(event_name=event_name, **kwargs)

    def _take_control_actions(self, event_name: str, **kwargs):
        """Invokes the act() method for all the operations registered for a given event. \
            Note here that the eval() is invoked with `__builtins__` set to None. \
            This is a precaution to restric the scope of eval(), to only the \
            fields produced by the metrics.

        Args:
            event_name: str. Event name.
            kwargs: List of arguments (key, value)-pairs.
        """
        if event_name in self.control_actions_on_event:
            for control_action in self.control_actions_on_event[event_name]:
                rule_succeeded = False
                try:
                    # pylint: disable=eval-used
                    rule_succeeded = eval(
                        control_action.rule, {"__builtins__": None}, self.metrics
                    )
                except TypeError as et:
                    raise TypeError("Rule failed due to incorrect type usage") from et
                except ValueError as ev:
                    raise ValueError(
                        "Rule failed due to use of disallowed packages"
                    ) from ev
                except NameError as en:
                    raise NameError(
                        "Rule failed due to use of disallowed variables"
                    ) from en
                if rule_succeeded:
                    for operation_action in control_action.operation_actions:
                        logger.info(
                            "Taking %s action in %s",
                            operation_action.action,
                            control_action.name,
                        )
                        operation_action.instance.act(
                            action=operation_action.action,
                            event_name=event_name,
                            **kwargs,
                        )

    def _actions_on_event(self, event_name: str, **kwargs):
        """Invokes all functions associated with an event.

        Args:
            event_name: str. Event name.
            kwargs: List of arguments (key, value)-pairs.
        """
        self._compute_metrics(event_name, **kwargs)
        self._take_control_actions(event_name, **kwargs)

    def _validate_rule(self, rule):
        """Validates the rule to check if there are any import attempts

        Returns:
            bool
        """
        return re.search(r"__", rule) is None

    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """This event gets the training arguments which is finally used by the trainer loop. \
            All metric and operation validation is performed here using these arguments. \
            Following this, validated metrics and operations instances are registered for use.

        Args:
            args: TrainingArguments. Training arguments for the trainer loop.
            state: TrainerState. Current trainer state.
            control: TrainerControl. Trainer control object.
            kwargs: List of arguments (key, value)-pairs.
        """
        # Training arguments, state and controls are folded into kwargs \
        # to be passed off to handlers
        kwargs["args"] = args
        kwargs["state"] = state
        kwargs["control"] = control

        # Check if there any metrics listed in the configuration
        if CONTROLLER_METRICS_KEY not in self.trainer_controller_config:
            logger.warning("Trainer controller config has no metrics.")

        # Metric handler validation and registration is performed here.
        for metric_name, metric_config in self.trainer_controller_config[
            CONTROLLER_METRICS_KEY
        ].items():
            # Get the metric class name from the config section.
            metric_handler_name = next(iter(metric_config.keys()))
            # Get the handler class using the metric class name.
            if metric_handler_name not in self.metric_handlers:
                raise KeyError(f"Undefined metric handler {metric_handler_name}")
            metric_handler = self.metric_handlers[metric_handler_name]
            # Get the metric handler class arguments specified in the config.
            metric_args = metric_config[metric_handler_name]
            if metric_args is None:
                metric_args = {}
            # Metric handler instance is created here.
            obj = metric_handler(name=metric_name, **metric_args, **kwargs)
            # Add metric instances to the events.
            for event_name in obj.get_events():
                if event_name in self.valid_events:
                    if event_name not in self.metrics_on_event:
                        self.metrics_on_event[event_name] = []
                    self.metrics_on_event[event_name].append(obj)
                else:
                    raise KeyError(
                        "Event name (%s) is not valid in metric %s"
                        % (event_name, metric_name)
                    )

        # Check if there any operations listed in the configuration
        if OPERATIONS_KEY in self.trainer_controller_config:
            # Operation handler validation and registration is performed here.
            for operation_name, operation_config in self.trainer_controller_config[
                OPERATIONS_KEY
            ].items():
                # Get the operation class name from the config section.
                operation_handler_name = next(iter(operation_config.keys()))
                # Get the handler class arguments using the operation class name.
                operation_args = operation_config[operation_handler_name]
                if operation_args is None:
                    operation_args = {}
                # Operation handler instance is created here.
                operation = self.operation_handlers[operation_handler_name](
                    **operation_args, **kwargs
                )
                # Add operation action instances.
                for action_name in operation.get_actions():
                    self.operation_actions[
                        operation_name + "." + action_name
                    ] = OperationAction(instance=operation, action=action_name)

        # Initialize controllers with respect to events.
        if CONTROLLERS_KEY in self.trainer_controller_config:
            for controller in self.trainer_controller_config[CONTROLLERS_KEY]:
                for event_name in controller[CONTROLLER_TRIGGERS_KEY]:
                    if event_name not in self.valid_events:
                        raise KeyError(
                            "Controller %s has an invalid event (%s)"
                            % (controller[CONTROLLER_NAME_KEY], event_name)
                        )
                    # Generates the byte-code for the rule from trainer configuration
                    if not self._validate_rule(controller[CONTROLLER_RULE_KEY]):
                        raise ValueError(
                            "Rule for control %s is invalid"
                            % (controller[CONTROLLER_NAME_KEY])
                        )
                    control = Control(
                        name=controller[CONTROLLER_NAME_KEY],
                        rule=compile(controller[CONTROLLER_RULE_KEY], "", "eval"),
                        operation_actions=[],
                    )
                    for control_operation_name in controller[CONTROLLER_OPERATIONS_KEY]:
                        if control_operation_name not in self.operation_actions:
                            raise KeyError(
                                "Invalid operation %s for control %s"
                                % (
                                    control_operation_name,
                                    controller[CONTROLLER_NAME_KEY],
                                )
                            )
                        control.operation_actions.append(
                            self.operation_actions[control_operation_name]
                        )
                    if event_name not in self.control_actions_on_event:
                        self.control_actions_on_event[event_name] = []
                    self.control_actions_on_event[event_name].append(control)
        self._actions_on_event(event_name="on_init_end", **kwargs)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Training arguments, state and controls are folded into kwargs to be passed off to
        # handlers
        kwargs["args"] = args
        kwargs["state"] = state
        kwargs["control"] = control
        self._actions_on_event(event_name="on_step_end", **kwargs)

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Training arguments, state and controls are folded into kwargs to be passed off to
        # handlers
        kwargs["args"] = args
        kwargs["state"] = state
        kwargs["control"] = control
        self._actions_on_event(event_name="on_epoch_begin", **kwargs)

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Training arguments, state and controls are folded into kwargs to be passed off to
        # handlers
        kwargs["args"] = args
        kwargs["state"] = state
        kwargs["control"] = control
        self._actions_on_event(event_name="on_epoch_end", **kwargs)

    def on_prediction_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Training arguments, state and controls are folded into kwargs to be passed off to
        # handlers
        kwargs["args"] = args
        kwargs["state"] = state
        kwargs["control"] = control
        self._actions_on_event(event_name="on_prediction_step", **kwargs)

    def on_predict(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics,
        **kwargs,
    ):
        # Training arguments, state and controls are folded into kwargs to be passed off to
        # handlers
        kwargs["args"] = args
        kwargs["state"] = state
        kwargs["control"] = control
        kwargs["metrics"] = metrics
        self._actions_on_event(event_name="on_predict", **kwargs)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Training arguments, state and controls are folded into kwargs to be passed off to
        # handlers
        kwargs["args"] = args
        kwargs["state"] = state
        kwargs["control"] = control
        self._actions_on_event(event_name="on_log", **kwargs)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Training arguments, state and controls are folded into kwargs to be passed off to
        # handlers
        kwargs["args"] = args
        kwargs["state"] = state
        kwargs["control"] = control
        self._actions_on_event(event_name="on_train_end", **kwargs)

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Training arguments, state and controls are folded into kwargs to be passed off to
        # handlers
        kwargs["args"] = args
        kwargs["state"] = state
        kwargs["control"] = control
        self._actions_on_event(event_name="on_train_begin", **kwargs)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Training arguments, state and controls are folded into kwargs to be passed off to
        # handlers
        kwargs["args"] = args
        kwargs["state"] = state
        kwargs["control"] = control
        self._actions_on_event(event_name="on_evaluate", **kwargs)
