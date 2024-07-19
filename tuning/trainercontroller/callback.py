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
from typing import Dict, List, Union
import inspect
import os
import re

# Third Party
from simpleeval import EvalWithCompoundTypes, FeatureNotAvailable, NameNotDefined
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.utils import logging
import yaml

# Local
from tuning.trainercontroller.control import Control, OperationAction, Rule
from tuning.trainercontroller.controllermetrics import (
    handlers as default_metric_handlers,
)
from tuning.trainercontroller.controllermetrics.metricshandler import MetricHandler
from tuning.trainercontroller.operations import Operation
from tuning.trainercontroller.operations import (
    operation_handlers as default_operation_handlers,
)
from tuning.trainercontroller.patience import PatienceControl
from tuning.utils.evaluator import MetricUnavailableError, RuleEvaluator

logger = logging.get_logger(__name__)

# Configuration keys
CONTROLLER_METRICS_KEY = "controller_metrics"
OPERATIONS_KEY = "operations"
CONTROLLERS_KEY = "controllers"
ARGS_KEY = "arguments"

CONTROLLER_NAME_KEY = "name"
CONTROLLER_CLASS_KEY = "class"
CONTROLLER_RULE_KEY = "rule"
CONTROLLER_CONFIG_KEY = "config"
CONTROLLER_PATIENCE_CONFIG_KEY = "patience"
CONTROLLER_TRIGGERS_KEY = "triggers"
CONTROLLER_OPERATIONS_KEY = OPERATIONS_KEY

# Default operations / metrics to register
DEFAULT_OPERATIONS = {"operations": [{"name": "hfcontrols", "class": "HFControls"}]}
DEFAULT_METRICS = {}

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
                    self.trainer_controller_config: dict = yaml.safe_load(f)
                if not isinstance(self.trainer_controller_config, dict):
                    raise TypeError(
                        "expected the trainer controller config YAML file"
                        "to contain a dictionary. actual type: %s"
                        % (type(self.trainer_controller_config))
                    )
            else:
                raise FileNotFoundError(
                    f"Trainer controller configuration \
                                        [{trainer_controller_config}] does NOT exist"
                )
        else:
            self.trainer_controller_config = trainer_controller_config

        if CONTROLLER_METRICS_KEY not in self.trainer_controller_config:
            self.trainer_controller_config[CONTROLLER_METRICS_KEY] = []

        if OPERATIONS_KEY not in self.trainer_controller_config:
            self.trainer_controller_config[OPERATIONS_KEY] = []

        if (
            DEFAULT_METRICS
            and CONTROLLER_METRICS_KEY in DEFAULT_METRICS
            and len(DEFAULT_METRICS[CONTROLLER_METRICS_KEY]) > 0
        ):
            self_controller_metrics = self.trainer_controller_config[
                CONTROLLER_METRICS_KEY
            ]
            default_controller_metrics: list[dict] = DEFAULT_METRICS[
                CONTROLLER_METRICS_KEY
            ]
            for metric_obj in default_controller_metrics:
                metric_name: str = metric_obj[CONTROLLER_NAME_KEY]
                found = False
                for self_controller_metric in self_controller_metrics:
                    if self_controller_metric[CONTROLLER_NAME_KEY] == metric_name:
                        found = True
                        break
                if not found:
                    self_controller_metrics.append(metric_obj)

        if (
            DEFAULT_OPERATIONS
            and OPERATIONS_KEY in DEFAULT_OPERATIONS
            and len(DEFAULT_OPERATIONS[OPERATIONS_KEY]) > 0
        ):
            self_controller_operations = self.trainer_controller_config[OPERATIONS_KEY]
            default_controller_operations: list[dict] = DEFAULT_OPERATIONS[
                OPERATIONS_KEY
            ]
            for op_obj in default_controller_operations:
                op_name: str = op_obj[CONTROLLER_NAME_KEY]
                found = False
                for self_controller_operation in self_controller_operations:
                    if self_controller_operation[CONTROLLER_NAME_KEY] == op_name:
                        found = True
                        break
                if not found:
                    self_controller_operations.append(op_obj)

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
        self.metric_handlers: dict[str, type[MetricHandler]] = {}
        self.metrics_on_event: dict[str, list[MetricHandler]] = {}
        self.register_metric_handlers(default_metric_handlers)

        # Supported operations
        self.operation_handlers: dict[str, type[Operation]] = {}
        self.operation_actions = {}
        self.register_operation_handlers(default_operation_handlers)

        # controls
        self.control_actions_on_event: Dict[str, list[Control]] = {}

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
        """Invokes the act() method for all the operations registered for a given event.

        Args:
            event_name: str. Event name.
            kwargs: List of arguments (key, value)-pairs.
        """
        if event_name in self.control_actions_on_event:
            evaluator = RuleEvaluator(metrics=self.metrics)
            for control_action in self.control_actions_on_event[event_name]:
                rule_succeeded = False
                try:
                    rule_succeeded = evaluator.eval(
                        expr=control_action.rule.rule,
                        previously_parsed=control_action.rule.rule_ast,
                    )
                    if not isinstance(rule_succeeded, bool):
                        raise TypeError(
                            "expected the rule to evaluate to a boolean. actual type: %s"
                            % (type(rule_succeeded))
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
                except NameNotDefined as en1:
                    raise NameError(
                        "Rule failed because some of the variables are not defined"
                    ) from en1
                except FeatureNotAvailable as ef:
                    raise NotImplementedError(
                        "Rule failed because it uses some unsupported features"
                    ) from ef
                except MetricUnavailableError as em:
                    logger.warning("Ignoring the rule because %s", em)
                    continue
                if (
                    control_action.patience is not None
                    and control_action.patience.should_tolerate(
                        rule_outcome=rule_succeeded,
                        event_name=event_name,
                        control_name=control_action.name,
                    )
                ):
                    continue
                if rule_succeeded:
                    for operation_action in control_action.operation_actions:
                        logger.info(
                            "Taking [%s] action in controller [%s]",
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
        if (
            CONTROLLER_METRICS_KEY not in self.trainer_controller_config
            or len(self.trainer_controller_config[CONTROLLER_METRICS_KEY]) == 0
        ):
            logger.warning("Trainer controller config has no metrics.")

        # Metric handler validation and registration is performed here.
        for metric_config in self.trainer_controller_config[CONTROLLER_METRICS_KEY]:
            metric_name = metric_config[CONTROLLER_NAME_KEY]
            # Get the metric class name from the config section.
            metric_handler_name = metric_config[CONTROLLER_CLASS_KEY]
            # Get the handler class using the metric class name.
            if metric_handler_name not in self.metric_handlers:
                raise KeyError(f"Undefined metric handler {metric_handler_name}")
            metric_handler_class = self.metric_handlers[metric_handler_name]
            # Get the metric handler class arguments specified in the config.
            metric_args = metric_config[ARGS_KEY] if ARGS_KEY in metric_config else {}
            # Metric handler instance is created here.
            metric_handler = metric_handler_class(
                name=metric_name, **metric_args, **kwargs
            )
            # Initialize the metric with a None value so that
            # the evaluator knows that the metric is unavailable.
            self.metrics[metric_handler.get_name()] = None
            # Add metric instances to the events.
            for event_name in metric_handler.get_events():
                if event_name in self.valid_events:
                    if event_name not in self.metrics_on_event:
                        self.metrics_on_event[event_name] = []
                    self.metrics_on_event[event_name].append(metric_handler)
                else:
                    raise KeyError(
                        "Event name (%s) is not valid in metric %s"
                        % (event_name, metric_name)
                    )

        # Check if there any operations listed in the configuration
        if (
            OPERATIONS_KEY in self.trainer_controller_config
            and len(self.trainer_controller_config[OPERATIONS_KEY]) > 0
        ):
            # Operation handler validation and registration is performed here.
            for operation_config in self.trainer_controller_config[OPERATIONS_KEY]:
                operation_name = operation_config[CONTROLLER_NAME_KEY]
                # Get the operation class name from the config section.
                operation_handler_name = operation_config[CONTROLLER_CLASS_KEY]
                # Get the handler class arguments using the operation class name.
                operation_args = (
                    operation_config[ARGS_KEY] if ARGS_KEY in operation_config else {}
                )
                # Operation handler instance is created here.
                operation_handler_class = self.operation_handlers[
                    operation_handler_name
                ]
                operation_handler = operation_handler_class(
                    name=operation_name, **operation_args, **kwargs
                )
                # Add operation action instances.
                for action_name in operation_handler.get_actions():
                    op_key = operation_name + "." + action_name
                    if op_key in self.operation_actions:
                        logger.warning(
                            "Trying to add the operation '%s' when it already exists, ignoring...",
                            op_key,
                        )
                        continue
                    self.operation_actions[op_key] = OperationAction(
                        instance=operation_handler, action=action_name
                    )

        # Initialize controllers with respect to events.
        if CONTROLLERS_KEY in self.trainer_controller_config:
            for controller in self.trainer_controller_config[CONTROLLERS_KEY]:
                controller_name: str = controller[CONTROLLER_NAME_KEY]
                controller_ops: list[str] = controller[CONTROLLER_OPERATIONS_KEY]
                controller_rule: str = controller[CONTROLLER_RULE_KEY]
                if not self._validate_rule(controller_rule):
                    raise ValueError(
                        "Rule for control %s is invalid" % (controller_name)
                    )
                for event_name in controller[CONTROLLER_TRIGGERS_KEY]:
                    if event_name not in self.valid_events:
                        raise KeyError(
                            "Controller %s has an invalid event (%s)"
                            % (controller_name, event_name)
                        )
                    # Generates the byte-code for the rule from the trainer configuration
                    control = Control(
                        name=controller[CONTROLLER_NAME_KEY],
                        rule=Rule(
                            rule=controller_rule,
                            rule_ast=EvalWithCompoundTypes.parse(expr=controller_rule),
                        ),
                        operation_actions=[],
                    )
                    if CONTROLLER_CONFIG_KEY in controller:
                        control.config = controller[CONTROLLER_CONFIG_KEY]
                    if CONTROLLER_PATIENCE_CONFIG_KEY in controller:
                        control.patience = PatienceControl(
                            **controller[CONTROLLER_PATIENCE_CONFIG_KEY]
                        )
                    for control_operation_name in controller_ops:
                        if control_operation_name not in self.operation_actions:
                            raise KeyError(
                                "Invalid operation %s for control %s"
                                % (
                                    control_operation_name,
                                    controller_name,
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
