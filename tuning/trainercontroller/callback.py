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

import inspect, re, yaml, os
from typing import List
from importlib import resources as impresources

from transformers import TrainerCallback
from transformers.utils import logging
from transformers import IntervalStrategy, TrainerState, TrainerControl, TrainingArguments

from tuning.trainercontroller import controllermetrics
from tuning.trainercontroller.controllermetrics import handlers as default_metric_handlers
from tuning.trainercontroller.controllermetrics.metricshandler import MetricHandler
from tuning.trainercontroller import operations
from tuning.trainercontroller.operations import operation_handlers as default_operation_handlers
from tuning.trainercontroller.operations import Operation
from tuning.trainercontroller.control import Control, OperationAction

logger = logging.get_logger(__name__)

CONTROLLER_METRICS_KEY = "controller-metrics"
OPERATIONS_KEY = "operations"
CONTROLLERS_KEY = "controllers"

CONTROLLER_NAME_KEY = "name"
CONTROLLER_TRIGGERS_KEY = "triggers"
CONTROLLER_RULE_KEY = "rule"
CONTROLLER_OPERATIONS_KEY = "operations"

class TrainerControllerCallback(TrainerCallback):
    """Implements the policy driven trainer loop control based on policy definition file and metrics"""
    
    def __init__(self, trainer_controller_config: dict):
        """Initializes the callback for policy-driven trainer control.

        Args:
            trainer_controller_config: Trainer controller configuration
        """
        if isinstance(trainer_controller_config, str):
            if os.path.exists(trainer_controller_config):
                with open(trainer_controller_config, "r") as f:
                    self.trainer_controller_config = yaml.safe_load(f)
            else:
                raise FileNotFoundError(f"Trainer controller configuration [{trainer_controller_config}] does NOT exist")
        else:
            self.trainer_controller_config = trainer_controller_config

        default_metrics_config_yaml = (impresources.files(controllermetrics) / 'metrics.yaml')
        with default_metrics_config_yaml.open("r") as f:
            default_metrics_config = yaml.safe_load(f)
        if default_metrics_config is not None and CONTROLLER_METRICS_KEY in default_metrics_config and len(default_metrics_config[CONTROLLER_METRICS_KEY]) > 0:
            for metric_name in default_metrics_config[CONTROLLER_METRICS_KEY].keys():
                if metric_name not in self.trainer_controller_config[CONTROLLER_METRICS_KEY]:
                    self.trainer_controller_config[CONTROLLER_METRICS_KEY][metric_name] = default_metrics_config[CONTROLLER_METRICS_KEY][metric_name]

        default_operations_config_yaml = (impresources.files(operations) / 'operations.yaml')
        with default_operations_config_yaml.open("r") as f:
            default_operations_config = yaml.safe_load(f)
        if default_operations_config is not None and OPERATIONS_KEY in default_operations_config and len(default_operations_config[OPERATIONS_KEY]) > 0:
            for operation_name in default_operations_config[OPERATIONS_KEY].keys():
                if OPERATIONS_KEY not in self.trainer_controller_config:
                    self.trainer_controller_config[OPERATIONS_KEY] = {}
                if operation_name not in self.trainer_controller_config[OPERATIONS_KEY]:
                    self.trainer_controller_config[OPERATIONS_KEY][operation_name] = default_operations_config[OPERATIONS_KEY][operation_name]

        # Load list of valid events
        self.valid_events = set()
        for callback_method_name, _ in inspect.getmembers(self, predicate=inspect.ismethod):
            if re.search(r'^on_', callback_method_name) != None:
                self.valid_events.add(callback_method_name)

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

        self.metrics = {}

    def register_metric_handlers(self, handlers: List[MetricHandler]):
        for handler in handlers:
            self.metric_handlers[handler.__name__]=handler

    def register_operation_handlers(self, operation_handlers: List[Operation]):
        for operation_handler in operation_handlers:
            self.operation_handlers[operation_handler.__name__]=operation_handler

    def _compute_metrics(self, event_name: str, **kwargs):
        if event_name in self.metrics_on_event:
            for m in self.metrics_on_event[event_name]:
                self.metrics[m.get_name()] = m.compute(event_name=event_name, **kwargs)

    def _take_control_actions(self, event_name: str, **kwargs):
        if event_name in self.control_actions_on_event:
            for control_action in self.control_actions_on_event[event_name]:
                if eval(control_action.rule, {'__builtins__': None}, self.metrics):
                    for operation_action in control_action.operation_actions:
                        logger.info(f"Taking {operation_action.action} action in {control_action.name}")
                        operation_action.instance.act(action=operation_action.action, event_name=event_name, **kwargs)

    def _actions_on_event(self, event_name: str, **kwargs):
        self._compute_metrics(event_name, **kwargs)
        self._take_control_actions(event_name, **kwargs)

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of the initialization of the Trainer.
        """
        kwargs["args"] = args
        kwargs["state"] = state
        kwargs["control"] = control

        # Initializing the metric objects for the config
        if CONTROLLER_METRICS_KEY not in self.trainer_controller_config:
            logger.warn("Trainer controller config has no metrics.")

        for metric_name, metric_config in self.trainer_controller_config[CONTROLLER_METRICS_KEY].items():
            metric_handler_name = next(iter(metric_config.keys()))
            metric_handler = self.metric_handlers[metric_handler_name]
            metric_args = metric_config[metric_handler_name]
            if metric_args == None:
                metric_args = {}
            obj = metric_handler(name=metric_name, **metric_args, **kwargs)
            for event_name in obj.get_events():
                if event_name in self.valid_events:
                    if event_name not in self.metrics_on_event:
                        self.metrics_on_event[event_name]=[]  
                    self.metrics_on_event[event_name].append(obj)
                else:
                    raise KeyError(f"Event name ({event_name}) is not valid in metric {metric_name}")

        # Initializing the operation objects for the config
        if OPERATIONS_KEY in self.trainer_controller_config:
            for operation_name, operation_config in self.trainer_controller_config[OPERATIONS_KEY].items():
                operation_handler_name = next(iter(operation_config.keys()))
                operation_args = operation_config[operation_handler_name]
                if operation_args == None:
                    operation_args = {}
                operation = self.operation_handlers[operation_handler_name](**operation_args, **kwargs)
                for action_name in operation.get_actions():
                    self.operation_actions[operation_name+"."+action_name] = OperationAction(instance=operation, action=action_name)

        # Initialize controllers
        if CONTROLLERS_KEY in self.trainer_controller_config:
            for controller in self.trainer_controller_config[CONTROLLERS_KEY]:
                for event_name in controller[CONTROLLER_TRIGGERS_KEY]:
                    if event_name not in self.valid_events:
                        raise KeyError(f"Event name ({event_name}) is not valid in control {controller[CONTROLLER_NAME_KEY]}")
                    control = Control(name=controller[CONTROLLER_NAME_KEY], rule = compile(controller[CONTROLLER_RULE_KEY], '', 'eval'), operation_actions = [])
                    for control_operation_name in controller[CONTROLLER_OPERATIONS_KEY]:
                        control.operation_actions.append(self.operation_actions[control_operation_name])
                    if event_name not in self.control_actions_on_event:
                        self.control_actions_on_event[event_name] = []
                    self.control_actions_on_event[event_name].append(control)
                        
        self._actions_on_event(event_name='on_init_end', **kwargs)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        kwargs["args"] = args
        kwargs["state"] = state
        kwargs["control"] = control
        self._actions_on_event(event_name='on_step_end', **kwargs)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        kwargs["args"] = args
        kwargs["state"] = state
        kwargs["control"] = control
        self._actions_on_event(event_name='on_epoch_begin', **kwargs)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        kwargs["args"] = args
        kwargs["state"] = state
        kwargs["control"] = control
        self._actions_on_event(event_name='on_epoch_end', **kwargs)

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        kwargs["args"] = args
        kwargs["state"] = state
        kwargs["control"] = control
        self._actions_on_event(event_name='on_prediction_step', **kwargs)

    def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        kwargs["args"] = args
        kwargs["state"] = state
        kwargs["control"] = control
        self._actions_on_event(event_name='on_predict', **kwargs)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        kwargs["args"] = args
        kwargs["state"] = state
        kwargs["control"] = control
        self._actions_on_event(event_name='on_log', **kwargs)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        kwargs["args"] = args
        kwargs["state"] = state
        kwargs["control"] = control
        self._actions_on_event(event_name='on_train_end', **kwargs)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        kwargs["args"] = args
        kwargs["state"] = state
        kwargs["control"] = control
        self._actions_on_event(event_name='on_train_begin', **kwargs)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        kwargs["args"] = args
        kwargs["state"] = state
        kwargs["control"] = control
        self._actions_on_event(event_name='on_evaluate', **kwargs)