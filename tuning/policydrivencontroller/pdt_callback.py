from transformers import TrainerCallback
from transformers.utils import logging
import numpy as np
from transformers import IntervalStrategy
from typing import Optional, Union
import json
import yaml
import os
import copy
from .controllermetrics import metrics as contmetrics

logger = logging.get_logger(__name__)

class PolicyDrivenTrainerControl(TrainerCallback):
    """Implements the policy driven trainer loop control based on policy definition file and metrics"""
    
    def __init__(self, train_control_args, training_args):
        self.__controllers = {}
        if os.path.exists(train_control_args.traning_control_definition_file):
            with open(train_control_args.traning_control_definition_file, "r") as f:
                self.training_control_def = yaml.safe_load(f)
                for controller in self.training_control_def['controllers']:
                    name = controller['name']
                    controller_metric_objs = []
                    for cm in controller['controller-metrics']:
                        obj = None
                        try:
                            class_type = getattr(contmetrics, cm['handler-class'])
                            if 'arguments' in cm:
                                obj = class_type(cm['name'], cm['arguments'])
                            else:
                                obj = class_type(cm['name'])
                        except Exception as e:
                            logger.fatal(e)
                        assert (obj.validate(training_args)), 'Controller metric class [%s] cannot be computed because the training args do not support it' % (cm['handler-class'])
                        controller_metric_objs.append(obj)
                    self.__controllers[name] = controller_metric_objs
        else:
            logger.warn("Controller configuration [%s] does NOT exist" % train_control_args.traning_control_definition_file)

    def __apply_control(self, cb, control):
        if 'should_training_stop' in cb['control-operation']:
            control.should_training_stop = cb['control-operation']['should_training_stop']
        elif 'should_epoch_stop' in cb['control-operation']:
            control.should_epoch_stop = cb['control-operation']['should_epoch_stop']
        elif 'should_save' in cb['control-operation']:
            control.should_save = cb['control-operation']['should_save']
        elif 'should_evaluate' in cb['control-operation']:
            control.should_evaluate = cb['control-operation']['should_evaluate']
        elif 'should_log' in cb['control-operation']:
            control.should_log = cb['control-operation']['should_log']

    def __loop_through_controllers(self, state, control, args, trigger_filter, metrics=None):
        controllers = self.training_control_def['controllers']
        num_controllers = len(controllers)
        for i in range(num_controllers):
            controller = controllers[i]
            name = controller['name']
            # logger.warn('*********CONTROLLER-BEGIN: %s************' % (name))
            controller_metrics_objs = self.__controllers[name]
            trigger_set = set(controller['triggers'])
            if trigger_filter not in trigger_set:
                continue
            metric_result = {}
            for i in range(len(controller['controller-metrics'])):
                cm_obj = controller_metrics_objs[i]
                cm_res = cm_obj.compute(state, args, metrics)
                if cm_res == None:
                    continue
                metric_result.update(cm_res)
            for rule in controller['rules']:
                try:
                    mr_copy = copy.deepcopy(metric_result)
                    rule_outcome = eval(rule, metric_result)
                    if rule_outcome:
                        logger.warn('[%s] metrics so far: %s' % (name, str(mr_copy)))
                        logger.warn('[%s] rule[%s] TRIGGERED' % (name, str(rule)))
                        self.__apply_control(controller, control)
                except Exception as e:
                    pass
            # logger.warn('******CONTROLLER-END: %s*************' % (name))

    def on_step_end(self, args, state, control, **kwargs):
        self.__loop_through_controllers(state, control, args, 'on_step_end')

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.__loop_through_controllers(state, control, args, 'on_epoch_begin')

    def on_epoch_end(self, args, state, control, **kwargs):
        self.__loop_through_controllers(state, control, args, 'on_epoch_end')

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        self.__loop_through_controllers(state, control, args, 'on_prediction_step')

    def on_predict(self, args, state, control, **kwargs):
        self.__loop_through_controllers(state, control, args, 'on_predict')

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.__loop_through_controllers(state, control, args, 'on_log')

    def on_train_end(self, args, state, control, **kwargs):
        self.__loop_through_controllers(state, control, args, 'on_train_end')

    def on_train_begin(self, args, state, control, **kwargs):
        self.__loop_through_controllers(state, control, args, 'on_train_end')

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        self.__loop_through_controllers(state, control, args, 'on_evaluate', metrics)