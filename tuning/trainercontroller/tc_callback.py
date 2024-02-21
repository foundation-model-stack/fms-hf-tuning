from transformers import TrainerCallback
from transformers.utils import logging
import numpy as np
from transformers import IntervalStrategy, TrainerControl
from typing import Optional, Union
import json
import yaml
import os
import copy
from .controllermetrics import metrics as contmetrics

logger = logging.get_logger(__name__)

class TrainerControllerCallback(TrainerCallback):
    """Implements the policy driven trainer loop control based on policy definition file and metrics"""
    
    def __init__(self, trainer_controller_args, training_args):
        """Initializes the callback for policy-driven trainer control.

        Args:
            trainer_controller_args: File path for trainer control definition file
            training_args: TrainingArguments object
        """
        self.__controllers = {}
        if os.path.exists(trainer_controller_args.trainer_controller_config_file):
            with open(trainer_controller_args.trainer_controller_config_file, "r") as f:
                self.trainer_controller_config = yaml.safe_load(f)
                self.__validate_config(self.trainer_controller_config)
                for controller in self.trainer_controller_config['controllers']:
                    name = controller['name']
                    controller_metric_objs = []
                    for cm in controller['controller-metrics']:
                        obj = None
                        try:
                            # Get the controller-metric class type
                            class_type = getattr(contmetrics, cm['handler-class'])
                            # Initialize the controller-metric instance
                            if 'arguments' in cm:
                                obj = class_type(**cm['arguments'])
                            else:
                                obj = class_type()
                        except Exception as e:
                            logger.fatal(e)
                        assert (obj.validate(training_args)), 'Controller metric class [%s] cannot be computed because the training args do not support it' % (cm['handler-class'])
                        controller_metric_objs.append(obj)
                    self.__controllers[name] = controller_metric_objs
        else:
            raise ValueError("Trainer controller configuration [%s] does NOT exist" % trainer_controller_args.trainer_controller_config_file)

    def __validate_config(self, config):
        assert 'controllers' in config and len(config['controllers']) > 0, "List of controllers missing in config"
        for c in config['controllers']:
            assert 'name' in c, f"Controller should have a name"
            assert ('triggers' in c) and (len(c['triggers']) > 0), f"Triggers not specified for controller {c['name']}"
            assert ('rule' in c) and (len(c['rule']) > 0), f"Rule not specified for controller {c['name']}"
            try:
                compile(c['rule'], '<stdin>', 'eval')
            except Exception as e:
                raise SyntaxError(f"Rule for controller {c['name']} has this error {e}")
            assert ('controller-metrics' in c) and (len(c['controller-metrics']) > 0), f"List of controller metrics missing for controller {c['name']}"
            for cm in c['controller-metrics']:
                assert ('name' in cm), f"Controller metric should have a name"
                assert ('handler-class' in cm), f"Handler class not specified for controller metric {cm['name']}"
            assert ('control-operations' in c) and (len(c['control-operations']) > 0), f"Control operations not specified for controller {c['name']}"
            for k, v in c['control-operations'].items():
                try:
                    getattr(TrainerControl(), k)
                except Exception as e:
                    raise AttributeError(f"Control operation {k} is not invalid for controller {c['name']}")

                assert (isinstance(v, bool) or isinstance(v, int)),  f"Control operation {k} is assigned invalid value for controller {c['name']}"


    def __apply_control(self, cb, control):
        """Given a controller-block, applies the control operation to the training loop.

        Args:
            cb: Controller block dictionary
            control: TrainerControl object

        Returns:
            None.
        """
        for k, v in cb['control-operations'].items():
            setattr(control, k, v)

    def __loop_through_controllers(self, state, control, args, trigger_filter, metrics=None):
        """Loops through the controllers computing the controller-metrics and validating the rules. Once any rule gets validated, the corresponding control is applied to the trainer loop.

        Args:
            state: TrainingState object
            control: TrainerControl object
            args: TrainingArguments object
            trigger_filter: string which specifies the trigger event invoking this function
            metrics: [optional] specifies the evaluation metric

        Returns:
            None.
        """
        controllers = self.trainer_controller_config['controllers']
        num_controllers = len(controllers)
        for i in range(num_controllers):
            controller = controllers[i]
            name = controller['name']
            controller_metrics_objs = self.__controllers[name]
            trigger_set = set(controller['triggers'])
            if trigger_filter not in trigger_set:
                continue
            metric_result = {}
            for i in range(len(controller['controller-metrics'])):
                cm_data = controller['controller-metrics'][i]
                cm_obj = controller_metrics_objs[i]
                cm_res = cm_obj.compute(state, args, metrics)
                if cm_res == None:
                    continue
                metric_result[cm_data["name"]]= cm_res
            rule = controller['rule']
            try:
                if eval(rule, metric_result):
                    logger.warn('<%s> rule[%s] triggered' % (name, str(rule)))
                    self.__apply_control(controller, control)
            except Exception as e:
                pass

    def on_step_end(self, args, state, control, **kwargs):
        """Event triggered when step ends.

        Args:
            args: TrainingArguments object
            state: TrainerState object
            control: TrainerControl object
            kwargs: Miscellaneous arguments

        Returns:
            None.
        """
        self.__loop_through_controllers(state, control, args, 'on_step_end')

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Event triggered when epoch begins.

        Args:
            args: TrainingArguments object
            state: TrainerState object
            control: TrainerControl object
            kwargs: Miscellaneous arguments

        Returns:
            None.
        """
        self.__loop_through_controllers(state, control, args, 'on_epoch_begin')

    def on_epoch_end(self, args, state, control, **kwargs):
        """Event triggered when epoch ends.

        Args:
            args: TrainingArguments object
            state: TrainerState object
            control: TrainerControl object
            kwargs: Miscellaneous arguments

        Returns:
            None.
        """
        self.__loop_through_controllers(state, control, args, 'on_epoch_end')

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        """Event triggered when prediction is performed.

        Args:
            args: TrainingArguments object
            state: TrainerState object
            control: TrainerControl object
            eval_dataloader: Data loader object
            kwargs: Miscellaneous arguments

        Returns:
            None.
        """
        self.__loop_through_controllers(state, control, args, 'on_prediction_step')

    def on_predict(self, args, state, control, **kwargs):
        """Event triggered when predict event occurs.

        Args:
            args: TrainingArguments object
            state: TrainerState object
            control: TrainerControl object
            kwargs: Miscellaneous arguments

        Returns:
            None.
        """
        self.__loop_through_controllers(state, control, args, 'on_predict')

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Event triggered when logging event happens.

        Args:
            args: TrainingArguments object
            state: TrainerState object
            control: TrainerControl object
            logs: logs data
            kwargs: Miscellaneous arguments

        Returns:
            None.
        """
        self.__loop_through_controllers(state, control, args, 'on_log')

    def on_train_end(self, args, state, control, **kwargs):
        """Event triggered when training ends.

        Args:
            args: TrainingArguments object
            state: TrainerState object
            control: TrainerControl object
            kwargs: Miscellaneous arguments

        Returns:
            None.
        """
        self.__loop_through_controllers(state, control, args, 'on_train_end')

    def on_train_begin(self, args, state, control, **kwargs):
        """Event triggered when training begins.

        Args:
            args: TrainingArguments object
            state: TrainerState object
            control: TrainerControl object
            kwargs: Miscellaneous arguments

        Returns:
            None.
        """
        self.__loop_through_controllers(state, control, args, 'on_train_end')

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """Event triggered when evaluation step occurs.

        Args:
            args: TrainingArguments object
            state: TrainerState object
            control: TrainerControl object
            kwargs: Miscellaneous arguments

        Returns:
            None.
        """
        self.__loop_through_controllers(state, control, args, 'on_evaluate', metrics)