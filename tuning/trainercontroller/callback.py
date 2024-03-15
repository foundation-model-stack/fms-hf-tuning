import os, yaml
from typing import Optional, Union

from transformers import TrainerCallback
from transformers.utils import logging
from transformers import IntervalStrategy, TrainerState, TrainerControl, TrainingArguments
from tuning.trainercontroller import controllermetrics
import tuning.config.configs as config
from tuning.trainercontroller.validator import Validator

logger = logging.get_logger(__name__)

class TrainerControllerCallback(TrainerCallback):
    """Implements the policy driven trainer loop control based on policy definition file and metrics"""
    # # TODO: Remove this
    # __control_op_prefixes = ['should_']
    
    # TODO: Change to use config contents instead of file
    def __init__(self, trainer_controller_config: dict):
        """Initializes the callback for policy-driven trainer control.

        Args:
            trainer_controller_config: Trainer controller configuration
        """
        self.trainer_controller_config = trainer_controller_config
        self._validator = Validator({TrainerControl.__name__: {'class': TrainerControl, 'filter': r'^should_'}, TrainerCallback.__name__: {'class': TrainerCallback, 'filter': r'^on_'}})
        self._rule_byte_code = {}
        # TODO: Change double underscores into single underscores
        self._validate_config()
        self._init_metrics()

    def _init_metrics(self):
        self._metrics_map = {}
        self._unique_metric_list = []
        for metric in self.trainer_controller_config['controller-metrics']:
            # TODO: Handle when the handler-class does not exist
            if metric['handler-class'] not in controllermetrics.handlers:
                raise KeyError(f"Trainer controller metric handler class {metric['handler-class']} not defined")
            else:
                class_type = controllermetrics.handlers[metric['handler-class']]
                if 'arguments' in cm:
                    obj = class_type(cm['name'], self._validator, **cm['arguments'])
                else:
                    obj = class_type(cm['name'], self._validator)
            event_list = obj.get_events()
            self._unique_metric_list.append(obj)
            for e in event_list:
                metric_list = []
                if e in self._metrics_map:
                    metric_list = self._metrics_map[e]
                metric_list.append(obj)
                self._metrics_map[e] = metric_list

    def _validate_config(self):
        # TODO: Create schema and use it for validation
        # TODO: Print a warn message if there are no controls
        config = self.trainer_controller_config
        if 'controller-metrics' not in config or len(config['controller-metrics']) == 0:
             logger.warn("List of controller-metrics missing in config")
        else:
            for cm in config['controller-metrics']:
                assert ('name' in cm), f"Controller metric should have a name"
                assert ('handler-class' in cm), f"Handler class not specified for controller metric {cm['name']}"
        if 'controllers' not in config or len(config['controllers']) == 0:
             logger.warn("List of controllers missing in config")
        else:
            for c in config['controllers']:
                assert 'name' in c, f"Controller should have a name"
                assert ('triggers' in c) and (len(c['triggers']) > 0), f"Triggers not specified for controller {c['name']}"
                for trigger in c['triggers']:
                    if not self._validator(TrainerCallback, trigger):
                        raise KeyError(f"Trigger {k} is invalid for controller {c['name']}")
                assert ('control-operations' in c) and (len(c['control-operations']) > 0), f"Control operations not specified for controller {c['name']}"
                for k, v in c['control-operations'].items():
                    if not self._validator(TrainerControl, k):
                        raise KeyError(f"Control operation {k} is invalid for controller {c['name']}")
                assert (isinstance(v, bool) or isinstance(v, int)),  f"Control operation {k} is assigned invalid value for controller {c['name']}"
                assert ('rule' in c) and (c['rule'] != None) and (len(c['rule']) > 0), f"Rule not specified for controller {c['name']}"
                try:
                    #TODO: Store and use the bytecode
                    self._rule_byte_code[c['name']] = compile(c['rule'], '', 'eval')
                except SyntaxError as e:
                    raise SyntaxError(f"Rule [{c['rule']}] for controller {c['name']} has this error {e}")

    def _loop_through_controllers(self, state: TrainerState, control: TrainerControl, args: TrainingArguments, trigger_filter: str, metrics=None):
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
        metric_result = {}
        if trigger_filter in self._metrics_map:
            metrics = self._metrics_map[trigger_filter]
            for m in metrics:
                cm_res = m._compute(state, trigger_filter, args, metrics)
                if cm_res == None:
                    continue
                metric_result.update(cm_res)
        controllers = self.trainer_controller_config['controllers']
        num_controllers = len(controllers)
        for i in range(num_controllers):
            controller = controllers[i]
            trigger_set = set(controller['triggers'])
            if trigger_filter not in trigger_set:
                continue
            rule = controller['rule']
            try:
                logger.warn(f'rule[{rule}]: Metric so far: {str(metric_result)}')
                if eval(self._rule_byte_code[controller['name']], metric_result):
                    logger.warn('rule[%s] triggered' % (str(rule)))
                    for k, v in controller['control-operations'].items():
                        setattr(control, k, v)
            except NameError as e:
                logger.warn(e)
        return control

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of the initialization of the Trainer.
        """
        if len(self._unique_metric_list) == 0:
            return
        for obj in self._unique_metric_list:
            assert (obj.validate(args)), 'Controller metric class [%s] cannot be computed because the training args do not support it' % (cm['handler-class'])

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Event triggered when step ends.

        Args:
            args: TrainingArguments object
            state: TrainerState object
            control: TrainerControl object
            kwargs: [optional] Miscellaneous arguments

        Returns:
           TrainerControl object.
        """
        self._loop_through_controllers(state, control, args, 'on_step_end')

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Event triggered when epoch begins.

        Args:
            args: TrainingArguments object
            state: TrainerState object
            control: TrainerControl object
            kwargs: [optional] Miscellaneous arguments

        Returns:
            TrainerControl object.
        """
        self._loop_through_controllers(state, control, args, 'on_epoch_begin')

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Event triggered when epoch ends.

        Args:
            args: TrainingArguments object
            state: TrainerState object
            control: TrainerControl object
            kwargs: [optional] Miscellaneous arguments

        Returns:
            TrainerControl object.
        """
        self._loop_through_controllers(state, control, args, 'on_epoch_end')

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, eval_dataloader=None, **kwargs):
        """Event triggered when prediction is performed.

        Args:
            args: TrainingArguments object
            state: TrainerState object
            control: TrainerControl object
            eval_dataloader: Data loader object
            kwargs: [optional] Miscellaneous arguments

        Returns:
            TrainerControl object.
        """
        self._loop_through_controllers(state, control, args, 'on_prediction_step')

    def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Event triggered when predict event occurs.

        Args:
            args: TrainingArguments object
            state: TrainerState object
            control: TrainerControl object
            kwargs: [optional] Miscellaneous arguments

        Returns:
            TrainerControl object.
        """
        self._loop_through_controllers(state, control, args, 'on_predict')

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Event triggered when logging event happens.

        Args:
            args: TrainingArguments object
            state: TrainerState object
            control: TrainerControl object
            logs: [optional] logs data
            kwargs: [optional] Miscellaneous arguments

        Returns:
            TrainerControl object.
        """
        self._loop_through_controllers(state, control, args, 'on_log')

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Event triggered when training ends.

        Args:
            args: TrainingArguments object
            state: TrainerState object
            control: TrainerControl object
            kwargs: [optional] Miscellaneous arguments

        Returns:
            TrainerControl object.
        """
        self._loop_through_controllers(state, control, args, 'on_train_end')

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Event triggered when training begins.

        Args:
            args: TrainingArguments object
            state: TrainerState object
            control: TrainerControl object
            kwargs: [optional] Miscellaneous arguments

        Returns:
            TrainerControl object.
        """
        self._loop_through_controllers(state, control, args, 'on_train_end')

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        """Event triggered when evaluation step occurs.

        Args:
            args: TrainingArguments object
            state: TrainerState object
            control: TrainerControl object
            kwargs: [optional] Miscellaneous arguments

        Returns:
            TrainerControl object.
        """
        self._loop_through_controllers(state, control, args, 'on_evaluate', metrics)