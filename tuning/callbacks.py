# OM NAMO GANAPATHAYEN NAMAHA
from transformers import TrainerCallback
from transformers.utils import logging
import numpy as np
from transformers import IntervalStrategy
from typing import Optional, Union
import json
import yaml
import os
from json_logic import jsonLogic
import tuning.tcprocessors as tcprop

logger = logging.get_logger(__name__)

class PolicyDrivenTrainerControl(TrainerCallback):
    """
    A [`TrainerCallback`] that handles early stopping.

    Args:
        early_stopping_patience (`int`):
            Use with `metric_for_best_model` to stop training when the specified metric worsens for
            `early_stopping_patience` evaluation calls.
        early_stopping_threshold(`float`, *optional*):
            Use with TrainingArguments `metric_for_best_model` and `early_stopping_patience` to denote how much the
            specified metric must improve to satisfy early stopping conditions. `

    This callback depends on [`TrainingArguments`] argument *load_best_model_at_end* functionality to set best_metric
    in [`TrainerState`]. Note that if the [`TrainingArguments`] argument *save_steps* differs from *eval_steps*, the
    early stopping will not occur until the next save step.
    """

    def __init__(self, train_control_args):
        # logger.warn('ENTERRING PolicyDrivenTrainerControl.....')
        self.early_stopping_patience = train_control_args.early_stopping_patience
        self.early_stopping_threshold = train_control_args.early_stopping_threshold
        # early_stopping_patience_counter denotes the number of times validation metrics failed to improve.
        self.early_stopping_patience_counter = 0
        self.__tcProcessors = []
        if os.path.exists(train_control_args.traning_control_definition_file):
            # logger.warn("TCCONFIG [%s] EXISTS!!!!" % train_control_args.traning_control_definition_file)
            with open(train_control_args.traning_control_definition_file, "r") as f:
                self.training_control_def = yaml.safe_load(f)
                # logger.warn('CCCC #1: %s ' % (str(self.training_control_def['spec']['controlDefinition'])))
                for c in self.training_control_def['spec']['controlDefinition']:
                    try:
                        # logger.warn('CCCC #2: %s ==== %s' % (repr(c['controlBlock']['class']['name']), repr(c['controlBlock']['class']['config'])))
                        classType = getattr(tcprop, c['controlBlock']['class']['name'])
                        if 'config' in c['controlBlock']['class']:
                            cfg = c['controlBlock']['class']['config']
                            self.__tcProcessors.append(classType(cfg))
                        else:
                            self.__tcProcessors.append(classType())
                    except Exception as e:
                        logger.fatal(e)
        else:
            logger.warn("TCCONFIG [%s] does NOT exist" % train_control_args.traning_control_definition_file)

    def apply_control(self, cb, control):
        cbInfo = cb['controlBlock']
        if 'should_training_stop' in cbInfo['control-operation']:
            control.should_training_stop = cbInfo['control-operation']['should_training_stop']
        elif 'should_epoch_stop' in cbInfo['control-operation']:
            control.should_epoch_stop = cbInfo['control-operation']['should_epoch_stop']
        elif 'should_save' in cbInfo['control-operation']:
            control.should_save = cbInfo['control-operation']['should_save']
        elif 'should_evaluate' in cbInfo['control-operation']:
            control.should_evaluate = cbInfo['control-operation']['should_evaluate']
        elif 'should_log' in cbInfo['control-operation']:
            control.should_log = cbInfo['control-operation']['should_log']

    def loop_through_control_blocks(self, state, control, filterByTrigger):
        controlBlocks = self.training_control_def['spec']['controlDefinition']
        numControlBlocks = len(controlBlocks)
        for i in range(numControlBlocks):
            tcp = self.__tcProcessors[i]
            cb = controlBlocks[i]
            if cb['controlBlock']['trigger'] != filterByTrigger:
                continue
            tcp.compute(state)
            data = tcp.get_result()
            if data == None:
                continue
            logger.warn('RESULT SO FAR: %s' % (repr(data)))
            rules = json.loads(cb['controlBlock']['rules'])
            trigger = jsonLogic(rules, data)
            if trigger:
                logger.warn('RESULT SO FAR TRIGGERED TRAINING STOP: %s' % (repr(data)))
                self.apply_control(cb, control)

    def on_step_end(self, args, state, control, **kwargs):
        # logger.warn('ON_STEPEND: %s' % repr(args))
        self.loop_through_control_blocks(state, control, 'on_step_end')

    def on_epoch_begin(self, args, state, control, **kwargs):
        pass

    def on_epoch_end(self, args, state, control, **kwargs):
        # logger.warn('STATE: %s' % (repr(state)))
        self.loop_through_control_blocks(state, control, 'on_epoch_end')

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        self.loop_through_control_blocks(state, control, 'on_prediction_step')

    def on_predict(self, args, state, control, **kwargs):
        self.loop_through_control_blocks(state, control, 'on_predict')

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.loop_through_control_blocks(state, control, 'on_log')

    def check_metric_value(self, args, state, control, metric_value):
        # best_metric is set by code for load_best_model
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or (
            operator(metric_value, state.best_metric)
            and abs(metric_value - state.best_metric) > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1

    def on_train_end(self, args, state, control, **kwargs):
        self.loop_through_control_blocks(state, control, 'on_train_end')

    def on_train_begin(self, args, state, control, **kwargs):
        assert args.load_best_model_at_end, "EarlyStoppingCallback requires load_best_model_at_end = True"
        assert (
            args.metric_for_best_model is not None
        ), "EarlyStoppingCallback requires metric_for_best_model is defined"
        assert (
            args.evaluation_strategy != IntervalStrategy.NO
        ), "EarlyStoppingCallback requires IntervalStrategy of steps or epoch"

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            logger.warning(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping"
                " is disabled"
            )
            return

        self.check_metric_value(args, state, control, metric_value)
        # if self.early_stopping_patience_counter >= self.early_stopping_patience:
        #     control.should_training_stop = True