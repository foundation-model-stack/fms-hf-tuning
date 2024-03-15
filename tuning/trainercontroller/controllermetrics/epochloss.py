import statistics as stats
from collections import deque
import math
from transformers.utils import logging
from . import MetricHandlerWithCache
from transformers import TrainerState, TrainingArguments, IntervalStrategy
from typing import Any
from tuning.trainercontroller.validator import Validator

logger = logging.get_logger(__name__)

class EpochLoss(MetricHandlerWithCache):
    """Implements the controller metric which evaluates loss-per-epoch over a user-defined window"""

    def __init__(self, name: str, validator: Validator, window_size: int):
        """Initializes the handler

        Args:
            name: Name for the metric
            validator: Instance of Validator. Performs event validation
            window_size: Size of the window (number of epochs stored in cache)
        """
        self.__window_size = window_size
        super().__init__(name, 
                        validator, 
                        {'avg_loss': deque(), \
                        'std_loss': deque(), \
                        'final_loss': deque(), \
                        'final_eval_loss': deque()})
        self.register_events(['on_epoch_end'])
        
    def validate(self, training_args: TrainingArguments):
        """Validate the training arguments (e.g logging_steps) are compatible with the computation of this metric

        Args:
            training_args: Training arguments

        Returns:
            bool
        """
        return training_args.logging_strategy == IntervalStrategy.STEPS and \
            training_args.logging_steps == 1

    def compute_metrics_on_cache(self) -> Any:
        window_full = self.slide_the_window(self.__window_size)
        if not window_full:
            logger.warn('Window not full')
            return None
        return self.get_cache()

    def compute(self, training_state: TrainerState, event_name: str, training_args: TrainingArguments=None, metrics=None) -> Any:
        """Computes the controller-metric (epoch-loss over window) and exposes the values of the variables used by the rules.

        Args:
            training_state: TrainerState object
            event_name: Name of the event which is invoking the metric handler
            training_args: [optional] TrainingArguments object
            metrics: [optional] metrics data

        Returns:
            dict
        """
        previous_epoch = -1
        loss_array = None
        final_loss = None
        final_eval_loss = None
        added = False
        size_of_log_history = len(training_state.log_history)
        for i in range(size_of_log_history - 1, -1, -1):
            log = training_state.log_history[i]
            if 'loss' in log:
                loss = log['loss']
                if final_loss == None:
                    final_loss = loss
            if 'eval_loss' in log:
                eval_loss = log['eval_loss']
                if final_eval_loss == None:
                    final_eval_loss = eval_loss
            epoch = math.ceil(log['epoch'])
            if previous_epoch == -1:
                previous_epoch = epoch
                loss_array = []
            if epoch != previous_epoch and loss_array != None and len(loss_array) > 0:
                if len(loss_array) >= 2:
                    epoch_avg_loss = stats.mean(loss_array)
                    epoch_std_loss = stats.stdev(loss_array)
                else:
                    epoch_avg_loss = loss_array[0]
                    epoch_std_loss = 0
                self.add_to_cache(avg_loss=epoch_avg_loss, \
                                std_loss=epoch_std_loss, \
                                final_loss=final_loss, \
                                final_eval_loss=final_eval_loss)
                added = True
                break
            loss_array.append(loss)
        if len(loss_array) > 0 and added == False:
            if len(loss_array) >= 2:
                epoch_avg_loss = stats.mean(loss_array)
                epoch_std_loss = stats.stdev(loss_array)
            else:
                epoch_avg_loss = loss_array[0]
                epoch_std_loss = 0
            self.add_to_cache(avg_loss=epoch_avg_loss, std_loss=epoch_std_loss, final_loss=final_loss, final_eval_loss=final_eval_loss)
        return self.compute_metrics_on_cache()
