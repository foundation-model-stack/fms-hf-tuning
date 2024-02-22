import numpy as np
from transformers import TrainerState, IntervalStrategy
from transformers.utils import logging
from collections import deque
import math
import copy

logger = logging.get_logger(__name__)

class MetricHandler:
    """Base class for the controller-metrics"""
    def validate(self, training_args) -> bool:
        pass
    
    def compute(self, training_state, training_args=None, metrics=None) -> dict:
        pass   

class StepLoss(MetricHandler):
    """Implements the controller metric which evaluates loss-per-step"""
    
    def __init__(self, window_size):
        # Initialize the handler arguments
        self.__window_size = window_size

    def validate(self, training_args):
        """Validate the training arguments (e.g logging_steps) are compatible with the computation of this metric

        Args:
            training_args: Training arguments

        Returns:
            bool
        """
        return training_args.logging_strategy == 'steps' and \
            training_args.logging_steps == 1

    def compute(self, training_state, training_args=None, metrics=None):
        """Computes the controller-metric (step-loss over window) and exposes the values of the variables used by the rules.

        Args:
            training_state: TrainerState object
            training_args: TrainingArguments object
            metrics: [optional] metrics data

        Returns:
            dict
        """
        # Compute the metric using the training state
        loss_values = [l['loss'] for l in training_state.log_history if 'loss' in l]
        n = len(loss_values)
        if n <= self.__window_size:
            return None
        window = loss_values[n-self.__window_size:n]
        consistently_increasing = True
        for i in range(len(window)-1):
            if window[i] > window[i+1]:
                consistently_increasing = False
                break
        w = np.array(window)
        avg_loss = np.mean(w)
        std_loss = np.std(w, dtype=np.float64)
        exposed_data = {'consistently_increasing': int(consistently_increasing), \
                    'average_loss': avg_loss,\
                    'std_loss': std_loss,\
                    'window': window}
        return exposed_data

class EpochLoss(MetricHandler):
    """Implements the controller metric which evaluates loss-per-epoch over a user-defined window"""

    def __init__(self, window_size):
        # Initialize the handler arguments
        self.__window_size = window_size
        self.__cache = []
        
    def validate(self, training_args):
        """Validate the training arguments (e.g logging_steps) are compatible with the computation of this metric

        Args:
            training_args: Training arguments

        Returns:
            bool
        """
        return training_args.logging_strategy == 'steps' and \
            training_args.logging_steps == 1

    def __externalize_data(self):
        """Create the dictionary of exposed variables used by rules.

        Returns:
            dict
        """
        if len(self.__cache) < self.__window_size:
            return None
        exposed_data = {}
        for elem in reversed(self.__cache):
            for k, v in elem.items():
                l = []
                if k in exposed_data:
                    l = exposed_data[k]
                l.append(v)
                exposed_data[k] = l
        return exposed_data

    def compute(self, training_state, training_args=None, metrics=None):
        """Computes the controller-metric (epoch-loss over window) and exposes the values of the variables used by the rules.

        Args:
            training_state: TrainerState object
            training_args: TrainingArguments object
            metrics: [optional] metrics data

        Returns:
            dict
        """
        previous_epoch = -1
        loss_array = None
        logs_latest_first = list(reversed(training_state.log_history))
        final_loss = None
        added = False
        for i in range(len(logs_latest_first)):
            log = logs_latest_first[i]
            try:
                loss = log['loss']
                if i == 0:
                    final_loss = loss
                epoch = math.ceil(log['epoch'])
                if previous_epoch == -1:
                    previous_epoch = epoch
                    loss_array = []
                if epoch != previous_epoch and loss_array != None:
                    if len(loss_array) > 0:
                        l = np.array(loss_array)
                        epoch_avg_loss = np.mean(l)
                        epoch_std_loss = np.std(l, dtype=np.float64)
                        self.__cache.append({'epoch': epoch, 'avg_loss': epoch_avg_loss, 'std_loss': epoch_std_loss, 'final_loss': final_loss})
                        added = True
                        break
                loss_array.append(loss)
            except:
                # Ignoring log lines not containing relevant fields
                continue
        if len(loss_array) > 0 and added == False:
            l = np.array(loss_array)
            epoch_avg_loss = np.mean(l)
            epoch_std_loss = np.std(l, dtype=np.float64)
            self.__cache.append({'epoch': epoch, 'avg_loss': epoch_avg_loss, 'std_loss': epoch_std_loss, 'final_loss': final_loss})
        return self.__externalize_data()
        
            
class EvalMetricBasedControl(MetricHandler):
    """Implements the controller metric which computes and evaluates metrics conditions on evaluation"""

    def __init__(self, early_stopping_patience=1.0, early_stopping_threshold=0.0):
        # Initialize the handler arguments
        self.__early_stopping_patience = early_stopping_patience
        self.__early_stopping_threshold = early_stopping_threshold
        self.__early_stopping_patience_counter = 0

    def validate(self, training_args):
        """Validate the training arguments (e.g logging_steps) are compatible with the computation of this metric

        Args:
            training_args: Training arguments

        Returns:
            bool
        """
        return (training_args.load_best_model_at_end == True or \
        training_args.metric_for_best_model is not None or \
        training_args.evaluation_strategy != IntervalStrategy.NO or \
        training_args.save_strategy == IntervalStrategy.EPOCH)

    def __check_metric_value(self, args, state, metric_value):
        """Checks the best metric value and compares it with a threshold
        Args:
            training_state: TrainerState object
            args: TrainingArguments object
            metric_value: [optional] metrics data
        """
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or (
            operator(metric_value, state.best_metric)
            and abs(metric_value - state.best_metric) > self.__early_stopping_threshold
        ):
            self.__early_stopping_patience_counter = 0
        else:
            self.__early_stopping_patience_counter += 1

    def compute(self, training_state, training_args=None, metrics=None):
        """Computes the controller-metric (evaluation metric) and exposes the values of the variables used by the rules.

        Args:
            training_state: TrainerState object
            training_args: TrainingArguments object
            metrics: [optional] metrics data

        Returns:
            dict
        """
        metric_to_check = training_args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            logger.warning(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping"
                " is disabled"
            )
            return None

        self.__check_metric_value(training_args, training_state, metric_value)
        return {
                self.__name + '_' + 'early_stopping_patience_counter': self.__early_stopping_patience_counter, \
                self.__name + "_" + 'early_stopping_patience': self.__early_stopping_patience
            }
