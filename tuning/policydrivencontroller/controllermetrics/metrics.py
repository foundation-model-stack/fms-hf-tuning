import numpy as np
from transformers import TrainerState, IntervalStrategy
from transformers.utils import logging
from collections import deque
import math
import copy

logger = logging.get_logger(__name__)

class MetricHandler:
    def validate(self, training_args):
        pass
    
    def compute(self, training_state, training_args=None, metrics=None):
        pass   

class WindowStepLoss(MetricHandler):
    __exposed_variable_prefix = 'wsl'

    def __init__(self, name, args):
        # Initialize the handler arguments
        self.__name = name
        self.__args = args

    def validate(self, training_args):
        # Validate the training arguments (e.g logging_steps) are
        # compatible with the computation of this metric
        return training_args.logging_strategy == 'steps' and \
            training_args.logging_steps == 1

    def compute(self, training_state, training_args=None, metrics=None):
        # Compute the metric using the training state
        loss_values = [l['loss'] for l in training_state.log_history if 'loss' in l]
        n = len(loss_values)
        if n == 0:
            logger.warn('(N=0) Number of LOSS values: %d, Given window size: %d' % (n, self.__args['window-size']))
            return None
        if n <= self.args['window-size']:
            logger.warn('(N<W) Number of LOSS values: %d, Given window size: %d' % (n, self.__args['window-size']))
            return None
        window = loss_values[n-self.__args['window-size']:n]
        consistently_increasing = True
        for i in range(len(window)-1):
            if window[i] > window[i+1]:
                consistently_increasing = False
                break
        w = np.array(window)
        avg_loss = np.mean(w)
        std_loss = np.std(w, dtype=np.float64)
        first_and_last_loss = window[0] < window[len(window)-1]
        return {self.__name + '_' + 'consistently_increasing': int(consistently_increasing), \
                self.__name + '_' + 'average_loss': avg_loss,\
                self.__name + '_' + 'std_loss': std_loss,\
                self.__name + '_' + 'first_and_last_loss': first_and_last_loss}

class EpochLoss(MetricHandler):

    def __init__(self, name, args=None):
        # Initialize the handler arguments
        self.__name = name
        self.__args = args
        self.__cache = deque()
        
    def validate(self, training_args):
        # Validate the training arguments (e.g logging_steps) are
        # compatible with the computation of this metric
        return training_args.logging_strategy == 'steps' and \
            training_args.logging_steps == 1

    def __externalize_data(self):
        if len(self.__cache) < self._args['window-size']:
            return None
        dq = copy.deepcopy(self.__cache)
        key_prefix = self.__name + "_"
        exposed_data = {}
        for i in range(len(dq)):
            elem = dq.pop()
            if i == 0:
                for k, v in elem.items():
                    key = key_prefix + k + '_epoch[n]'
                    exposed_data[key] =  v
            else:
                for k, v in elem.items():
                    key = key_prefix + k + '_epoch[n-' + str(i)+ ']'
                    exposed_data[key] =  v
        self.__cache = dq
        return exposed_data

    def compute(self, training_state, training_args=None, metrics=None):
        # Compute the metric using the training state
        previous_epoch = -1
        loss_array = None
        logs_latest_first = training_state.log_history.reverse()
        latest_log = None
        for i in range(len(logs_latest_first)):
            log = logs_latest_first[i]
            if 'epoch' not in log and 'loss' not in log:
                continue
            if i == 0:
                latest_log = log
            epoch = math.ceil(log['epoch'])
            if previous_epoch == -1:
                previous_epoch = epoch
                loss_array = []
            if epoch != previous_epoch and loss_array != None and len(loss_array) > 0:
                l = np.array(loss_array)
                epoch_avg_loss = np.mean(l)
                epoch_std_loss = np.std(l, dtype=np.float64)
                self.cache.append({'epoch': epoch, 'avg_loss': epoch_avg_loss, 'std_loss': epoch_std_loss, 'end_loss': latest_log})
                break
            loss_array.append(log['loss'])
        return self.__externalize_data()
        
            
class EvalMetricBasedControl(MetricHandler):

    def __init__(self, name, args=None):
        # Initialize the handler arguments
        self.__name = name
        if args == None:
            self.early_stopping_patience = 1.0
            self.early_stopping_threshold = 0.0
        else:
            self.__early_stopping_patience = args.early_stopping_patience
            self.__early_stopping_threshold = args.early_stopping_threshold
        self.early_stopping_patience_counter = 0

    def validate(self, training_args):
        # Validate the training arguments (e.g logging_steps) are
        # compatible with the computation of this metric
        return (training_args.load_best_model_at_end == True and \
        training_args.metric_for_best_model is not None and \
        training_args.evaluation_strategy != IntervalStrategy.NO and \
        training_args.save_strategy == IntervalStrategy.EPOCH)

    def __check_metric_value(self, args, state, metric_value):
        # best_metric is set by code for load_best_model
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or (
            operator(metric_value, state.best_metric)
            and abs(metric_value - state.best_metric) > self.__early_stopping_threshold
        ):
            self.__early_stopping_patience_counter = 0
        else:
            self.__early_stopping_patience_counter += 1

    def compute(self, training_state, training_args=None, metrics=None):
        # Compute the metric using the training state
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
