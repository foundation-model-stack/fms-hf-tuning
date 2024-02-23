from transformers import IntervalStrategy
from transformers.utils import logging
import numpy as np
from . import MetricHandler

logger = logging.get_logger(__name__)        
            
class EvalMetricBasedControl(MetricHandler):
    """Implements the controller metric which computes and evaluates metrics conditions on evaluation"""

    def __init__(self, name, early_stopping_patience=1.0, early_stopping_threshold=0.0):
        # Initialize the handler arguments
        self.__early_stopping_patience = early_stopping_patience
        self.__early_stopping_threshold = early_stopping_threshold
        self.__early_stopping_patience_counter = 0
        super().__init__(name)

    def validate(self, training_args):
        """Validate the training arguments (e.g logging_steps) are compatible with the computation of this metric

        Args:
            training_args: Training arguments

        Returns:
            bool
        """
        return (training_args.load_best_model_at_end == True and \
        training_args.metric_for_best_model != None and \
        training_args.evaluation_strategy != IntervalStrategy.NO and \
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
        return {self.get_name(): {
                'early_stopping_patience_counter': self.__early_stopping_patience_counter, \
                'early_stopping_patience': self.__early_stopping_patience
            }}
