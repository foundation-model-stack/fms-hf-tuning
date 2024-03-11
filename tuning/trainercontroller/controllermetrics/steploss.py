import statistics as stats
from collections import deque
from . import MetricHandlerWithCache

class StepLoss(MetricHandlerWithCache):
    """Implements the controller metric which evaluates loss-per-step"""
    
    def __init__(self, name, window_size):
        # Initialize the handler arguments
        self.__window_size = window_size
        super().__init__(name, {'loss': deque()})

    def validate(self, training_args) -> bool:
        """Validate the training arguments (e.g logging_steps) are compatible with the computation of this metric

        Args:
            training_args: Training arguments

        Returns:
            bool
        """
        return training_args.logging_strategy == 'steps' and \
            training_args.logging_steps == 1

    def compute_metrics_on_cache(self):
        if not self.slide_the_window(self.__window_size):
            return None  
        cache = self.get_cache()
        consistently_increasing = True
        for i in range(len(cache['loss'])-1):
            if cache['loss'][i] > cache['loss'][i+1]:
                consistently_increasing = False
                break
        avg_loss = stats.mean(cache['loss'])
        std_loss = stats.stdev(cache['loss'])
        return {self.get_name(): {'consistently_increasing': int(consistently_increasing), \
            'average_loss': avg_loss,\
            'std_loss': std_loss,\
            'window': cache['loss']}}

    def compute(self, training_state, training_args=None, metrics=None) -> dict:
        """Computes the controller-metric (step-loss over window) and exposes the values of the variables used by the rules.

        Args:
            training_state: TrainerState object
            training_args: TrainingArguments object
            metrics: [optional] metrics data

        Returns:
            dict
        """
        size_of_log_history = len(training_state.log_history)
        for i in range(size_of_log_history - 1, -1, -1):
            log = training_state.log_history[i]
            if 'loss' not in log:
                continue
            self.add_to_cache(loss=log['loss'])
            return self.compute_metrics_on_cache()
