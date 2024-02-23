import statistics as stats
from collections import deque
import math
from . import MetricHandlerWithCache

class EpochLoss(MetricHandlerWithCache):
    """Implements the controller metric which evaluates loss-per-epoch over a user-defined window"""

    def __init__(self, name, window_size):
        # Initialize the handler arguments
        self.__window_size = window_size
        super().__init__(name, {'avg_loss': deque(), \
                        'std_loss': deque(), \
                        'final_loss': deque(), \
                        'final_eval_loss': deque()})
        
    def validate(self, training_args):
        """Validate the training arguments (e.g logging_steps) are compatible with the computation of this metric

        Args:
            training_args: Training arguments

        Returns:
            bool
        """
        return training_args.logging_strategy == 'steps' and \
            training_args.logging_steps == 1

    def add_to_cache(self, **kwargs):
        cache = self.get_cache()
        for k, v in kwargs.items():
            cache[k].append(v)

    def compute_metrics_on_cache(self):
        window_full = self.slide_the_window(self.__window_size)
        if not window_full:
            return None
        return {self.get_name(): self.get_cache()}

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
        final_loss = None
        final_eval_loss = None
        added = False
        size_of_log_history = len(training_state.log_history)
        for i in range(size_of_log_history - 1, -1, -1):
            log = training_state.log_history[i]
            try:
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
                    epoch_avg_loss = stats.mean(loss_array)
                    epoch_std_loss = stats.stdev(loss_array)
                    self.add_to_cache(avg_loss=epoch_avg_loss, std_loss=epoch_std_loss, final_loss=final_loss, final_eval_loss=final_eval_loss)
                    added = True
                    break
                loss_array.append(loss)
            except:
                # Ignoring log lines not containing relevant fields
                continue
        if len(loss_array) > 0 and added == False:
            epoch_avg_loss = stats.mean(loss_array)
            epoch_std_loss = stats.stdev(loss_array)
            self.add_to_cache(avg_loss=epoch_avg_loss, std_loss=epoch_std_loss, final_loss=final_loss, final_eval_loss=final_eval_loss)
        return self.compute_metrics_on_cache()
