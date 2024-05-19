# Copyright The IBM Tuning Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-License-Identifier: Apache-2.0
# https://spdx.dev/learn/handling-license-info/

# Standard
from typing import Any
from collections import deque

# Third Party
from transformers import TrainerState
from transformers.utils import logging

# Local
from tuning.trainercontroller.controllermetrics.metricshandler import MetricHandler

logger = logging.get_logger(__name__)

class HistoryBasedMetric(MetricHandler):
    """Implements the controller metric which evaluates loss-per-step"""
    _METRICS_KEY = "metrics"
    _TRAINING_LOSS_KEY = "loss"
    
    def __init__(self, **kwargs):
        """Initializes the metric handler, by registering the event \
            list and arguments with base handler.

        Args:
            kwargs: List of arguments (key, value)-pairs
        """
        self._window = {self._TRAINING_LOSS_KEY: deque(), 
                        self._METRICS_KEY: deque()}
        self._window_size = kwargs.get("window-size")
        if self._window_size is None:
            self._window_size = 1
        super().__init__(events=["on_log", "on_evaluate"], **kwargs)
    
    def _add_to_window(self, data_type, data):
        self._window[data_type].append(data)

    def _slide_the_window(self, data_type):
        if len(self._window[data_type]) < self._window_size:
            return False        
        if len(self._window[data_type]) == self._window_size:
            return True
        self._window[data_type].popleft()
        return True

    def validate(self) -> bool:
        """Validate the training arguments (e.g logging_steps) are \
            compatible with the computation of this metric.

        Returns:
            bool
        """
        return True

    def compute(self, state: TrainerState = None, **kwargs) -> Any:
        """Exposes  the window of loss and metrics values in the log.

        Args:
            state: TrainerState object
            kwargs: Remaining event arguments

        Returns:
            Any. The exposed variables are returned here.
        """
        if self._METRICS_KEY in kwargs:
            self._add_to_window(self._METRICS_KEY, kwargs[self._METRICS_KEY])
        else:
            size_of_log_history = len(state.log_history)
            for i in range(size_of_log_history - 1, -1, -1):
                log = state.log_history[i]
                if self._TRAINING_LOSS_KEY in log:
                    self._add_to_window(self._TRAINING_LOSS_KEY, 
                                       float(log[self._TRAINING_LOSS_KEY]))
                    break
        return self._window
