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
from collections import deque
from typing import Any

# Third Party
from transformers import TrainerState
from transformers.utils import logging

# Local
from tuning.trainercontroller.controllermetrics.metricshandler import MetricHandler

logger = logging.get_logger(__name__)
METRICS_KEY = "metrics"
LOG_LOSS_KEY = "loss"
TRAINING_LOSS_KEY = "training_loss"
WINDOW_SIZE = "window_size"
STEP_KEY = "steps"
EPOCH_KEY = "epoch"


class HistoryBasedMetric(MetricHandler):
    """Implements the controller metric which evaluates loss-per-step"""

    def __init__(self, window_size=1, **kwargs):
        """Initializes the metric handler, by registering the event \
            list and arguments with base handler.

        Args:
            kwargs: List of arguments (key, value)-pairs
        """
        self._window = {
            TRAINING_LOSS_KEY: {},
            METRICS_KEY: {},
            WINDOW_SIZE: window_size,
        }
        super().__init__(events=["on_log", "on_evaluate"], **kwargs)

    def _add_and_slide(self, data_type: str, data: dict) -> bool:
        """Add field values to vectors for each field in the data source.

        Args:
            type: Data type.
            data_source: Keys in data source.

        Returns:
            bool
        """
        data_sources = list(self._window[data_type].keys())
        for data_source in data_sources:
            self._window[data_type][data_source].append(data[data_source])
        window_size = self._window[WINDOW_SIZE]
        if window_size < 0:
            return True
        # All metrics in a data_type group are expected to computed together
        if len(self._window[data_type][data_sources[0]]) < window_size:
            return False
        if len(self._window[data_type][data_sources[0]]) == window_size:
            return True
        for data_source in data_sources:
            self._window[data_type][data_source].popleft()
        return True

    def validate(self) -> bool:
        """Validate the training arguments (e.g logging_steps) are \
            compatible with the computation of this metric.

        Returns:
            bool
        """
        return True

    def _create_vectors_if_not_exists(self, data_type: str, data_sources: list):
        """Creates vectors for each field in the data source.

        Args:
            data_type: Data type.
            data_source: Keys in data source.
        """
        if len(self._window[data_type]) > 0:
            return
        for data_source_name in data_sources:
            self._window[data_type][data_source_name] = deque()

    def compute(self, state: TrainerState = None, **kwargs) -> Any:
        """Exposes  the window of loss and metrics values in the log.

        Args:
            state: TrainerState object
            kwargs: Remaining event arguments

        Returns:
            Any. The exposed variables are returned here.
        """
        if METRICS_KEY in kwargs:
            metrics = kwargs[METRICS_KEY]
            metrics[STEP_KEY] = state.global_step
            metrics[EPOCH_KEY] = state.epoch
            self._create_vectors_if_not_exists(METRICS_KEY, list(metrics.keys()))
            self._add_and_slide(METRICS_KEY, metrics)
        else:
            self._create_vectors_if_not_exists(
                TRAINING_LOSS_KEY, [LOG_LOSS_KEY, STEP_KEY, EPOCH_KEY]
            )
            size_of_log_history = len(state.log_history)
            for i in range(size_of_log_history - 1, -1, -1):
                log = state.log_history[i]
                if LOG_LOSS_KEY in log:
                    data = {
                        LOG_LOSS_KEY: float(log[LOG_LOSS_KEY]),
                        STEP_KEY: state.global_step,
                        EPOCH_KEY: float(log[EPOCH_KEY]),
                    }
                    loss_data = self._window[TRAINING_LOSS_KEY][LOG_LOSS_KEY]
                    epoch_data = self._window[TRAINING_LOSS_KEY][EPOCH_KEY]
                    if (
                        len(loss_data) == 0
                        or loss_data[-1] != data[LOG_LOSS_KEY]
                        or epoch_data[-1] != data[EPOCH_KEY]
                    ):
                        self._add_and_slide(TRAINING_LOSS_KEY, data)
                    break
        return self._window
