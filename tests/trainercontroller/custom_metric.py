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
from dataclasses import dataclass
from typing import Any

# Third Party
from transformers import TrainerState
import pytest

# Local
from tuning.trainercontroller.controllermetrics.metricshandler import MetricHandler


class CustomMetric(MetricHandler):
    """Implements a custom metric for testing"""

    def __init__(self, **kwargs):
        """Initializes the metric handler, by registering the event list and arguments with base handler.

        Args:
            kwargs: List of arguments (key, value)-pairs
        """
        super().__init__(events=["on_log"], **kwargs)

    def validate(self) -> bool:
        """Validate the training arguments (e.g logging_steps) are compatible with the computation of this metric.

        Returns:
            bool
        """
        return True

    def compute(self, state: TrainerState = None, **kwargs) -> Any:
        """Just returns True (for testing purposes only).

        Args:
            state: TrainerState object
            kwargs: Remaining event arguments

        Returns:
            Any. The exposed variables are returned here.
        """
        return True
