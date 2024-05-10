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
from typing import Any, List
import abc

# Third Party
from transformers import TrainingArguments


class MetricHandlerException(Exception):
    """Initializes the metric handler exception class"""

    def __init__(self, name):
        super().__init__(f"Metric handler {name} failed validation.")


class MetricHandler(metaclass=abc.ABCMeta):
    """Base class for the controller-metrics"""

    def __init__(self, name: str, events: List[str], args: TrainingArguments, **kwargs):
        """Initializes the metric handler base class

        Args:
            name: str. Name of the metric handler
            event: List[str]. List of events for with the metric computation has to be performed.
            args: TrainingArguments. Training arguments.
            kwargs: List of arguments (key, value)-pairs
        """
        self._name = name
        self._events = events
        self.training_args = args
        self.kwargs = kwargs
        if not self.validate():
            raise MetricHandlerException(name)

    def get_name(self):
        """Returns the name of the handler.

        Returns:
            str
        """
        return self._name

    def get_events(self):
        """Returns the list of events for the metric.

        Returns:
            str
        """
        return self._events

    @abc.abstractmethod
    def validate(self) -> bool:
        """Validate the training arguments (e.g logging_steps) are compatible with
           the computation of this metric, and log the errors, and return False when
           the metric is incompatible with the configuration

        Returns:
            bool
        """

    @abc.abstractmethod
    def compute(self, **kwargs) -> Any:
        """Computes the controller-metric returns the metric.

        Args:
            kwargs: Remaining event arguments. List of arguments (key, value)-pairs.

        Returns:
            Any
        """
