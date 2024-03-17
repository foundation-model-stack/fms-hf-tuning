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

import abc
from typing import Any, List

from transformers import TrainingArguments

class MetricHandler(metaclass=abc.ABCMeta):

    """Base class for the controller-metrics"""
    def __init__(self, name: str, events: List[str], args: TrainingArguments, **kwargs):
        self._name = name
        self._events = events
        self.training_args = args
        if not self.validate():
            raise Exception(f"Metric handler {name} failed validation.")

    def get_name(self):
        return self._name

    def get_events(self):
        return self._events

    @abc.abstractmethod 
    def validate(self) -> bool:
        """Validate the training arguments (e.g logging_steps) are compatible with 
           the computation of this metric, and log the errors, and return False when
           the metric is incompatible with the configuration

        Returns:
            bool
        """
        pass

    @abc.abstractmethod 
    def compute(self, event_name: str, **kwargs) -> Any:
        """Computes the controller-metric returns the metric.

        Args:
            event_name: Name of the event which is invoking the metric handler
            kwargs: Remaining event arguments

        Returns:
            dict
        """
        pass
