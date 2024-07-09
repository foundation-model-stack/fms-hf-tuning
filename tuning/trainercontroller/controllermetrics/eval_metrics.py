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

# Third Party
from transformers.utils import logging

# Local
from tuning.trainercontroller.controllermetrics.metricshandler import MetricHandler

logger = logging.get_logger(__name__)


class EvalMetrics(MetricHandler):
    """Implements the controller metric which exposes the evaluation metrics"""

    def __init__(self, **kwargs):
        """Initializes the metric handler, by registering the event \
            list and arguments with base handler.

        Args:
            kwargs: List of arguments (key, value)-pairs
        """
        source_events_to_check = {"on_evaluate", "on_predict"}
        source_event = kwargs.get("source_event")
        if source_event is None:
            source_event = "on_evaluate"
        if source_event in source_events_to_check:
            super().__init__(
                events=[
                    source_event,
                ],
                **kwargs,
            )
        else:
            raise ValueError(
                "Specified source event [%s] is invalid for EvalMetrics"
                % (source_event)
            )

    def validate(self) -> bool:
        """Validate the training arguments (e.g logging_steps) are \
            compatible with the computation of this metric.

        Returns:
            bool
        """
        return True

    def compute(self, **kwargs) -> Any:
        """Exposes the trainer state.

        Args:
            kwargs: Remaining event arguments

        Returns:
            dict. Trainer state as a dictionary
        """
        return kwargs["metrics"]
