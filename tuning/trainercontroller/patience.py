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

# Third Party
from transformers.utils import logging

# Resets the patience if the rule outcome happens to be false.
# Here, the expectation is to have unbroken "True"s for patience
# to be up-countered.
# E.g. For patience threshold, patience_threshold=3, rule outcome
# has to be T, T, T, T (each is an event
# then patience is reset at the third event when outcome is F.
MODE_RESET_ON_FAILURE = "reset_on_failure"

# This mode does not reset patience. E.g if rule outcome is T, T, F, T, T,
# then the patience counter is not reset at F. Instead, the patience threshold
# will be exceeded afer the fifth event.
MODE_NO_RESET_ON_FAILURE = "no_reset_on_failure"

logger = logging.get_logger(__name__)


class PatienceControl:
    """Implements the patience control for every rule"""

    # pylint: disable=unused-argument
    def __init__(self, patience_threshold=1, mode=MODE_RESET_ON_FAILURE, **kwargs):
        self._patience_threshold = patience_threshold
        self._patience_counter = 0
        self._mode = mode

    def should_tolerate(
        self, rule_outcome: bool, event_name=None, control_name=None, **kwargs
    ) -> bool:
        if rule_outcome:
            self._patience_counter = self._patience_counter + 1
        elif self._mode == MODE_RESET_ON_FAILURE:
            self._patience_counter = 0
        if self._patience_counter <= self._patience_threshold:
            logger.debug(
                "Control {} triggered on event {}: "
                "Enforcing patience [patience_counter = {:.2f}, "
                "patience_threshold = {:.2f}]".format(
                    control_name,
                    event_name,
                    self._patience_counter,
                    self._patience_threshold,
                )
            )
            return True
        logger.debug(
            "Control {} triggered on event {}: "
            "Exceeded patience [patience_counter = {:.2f}, "
            "patience_threshold = {:.2f}]".format(
                control_name,
                event_name,
                self._patience_counter,
                self._patience_threshold,
            )
        )
        self._patience_counter = 0
        return False
