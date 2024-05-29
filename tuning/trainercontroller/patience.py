# OM NAMO GANAPATHAYEN NAMAHA
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

MODE_RESET_ON_FAILURE = "reset_on_failure"

logger = logging.get_logger(__name__)


class PatienceControl:
    """Implements the patience control for every rule"""

    # pylint: disable=unused-argument
    def __init__(self, patience_threshold=1, mode=MODE_RESET_ON_FAILURE, **kwargs):
        self._patience_threshold = patience_threshold
        self._patience_counter = 0
        self._mode = mode

    def should_tolerate(self, rule_outcome: bool) -> bool:
        if rule_outcome:
            self._patience_counter = self._patience_counter + 1
        else:
            if self._mode == MODE_RESET_ON_FAILURE:
                self._patience_counter = 0
        if self._patience_counter <= self._patience_threshold:
            logging.info(
                "Enforcing patience [patience_counter = %d, patience_threshold = %d]"
                % (self._patience_counter, self._patience_threshold)
            )
            return True
        logging.info(
            "Exceeded patience [patience_counter = %d, patience_threshold = %d]"
            % (self._patience_counter, self._patience_threshold)
        )
        return False
