# Copyright The FMS HF Tuning Authors
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

# Standard
import logging
import os

# The USER_ERROR_EXIT_CODE will be thrown when the process must exit
# as result of a user input error. User-related errors should be
# >= 1 and <=127 due to how some kubernetes operators interpret them.
USER_ERROR_EXIT_CODE = 1
# The INTERNAL_ERROR_EXIT_CODE will be thrown when training
# abnormally terminates, and it is not clearly fault of the user.
# System-level errors should be >= 128 and <= 254
INTERNAL_ERROR_EXIT_CODE = 203


def write_termination_log(text, log_file="error.log"):
    """Writes text to termination log.

    Args:
        text: str
        log_file: Optional[str]
    """
    log_file = os.environ.get("TERMINATION_LOG_FILE", log_file)
    try:
        with open(log_file, "a", encoding="utf-8") as handle:
            handle.write(text)
    except Exception as e:  # pylint: disable=broad-except
        logging.warning("Unable to write termination log due to error {}".format(e))
