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


def set_log_level(train_args, logger_name=None):
    """Set log level of python native logger and TF logger via argument from CLI or env variable.

    Args:
        train_args
            Training arguments for training model.
        logger_name
            Logger name with which the logger is instantiated.

    Returns:
        train_args
            Updated training arguments for training model.
        train_logger
            Logger with updated effective log level
    """

    # Clear any existing handlers if necessary
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure Python native logger and transformers log level
    # If CLI arg is passed, assign same log level to python native logger
    log_level = "WARNING"
    if train_args.log_level != "passive":
        log_level = train_args.log_level

    # If CLI arg not is passed and env var LOG_LEVEL is set,
    # assign same log level to both logger
    elif os.environ.get("LOG_LEVEL"):
        log_level = os.environ.get("LOG_LEVEL")
        train_args.log_level = (
            log_level.lower()
            if not os.environ.get("TRANSFORMERS_VERBOSITY")
            else os.environ.get("TRANSFORMERS_VERBOSITY")
        )

    logging.basicConfig(
        format="%(levelname)s:%(filename)s:%(message)s", level=log_level.upper()
    )

    if logger_name:
        train_logger = logging.getLogger(logger_name)
    else:
        train_logger = logging.getLogger()
    return train_args, train_logger
