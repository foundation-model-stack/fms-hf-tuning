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
from typing import Dict
import logging
import os

# Third Party
from accelerate.state import PartialState
import datasets
import transformers

DEFAULT_LOG_LEVEL_MAIN = "INFO"
DEFAULT_LOG_LEVEL_WORKERS = "WARNING"


def set_log_level(logger_name="fms-hf-tuning", level=None):
    """Set log level of python native logger and TF logger via argument from CLI or env variable.

    Args:
        logger_name
            Logger name with which the logger is instantiated.
        level
            Requested level of the logger

    Returns:
        train_logger
            Logger with updated effective log level
        level
            Level of the logger initialized
    """

    # Clear any existing handlers if necessary
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure Python native logger and transformers log level
    # If CLI arg is passed, assign same log level to python native logger
    lowest_log_level = DEFAULT_LOG_LEVEL_MAIN
    if level != "passive":
        lowest_log_level = level
    elif os.environ.get("LOG_LEVEL"):
        # If CLI arg not is passed and env var LOG_LEVEL is set,
        # assign same log level to both logger
        lowest_log_level = (
            os.environ.get("LOG_LEVEL").lower()
            if not os.environ.get("TRANSFORMERS_VERBOSITY")
            else os.environ.get("TRANSFORMERS_VERBOSITY")
        )

    state = PartialState()
    rank = state.process_index

    log_on_all = os.environ.get("LOG_ON_ALL_PROCESSES")
    if log_on_all:
        log_level = lowest_log_level or DEFAULT_LOG_LEVEL_MAIN
    else:
        if state.is_local_main_process:
            log_level = lowest_log_level or DEFAULT_LOG_LEVEL_MAIN
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            log_level = DEFAULT_LOG_LEVEL_WORKERS
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()

    log_format = f"Rank-{rank} [%(levelname)s]:%(filename)s:%(funcName)s: %(message)s"

    logging.basicConfig(
        format=log_format,
        level=log_level.upper(),
    )

    if logger_name:
        train_logger = logging.getLogger(logger_name)
    else:
        train_logger = logging.getLogger()

    return train_logger, log_level.lower()


def pretty_print_args(args: Dict):
    dump = "\n========================= Flat Arguments =========================\n"
    for name, arg in args.items():
        if arg:
            dump += f"---------------------------- {name} -----------------------\n"
            if hasattr(arg, "__dict__"):
                arg = vars(arg)
            max_len = max(len(k) for k in arg.keys())
            for k, v in sorted(arg.items()):
                dump += f"  {k:<{max_len}} : {v}\n"
    dump += "========================= Arguments Done =========================\n"
    return dump
