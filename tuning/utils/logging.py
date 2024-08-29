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
from typing import Optional
import logging
import os

# Third Party
import aim

# Local
from tuning.config.tracker_configs import AimConfig


class AimLogger(logging.Logger):
    """Used for sending logs to AIM."""

    aim_run: aim.Run

    def __init__(self, aim_config: AimConfig, level=logging.NOTSET):
        super().__init__(name=aim_config.experiment, level=level)
        aim_repo = aim_config.aim_repo if aim_config.aim_repo else aim_config.aim_url
        self.aim_run = aim.Run(repo=aim_repo, experiment=aim_config.experiment)

    def debug(self, msg: str, *args, **kwargs):
        if self.isEnabledFor(logging.DEBUG):
            msg_filled = msg.format(*args, **kwargs)
            self.aim_run.log_debug(msg_filled)
            super().debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        if self.isEnabledFor(logging.INFO):
            msg_filled = msg.format(*args, **kwargs)
            self.aim_run.log_debug(msg_filled)
            super().info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.WARNING):
            msg_filled = msg.format(*args, **kwargs)
            self.aim_run.log_debug(msg_filled)
            super().warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.ERROR):
            msg_filled = msg.format(*args, **kwargs)
            self.aim_run.log_debug(msg_filled)
            super().error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.CRITICAL):
            msg_filled = msg.format(*args, **kwargs)
            self.aim_run.log_debug(msg_filled)
            super().critical(msg, *args, **kwargs)


__LOGGER: Optional[logging.Logger] = None


def get_logger(
    aim_config: Optional[AimConfig] = None,
    logger_name: Optional[str] = None,
    level=logging.NOTSET,
) -> logging.Logger:
    """Get the singleton instance of logger."""
    # pylint: disable=global-statement
    global __LOGGER
    if not __LOGGER:
        if aim_config and (aim_config.aim_repo or aim_config.aim_url):
            # ignore logger_name if the AIM experiment name is provided
            __LOGGER = AimLogger(aim_config=aim_config, level=level)
            __LOGGER.info("AIM logger intialized")
        elif logger_name:
            __LOGGER = logging.getLogger(logger_name)
        else:
            __LOGGER = logging.getLogger()
    return __LOGGER


def set_log_level(
    train_args,
    logger_name: Optional[str] = None,
    aim_config: Optional[AimConfig] = None,
):
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

    train_logger = get_logger(
        aim_config=aim_config, logger_name=logger_name, level=log_level
    )
    return train_args, train_logger
