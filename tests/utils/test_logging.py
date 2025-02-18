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

# SPDX-License-Identifier: Apache-2.0
# https://spdx.dev/learn/handling-license-info/

# Standard
from unittest import mock
import copy
import logging
import os

# First Party
from tests.test_sft_trainer import TRAIN_ARGS

# Local
from tuning.utils.logging import set_log_level


def test_set_log_level_for_logger_default():
    """
    Ensure that the correct log level is being set for python native logger and
    transformers logger when no env var or CLI flag is passed
    """

    with mock.patch.dict(os.environ, {}, clear=True):
        train_args = copy.deepcopy(TRAIN_ARGS)
        training_args, logger = set_log_level(train_args)
        assert logger.getEffectiveLevel() == logging.WARNING
        assert training_args.log_level == "passive"


def test_set_log_level_for_logger_with_env_var():
    """
    Ensure that the correct log level is being set for python native logger and
    transformers logger when env var LOG_LEVEL is used
    """

    with mock.patch.dict(os.environ, {"LOG_LEVEL": "info"}, clear=True):
        train_args_env = copy.deepcopy(TRAIN_ARGS)
        training_args, logger = set_log_level(train_args_env)
        assert logger.getEffectiveLevel() == logging.INFO
        assert training_args.log_level == "info"


def test_set_log_level_for_logger_with_set_verbosity_and_cli():
    """
    Ensure that the correct log level is being set for python native logger and
    log_level of transformers logger is unchanged when env var TRANSFORMERS_VERBOSITY is used
    and CLI flag is passed
    """

    with mock.patch.dict(os.environ, {"TRANSFORMERS_VERBOSITY": "info"}, clear=True):
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.log_level = "error"
        training_args, logger = set_log_level(train_args)
        assert logger.getEffectiveLevel() == logging.ERROR
        assert training_args.log_level == "error"


def test_set_log_level_for_logger_with_env_var_and_cli():
    """
    Ensure that the correct log level is being set for python native logger and
    transformers logger when env var LOG_LEVEL is used and CLI flag is passed.
    In this case, CLI arg takes precedence over the set env var LOG_LEVEL.
    """

    with mock.patch.dict(os.environ, {"LOG_LEVEL": "info"}, clear=True):
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.log_level = "error"
        training_args, logger = set_log_level(train_args)
        assert logger.getEffectiveLevel() == logging.ERROR
        assert training_args.log_level == "error"
