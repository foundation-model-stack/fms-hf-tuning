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
import copy
import tempfile

# First Party
from tests.test_sft_trainer import (
    DATA_ARGS,
    MODEL_ARGS,
    TRAIN_ARGS,
    _get_checkpoint_path,
    _test_run_causallm_ft,
    _test_run_inference,
    _validate_training,
)

# Local
from tuning import sft_trainer
from tuning.config.tracker_configs import FileLoggingTrackerConfig, TrackerConfigFactory

## File logging tracker tests


def test_run_with_file_logging_tracker():
    """Ensure that training succeeds with a good tracker name"""
    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.trackers = ["file_logger"]

        _test_run_causallm_ft(TRAIN_ARGS, MODEL_ARGS, DATA_ARGS, tempdir)
        _test_run_inference(_get_checkpoint_path(tempdir))


def test_sample_run_with_file_logger_updated_filename():
    """Ensure that file_logger filename can be updated"""

    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir

        train_args.trackers = ["file_logger"]

        logs_file = "new_train_logs.jsonl"

        tracker_configs = TrackerConfigFactory(
            file_logger_config=FileLoggingTrackerConfig(
                training_logs_filename=logs_file
            )
        )

        sft_trainer.train(
            MODEL_ARGS, DATA_ARGS, train_args, tracker_configs=tracker_configs
        )

        # validate ft tuning configs
        _validate_training(tempdir, train_logs_file=logs_file)
