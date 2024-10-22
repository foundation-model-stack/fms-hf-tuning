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
import os

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
from tuning.config.tracker_configs import TimingTrackerConfig, TrackerConfigFactory

## Timing tracker tests


def test_run_with_timing_tracker():
    """Ensure that training succeeds with the timing tracker enabled."""
    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.trackers = ["timer"]

        _test_run_causallm_ft(TRAIN_ARGS, MODEL_ARGS, DATA_ARGS, tempdir)
        _test_run_inference(_get_checkpoint_path(tempdir))


def test_sample_run_with_timing_tracker_updated_filename():
    """Ensure that the timing tracker filename can be updated."""
    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir

        train_args.trackers = ["timer"]

        logs_file = "custom_timing_logs.jsonl"

        tracker_configs = TrackerConfigFactory(
            timer_config=TimingTrackerConfig(timing_logs_filename=logs_file)
        )

        sft_trainer.train(
            MODEL_ARGS, DATA_ARGS, train_args, tracker_configs=tracker_configs
        )

        # Validate that the timing logs file was created with the correct name
        train_logs_file_path = "{}/{}".format(tempdir, logs_file)
        train_log_contents = ""
        with open(train_logs_file_path, encoding="utf-8") as f:
            train_log_contents = f.read()
        assert os.path.exists(train_logs_file_path) is True
        assert os.path.getsize(train_logs_file_path) > 0
        assert "train_runtime" in train_log_contents