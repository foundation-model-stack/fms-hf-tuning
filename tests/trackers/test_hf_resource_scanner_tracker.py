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
import os
import tempfile

# Third Party
from transformers.utils.import_utils import _is_package_available
import pytest

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
from tuning.config.tracker_configs import HFResourceScannerConfig, TrackerConfigFactory

## HF Resource Scanner Tracker Tests


@pytest.mark.skipif(
    not _is_package_available("HFResourceScanner"),
    reason="Only runs if HFResourceScanner is installed",
)
def test_run_with_hf_resource_scanner_tracker():
    """Ensure that training succeeds with a good tracker name"""
    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.trackers = ["hf_resource_scanner"]

        _test_run_causallm_ft(TRAIN_ARGS, MODEL_ARGS, DATA_ARGS, tempdir)
        _test_run_inference(_get_checkpoint_path(tempdir))


@pytest.mark.skipif(
    not _is_package_available("HFResourceScanner"),
    reason="Only runs if HFResourceScanner is installed",
)
def test_sample_run_with_hf_resource_scanner_updated_filename():
    """Ensure that hf_resource_scanner output filename can be updated"""

    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.gradient_accumulation_steps = 1
        train_args.output_dir = tempdir

        # add hf_resource_scanner to the list of requested tracker
        train_args.trackers = ["hf_resource_scanner"]

        scanner_output_file = "scanner_output.json"

        tracker_configs = TrackerConfigFactory(
            hf_resource_scanner_config=HFResourceScannerConfig(
                scanner_output_filename=os.path.join(tempdir, scanner_output_file)
            )
        )

        sft_trainer.train(
            MODEL_ARGS, DATA_ARGS, train_args, tracker_configs=tracker_configs
        )

        # validate ft tuning configs
        _validate_training(tempdir, check_scanner_file=True)
