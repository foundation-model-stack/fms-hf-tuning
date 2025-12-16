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

# Third Party
import pytest

# First Party
from tests.test_sft_trainer import DATA_ARGS, MODEL_ARGS, TRAIN_ARGS

# Local
from tuning import sft_trainer


def test_run_with_bad_tracker_config():
    """Ensure that train() raises error with bad tracker configs"""

    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir

        with pytest.raises(
            ValueError,
            match="tracker configs should adhere to the TrackerConfigs type",
        ):
            sft_trainer.train(
                copy.deepcopy(MODEL_ARGS),
                copy.deepcopy(DATA_ARGS),
                train_args,
                tracker_configs="NotSupposedToBeHere",
            )


def test_run_with_bad_tracker_name():
    """Ensure that train() raises error with bad tracker name"""

    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir

        bad_name = "NotAValidTracker"
        train_args.trackers = [bad_name]

        # ensure bad tracker name gets called out
        with pytest.raises(
            ValueError, match=r"Requested Tracker {} not found.".format(bad_name)
        ):
            sft_trainer.train(
                copy.deepcopy(MODEL_ARGS),
                copy.deepcopy(DATA_ARGS),
                train_args,
            )
