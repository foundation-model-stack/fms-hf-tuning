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
import json
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
    _test_run_inference,
    _validate_training,
)

# Local
from tuning import sft_trainer
from tuning.config.tracker_configs import AimConfig, TrackerConfigFactory

aim_not_available = not _is_package_available("aim")


@pytest.fixture(name="aimrepo", scope="module", autouse=True)
def fixture_aimrepo():

    if aim_not_available:
        yield None
        return

    # if Aim is installed, this fixture sets up an aim repo for the tests to follow
    # yeilds the aimstack repo path which is cleaned up later.
    with tempfile.TemporaryDirectory() as aimstackrepo_path:
        os.system("cd " + aimstackrepo_path + " ; aim init")
        yield aimstackrepo_path
        return


@pytest.mark.skipif(aim_not_available, reason="Requires aimstack to be installed")
def test_run_with_aim_tracker_name_but_no_args():
    """Ensure that train() raises error with aim tracker name but no args"""

    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir

        train_args.trackers = ["aim"]

        with pytest.raises(
            ValueError,
            match="Aim tracker requested but repo or server is not specified.",
        ):
            sft_trainer.train(MODEL_ARGS, DATA_ARGS, train_args)


@pytest.mark.skipif(aim_not_available, reason="Requires aimstack to be installed")
def test_e2e_run_with_aim_tracker(aimrepo):
    """Ensure that training succeeds with aim tracker"""

    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir

        # This should not mean file logger is not present.
        # code will add it by default
        # The below validate_training check will test for that too.
        train_args.trackers = ["aim"]

        tracker_configs = TrackerConfigFactory(
            aim_config=AimConfig(experiment="unit_test", aim_repo=aimrepo)
        )

        sft_trainer.train(
            MODEL_ARGS, DATA_ARGS, train_args, tracker_configs=tracker_configs
        )

        # validate ft tuning configs
        _validate_training(tempdir)

        # validate inference
        _test_run_inference(checkpoint_path=_get_checkpoint_path(tempdir))


@pytest.mark.skipif(aim_not_available, reason="Requires aimstack to be installed")
def test_e2e_run_with_aim_runid_export_default_path(aimrepo):
    """Ensure that aim outputs runid hash in the output dir by default"""

    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir

        # This should not mean file logger is not present.
        # code will add it by default
        # The below validate_training check will test for that too.
        train_args.trackers = ["aim"]

        tracker_configs = TrackerConfigFactory(
            aim_config=AimConfig(experiment="unit_test", aim_repo=aimrepo)
        )

        sft_trainer.train(
            MODEL_ARGS, DATA_ARGS, train_args, tracker_configs=tracker_configs
        )

        # validate ft tuning configs
        _validate_training(tempdir)

        runid_file = os.path.join(tempdir, "aimstack_tracker.json")

        assert os.path.exists(runid_file) is True
        assert os.path.getsize(runid_file) > 0

        with open(runid_file, "r", encoding="utf-8") as f:
            content = json.loads(f.read())
            assert "run_hash" in content
