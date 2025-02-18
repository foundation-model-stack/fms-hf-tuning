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
from tuning.config.tracker_configs import MLflowConfig, TrackerConfigFactory

mlflow_not_available = not _is_package_available("mlflow")


@pytest.mark.skipif(mlflow_not_available, reason="Requires mlflow to be installed")
def test_run_with_mlflow_tracker_name_but_no_args():
    """Ensure that train() raises error with mlflow tracker name but no args"""

    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir

        train_args.trackers = ["mlflow"]

        with pytest.raises(
            ValueError,
            match="mlflow tracker requested but mlflow_uri is not specified.",
        ):
            sft_trainer.train(MODEL_ARGS, DATA_ARGS, train_args)


@pytest.mark.skipif(mlflow_not_available, reason="Requires mlflow to be installed")
def test_e2e_run_with_mlflow_tracker():
    """Ensure that training succeeds with mlflow tracker"""

    # mlflow performs a cleanup at callback close time which happens post the
    # delete of this directory so we run into two issues
    # 1. the temp directory cannot be cleared as it has open pointer by mlflow
    # 2. mlflow complaints that it cannot find a run which it just created.
    # this is a race condition which is fixed with mkdtemp() which doesn't delete
    tempdir = tempfile.mkdtemp()

    train_args = copy.deepcopy(TRAIN_ARGS)
    train_args.output_dir = tempdir

    # This should not mean file logger is not present.
    # code will add it by default
    # The below validate_training check will test for that too.
    train_args.trackers = ["mlflow"]

    mlflow_path = os.path.join(tempdir, "mlflow")

    tracker_configs = TrackerConfigFactory(
        mlflow_config=MLflowConfig(
            mlflow_experiment="unit_test",
            mlflow_tracking_uri=f"file://{mlflow_path}",
        )
    )

    sft_trainer.train(
        MODEL_ARGS, DATA_ARGS, train_args, tracker_configs=tracker_configs
    )

    # validate ft tuning configs
    _validate_training(tempdir)

    assert os.path.exists(mlflow_path) and os.path.isdir(mlflow_path)

    # validate inference
    _test_run_inference(checkpoint_path=_get_checkpoint_path(tempdir))


@pytest.mark.skipif(mlflow_not_available, reason="Requires mlflow to be installed")
def test_e2e_run_with_mlflow_runuri_export_default_path():
    """Ensure that mlflow outputs run uri in the output dir by default"""

    tempdir = tempfile.mkdtemp()
    train_args = copy.deepcopy(TRAIN_ARGS)
    train_args.output_dir = tempdir

    train_args.trackers = ["mlflow"]

    mlflow_path = os.path.join(tempdir, "mlflow")

    tracker_configs = TrackerConfigFactory(
        mlflow_config=MLflowConfig(
            mlflow_experiment="unit_test",
            mlflow_tracking_uri=f"file://{mlflow_path}",
        )
    )

    sft_trainer.train(
        MODEL_ARGS, DATA_ARGS, train_args, tracker_configs=tracker_configs
    )

    # validate ft tuning configs
    _validate_training(tempdir)

    assert os.path.exists(mlflow_path) and os.path.isdir(mlflow_path)

    run_uri_file = os.path.join(tempdir, "mlflow_tracker.json")

    assert os.path.exists(run_uri_file) is True
    assert os.path.getsize(run_uri_file) > 0

    with open(run_uri_file, "r", encoding="utf-8") as f:
        content = json.loads(f.read())
        assert "run_uri" in content
