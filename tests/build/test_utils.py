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
import copy
import json
import os
from unittest.mock import patch

# Third Party
import pytest

# Local
from build.utils import process_accelerate_launch_args

HAPPY_PATH_DUMMY_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "dummy_job_config.json"
)


# Note: job_config dict gets modified during processing training args
@pytest.fixture(name="job_config", scope="session")
def fixture_job_config():
    with open(HAPPY_PATH_DUMMY_CONFIG_PATH, "r", encoding="utf-8") as f:
        dummy_job_config_dict = json.load(f)
    return dummy_job_config_dict


def test_process_accelerate_launch_args(job_config):
    args = process_accelerate_launch_args(job_config)
    # json config values used
    assert args.use_fsdp is True
    assert args.fsdp_backward_prefetch_policy == "TRANSFORMER_BASED_WRAP"
    assert args.env == ["env1", "env2"]
    assert args.training_script == "tuning.sft_trainer"
    assert args.config_file == "fixtures/accelerate_fsdp_defaults.yaml"

    # default values
    assert args.tpu_use_cluster is False
    assert args.mixed_precision is None


@patch("torch.cuda.device_count", return_value=1)
def test_accelerate_launch_args_user_set_num_processes_ignored(job_config):
    job_config_copy = copy.deepcopy(job_config)
    job_config_copy["accelerate_launch_args"]["num_processes"] = "3"
    args = process_accelerate_launch_args(job_config_copy)
    # determine number of processes by number of GPUs available
    assert args.num_processes == 1

    # if single-gpu, CUDA_VISIBLE_DEVICES set
    assert os.getenv("CUDA_VISIBLE_DEVICES") == "0"


@patch.dict(os.environ, {"SET_NUM_PROCESSES_TO_NUM_GPUS": "False"})
def test_accelerate_launch_args_user_set_num_processes(job_config):
    job_config_copy = copy.deepcopy(job_config)
    job_config_copy["accelerate_launch_args"]["num_processes"] = "3"

    args = process_accelerate_launch_args(job_config_copy)
    # json config values used
    assert args.num_processes == 3
    assert args.config_file == "fixtures/accelerate_fsdp_defaults.yaml"


def test_accelerate_launch_args_default_fsdp_config_multigpu(job_config):
    with patch("torch.cuda.device_count", return_value=2):
        with patch("os.path.exists", return_value=True):
            job_config_copy = copy.deepcopy(job_config)
            job_config_copy["accelerate_launch_args"].pop("config_file")

            assert "config_file" not in job_config_copy["accelerate_launch_args"]

            args = process_accelerate_launch_args(job_config_copy)

            # use default config file
            assert args.config_file == "/app/accelerate_fsdp_defaults.yaml"
            # determine number of processes by number of GPUs available
            assert args.num_processes == 2


@patch("os.path.exists")
def test_process_accelerate_launch_custom_config_file(patch_path_exists):
    patch_path_exists.return_value = True

    dummy_config_path = "dummy_fsdp_config.yaml"

    # When user passes custom fsdp config file, use custom config and accelerate
    # launch will use `num_processes` from config
    temp_job_config = {"accelerate_launch_args": {"config_file": dummy_config_path}}
    args = process_accelerate_launch_args(temp_job_config)
    assert args.config_file == dummy_config_path
    assert args.num_processes is None

    # When user passes custom fsdp config file and also `num_processes` as a param,
    # use custom config and overwrite num_processes from config with param
    temp_job_config = {"accelerate_launch_args": {"config_file": dummy_config_path}}
    args = process_accelerate_launch_args(temp_job_config)
    assert args.config_file == dummy_config_path
