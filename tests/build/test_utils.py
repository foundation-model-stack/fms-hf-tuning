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
from tuning.config.peft_config import LoraConfig, PromptTuningConfig
from build.utils import process_launch_training_args, process_accelerate_launch_args

HAPPY_PATH_DUMMY_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "dummy_job_config.json"
)


# Note: job_config dict gets modified during process_launch_training_args
@pytest.fixture(scope="session")
def job_config():
    with open(HAPPY_PATH_DUMMY_CONFIG_PATH, "r", encoding="utf-8") as f:
        dummy_job_config_dict = json.load(f)
    return dummy_job_config_dict


def test_process_launch_training_args(job_config):
    job_config_copy = copy.deepcopy(job_config)
    (
        model_args,
        data_args,
        training_args,
        tune_config,
        merge_model,
    ) = process_launch_training_args(job_config_copy)
    assert str(model_args.torch_dtype) == "torch.bfloat16"
    assert data_args.dataset_text_field == "output"
    assert training_args.output_dir == "bloom-twitter"
    assert tune_config == None
    assert merge_model == False


def test_process_launch_training_args_defaults(job_config):
    job_config_defaults = copy.deepcopy(job_config)
    assert "torch_dtype" not in job_config_defaults
    assert job_config_defaults["use_flash_attn"] == False
    assert "save_strategy" not in job_config_defaults
    model_args, _, training_args, _, _ = process_launch_training_args(
        job_config_defaults
    )
    assert str(model_args.torch_dtype) == "torch.bfloat16"
    assert model_args.use_flash_attn == False
    assert training_args.save_strategy.value == "epoch"


def test_process_launch_training_args_peft_method(job_config):
    job_config_pt = copy.deepcopy(job_config)
    job_config_pt["peft_method"] = "pt"
    _, _, _, tune_config, merge_model = process_launch_training_args(job_config_pt)
    assert type(tune_config) == PromptTuningConfig
    assert merge_model == False

    job_config_lora = copy.deepcopy(job_config)
    job_config_lora["peft_method"] = "lora"
    _, _, _, tune_config, merge_model = process_launch_training_args(job_config_lora)
    assert type(tune_config) == LoraConfig
    assert merge_model == True


def test_process_accelerate_launch_args(job_config):
    job_config_copy = copy.deepcopy(job_config)
    args = process_accelerate_launch_args(job_config_copy)
    assert args.config_file == "fixtures/accelerate_fsdp_defaults.yaml"
    assert args.use_fsdp == True
    assert args.tpu_use_cluster == False


@patch("os.path.exists")
def test_process_accelerate_launch_custom_fsdp(patch_path_exists):
    patch_path_exists.return_value = True

    dummy_fsdp_path = "dummy_fsdp_config.yaml"

    # When user passes custom fsdp config file, use custom config and accelerate
    # launch will use `num_processes` from config
    temp_job_config = {"accelerate_launch_args": {"config_file": dummy_fsdp_path}}
    args = process_accelerate_launch_args(temp_job_config)
    assert args.config_file == dummy_fsdp_path
    assert args.num_processes == None

    # When user passes custom fsdp config file and also `num_processes` as a param, use custom config and
    # overwrite num_processes from config with param
    temp_job_config = {
        "accelerate_launch_args": {
            "config_file": dummy_fsdp_path,
            "num_processes": 3,
        }
    }
    args = process_accelerate_launch_args(temp_job_config)
    assert args.config_file == dummy_fsdp_path
    assert args.num_processes == 3
