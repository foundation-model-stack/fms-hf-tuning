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
import tempfile

# Third Party
import pytest
import filecmp

# Local
from build.utils import process_accelerate_launch_args
from build.utils import copy_checkpoint

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
    assert args.fsdp_backward_prefetch == "TRANSFORMER_BASED_WRAP"
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


class CopyCheckpointTestConfig:
    def __init__(self, temp_root):

        # Create the following file tree for testing:
        # test_root
        #   test_copytree_source
        #      tf1.txt
        #      tf2.txt
        #      tf3.txt
        #      subdir1
        #         tf4.txt
        #         tf5.txt
        #         tf6.txt

        self.test_root = temp_root
        self.source_dir = os.path.join(self.test_root, "test_copytree_source")
        self.source_sub_dir = os.path.join(self.source_dir, "subdir1")

        os.mkdir(self.source_dir)
        for file_number in range(2):
            with open(
                os.path.join(self.source_dir, f"tf{file_number+1}.txt"),
                "a",
                encoding="utf-8",
            ) as f:
                f.close()

        os.mkdir(self.source_sub_dir)
        for file_number in range(2):
            with open(
                os.path.join(self.source_sub_dir, f"tf{file_number+4}.txt"),
                "a",
                encoding="utf-8",
            ) as f:
                f.close()

    def are_dir_trees_equal(self, dir1, dir2):

        dirs_cmp = filecmp.dircmp(dir1, dir2)
        if (
            len(dirs_cmp.left_only) > 0
            or len(dirs_cmp.right_only) > 0
            or len(dirs_cmp.funny_files) > 0
        ):
            return False
        (_, mismatch, errors) = filecmp.cmpfiles(
            dir1, dir2, dirs_cmp.common_files, shallow=False
        )
        if len(mismatch) > 0 or len(errors) > 0:
            return False
        for common_dir in dirs_cmp.common_dirs:
            new_dir1 = os.path.join(dir1, common_dir)
            new_dir2 = os.path.join(dir2, common_dir)
            if not self.are_dir_trees_equal(new_dir1, new_dir2):
                return False
        return True


def test_copy_checkpoint_dest_dir_does_not_exist():

    # Init source directory
    with tempfile.TemporaryDirectory() as test_root:
        config = CopyCheckpointTestConfig(test_root)

        target_dir_does_not_exist = os.path.join(
            config.test_root, "test_copytree_target"
        )

        # Execute the copy
        copy_checkpoint(config.source_dir, target_dir_does_not_exist)
        assert config.are_dir_trees_equal(config.source_dir, target_dir_does_not_exist)


def test_copy_checkpoint_dest_dir_does_exist():

    # Init source directory
    with tempfile.TemporaryDirectory() as test_root:
        config = CopyCheckpointTestConfig(test_root)

        # Init target directory
        target_dir_does_exist = os.path.join(config.test_root, "test_copytree_target2")
        os.mkdir(target_dir_does_exist)
        # Add a file to the target. This file will be overwritten during the copy.
        with open(
            os.path.join(target_dir_does_exist, "tf1.txt"),
            "a",
            encoding="utf-8",
        ) as f:
            f.close()
        # Add a file to the target. This file does not exist in source.
        with open(
            os.path.join(target_dir_does_exist, "tf9.txt"),
            "a",
            encoding="utf-8",
        ) as f:
            f.close()
        # Execute the copy
        copy_checkpoint(config.source_dir, target_dir_does_exist)
        assert os.path.exists(os.path.join(target_dir_does_exist, "tf9.txt"))
        # Remove it so we can validate the dir trees are equal.
        os.remove(os.path.join(target_dir_does_exist, "tf9.txt"))
        assert config.are_dir_trees_equal(config.source_dir, target_dir_does_exist)


def test_copy_checkpoint_dest_dir_not_writeable():

    # Init source directory
    with tempfile.TemporaryDirectory() as test_root:
        config = CopyCheckpointTestConfig(test_root)

        # Init target directory
        target_dir_not_writeable = os.path.join(
            config.test_root, "test_copytree_notwriteable"
        )

        os.makedirs(target_dir_not_writeable, mode=0o446)

        # Execute the copy. Should FAIL
        with pytest.raises(PermissionError) as e:
            copy_checkpoint(config.source_dir, target_dir_not_writeable)
        assert "Permission denied:" in str(e.value)


def test_copy_checkpoint_source_dir_does_not_exist():
    with pytest.raises(FileNotFoundError) as e:
        copy_checkpoint("/doesnotexist", "/tmp")
    assert "No such file or directory" in str(e.value)
