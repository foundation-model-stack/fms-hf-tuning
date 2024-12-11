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

"""Unit Tests for accelerate_launch script.
"""

# Standard
import os
import tempfile
import glob

# Third Party
import pytest

# First Party
from build.accelerate_launch import main
from build.utils import serialize_args, get_highest_checkpoint
from tests.artifacts.testdata import TWITTER_COMPLAINTS_DATA_JSONL
from tuning.utils.error_logging import (
    USER_ERROR_EXIT_CODE,
    INTERNAL_ERROR_EXIT_CODE,
)
from tuning.config.tracker_configs import FileLoggingTrackerConfig

SCRIPT = "tuning/sft_trainer.py"
MODEL_NAME = "Maykeye/TinyLLama-v0"
BASE_KWARGS = {
    "model_name_or_path": MODEL_NAME,
    "training_data_path": TWITTER_COMPLAINTS_DATA_JSONL,
    "num_train_epochs": 5,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 0.00001,
    "weight_decay": 0,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "logging_steps": 1,
    "include_tokens_per_second": True,
    "packing": False,
    "response_template": "\n### Label:",
    "dataset_text_field": "output",
    "use_flash_attn": False,
    "torch_dtype": "float32",
    "max_seq_length": 4096,
}
BASE_PEFT_KWARGS = {
    **BASE_KWARGS,
    **{
        "peft_method": "pt",
        "prompt_tuning_init": "RANDOM",
        "num_virtual_tokens": 8,
        "prompt_tuning_init_text": "hello",
        "save_strategy": "epoch",
        "output_dir": "tmp",
    },
}
BASE_LORA_KWARGS = {
    **BASE_KWARGS,
    **{
        "peft_method": "lora",
        "r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
    },
}


def setup_env(tempdir):
    os.environ["TRAINING_SCRIPT"] = SCRIPT
    os.environ["PYTHONPATH"] = "./:$PYTHONPATH"
    os.environ["TERMINATION_LOG_FILE"] = tempdir + "/termination-log"


def cleanup_env():
    os.environ.pop("TRAINING_SCRIPT", None)
    os.environ.pop("PYTHONPATH", None)
    os.environ.pop("TERMINATION_LOG_FILE", None)


def test_successful_ft():
    """Check if we can bootstrap and fine tune causallm models"""
    with tempfile.TemporaryDirectory() as tempdir:
        setup_env(tempdir)
        TRAIN_KWARGS = {**BASE_KWARGS, **{"output_dir": tempdir}}
        serialized_args = serialize_args(TRAIN_KWARGS)
        os.environ["SFT_TRAINER_CONFIG_JSON_ENV_VAR"] = serialized_args

        assert main() == 0
        _validate_termination_files_when_tuning_succeeds(tempdir)
        checkpoint = os.path.join(tempdir, get_highest_checkpoint(tempdir))
        _validate_training_output(checkpoint, "ft")


def test_successful_pt():
    """Check if we can bootstrap and peft tune causallm models"""
    with tempfile.TemporaryDirectory() as tempdir:
        setup_env(tempdir)
        TRAIN_KWARGS = {**BASE_PEFT_KWARGS, **{"output_dir": tempdir}}
        serialized_args = serialize_args(TRAIN_KWARGS)
        os.environ["SFT_TRAINER_CONFIG_JSON_ENV_VAR"] = serialized_args

        assert main() == 0
        _validate_termination_files_when_tuning_succeeds(tempdir)
        checkpoint = os.path.join(tempdir, get_highest_checkpoint(tempdir))
        _validate_training_output(checkpoint, "pt")


def test_successful_lora():
    """Check if we can bootstrap and LoRA tune causallm models"""
    with tempfile.TemporaryDirectory() as tempdir:
        setup_env(tempdir)
        TRAIN_KWARGS = {**BASE_LORA_KWARGS, **{"output_dir": tempdir}}
        serialized_args = serialize_args(TRAIN_KWARGS)
        os.environ["SFT_TRAINER_CONFIG_JSON_ENV_VAR"] = serialized_args

        assert main() == 0
        _validate_termination_files_when_tuning_succeeds(tempdir)
        checkpoint = os.path.join(tempdir, get_highest_checkpoint(tempdir))
        _validate_training_output(checkpoint, "lora")


def test_lora_save_model_dir_separate_dirs():
    """Run LoRA tuning with separate save_model_dir and output_dir.
    Verify model saved to save_model_dir and checkpoints saved to
    output_dir.
    """
    with tempfile.TemporaryDirectory() as tempdir:
        output_dir = os.path.join(tempdir, "output_dir")
        save_model_dir = os.path.join(tempdir, "save_model_dir")
        setup_env(tempdir)
        TRAIN_KWARGS = {
            **BASE_LORA_KWARGS,
            **{
                "output_dir": output_dir,
                "save_model_dir": save_model_dir,
                "save_total_limit": 1,
            },
        }
        serialized_args = serialize_args(TRAIN_KWARGS)
        os.environ["SFT_TRAINER_CONFIG_JSON_ENV_VAR"] = serialized_args

        assert main() == 0
        _validate_termination_files_when_tuning_succeeds(output_dir)
        _validate_training_output(save_model_dir, "lora")

        # purpose here is to see if only one checkpoint is saved
        checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        assert len(checkpoints) == 1


def test_lora_save_model_dir_same_dir_as_output_dir():
    """Run LoRA tuning with same save_model_dir and output_dir.
    Verify checkpoints, logs, and model saved to path.
    """
    with tempfile.TemporaryDirectory() as tempdir:
        setup_env(tempdir)
        TRAIN_KWARGS = {
            **BASE_LORA_KWARGS,
            **{
                "output_dir": tempdir,
                "save_model_dir": tempdir,
                "gradient_accumulation_steps": 1,
            },
        }
        serialized_args = serialize_args(TRAIN_KWARGS)
        os.environ["SFT_TRAINER_CONFIG_JSON_ENV_VAR"] = serialized_args

        assert main() == 0
        # check logs, checkpoint dir, and model exists in path
        _validate_termination_files_when_tuning_succeeds(tempdir)
        # check that model exists in output_dir and checkpoint dir
        _validate_training_output(tempdir, "lora")
        checkpoint_path = os.path.join(tempdir, get_highest_checkpoint(tempdir))
        _validate_training_output(checkpoint_path, "lora")

        # number of checkpoints should equal number of epochs
        checkpoints = glob.glob(os.path.join(tempdir, "checkpoint-*"))
        assert len(checkpoints) == TRAIN_KWARGS["num_train_epochs"]


def test_lora_save_model_dir_same_dir_as_output_dir_save_strategy_no():
    """Run LoRA tuning with same save_model_dir and output_dir and
    save_strategy=no. Verify no checkpoints created, only
    logs and final model.
    """
    with tempfile.TemporaryDirectory() as tempdir:
        setup_env(tempdir)
        TRAIN_KWARGS = {
            **BASE_LORA_KWARGS,
            **{"output_dir": tempdir, "save_model_dir": tempdir, "save_strategy": "no"},
        }
        serialized_args = serialize_args(TRAIN_KWARGS)
        os.environ["SFT_TRAINER_CONFIG_JSON_ENV_VAR"] = serialized_args

        assert main() == 0
        # check that model and logs exists in output_dir
        _validate_termination_files_when_tuning_succeeds(tempdir)
        _validate_training_output(tempdir, "lora")

        # no checkpoints should be created
        checkpoints = glob.glob(os.path.join(tempdir, "checkpoint-*"))
        assert len(checkpoints) == 0


def test_lora_with_lora_post_process_for_vllm_set_to_true():
    with tempfile.TemporaryDirectory() as tempdir:
        setup_env(tempdir)
        TRAIN_KWARGS = {
            **BASE_LORA_KWARGS,
            **{
                "output_dir": tempdir,
                "save_model_dir": tempdir,
                "lora_post_process_for_vllm": True,
            },
        }
        serialized_args = serialize_args(TRAIN_KWARGS)
        os.environ["SFT_TRAINER_CONFIG_JSON_ENV_VAR"] = serialized_args

        assert main() == 0
        # check that model and logs exists in output_dir
        _validate_termination_files_when_tuning_succeeds(tempdir)
        _validate_training_output(tempdir, "lora")

        for _, dirs, _ in os.walk(tempdir, topdown=False):
            for name in dirs:
                if "checkpoint-" in name.lower():
                    new_embeddings_file_path = os.path.join(
                        tempdir, name, "new_embeddings.safetensors"
                    )
                    assert os.path.exists(new_embeddings_file_path)

        # check for new_embeddings.safetensors
        new_embeddings_file_path = os.path.join(tempdir, "new_embeddings.safetensors")
        assert os.path.exists(new_embeddings_file_path)


def test_bad_script_path():
    """Check for appropriate error for an invalid training script location"""
    with tempfile.TemporaryDirectory() as tempdir:
        setup_env(tempdir)
        TRAIN_KWARGS = {**BASE_PEFT_KWARGS, **{"output_dir": tempdir}}
        serialized_args = serialize_args(TRAIN_KWARGS)
        os.environ["SFT_TRAINER_CONFIG_JSON_ENV_VAR"] = serialized_args
        os.environ["TRAINING_SCRIPT"] = "/not/here"

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            main()
        assert pytest_wrapped_e.type == SystemExit
        assert pytest_wrapped_e.value.code == INTERNAL_ERROR_EXIT_CODE
        assert os.stat(tempdir + "/termination-log").st_size > 0


def test_blank_env_var():
    with tempfile.TemporaryDirectory() as tempdir:
        setup_env(tempdir)
        os.environ["SFT_TRAINER_CONFIG_JSON_ENV_VAR"] = ""
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            main()
        assert pytest_wrapped_e.type == SystemExit
        assert pytest_wrapped_e.value.code == USER_ERROR_EXIT_CODE
        assert os.stat(tempdir + "/termination-log").st_size > 0


def test_faulty_file_path():
    with tempfile.TemporaryDirectory() as tempdir:
        setup_env(tempdir)
        faulty_path = os.path.join(tempdir, "non_existent_file.pkl")
        TRAIN_KWARGS = {
            **BASE_PEFT_KWARGS,
            **{"training_data_path": faulty_path, "output_dir": tempdir},
        }
        serialized_args = serialize_args(TRAIN_KWARGS)
        os.environ["SFT_TRAINER_CONFIG_JSON_ENV_VAR"] = serialized_args
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            main()
        assert pytest_wrapped_e.type == SystemExit
        assert pytest_wrapped_e.value.code == USER_ERROR_EXIT_CODE
        assert os.stat(tempdir + "/termination-log").st_size > 0


def test_bad_base_model_path():
    with tempfile.TemporaryDirectory() as tempdir:
        setup_env(tempdir)
        TRAIN_KWARGS = {
            **BASE_PEFT_KWARGS,
            **{"model_name_or_path": "/wrong/path"},
        }
        serialized_args = serialize_args(TRAIN_KWARGS)
        os.environ["SFT_TRAINER_CONFIG_JSON_ENV_VAR"] = serialized_args
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            main()
        assert pytest_wrapped_e.type == SystemExit
        assert pytest_wrapped_e.value.code == USER_ERROR_EXIT_CODE
        assert os.stat(tempdir + "/termination-log").st_size > 0


def test_config_parsing_error():
    with tempfile.TemporaryDirectory() as tempdir:
        setup_env(tempdir)
        TRAIN_KWARGS = {
            **BASE_PEFT_KWARGS,
            **{"num_train_epochs": "five"},
        }  # Intentional type error
        serialized_args = serialize_args(TRAIN_KWARGS)
        os.environ["SFT_TRAINER_CONFIG_JSON_ENV_VAR"] = serialized_args
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            main()
        assert pytest_wrapped_e.type == SystemExit
        assert pytest_wrapped_e.value.code == USER_ERROR_EXIT_CODE
        assert os.stat(tempdir + "/termination-log").st_size > 0


def _validate_termination_files_when_tuning_succeeds(base_dir):
    # check termination log and .complete files
    assert os.path.exists(os.path.join(base_dir, "/termination-log")) is False
    assert os.path.exists(os.path.join(base_dir, ".complete")) is True
    assert (
        os.path.exists(
            os.path.join(base_dir, FileLoggingTrackerConfig.training_logs_filename)
        )
        is True
    )


def _validate_training_output(base_dir, tuning_technique):
    if tuning_technique == "ft":
        assert len(glob.glob(f"{base_dir}/model*.safetensors")) > 0
        assert os.path.exists(base_dir + "/adapter_config.json") is False
    else:
        assert os.path.exists(base_dir + "/adapter_config.json") is True
        assert os.path.exists(base_dir + "/adapter_model.safetensors") is True


def test_cleanup():
    # This runs to unset env variables that could disrupt other tests
    cleanup_env()
    assert True
