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
import base64
import os
import pickle
import tempfile

# Third Party
import pytest

# First Party
from build.accelerate_launch import main
from build.utils import INTERNAL_ERROR_EXIT_CODE, USER_ERROR_EXIT_CODE
from tests.data import TWITTER_COMPLAINTS_DATA

SCRIPT = "build/launch_training.py"
MODEL_NAME = "Maykeye/TinyLLama-v0"
BASE_PEFT_KWARGS = {
    "model_name_or_path": MODEL_NAME,
    "training_data_path": TWITTER_COMPLAINTS_DATA,
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
    "peft_method": "pt",
    "prompt_tuning_init": "RANDOM",
    "num_virtual_tokens": 8,
    "prompt_tuning_init_text": "hello",
    "tokenizer_name_or_path": MODEL_NAME,
    "save_strategy": "epoch",
    "output_dir": "tmp",
}


def serialize_args(args_json):
    message_bytes = pickle.dumps(args_json)
    base64_bytes = base64.b64encode(message_bytes)
    return base64_bytes.decode("ascii")


def setup_env(tempdir):
    os.environ["LAUNCH_TRAINING_SCRIPT"] = SCRIPT
    os.environ["PYTHONPATH"] = "./:$PYTHONPATH"
    os.environ["TERMINATION_LOG_FILE"] = tempdir + "/termination-log"


def cleanup_env():
    os.environ.pop("LAUNCH_TRAINING_SCRIPT", None)
    os.environ.pop("PYTHONPATH", None)
    os.environ.pop("TERMINATION_LOG_FILE", None)


def test_successful_pt():
    """Check if we can bootstrap and peft tune causallm models"""
    with tempfile.TemporaryDirectory() as tempdir:
        setup_env(tempdir)
        TRAIN_KWARGS = {**BASE_PEFT_KWARGS, **{"output_dir": tempdir}}
        serialized_args = serialize_args(TRAIN_KWARGS)
        os.environ["SFT_TRAINER_CONFIG_JSON_ENV_VAR"] = serialized_args

        assert main() == 0
        assert os.path.exists(tempdir + "/termination-log") is False


def test_bad_script_path():
    """Check for appropriate error for an invalid training script location"""
    with tempfile.TemporaryDirectory() as tempdir:
        setup_env(tempdir)
        TRAIN_KWARGS = {**BASE_PEFT_KWARGS, **{"output_dir": tempdir}}
        serialized_args = serialize_args(TRAIN_KWARGS)
        os.environ["SFT_TRAINER_CONFIG_JSON_ENV_VAR"] = serialized_args
        os.environ["LAUNCH_TRAINING_SCRIPT"] = "/not/here"

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


def test_cleanup():
    # This runs to unset env variables that could disrupt other tests
    cleanup_env()
    assert True
