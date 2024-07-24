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
import base64
import os
import pickle

# Third Party
from peft import LoraConfig, PromptTuningConfig
import pytest

# First Party
from tests.build.test_utils import HAPPY_PATH_DUMMY_CONFIG_PATH

# Local
from tuning.config import peft_config
from tuning.utils import config_utils


def test_get_hf_peft_config_returns_None_for_FT():
    expected_config = None
    assert expected_config == config_utils.get_hf_peft_config("", None, "")


def test_get_hf_peft_config_returns_Lora_config_correctly():
    # Test that when a value is not defined, the default values are used
    # Default values: r=8, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"]
    tuning_config = peft_config.LoraConfig(r=3, lora_alpha=3)

    config = config_utils.get_hf_peft_config("CAUSAL_LM", tuning_config, "")
    assert isinstance(config, LoraConfig)
    assert config.task_type == "CAUSAL_LM"
    assert config.r == 3
    assert config.lora_alpha == 3
    assert config.lora_dropout == 0.05  # default value from peft_config.LoraConfig
    assert config.target_modules == {
        "q_proj",
        "v_proj",
    }  # default value from peft_config.LoraConfig

    # Test that when target_modules is ["all-linear"], we convert it to str type "all-linear"
    tuning_config = peft_config.LoraConfig(r=234, target_modules=["all-linear"])

    config = config_utils.get_hf_peft_config("CAUSAL_LM", tuning_config, "")
    assert isinstance(config, LoraConfig)
    assert config.r == 234
    assert config.target_modules == "all-linear"
    assert config.lora_dropout == 0.05  # default value from peft_config.LoraConfig


def test_get_hf_peft_config_returns_PT_config_correctly():
    # Test that the prompt tuning config is set properly for each field
    # when a value is not defined, the default values are used
    # Default values:
    # prompt_tuning_init="TEXT",
    # prompt_tuning_init_text="Classify if the tweet is a complaint or not:"
    tuning_config = peft_config.PromptTuningConfig(num_virtual_tokens=12)

    config = config_utils.get_hf_peft_config("CAUSAL_LM", tuning_config, "foo/bar/path")
    assert isinstance(config, PromptTuningConfig)
    assert config.task_type == "CAUSAL_LM"
    assert config.prompt_tuning_init == "TEXT"
    assert config.num_virtual_tokens == 12
    assert (
        config.prompt_tuning_init_text == "Classify if the tweet is a complaint or not:"
    )
    assert config.tokenizer_name_or_path == "foo/bar/path"

    # Test that tokenizer path is allowed to be None only when prompt_tuning_init is not TEXT
    tuning_config = peft_config.PromptTuningConfig(prompt_tuning_init="RANDOM")
    config = config_utils.get_hf_peft_config(None, tuning_config, None)
    assert isinstance(config, PromptTuningConfig)
    assert config.tokenizer_name_or_path is None

    tuning_config = peft_config.PromptTuningConfig(prompt_tuning_init="TEXT")
    with pytest.raises(ValueError) as err:
        config_utils.get_hf_peft_config(None, tuning_config, None)
        assert "tokenizer_name_or_path can't be None" in err.value


def test_create_tuning_config():
    # Test that LoraConfig is created for peft_method Lora
    # and fields are set properly
    tune_config = config_utils.create_tuning_config("lora", foo="x", r=234)
    assert isinstance(tune_config, peft_config.LoraConfig)
    assert tune_config.r == 234
    assert tune_config.lora_alpha == 32
    assert tune_config.lora_dropout == 0.05

    # Test that PromptTuningConfig is created for peft_method pt
    # and fields are set properly
    tune_config = config_utils.create_tuning_config(
        "pt", foo="x", prompt_tuning_init="RANDOM"
    )
    assert isinstance(tune_config, peft_config.PromptTuningConfig)
    assert tune_config.prompt_tuning_init == "RANDOM"

    # Test that None is created for peft_method "None" or None
    # and fields are set properly
    tune_config = config_utils.create_tuning_config("None", foo="x")
    assert tune_config is None

    tune_config = config_utils.create_tuning_config(None, foo="x")
    assert tune_config is None

    # Test that this function does not recognize any other peft_method
    with pytest.raises(AssertionError) as err:
        tune_config = config_utils.create_tuning_config("hello", foo="x")
        assert err.value == "peft config hello not defined in peft.py"


def test_update_config_can_handle_dot_for_nested_field():
    # Test update_config allows nested field
    config = peft_config.LoraConfig(r=5)
    assert config.lora_alpha == 32  # default value is 32

    # update lora_alpha to 98
    kwargs = {"LoraConfig.lora_alpha": 98}
    config_utils.update_config(config, **kwargs)
    assert config.lora_alpha == 98

    # update an unknown field
    kwargs = {"LoraConfig.foobar": 98}
    config_utils.update_config(config, **kwargs)  # nothing happens


def test_update_config_can_handle_multiple_config_updates():
    # update a tuple of configs
    config = (peft_config.LoraConfig(r=5), peft_config.LoraConfig(r=7))
    kwargs = {"r": 98}
    config_utils.update_config(config, **kwargs)
    assert config[0].r == 98
    assert config[1].r == 98


def test_get_json_config_can_load_from_path_or_envvar():
    # Load from path
    if "SFT_TRAINER_CONFIG_JSON_ENV_VAR" in os.environ:
        del os.environ["SFT_TRAINER_CONFIG_JSON_ENV_VAR"]
    os.environ["SFT_TRAINER_CONFIG_JSON_PATH"] = HAPPY_PATH_DUMMY_CONFIG_PATH

    job_config = config_utils.get_json_config()
    assert job_config is not None
    assert job_config["model_name_or_path"] == "bigscience/bloom-560m"

    # Load from envvar
    config_json = {"model_name_or_path": "foobar"}
    message_bytes = pickle.dumps(config_json)
    base64_bytes = base64.b64encode(message_bytes)
    encoded_json = base64_bytes.decode("ascii")
    os.environ["SFT_TRAINER_CONFIG_JSON_ENV_VAR"] = encoded_json

    job_config = config_utils.get_json_config()
    assert job_config is not None
    assert job_config["model_name_or_path"] == "foobar"
