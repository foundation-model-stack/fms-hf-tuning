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
import logging
import os
import pickle

# Third Party
from datasets import Dataset, Features, Value
from peft import LoraConfig, PromptTuningConfig
import pytest

# First Party
from tests.build.test_utils import HAPPY_PATH_DUMMY_CONFIG_PATH

# Local
from tuning.config import peft_config
from tuning.utils import config_utils, utils


def test_get_hf_peft_config_returns_None_for_tuning_config_None():
    """Test that when tuning_config is None, the function returns None"""
    expected_config = None
    assert expected_config == config_utils.get_hf_peft_config("", None, "")


def test_get_hf_peft_config_returns_lora_config_correctly():
    """Test that tuning_config fields are passed to LoraConfig correctly,
    If not defined, the default values are used
    """
    tuning_config = peft_config.LoraConfig(r=3, lora_alpha=3)

    config = config_utils.get_hf_peft_config("CAUSAL_LM", tuning_config, "")
    assert isinstance(config, LoraConfig)
    assert config.task_type == "CAUSAL_LM"
    assert config.r == 3
    assert config.lora_alpha == 3
    assert (
        config.lora_dropout == 0.05
    )  # default value from local peft_config.LoraConfig
    assert (
        config.target_modules is None
    )  # default value from local peft_config.LoraConfig
    assert config.init_lora_weights is True  # default value from HF peft.LoraConfig
    assert (
        config.megatron_core == "megatron.core"
    )  # default value from HF peft.LoraConfig


def test_get_hf_peft_config_ignores_tokenizer_path_for_lora_config():
    """Test that if tokenizer is given with a LoraConfig, it is ignored"""
    tuning_config = peft_config.LoraConfig(r=3, lora_alpha=3)

    config = config_utils.get_hf_peft_config(
        task_type="CAUSAL_LM",
        tuning_config=tuning_config,
        tokenizer_name_or_path="foo/bar/path",
    )
    assert isinstance(config, LoraConfig)
    assert config.task_type == "CAUSAL_LM"
    assert config.r == 3
    assert config.lora_alpha == 3
    assert not hasattr(config, "tokenizer_name_or_path")


def test_get_hf_peft_config_returns_lora_config_with_correct_value_for_all_linear():
    """Test that when target_modules is ["all-linear"], we convert it to str type "all-linear" """
    tuning_config = peft_config.LoraConfig(r=234, target_modules=["all-linear"])

    config = config_utils.get_hf_peft_config("CAUSAL_LM", tuning_config, "")
    assert isinstance(config, LoraConfig)
    assert config.target_modules == "all-linear"


def test_get_hf_peft_config_returns_pt_config_correctly():
    """Test that the prompt tuning config is set properly for each field
    When a value is not defined, the default values are used
    """
    tuning_config = peft_config.PromptTuningConfig(num_virtual_tokens=12)

    config = config_utils.get_hf_peft_config("CAUSAL_LM", tuning_config, "foo/bar/path")
    assert isinstance(config, PromptTuningConfig)
    assert config.task_type == "CAUSAL_LM"
    assert (
        config.prompt_tuning_init == "TEXT"
    )  # default value from local peft_config.PromptTuningConfig
    assert config.num_virtual_tokens == 12
    assert (
        config.prompt_tuning_init_text == "Classify if the tweet is a complaint or not:"
    )  # default value from local peft_config.PromptTuningConfig
    assert config.tokenizer_name_or_path == "foo/bar/path"
    assert config.num_layers is None  # default value from HF peft.PromptTuningConfig
    assert (
        config.inference_mode is False
    )  # default value from HF peft.PromptTuningConfig


def test_get_hf_peft_config_returns_pt_config_with_correct_tokenizer_path():
    """Test that tokenizer path is allowed to be None only when prompt_tuning_init is not TEXT
    Reference:
    https://github.com/huggingface/peft/blob/main/src/peft/tuners/prompt_tuning/config.py#L73
    """

    # When prompt_tuning_init is not TEXT, we can pass in None for tokenizer path
    tuning_config = peft_config.PromptTuningConfig(prompt_tuning_init="RANDOM")
    config = config_utils.get_hf_peft_config(
        task_type=None, tuning_config=tuning_config, tokenizer_name_or_path=None
    )
    assert isinstance(config, PromptTuningConfig)
    assert config.tokenizer_name_or_path is None

    # When prompt_tuning_init is TEXT, exception is raised if tokenizer path is None
    tuning_config = peft_config.PromptTuningConfig(prompt_tuning_init="TEXT")
    with pytest.raises(ValueError) as err:
        config_utils.get_hf_peft_config(
            task_type=None, tuning_config=tuning_config, tokenizer_name_or_path=None
        )
        assert "tokenizer_name_or_path can't be None" in err.value


def test_create_tuning_config_for_peft_method_lora():
    """Test that LoraConfig is created for peft_method Lora
    and fields are set properly.
    If unknown fields are passed, they are ignored
    """
    tune_config = config_utils.create_tuning_config("lora", foo="x", r=234)
    assert isinstance(tune_config, peft_config.LoraConfig)
    assert tune_config.r == 234
    assert tune_config.lora_alpha == 32
    assert tune_config.lora_dropout == 0.05
    assert not hasattr(tune_config, "foo")


def test_create_tuning_config_for_peft_method_pt():
    """Test that PromptTuningConfig is created for peft_method pt
    and fields are set properly
    """
    tune_config = config_utils.create_tuning_config(
        "pt", foo="x", prompt_tuning_init="RANDOM"
    )
    assert isinstance(tune_config, peft_config.PromptTuningConfig)
    assert tune_config.prompt_tuning_init == "RANDOM"


def test_create_tuning_config_for_peft_method_none():
    """Test that PromptTuningConfig is created for peft_method "None" or None"""
    tune_config = config_utils.create_tuning_config("None")
    assert tune_config is None

    tune_config = config_utils.create_tuning_config(None)
    assert tune_config is None


def test_create_tuning_config_does_not_recognize_any_other_peft_method():
    """Test that PromptTuningConfig is created for peft_method "None" or None,
    "lora" or "pt", and no other
    """
    with pytest.raises(AssertionError) as err:
        config_utils.create_tuning_config("hello", foo="x")
        assert err.value == "peft config hello not defined in peft.py"


def test_update_config_can_handle_dot_for_nested_field():
    """Test that the function can read dotted field for kwargs fields"""
    config = peft_config.LoraConfig(r=5)
    assert config.lora_alpha == 32  # default value is 32

    # update lora_alpha to 98
    kwargs = {"LoraConfig.lora_alpha": 98}
    config_utils.update_config(config, **kwargs)
    assert config.lora_alpha == 98


def test_update_config_does_nothing_for_unknown_field():
    """Test that the function does not change other config
    field values if a kwarg field is unknown
    """
    # foobar is an unknown field
    config = peft_config.LoraConfig(r=5)
    kwargs = {"LoraConfig.foobar": 98}
    config_utils.update_config(config, **kwargs)  # nothing happens
    assert config.r == 5  # did not change r value
    assert not hasattr(config, "foobar")


def test_update_config_can_handle_multiple_config_updates():
    """Test that the function can handle a tuple of configs"""
    config = (peft_config.LoraConfig(r=5), peft_config.LoraConfig(r=7))
    kwargs = {"r": 98}
    config_utils.update_config(config, **kwargs)
    assert config[0].r == 98
    assert config[1].r == 98


def test_get_json_config_can_load_from_path():
    """Test that the function get_json_config can read
    the json path from env var SFT_TRAINER_CONFIG_JSON_PATH
    """
    if "SFT_TRAINER_CONFIG_JSON_ENV_VAR" in os.environ:
        del os.environ["SFT_TRAINER_CONFIG_JSON_ENV_VAR"]
    os.environ["SFT_TRAINER_CONFIG_JSON_PATH"] = HAPPY_PATH_DUMMY_CONFIG_PATH

    job_config = config_utils.get_json_config()
    assert job_config is not None
    assert job_config["model_name_or_path"] == "bigscience/bloom-560m"


def test_get_json_config_can_load_from_envvar():
    """Test that the function get_json_config can read
    the json path from env var SFT_TRAINER_CONFIG_JSON_ENV_VAR
    """
    config_json = {"model_name_or_path": "foobar"}
    message_bytes = pickle.dumps(config_json)
    base64_bytes = base64.b64encode(message_bytes)
    encoded_json = base64_bytes.decode("ascii")
    os.environ["SFT_TRAINER_CONFIG_JSON_ENV_VAR"] = encoded_json

    job_config = config_utils.get_json_config()
    assert job_config is not None
    assert job_config["model_name_or_path"] == "foobar"


def test_validate_datasets_logs_warnings_on_mismatch(caplog):
    """Test that `validate_mergeable_datasets` logs warnings when
    datasets have different columns or dtypes."""
    # Create a reference dataset with columns col1:int64 and col2:string
    ds1 = Dataset.from_dict(
        {"col1": [1, 2], "col2": ["hello", "world"]},
        features=Features({"col1": Value("int64"), "col2": Value("string")}),
    )

    # Create a second dataset with a different column set and a different dtype for col1
    ds2 = Dataset.from_dict(
        {"col1": [0.1, 0.2], "col3": ["hi", "there"]},
        features=Features({"col1": Value("float64"), "col3": Value("string")}),
    )

    with caplog.at_level(logging.WARNING):
        utils.validate_mergeable_datasets([ds1, ds2])

    assert (
        "different columns" in caplog.text
    ), "Expected a warning about differing columns."
    assert (
        "expected int64" in caplog.text
    ), "Expected a warning about mismatching column dtypes."
