# Copyright The IBM Tuning Team
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

"""Unit Tests for SFT Trainer.
"""

# Standard
import os
import tempfile
import pytest
import copy
import json

# First Party
from scripts.run_inference import TunedCausalLM
from tests.data import TWITTER_COMPLAINTS_DATA
from tests.helpers import causal_lm_train_kwargs

# Local
from tuning import sft_trainer

MODEL_NAME = "Maykeye/TinyLLama-v0"
BASE_PEFT_KWARGS = {
    "model_name_or_path": MODEL_NAME,
    "data_path": TWITTER_COMPLAINTS_DATA,
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
    "model_max_length": 4096,
    "peft_method": "pt",
    "prompt_tuning_init": "RANDOM",
    "num_virtual_tokens": 8,
    "prompt_tuning_init_text": "hello",
    "tokenizer_name_or_path": MODEL_NAME,
    "save_strategy": "epoch",
    "output_dir": "tmp",
}

BASE_LORA_KWARGS = copy.deepcopy(BASE_PEFT_KWARGS)
BASE_LORA_KWARGS["peft_method"] = "lora"

def test_helper_causal_lm_train_kwargs():
    """Check happy path kwargs passed and parsed properly."""
    model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
        BASE_PEFT_KWARGS
    )

    assert model_args.model_name_or_path == MODEL_NAME
    assert model_args.use_flash_attn == False
    assert model_args.torch_dtype == "float32"

    assert data_args.data_path == TWITTER_COMPLAINTS_DATA
    assert data_args.response_template == "\n### Label:"
    assert data_args.dataset_text_field == "output"

    assert training_args.num_train_epochs == 5
    assert training_args.model_max_length == 4096
    assert training_args.save_strategy == "epoch"

    assert tune_config.prompt_tuning_init == "RANDOM"
    assert tune_config.prompt_tuning_init_text == "hello"
    assert tune_config.tokenizer_name_or_path == MODEL_NAME
    assert tune_config.num_virtual_tokens == 8

def test_run_train_requires_output_dir():
    """Check fails when output dir not provided."""
    updated_output_dir = copy.deepcopy(BASE_PEFT_KWARGS)
    updated_output_dir["output_dir"] = None
    model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
        updated_output_dir
    )
    with pytest.raises(TypeError):
        sft_trainer.train(model_args, data_args, training_args, tune_config)

def test_run_train_fails_data_path_not_exist():
    """Check fails when data path not found."""
    updated_output_path = copy.deepcopy(BASE_PEFT_KWARGS)
    updated_output_path["data_path"] = "fake/path"
    model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
        updated_output_path
    )
    with pytest.raises(FileNotFoundError):
        sft_trainer.train(model_args, data_args, training_args, tune_config)

def test_run_causallm_pt_and_inference():
    """Check if we can bootstrap and peft tune causallm models"""
    with tempfile.TemporaryDirectory() as tempdir:
        BASE_PEFT_KWARGS["output_dir"] = tempdir
        model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
            BASE_PEFT_KWARGS
        )
        sft_trainer.train(model_args, data_args, training_args, tune_config)
        _validate_training(tempdir)

        # validate peft tuning configs
        checkpoint_path = os.path.join(tempdir, "checkpoint-5")
        adapter_config = _get_adapter_config(checkpoint_path)
        assert adapter_config.get("task_type") == "CAUSAL_LM"
        assert adapter_config.get("peft_type") == "PROMPT_TUNING"
        assert adapter_config.get("tokenizer_name_or_path") == BASE_PEFT_KWARGS["tokenizer_name_or_path"]

        # Load the model
        loaded_model = TunedCausalLM.load(checkpoint_path)

        # Run inference on the text
        output_inference = loaded_model.run("### Text: @NortonSupport Thanks much.\n\n### Label:", max_new_tokens=50)
        assert len(output_inference) > 0
        assert "### Text: @NortonSupport Thanks much.\n\n### Label:" in output_inference

def test_run_causallm_pt_with_validation():
    """Check if we can bootstrap and peft tune causallm models with validation dataset"""
    with tempfile.TemporaryDirectory() as tempdir:
        validation_peft = copy.deepcopy(BASE_PEFT_KWARGS)
        validation_peft["output_dir"] = tempdir
        validation_peft["validation_data_path"] = TWITTER_COMPLAINTS_DATA
        validation_peft["evaluation_strategy"] = "epoch"
        model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
            validation_peft
        )

        assert data_args.validation_data_path == TWITTER_COMPLAINTS_DATA

        sft_trainer.train(model_args, data_args, training_args, tune_config)
        _validate_training(tempdir)

        eval_loss_file_path = os.path.join(tempdir, "eval_loss.jsonl")
        assert os.path.exists(eval_loss_file_path)
        assert os.path.getsize(eval_loss_file_path) > 0

def test_run_causallm_lora_and_inference():
    """Check if we can bootstrap and lora tune causallm models"""
    with tempfile.TemporaryDirectory() as tempdir:
        BASE_LORA_KWARGS["output_dir"] = tempdir
        model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
            BASE_LORA_KWARGS
        )
        sft_trainer.train(model_args, data_args, training_args, tune_config)
        _validate_training(tempdir)

        # validate peft tuning configs
        checkpoint_path = os.path.join(tempdir, "checkpoint-5")
        adapter_config = _get_adapter_config(checkpoint_path)
        assert adapter_config.get("task_type") == "CAUSAL_LM"
        assert adapter_config.get("peft_type") == "LORA"
        for module in ["q_proj", "v_proj"]: # default target_modules used
            assert module in adapter_config.get("target_modules")

        # Load the model
        loaded_model = TunedCausalLM.load(checkpoint_path)

        # Run inference on the text
        output_inference = loaded_model.run("Simply put, the theory of relativity states that ", max_new_tokens=50)
        assert len(output_inference) > 0
        assert "Simply put, the theory of relativity states that" in output_inference

def test_run_train_lora_target_modules():
    """Check runs lora tuning with given list of target modules."""
    with tempfile.TemporaryDirectory() as tempdir:
        lora_target_modules = copy.deepcopy(BASE_LORA_KWARGS)
        lora_target_modules["output_dir"] = tempdir
        lora_target_modules["target_modules"] = ["q_proj","k_proj","v_proj","o_proj"]

        model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
            lora_target_modules
        )
        sft_trainer.train(model_args, data_args, training_args, tune_config)
        _validate_training(tempdir)

        checkpoint_path = os.path.join(tempdir, "checkpoint-5")
        adapter_config = _get_adapter_config(checkpoint_path)
        for module in lora_target_modules["target_modules"]:
            assert module in adapter_config.get("target_modules")

def test_run_train_lora_target_modules_all_linear():
    """Check runs lora tuning with all linear target modules."""
    with tempfile.TemporaryDirectory() as tempdir:
        lora_target_modules = copy.deepcopy(BASE_LORA_KWARGS)
        lora_target_modules["output_dir"] = tempdir
        lora_target_modules["target_modules"] = ["all-linear"]

        model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
            lora_target_modules
        )
        sft_trainer.train(model_args, data_args, training_args, tune_config)
        _validate_training(tempdir)

        checkpoint_path = os.path.join(tempdir, "checkpoint-5")
        adapter_config = _get_adapter_config(checkpoint_path)
        llama_expected_modules = ["o_proj", "q_proj", "gate_proj", "down_proj", "k_proj", "up_proj", "v_proj"]
        for module in llama_expected_modules:
            assert module in adapter_config.get("target_modules")

def _validate_training(tempdir):
    assert any(x.startswith("checkpoint-") for x in os.listdir(tempdir))
    loss_file_path = "{}/train_loss.jsonl".format(tempdir)
    assert os.path.exists(loss_file_path)
    assert os.path.getsize(loss_file_path) > 0

def _get_adapter_config(dir_path):
    with open(os.path.join(dir_path, "adapter_config.json")) as f:
        return json.load(f)