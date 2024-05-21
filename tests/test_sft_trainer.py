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

"""Unit Tests for SFT Trainer.
"""

# Standard
import copy
import json
import os
import tempfile

# Third Party
from datasets.exceptions import DatasetGenerationError
import pytest
import torch
import transformers

# First Party
from scripts.run_inference import TunedCausalLM
from tests.data import EMPTY_DATA, MALFORMATTED_DATA, TWITTER_COMPLAINTS_DATA
from tests.helpers import causal_lm_train_kwargs

# Local
from tuning import sft_trainer
from tuning.config import peft_config

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

BASE_LORA_KWARGS = copy.deepcopy(BASE_PEFT_KWARGS)
BASE_LORA_KWARGS["peft_method"] = "lora"

BASE_FT_KWARGS = copy.deepcopy(BASE_PEFT_KWARGS)
BASE_FT_KWARGS["peft_method"] = None
del BASE_FT_KWARGS["prompt_tuning_init"]
del BASE_FT_KWARGS["prompt_tuning_init_text"]


def test_helper_causal_lm_train_kwargs():
    """Check happy path kwargs passed and parsed properly."""
    model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
        BASE_PEFT_KWARGS
    )

    assert model_args.model_name_or_path == MODEL_NAME
    assert model_args.use_flash_attn is False
    assert model_args.torch_dtype == "float32"

    assert data_args.training_data_path == TWITTER_COMPLAINTS_DATA
    assert data_args.response_template == "\n### Label:"
    assert data_args.dataset_text_field == "output"

    assert training_args.num_train_epochs == 5
    assert training_args.max_seq_length == 4096
    assert training_args.save_strategy == "epoch"

    assert tune_config.prompt_tuning_init == "RANDOM"
    assert tune_config.prompt_tuning_init_text == "hello"
    assert tune_config.tokenizer_name_or_path == MODEL_NAME
    assert tune_config.num_virtual_tokens == 8

    model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
        BASE_FT_KWARGS
    )
    assert tune_config is None
    model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
        BASE_LORA_KWARGS
    )
    assert isinstance(tune_config, peft_config.LoraConfig)


def test_run_train_requires_output_dir():
    """Check fails when output dir not provided."""
    updated_output_dir = copy.deepcopy(BASE_PEFT_KWARGS)
    updated_output_dir["output_dir"] = None
    model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
        updated_output_dir
    )
    with pytest.raises(TypeError):
        sft_trainer.train(model_args, data_args, training_args, tune_config)


def test_run_train_fails_training_data_path_not_exist():
    """Check fails when data path not found."""
    updated_output_path = copy.deepcopy(BASE_PEFT_KWARGS)
    updated_output_path["training_data_path"] = "fake/path"
    model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
        updated_output_path
    )
    with pytest.raises(FileNotFoundError):
        sft_trainer.train(model_args, data_args, training_args, tune_config)


############################# Prompt Tuning Tests #############################


def test_run_causallm_pt_and_inference():
    """Check if we can bootstrap and peft tune causallm models"""
    with tempfile.TemporaryDirectory() as tempdir:
        TRAIN_KWARGS = {**BASE_PEFT_KWARGS, **{"output_dir": tempdir}}

        model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
            TRAIN_KWARGS
        )
        sft_trainer.train(model_args, data_args, training_args, tune_config)

        # validate peft tuning configs
        _validate_training(tempdir)
        checkpoint_path = _get_checkpoint_path(tempdir)
        adapter_config = _get_adapter_config(checkpoint_path)
        _validate_adapter_config(adapter_config, "PROMPT_TUNING", BASE_PEFT_KWARGS)

        # Load the model
        loaded_model = TunedCausalLM.load(checkpoint_path)

        # Run inference on the text
        output_inference = loaded_model.run(
            "### Text: @NortonSupport Thanks much.\n\n### Label:", max_new_tokens=50
        )
        assert len(output_inference) > 0
        assert "### Text: @NortonSupport Thanks much.\n\n### Label:" in output_inference


def test_run_causallm_pt_init_text():
    """Check if we can bootstrap and peft tune causallm models with init text as 'TEXT'"""
    with tempfile.TemporaryDirectory() as tempdir:
        TRAIN_KWARGS = {
            **BASE_PEFT_KWARGS,
            **{"output_dir": tempdir, "prompt_tuning_init": "TEXT"},
        }
        model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
            TRAIN_KWARGS
        )
        sft_trainer.train(model_args, data_args, training_args, tune_config)

        # validate peft tuning configs
        _validate_training(tempdir)
        checkpoint_path = _get_checkpoint_path(tempdir)
        adapter_config = _get_adapter_config(checkpoint_path)
        _validate_adapter_config(adapter_config, "PROMPT_TUNING", TRAIN_KWARGS)


invalid_params_map = [
    ("num_train_epochs", 0, "num_train_epochs has to be an integer/float >= 1"),
    (
        "gradient_accumulation_steps",
        0,
        "gradient_accumulation_steps has to be an integer >= 1",
    ),
]


@pytest.mark.parametrize(
    "param_name,param_val,exc_msg",
    invalid_params_map,
    ids=["num_train_epochs", "grad_acc_steps"],
)
def test_run_causallm_pt_invalid_params(param_name, param_val, exc_msg):
    """Check if error is raised when invalid params are used to peft tune causallm models"""
    with tempfile.TemporaryDirectory() as tempdir:
        invalid_params = copy.deepcopy(BASE_PEFT_KWARGS)
        invalid_params["output_dir"] = tempdir
        invalid_params[param_name] = param_val
        model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
            invalid_params
        )

        with pytest.raises(ValueError, match=exc_msg):
            sft_trainer.train(model_args, data_args, training_args, tune_config)


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
        _validate_training(tempdir, check_eval=True)


############################# Lora Tests #############################

target_modules_val_map = [
    (None, ["q_proj", "v_proj"]),
    (
        ["q_proj", "k_proj", "v_proj", "o_proj"],
        ["q_proj", "k_proj", "v_proj", "o_proj"],
    ),
    (
        ["all-linear"],
        ["o_proj", "q_proj", "gate_proj", "down_proj", "k_proj", "up_proj", "v_proj"],
    ),
]


@pytest.mark.parametrize(
    "target_modules,expected",
    target_modules_val_map,
    ids=["default", "custom_target_modules", "all_linear_target_modules"],
)
def test_run_causallm_lora_and_inference(request, target_modules, expected):
    """Check if we can bootstrap and lora tune causallm models"""
    with tempfile.TemporaryDirectory() as tempdir:
        base_lora_kwargs = copy.deepcopy(BASE_LORA_KWARGS)
        base_lora_kwargs["output_dir"] = tempdir
        if "default" not in request._pyfuncitem.callspec.id:
            base_lora_kwargs["target_modules"] = target_modules

        model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
            base_lora_kwargs
        )
        sft_trainer.train(model_args, data_args, training_args, tune_config)

        # validate lora tuning configs
        _validate_training(tempdir)
        checkpoint_path = _get_checkpoint_path(tempdir)
        adapter_config = _get_adapter_config(checkpoint_path)
        _validate_adapter_config(adapter_config, "LORA", base_lora_kwargs)

        for module in expected:
            assert module in adapter_config.get("target_modules")

        # Load the model
        loaded_model = TunedCausalLM.load(checkpoint_path)

        # Run inference on the text
        output_inference = loaded_model.run(
            "Simply put, the theory of relativity states that ", max_new_tokens=50
        )
        assert len(output_inference) > 0
        assert "Simply put, the theory of relativity states that" in output_inference


############################# Finetuning Tests #############################


def test_run_causallm_ft_and_inference():
    """Check if we can bootstrap and finetune tune causallm models"""
    with tempfile.TemporaryDirectory() as tempdir:
        BASE_FT_KWARGS["output_dir"] = tempdir
        model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
            BASE_FT_KWARGS
        )
        # Just assuring no tuning config is passed for PT or LoRA
        assert tune_config is None

        sft_trainer.train(model_args, data_args, training_args, tune_config)

        # validate ft tuning configs
        _validate_training(tempdir)
        checkpoint_path = _get_checkpoint_path(tempdir)

        # Load the model
        loaded_model = TunedCausalLM.load(checkpoint_path)

        # Run inference on the text
        output_inference = loaded_model.run(
            "### Text: @NortonSupport Thanks much.\n\n### Label:", max_new_tokens=50
        )
        assert len(output_inference) > 0
        assert "### Text: @NortonSupport Thanks much.\n\n### Label:" in output_inference


############################# Helper functions #############################
def _validate_training(tempdir, check_eval=False):
    assert any(x.startswith("checkpoint-") for x in os.listdir(tempdir))
    train_logs_file_path = "{}/training_logs.jsonl".format(tempdir)
    train_log_contents = ""
    with open(train_logs_file_path, encoding="utf-8") as f:
        train_log_contents = f.read()

    assert os.path.exists(train_logs_file_path) is True
    assert os.path.getsize(train_logs_file_path) > 0
    assert "training_loss" in train_log_contents

    if check_eval:
        assert "validation_loss" in train_log_contents


def _get_checkpoint_path(dir_path):
    return os.path.join(dir_path, "checkpoint-5")


def _get_adapter_config(dir_path):
    with open(os.path.join(dir_path, "adapter_config.json"), encoding="utf-8") as f:
        return json.load(f)


def _validate_adapter_config(adapter_config, peft_type, base_kwargs):
    assert adapter_config.get("task_type") == "CAUSAL_LM"
    assert adapter_config.get("peft_type") == peft_type
    assert (
        (
            adapter_config.get("tokenizer_name_or_path")
            == base_kwargs["tokenizer_name_or_path"]
        )
        if peft_type == "PROMPT_TUNING"
        else True
    )


############################# Other Tests #############################
### Tests for a variety of edge cases and potentially problematic cases;
# some of these test directly test validation within external dependencies
# and validate errors that we expect to get from them which might be unintuitive.
# In such cases, it would probably be best for us to handle these things directly
# for better error messages, etc.

### Tests related to tokenizer configuration
def test_tokenizer_has_no_eos_token():
    """Ensure that if the model has no EOS token, it sets the default before formatting."""
    # This is a bit roundabout, but patch the tokenizer and export it and the model to a tempdir
    # that we can then reload out of for the train call, and clean up afterwards.
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        BASE_PEFT_KWARGS["model_name_or_path"]
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        BASE_PEFT_KWARGS["model_name_or_path"]
    )
    tokenizer.eos_token = None
    with tempfile.TemporaryDirectory() as tempdir:
        tokenizer.save_pretrained(tempdir)
        model.save_pretrained(tempdir)
        TRAIN_KWARGS = {
            **BASE_PEFT_KWARGS,
            **{"model_name_or_path": tempdir, "output_dir": tempdir},
        }
        model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
            TRAIN_KWARGS
        )
        # If we handled this badly, we would probably get something like a
        # TypeError: can only concatenate str (not "NoneType") to str error
        # when we go to apply the data formatter.
        sft_trainer.train(model_args, data_args, training_args, tune_config)
        _validate_training(tempdir)


### Tests for Bad dataset specification, i.e., data is valid, but the field we point it at isn't
def test_invalid_dataset_text_field():
    """Ensure that if we specify a dataset_text_field that doesn't exist, we get a KeyError."""
    TRAIN_KWARGS = {
        **BASE_PEFT_KWARGS,
        **{"dataset_text_field": "not found", "output_dir": "foo/bar/baz"},
    }
    model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
        TRAIN_KWARGS
    )
    with pytest.raises(KeyError):
        sft_trainer.train(model_args, data_args, training_args, tune_config)


### Tests for bad training data (i.e., data_path is an unhappy value or points to an unhappy thing)
def test_malformatted_data():
    """Ensure that malformatted data explodes due to failure to generate the dataset."""
    TRAIN_KWARGS = {
        **BASE_PEFT_KWARGS,
        **{"training_data_path": MALFORMATTED_DATA, "output_dir": "foo/bar/baz"},
    }
    model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
        TRAIN_KWARGS
    )
    with pytest.raises(DatasetGenerationError):
        sft_trainer.train(model_args, data_args, training_args, tune_config)


def test_empty_data():
    """Ensure that malformatted data explodes due to failure to generate the dataset."""
    TRAIN_KWARGS = {
        **BASE_PEFT_KWARGS,
        **{"training_data_path": EMPTY_DATA, "output_dir": "foo/bar/baz"},
    }
    model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
        TRAIN_KWARGS
    )
    with pytest.raises(DatasetGenerationError):
        sft_trainer.train(model_args, data_args, training_args, tune_config)


def test_data_path_is_a_directory():
    """Ensure that we get FileNotFoundError if we point the data path at a dir, not a file."""
    with tempfile.TemporaryDirectory() as tempdir:
        TRAIN_KWARGS = {
            **BASE_PEFT_KWARGS,
            **{"training_data_path": tempdir, "output_dir": tempdir},
        }
        model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
            TRAIN_KWARGS
        )
        # Confusingly, if we pass a directory for our data path, it will throw a
        # FileNotFoundError saying "unable to find '<data_path>'", since it can't
        # find a matchable file in the path.
        with pytest.raises(FileNotFoundError):
            sft_trainer.train(model_args, data_args, training_args, tune_config)


### Tests for bad tuning module configurations
def test_run_causallm_lora_with_invalid_modules():
    """Check that we throw a value error if the target modules for lora don't exist."""
    with tempfile.TemporaryDirectory() as tempdir:
        TRAIN_KWARGS = {
            **BASE_PEFT_KWARGS,
            **{"peft_method": "lora", "output_dir": tempdir},
        }
        model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
            TRAIN_KWARGS
        )
        # Defaults are q_proj / v_proj; this will fail lora as the torch module doesn't have them
        tune_config.target_modules = ["foo", "bar"]
        # Peft should throw a value error about modules not matching the base module
        with pytest.raises(ValueError):
            sft_trainer.train(model_args, data_args, training_args, tune_config)


### Direct validation tests based on whether or not packing is enabled
def test_no_packing_needs_dataset_text_field():
    """Ensure we need to set the dataset text field if packing is False"""
    with tempfile.TemporaryDirectory() as tempdir:
        TRAIN_KWARGS = {
            **BASE_PEFT_KWARGS,
            **{"dataset_text_field": None, "output_dir": tempdir},
        }
        model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
            TRAIN_KWARGS
        )
        with pytest.raises(ValueError):
            sft_trainer.train(model_args, data_args, training_args, tune_config)


# TODO: Fix this case
@pytest.mark.skip(reason="currently crashes before validation is done")
def test_no_packing_needs_reponse_template():
    """Ensure we need to set the response template if packing is False"""
    with tempfile.TemporaryDirectory() as tempdir:
        TRAIN_KWARGS = {
            **BASE_PEFT_KWARGS,
            **{"response_template": None, "output_dir": tempdir},
        }
        model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
            TRAIN_KWARGS
        )
        with pytest.raises(ValueError):
            sft_trainer.train(model_args, data_args, training_args, tune_config)


### Tests for model dtype edge cases
@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    reason="Only runs if bf16 is unsupported",
)
def test_bf16_still_tunes_if_unsupported():
    """Ensure that even if bf16 is not supported, tuning still works without problems."""
    assert not torch.cuda.is_bf16_supported()
    with tempfile.TemporaryDirectory() as tempdir:
        TRAIN_KWARGS = {
            **BASE_PEFT_KWARGS,
            **{"torch_dtype": "bfloat16", "output_dir": tempdir},
        }
        model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
            TRAIN_KWARGS
        )
        sft_trainer.train(model_args, data_args, training_args, tune_config)
        _validate_training(tempdir)


def test_bad_torch_dtype():
    """Ensure that specifying an invalid torch dtype yields a ValueError."""
    with tempfile.TemporaryDirectory() as tempdir:
        TRAIN_KWARGS = {
            **BASE_PEFT_KWARGS,
            **{"torch_dtype": "not a type", "output_dir": tempdir},
        }
        model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
            TRAIN_KWARGS
        )
        with pytest.raises(ValueError):
            sft_trainer.train(model_args, data_args, training_args, tune_config)
