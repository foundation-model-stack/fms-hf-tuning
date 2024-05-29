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
from tests.data import (
    EMPTY_DATA,
    MALFORMATTED_DATA,
    TWITTER_COMPLAINTS_DATA,
    TWITTER_COMPLAINTS_JSON_FORMAT,
)

# Local
from tuning import sft_trainer
from tuning.config import configs, peft_config

MODEL_NAME = "Maykeye/TinyLLama-v0"
MODEL_ARGS = configs.ModelArguments(
    model_name_or_path=MODEL_NAME, use_flash_attn=False, torch_dtype="float32"
)
DATA_ARGS = configs.DataArguments(
    training_data_path=TWITTER_COMPLAINTS_DATA,
    response_template="\n### Label:",
    dataset_text_field="output",
)
TRAIN_ARGS = configs.TrainingArguments(
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=0.00001,
    weight_decay=0,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=1,
    include_tokens_per_second=True,
    packing=False,
    max_seq_length=4096,
    save_strategy="epoch",
    output_dir="tmp",
)
PEFT_PT_ARGS = peft_config.PromptTuningConfig(
    prompt_tuning_init="RANDOM",
    num_virtual_tokens=8,
    prompt_tuning_init_text="hello",
    tokenizer_name_or_path=MODEL_NAME,
)

PEFT_LORA_ARGS = peft_config.LoraConfig(r=8, lora_alpha=32, lora_dropout=0.05)


def test_run_train_requires_output_dir():
    """Check fails when output dir not provided."""
    updated_output_dir_train_args = copy.deepcopy(TRAIN_ARGS)
    updated_output_dir_train_args.output_dir = None
    with pytest.raises(TypeError):
        sft_trainer.train(MODEL_ARGS, DATA_ARGS, updated_output_dir_train_args, None)


def test_run_train_fails_training_data_path_not_exist():
    """Check fails when data path not found."""
    updated_data_path_args = copy.deepcopy(DATA_ARGS)
    updated_data_path_args.training_data_path = "fake/path"
    with pytest.raises(FileNotFoundError):
        sft_trainer.train(MODEL_ARGS, updated_data_path_args, TRAIN_ARGS, None)


############################# Prompt Tuning Tests #############################


def test_run_causallm_pt_and_inference():
    """Check if we can bootstrap and peft tune causallm models"""
    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir

        sft_trainer.train(MODEL_ARGS, DATA_ARGS, train_args, PEFT_PT_ARGS)

        # validate peft tuning configs
        _validate_training(tempdir)
        checkpoint_path = _get_checkpoint_path(tempdir)
        adapter_config = _get_adapter_config(checkpoint_path)
        _validate_adapter_config(adapter_config, "PROMPT_TUNING", PEFT_PT_ARGS)

        # Load the model
        loaded_model = TunedCausalLM.load(checkpoint_path)

        # Run inference on the text
        output_inference = loaded_model.run(
            "### Text: @NortonSupport Thanks much.\n\n### Label:", max_new_tokens=50
        )
        assert len(output_inference) > 0
        assert "### Text: @NortonSupport Thanks much.\n\n### Label:" in output_inference


def test_run_causallm_pt_and_inference_with_formatting_data():
    """Check if we can bootstrap and peft tune causallm models
    This test needs the trainer to format data to a single sequence internally.
    """
    with tempfile.TemporaryDirectory() as tempdir:
        data_formatting_args = copy.deepcopy(DATA_ARGS)
        data_formatting_args.dataset_text_field = None
        data_formatting_args.data_formatter_template = (
            "### Text: {{Tweet text}} \n\n### Label: {{text_label}}"
        )

        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir

        sft_trainer.train(MODEL_ARGS, data_formatting_args, train_args, PEFT_PT_ARGS)

        # validate peft tuning configs
        _validate_training(tempdir)
        checkpoint_path = _get_checkpoint_path(tempdir)
        adapter_config = _get_adapter_config(checkpoint_path)
        _validate_adapter_config(adapter_config, "PROMPT_TUNING", PEFT_PT_ARGS)

        # Load the model
        loaded_model = TunedCausalLM.load(checkpoint_path)

        # Run inference on the text
        output_inference = loaded_model.run(
            "### Text: @NortonSupport Thanks much.\n\n### Label:", max_new_tokens=50
        )
        assert len(output_inference) > 0
        assert "### Text: @NortonSupport Thanks much.\n\n### Label:" in output_inference


def test_run_causallm_pt_and_inference_JSON_file_formatter():
    """Check if we can bootstrap and peft tune causallm models with JSON train file format"""
    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir
        data_args = copy.deepcopy(DATA_ARGS)
        data_args.training_data_path = TWITTER_COMPLAINTS_JSON_FORMAT
        data_args.dataset_text_field = None
        data_args.data_formatter_template = (
            "### Text: {{Tweet text}} \n\n### Label: {{text_label}}"
        )

        sft_trainer.train(MODEL_ARGS, data_args, train_args, PEFT_PT_ARGS)

        # validate peft tuning configs
        _validate_training(tempdir)
        checkpoint_path = _get_checkpoint_path(tempdir)
        adapter_config = _get_adapter_config(checkpoint_path)
        _validate_adapter_config(adapter_config, "PROMPT_TUNING", PEFT_PT_ARGS)

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
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir

        tuning_config = peft_config.PromptTuningConfig(
            prompt_tuning_init="TEXT",
            prompt_tuning_init_text="hello",
            tokenizer_name_or_path=MODEL_NAME,
        )

        sft_trainer.train(MODEL_ARGS, DATA_ARGS, train_args, tuning_config)

        # validate peft tuning configs
        _validate_training(tempdir)
        checkpoint_path = _get_checkpoint_path(tempdir)
        adapter_config = _get_adapter_config(checkpoint_path)
        _validate_adapter_config(adapter_config, "PROMPT_TUNING", tuning_config)


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
def test_run_causallm_pt_invalid_train_params(param_name, param_val, exc_msg):
    """Check if error is raised when invalid params are used to peft tune causallm models"""
    with tempfile.TemporaryDirectory() as tempdir:
        invalid_params = copy.deepcopy(TRAIN_ARGS)
        invalid_params.output_dir = tempdir
        setattr(invalid_params, param_name, param_val)

        with pytest.raises(ValueError, match=exc_msg):
            sft_trainer.train(MODEL_ARGS, DATA_ARGS, invalid_params, PEFT_PT_ARGS)


def test_run_causallm_pt_with_validation():
    """Check if we can bootstrap and peft tune causallm models with validation dataset"""
    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir
        train_args.eval_strategy = "epoch"
        data_args = copy.deepcopy(DATA_ARGS)
        data_args.validation_data_path = TWITTER_COMPLAINTS_DATA

        sft_trainer.train(MODEL_ARGS, data_args, train_args, PEFT_PT_ARGS)
        _validate_training(tempdir, check_eval=True)


def test_run_causallm_pt_with_validation_data_formatting():
    """Check if we can bootstrap and peft tune causallm models with validation dataset"""
    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir
        train_args.eval_strategy = "epoch"
        data_args = copy.deepcopy(DATA_ARGS)
        data_args.validation_data_path = TWITTER_COMPLAINTS_DATA
        data_args.dataset_text_field = None
        data_args.data_formatter_template = (
            "### Text: {{Tweet text}} \n\n### Label: {{text_label}}"
        )

        sft_trainer.train(MODEL_ARGS, data_args, train_args, PEFT_PT_ARGS)
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
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir
        base_lora_args = copy.deepcopy(PEFT_LORA_ARGS)
        if "default" not in request._pyfuncitem.callspec.id:
            base_lora_args.target_modules = target_modules

        sft_trainer.train(MODEL_ARGS, DATA_ARGS, train_args, base_lora_args)

        # validate lora tuning configs
        _validate_training(tempdir)
        checkpoint_path = _get_checkpoint_path(tempdir)
        adapter_config = _get_adapter_config(checkpoint_path)
        _validate_adapter_config(adapter_config, "LORA", base_lora_args)

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
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir

        sft_trainer.train(MODEL_ARGS, DATA_ARGS, train_args, None)

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


def _validate_adapter_config(adapter_config, peft_type, tuning_config):
    assert adapter_config.get("task_type") == "CAUSAL_LM"
    assert adapter_config.get("peft_type") == peft_type
    assert (
        (
            adapter_config.get("tokenizer_name_or_path")
            == tuning_config.tokenizer_name_or_path
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
        MODEL_ARGS.model_name_or_path
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ARGS.model_name_or_path
    )
    tokenizer.eos_token = None
    with tempfile.TemporaryDirectory() as tempdir:
        tokenizer.save_pretrained(tempdir)
        model.save_pretrained(tempdir)

        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir
        model_args = copy.deepcopy(MODEL_ARGS)
        train_args.model_name_or_path = tempdir

        # If we handled this badly, we would probably get something like a
        # TypeError: can only concatenate str (not "NoneType") to str error
        # when we go to apply the data formatter.
        sft_trainer.train(model_args, DATA_ARGS, train_args, PEFT_PT_ARGS)
        _validate_training(tempdir)


### Tests for Bad dataset specification, i.e., data is valid, but the field we point it at isn't
def test_invalid_dataset_text_field():
    """Ensure that if we specify a dataset_text_field that doesn't exist, we get a KeyError."""

    data_args = copy.deepcopy(DATA_ARGS)
    data_args.dataset_text_field = "not found"

    with pytest.raises(KeyError):
        sft_trainer.train(MODEL_ARGS, data_args, TRAIN_ARGS, PEFT_PT_ARGS)


### Tests that giving dataset_text_field as well as formatter template gives error
def test_invalid_dataset_text_field_and_formatter_template():
    """Only one of dataset_text_field or formatter can be supplied"""
    data_args = copy.deepcopy(DATA_ARGS)
    data_args.data_formatter_template = (
        "### Text: {{Tweet text}} \n\n### Label: {{text_label}}"
    )

    with pytest.raises(ValueError):
        sft_trainer.train(MODEL_ARGS, data_args, TRAIN_ARGS, PEFT_PT_ARGS)


### Tests passing formatter with invalid keys gives error
def test_invalid_formatter_template():
    data_args = copy.deepcopy(DATA_ARGS)
    data_args.dataset_text_field = None
    data_args.data_formatter_template = (
        "### Text: {{not found}} \n\n### Label: {{text_label}}"
    )

    with pytest.raises(KeyError):
        sft_trainer.train(MODEL_ARGS, data_args, TRAIN_ARGS, PEFT_PT_ARGS)


### Tests for bad training data (i.e., data_path is an unhappy value or points to an unhappy thing)
def test_malformatted_data():
    """Ensure that malformatted data explodes due to failure to generate the dataset."""
    data_args = copy.deepcopy(DATA_ARGS)
    data_args.training_data_path = MALFORMATTED_DATA

    with pytest.raises(DatasetGenerationError):
        sft_trainer.train(MODEL_ARGS, data_args, TRAIN_ARGS, PEFT_PT_ARGS)


def test_empty_data():
    """Ensure that malformatted data explodes due to failure to generate the dataset."""
    data_args = copy.deepcopy(DATA_ARGS)
    data_args.training_data_path = EMPTY_DATA

    with pytest.raises(DatasetGenerationError):
        sft_trainer.train(MODEL_ARGS, data_args, TRAIN_ARGS, PEFT_PT_ARGS)


def test_data_path_is_a_directory():
    """Ensure that we get FileNotFoundError if we point the data path at a dir, not a file."""
    with tempfile.TemporaryDirectory() as tempdir:
        data_args = copy.deepcopy(DATA_ARGS)
        data_args.training_data_path = tempdir

        # Confusingly, if we pass a directory for our data path, it will throw a
        # FileNotFoundError saying "unable to find '<data_path>'", since it can't
        # find a matchable file in the path.
        with pytest.raises(FileNotFoundError):
            sft_trainer.train(MODEL_ARGS, data_args, TRAIN_ARGS, PEFT_PT_ARGS)


### Tests for bad tuning module configurations
def test_run_causallm_lora_with_invalid_modules():
    """Check that we throw a value error if the target modules for lora don't exist."""
    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir
        # Defaults are q_proj / v_proj; this will fail lora as the torch module doesn't have them
        lora_config = copy.deepcopy(PEFT_LORA_ARGS)
        lora_config.target_modules = ["foo", "bar"]
        # Peft should throw a value error about modules not matching the base module
        with pytest.raises(ValueError):
            sft_trainer.train(MODEL_ARGS, DATA_ARGS, train_args, lora_config)


### Direct validation tests based on whether or not packing is enabled
def test_no_packing_needs_dataset_text_field_or_data_formatter_template():
    """Ensure we need to set the dataset text field if packing is False"""
    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir
        data_args = copy.deepcopy(DATA_ARGS)
        # One of dataset_text_field or data_formatter_template should be set
        data_args.dataset_text_field = None
        data_args.data_formatter_template = None

        with pytest.raises(ValueError):
            sft_trainer.train(MODEL_ARGS, data_args, train_args, PEFT_PT_ARGS)


# TODO: Fix this case
@pytest.mark.skip(reason="currently crashes before validation is done")
def test_no_packing_needs_reponse_template():
    """Ensure we need to set the response template if packing is False"""
    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir
        data_args = copy.deepcopy(DATA_ARGS)
        data_args.response_template = None

        with pytest.raises(ValueError):
            sft_trainer.train(MODEL_ARGS, data_args, train_args, PEFT_PT_ARGS)


### Tests for model dtype edge cases
@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    reason="Only runs if bf16 is unsupported",
)
def test_bf16_still_tunes_if_unsupported():
    """Ensure that even if bf16 is not supported, tuning still works without problems."""
    assert not torch.cuda.is_bf16_supported()
    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir
        model_args = copy.deepcopy(MODEL_ARGS)
        model_args.torch_dtype = "bfloat16"

        sft_trainer.train(model_args, DATA_ARGS, train_args, PEFT_PT_ARGS)
        _validate_training(tempdir)


def test_bad_torch_dtype():
    """Ensure that specifying an invalid torch dtype yields a ValueError."""
    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir
        model_args = copy.deepcopy(MODEL_ARGS)
        model_args.torch_dtype = "not a type"

        with pytest.raises(ValueError):
            sft_trainer.train(model_args, DATA_ARGS, train_args, PEFT_PT_ARGS)
