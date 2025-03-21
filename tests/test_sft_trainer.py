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

# pylint: disable=too-many-lines

# Standard
from dataclasses import asdict
import copy
import json
import os
import re
import tempfile

# Third Party
from datasets.exceptions import DatasetGenerationError, DatasetNotFoundError
from transformers.trainer_callback import TrainerCallback
import pytest
import torch
import transformers
import yaml

# First Party
from build.utils import serialize_args
from scripts.run_inference import TunedCausalLM
from tests.artifacts.predefined_data_configs import (
    DATA_CONFIG_DUPLICATE_COLUMNS,
    DATA_CONFIG_MULTIPLE_DATASETS_SAMPLING_YAML,
    DATA_CONFIG_MULTITURN_DATA_YAML,
    DATA_CONFIG_MULTITURN_GRANITE_3_1B_DATA_YAML,
    DATA_CONFIG_RENAME_SELECT_COLUMNS,
    DATA_CONFIG_SKIP_LARGE_COLUMNS_HANDLER,
    DATA_CONFIG_TOKENIZE_AND_APPLY_INPUT_MASKING_YAML,
    DATA_CONFIG_TOKENIZE_AND_TRAIN_WITH_HANDLER,
    DATA_CONFIG_YAML_STREAMING_INPUT_OUTPUT,
    DATA_CONFIG_YAML_STREAMING_PRETOKENIZED,
)
from tests.artifacts.testdata import (
    CHAT_DATA_MULTI_TURN,
    CHAT_DATA_MULTI_TURN_GRANITE_3_1B,
    CHAT_DATA_SINGLE_TURN,
    CUSTOM_TOKENIZER_TINYLLAMA,
    EMPTY_DATA,
    MALFORMATTED_DATA,
    MODEL_NAME,
    TWITTER_COMPLAINTS_DATA_ARROW,
    TWITTER_COMPLAINTS_DATA_DIR_JSON,
    TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_ARROW,
    TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
    TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSONL,
    TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_PARQUET,
    TWITTER_COMPLAINTS_DATA_JSON,
    TWITTER_COMPLAINTS_DATA_JSONL,
    TWITTER_COMPLAINTS_DATA_PARQUET,
    TWITTER_COMPLAINTS_TOKENIZED_ARROW,
    TWITTER_COMPLAINTS_TOKENIZED_JSON,
    TWITTER_COMPLAINTS_TOKENIZED_JSONL,
    TWITTER_COMPLAINTS_TOKENIZED_ONLY_INPUT_IDS_JSON,
    TWITTER_COMPLAINTS_TOKENIZED_PARQUET,
)

# Local
from tuning import sft_trainer
from tuning.config import configs, peft_config
from tuning.config.acceleration_configs.fast_moe import FastMoe, FastMoeConfig
from tuning.config.tracker_configs import FileLoggingTrackerConfig
from tuning.data.data_config import (
    DataConfig,
    DataHandlerConfig,
    DataPreProcessorConfig,
    DataSetConfig,
)
from tuning.data.data_handlers import (
    DataHandler,
    DataHandlerType,
    add_tokenizer_eos_token,
)
from tuning.utils.import_utils import is_fms_accelerate_available

MODEL_ARGS = configs.ModelArguments(
    model_name_or_path=MODEL_NAME, use_flash_attn=False, torch_dtype="float32"
)
DATA_ARGS = configs.DataArguments(
    training_data_path=TWITTER_COMPLAINTS_DATA_JSONL,
    response_template="\n### Label:",
    dataset_text_field="output",
)
TRAIN_ARGS = configs.TrainingArguments(
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
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
)

PEFT_LORA_ARGS = peft_config.LoraConfig(r=8, lora_alpha=32, lora_dropout=0.05)


def test_resume_training_from_checkpoint():
    """
    Test tuning resumes from the latest checkpoint, creating new checkpoints and the
    checkpoints created before resuming tuning is not affected.
    """
    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir

        sft_trainer.train(MODEL_ARGS, DATA_ARGS, train_args, None)
        _validate_training(tempdir)

        # Get trainer state of latest checkpoint
        init_trainer_state, _ = _get_latest_checkpoint_trainer_state(tempdir)
        assert init_trainer_state is not None

        # Resume training with higher epoch and same output dir
        train_args.num_train_epochs += 5
        sft_trainer.train(MODEL_ARGS, DATA_ARGS, train_args, None)
        _validate_training(tempdir)

        # Get trainer state of latest checkpoint
        final_trainer_state, _ = _get_latest_checkpoint_trainer_state(tempdir)
        assert final_trainer_state is not None

        assert final_trainer_state["epoch"] == init_trainer_state["epoch"] + 5
        assert final_trainer_state["global_step"] > init_trainer_state["global_step"]

        # Check if loss of 1st epoch after first tuning is same after
        # resuming tuning and not overwritten
        assert len(init_trainer_state["log_history"]) > 0

        init_log_history = init_trainer_state["log_history"][0]
        assert init_log_history["epoch"] == 1

        final_log_history = final_trainer_state["log_history"][0]
        assert final_log_history["epoch"] == 1

        assert init_log_history["loss"] == final_log_history["loss"]


def test_resume_training_from_checkpoint_with_flag_true():
    """
    Test tuning resumes from the latest checkpoint when flag is true,
    creating new checkpoints and the checkpoints created before resuming
    tuning is not affected.
    """
    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir
        train_args.resume_from_checkpoint = "True"

        sft_trainer.train(MODEL_ARGS, DATA_ARGS, train_args, None)
        _validate_training(tempdir)

        # Get trainer state of latest checkpoint
        init_trainer_state, _ = _get_latest_checkpoint_trainer_state(tempdir)
        assert init_trainer_state is not None

        # Get Training logs
        init_training_logs = _get_training_logs_by_epoch(tempdir)

        # Resume training with higher epoch and same output dir
        train_args.num_train_epochs += 5
        sft_trainer.train(MODEL_ARGS, DATA_ARGS, train_args, None)
        _validate_training(tempdir)

        # Get trainer state of latest checkpoint
        final_trainer_state, _ = _get_latest_checkpoint_trainer_state(tempdir)
        assert final_trainer_state is not None

        assert final_trainer_state["epoch"] == init_trainer_state["epoch"] + 5
        assert final_trainer_state["global_step"] > init_trainer_state["global_step"]

        final_training_logs = _get_training_logs_by_epoch(tempdir)

        assert (
            init_training_logs[0]["data"]["timestamp"]
            == final_training_logs[0]["data"]["timestamp"]
        )


def test_resume_training_from_checkpoint_with_flag_false():
    """
    Test when setting resume_from_checkpoint=False that tuning will start from scratch.
    """
    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir
        train_args.resume_from_checkpoint = "False"

        sft_trainer.train(MODEL_ARGS, DATA_ARGS, train_args, None)
        _validate_training(tempdir)

        # Get trainer state of latest checkpoint
        init_trainer_state, _ = _get_latest_checkpoint_trainer_state(tempdir)
        assert init_trainer_state is not None

        # Get Training log entry for epoch 1
        init_training_logs = _get_training_logs_by_epoch(tempdir, epoch=1)
        assert len(init_training_logs) == 1

        # Training again with higher epoch and same output dir
        train_args.num_train_epochs += 5
        sft_trainer.train(MODEL_ARGS, DATA_ARGS, train_args, None)
        _validate_training(tempdir)

        # Get Training log entry for epoch 1
        final_training_logs = _get_training_logs_by_epoch(tempdir, epoch=1)
        assert len(final_training_logs) == 2


def test_resume_training_from_checkpoint_with_flag_checkpoint_path_lora():
    """
    Test resume checkpoint from a specified checkpoint path for LoRA tuning.
    """
    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        lora_config = copy.deepcopy(PEFT_LORA_ARGS)
        train_args.output_dir = tempdir

        sft_trainer.train(MODEL_ARGS, DATA_ARGS, train_args, lora_config)
        _validate_training(tempdir)

        # Get trainer state and checkpoint_path of second last checkpoint
        init_trainer_state, checkpoint_path = _get_latest_checkpoint_trainer_state(
            tempdir, checkpoint_index=-2
        )
        assert init_trainer_state is not None

        # Resume training with higher epoch and same output dir
        train_args.num_train_epochs += 5
        train_args.resume_from_checkpoint = checkpoint_path
        sft_trainer.train(MODEL_ARGS, DATA_ARGS, train_args, lora_config)
        _validate_training(tempdir)

        # Get total_flos from trainer state of checkpoint_path and check if its same
        final_trainer_state = None
        trainer_state_file = os.path.join(checkpoint_path, "trainer_state.json")
        with open(trainer_state_file, "r", encoding="utf-8") as f:
            final_trainer_state = json.load(f)

        assert final_trainer_state["total_flos"] == init_trainer_state["total_flos"]


def _get_latest_checkpoint_trainer_state(dir_path: str, checkpoint_index: int = -1):
    """
    Get the trainer state from the latest or specified checkpoint directory.
    The trainer state is returned along with the path to the checkpoint.

    Args:
        dir_path (str): The directory path where checkpoint folders are located.
        checkpoint_index (int, optional): The index of the checkpoint to retrieve,
                                          based on the checkpoint number. The default
                                          is -1, which returns the latest checkpoint.

    Returns:
        trainer_state: The trainer state loaded from `trainer_state.json` in the
                            checkpoint directory.
        last_checkpoint: The path to the checkpoint directory.
    """
    trainer_state = None
    last_checkpoint = None
    checkpoints = [
        os.path.join(dir_path, d)
        for d in os.listdir(dir_path)
        if d.startswith("checkpoint")
    ]
    if checkpoints:
        last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[
            checkpoint_index
        ]
        trainer_state_file = os.path.join(last_checkpoint, "trainer_state.json")
        with open(trainer_state_file, "r", encoding="utf-8") as f:
            trainer_state = json.load(f)
    return trainer_state, last_checkpoint


def _get_training_logs_by_epoch(dir_path: str, epoch: int = None):
    """
    Load and optionally filter training_logs.jsonl file.
    If an epoch number is specified, the function filters the logs
    and returns only the entries corresponding to the specified epoch.

    Args:
        dir_path (str): The directory path where the `training_logs.jsonl` file is located.
        epoch (int, optional): The epoch number to filter logs by. If not specified,
                               all logs are returned.

    Returns:
        list: A list containing the training logs. If `epoch` is specified,
              only logs from the specified epoch are returned; otherwise, all logs are returned.
    """
    data_list = []
    with open(f"{dir_path}/training_logs.jsonl", "r", encoding="utf-8") as file:
        for line in file:
            json_data = json.loads(line)
            data_list.append(json_data)

    if epoch:
        mod_data_list = []
        for value in data_list:
            if value["data"]["epoch"] == epoch:
                mod_data_list.append(value)
        return mod_data_list
    return data_list


def test_run_train_fails_training_data_path_not_exist():
    """Check fails when data path not found."""
    updated_data_path_args = copy.deepcopy(DATA_ARGS)
    updated_data_path_args.training_data_path = "fake/path"
    with pytest.raises(DatasetNotFoundError):
        sft_trainer.train(MODEL_ARGS, updated_data_path_args, TRAIN_ARGS, None)


HAPPY_PATH_DUMMY_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "build", "dummy_job_config.json"
)


# Note: job_config dict gets modified during process training args
@pytest.fixture(name="job_config", scope="session")
def fixture_job_config():
    with open(HAPPY_PATH_DUMMY_CONFIG_PATH, "r", encoding="utf-8") as f:
        dummy_job_config_dict = json.load(f)
    return dummy_job_config_dict


############################# Arg Parsing Tests #############################


def test_parse_arguments(job_config):
    parser = sft_trainer.get_parser()
    job_config_copy = copy.deepcopy(job_config)
    (
        model_args,
        data_args,
        training_args,
        _,
        tune_config,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = sft_trainer.parse_arguments(parser, job_config_copy)
    assert str(model_args.torch_dtype) == "torch.bfloat16"
    assert data_args.dataset_text_field == "output"
    assert training_args.output_dir == "bloom-twitter"
    assert tune_config is None


def test_parse_arguments_defaults(job_config):
    parser = sft_trainer.get_parser()
    job_config_defaults = copy.deepcopy(job_config)
    assert "torch_dtype" not in job_config_defaults
    assert job_config_defaults["use_flash_attn"] is False
    assert "save_strategy" not in job_config_defaults
    (
        model_args,
        _,
        training_args,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = sft_trainer.parse_arguments(parser, job_config_defaults)
    assert str(model_args.torch_dtype) == "torch.bfloat16"
    assert model_args.use_flash_attn is False
    assert training_args.save_strategy.value == "epoch"


def test_parse_arguments_peft_method(job_config):
    parser = sft_trainer.get_parser()
    job_config_pt = copy.deepcopy(job_config)
    job_config_pt["peft_method"] = "pt"
    _, _, _, _, tune_config, _, _, _, _, _, _, _, _, _ = sft_trainer.parse_arguments(
        parser, job_config_pt
    )
    assert isinstance(tune_config, peft_config.PromptTuningConfig)

    job_config_lora = copy.deepcopy(job_config)
    job_config_lora["peft_method"] = "lora"
    _, _, _, _, tune_config, _, _, _, _, _, _, _, _, _ = sft_trainer.parse_arguments(
        parser, job_config_lora
    )
    assert isinstance(tune_config, peft_config.LoraConfig)
    assert not tune_config.target_modules
    assert "target_modules" not in job_config_lora


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

        _validate_adapter_config(
            adapter_config, "PROMPT_TUNING", MODEL_ARGS.model_name_or_path
        )

        # Load the model
        loaded_model = TunedCausalLM.load(checkpoint_path, MODEL_NAME)

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
        data_args = copy.deepcopy(DATA_ARGS)
        data_args.dataset_text_field = None
        data_args.data_formatter_template = (
            "### Text: {{Tweet text}} \n\n### Label: {{text_label}}"
        )

        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir

        sft_trainer.train(MODEL_ARGS, data_args, train_args, PEFT_PT_ARGS)

        # validate peft tuning configs
        _validate_training(tempdir)
        checkpoint_path = _get_checkpoint_path(tempdir)
        adapter_config = _get_adapter_config(checkpoint_path)
        _validate_adapter_config(
            adapter_config, "PROMPT_TUNING", MODEL_ARGS.model_name_or_path
        )

        # Load the model
        loaded_model = TunedCausalLM.load(checkpoint_path, MODEL_NAME)

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
        data_args.training_data_path = TWITTER_COMPLAINTS_DATA_JSON
        data_args.dataset_text_field = None
        data_args.data_formatter_template = (
            "### Text: {{Tweet text}} \n\n### Label: {{text_label}}"
        )

        sft_trainer.train(MODEL_ARGS, data_args, train_args, PEFT_PT_ARGS)

        # validate peft tuning configs
        _validate_training(tempdir)
        checkpoint_path = _get_checkpoint_path(tempdir)
        adapter_config = _get_adapter_config(checkpoint_path)
        _validate_adapter_config(
            adapter_config, "PROMPT_TUNING", MODEL_ARGS.model_name_or_path
        )

        # Load the model
        loaded_model = TunedCausalLM.load(checkpoint_path, MODEL_NAME)

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
        )

        sft_trainer.train(MODEL_ARGS, DATA_ARGS, train_args, tuning_config)

        # validate peft tuning configs
        _validate_training(tempdir)
        checkpoint_path = _get_checkpoint_path(tempdir)
        adapter_config = _get_adapter_config(checkpoint_path)
        _validate_adapter_config(
            adapter_config, "PROMPT_TUNING", MODEL_ARGS.model_name_or_path
        )


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


@pytest.mark.parametrize(
    "dataset_path",
    [TWITTER_COMPLAINTS_DATA_JSONL, TWITTER_COMPLAINTS_DATA_JSON],
)
def test_run_causallm_pt_with_validation(dataset_path):
    """Check if we can bootstrap and peft tune causallm models with validation dataset"""
    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir
        train_args.eval_strategy = "epoch"
        data_args = copy.deepcopy(DATA_ARGS)
        data_args.validation_data_path = dataset_path

        sft_trainer.train(MODEL_ARGS, data_args, train_args, PEFT_PT_ARGS)
        _validate_training(tempdir, check_eval=True)


@pytest.mark.parametrize(
    "dataset_path",
    [TWITTER_COMPLAINTS_DATA_JSONL, TWITTER_COMPLAINTS_DATA_JSON],
)
def test_run_causallm_pt_with_validation_data_formatting(dataset_path):
    """Check if we can bootstrap and peft tune causallm models with validation dataset"""
    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir
        train_args.eval_strategy = "epoch"
        data_args = copy.deepcopy(DATA_ARGS)
        data_args.validation_data_path = dataset_path
        data_args.dataset_text_field = None
        data_args.data_formatter_template = (
            "### Text: {{Tweet text}} \n\n### Label: {{text_label}}"
        )

        sft_trainer.train(MODEL_ARGS, data_args, train_args, PEFT_PT_ARGS)
        _validate_training(tempdir, check_eval=True)


@pytest.mark.parametrize(
    "dataset_path",
    [TWITTER_COMPLAINTS_DATA_JSONL, TWITTER_COMPLAINTS_DATA_JSON],
)
def test_run_causallm_pt_with_custom_tokenizer(dataset_path):
    """Check if we fail when custom tokenizer not having pad token is used in prompt tuning"""
    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        model_args = copy.deepcopy(MODEL_ARGS)
        model_args.tokenizer_name_or_path = model_args.model_name_or_path
        train_args.output_dir = tempdir
        train_args.eval_strategy = "epoch"
        data_args = copy.deepcopy(DATA_ARGS)
        data_args.validation_data_path = dataset_path
        with pytest.raises(ValueError):
            sft_trainer.train(model_args, data_args, train_args, PEFT_PT_ARGS)


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
        _validate_adapter_config(adapter_config, "LORA")

        for module in expected:
            assert module in adapter_config.get("target_modules")

        # Load the model
        loaded_model = TunedCausalLM.load(checkpoint_path, MODEL_NAME)

        # Run inference on the text
        output_inference = loaded_model.run(
            "Simply put, the theory of relativity states that ", max_new_tokens=50
        )
        assert len(output_inference) > 0
        assert "Simply put, the theory of relativity states that" in output_inference


def test_successful_lora_target_modules_default_from_main():
    """Check that if target_modules is not set, or set to None via JSON, the
    default value by model type will be using in LoRA tuning.
    The correct default target modules will be used for model type llama
    and will exist in the resulting adapter_config.json.
    https://github.com/huggingface/peft/blob/7b1c08d2b5e13d3c99b7d6ee83eab90e1216d4ba/
    src/peft/tuners/lora/model.py#L432
    """
    with tempfile.TemporaryDirectory() as tempdir:
        TRAIN_KWARGS = {
            **MODEL_ARGS.__dict__,
            **TRAIN_ARGS.__dict__,
            **DATA_ARGS.__dict__,
            **PEFT_LORA_ARGS.__dict__,
            **{"peft_method": "lora", "output_dir": tempdir},
        }
        serialized_args = serialize_args(TRAIN_KWARGS)
        os.environ["SFT_TRAINER_CONFIG_JSON_ENV_VAR"] = serialized_args

        sft_trainer.main()

        _validate_training(tempdir)
        # Calling LoRA tuning from the main results in 'added_tokens_info.json'
        assert "added_tokens_info.json" in os.listdir(tempdir)
        checkpoint_path = _get_checkpoint_path(tempdir)
        adapter_config = _get_adapter_config(checkpoint_path)
        _validate_adapter_config(adapter_config, "LORA")

        assert (
            "target_modules" in adapter_config
        ), "target_modules not found in adapter_config.json."

        assert set(adapter_config.get("target_modules")) == {
            "q_proj",
            "v_proj",
        }, "target_modules are not set to the default values."


############################# Finetuning Tests #############################
@pytest.mark.parametrize(
    "dataset_path",
    [
        TWITTER_COMPLAINTS_DATA_JSONL,
        TWITTER_COMPLAINTS_DATA_JSON,
        TWITTER_COMPLAINTS_DATA_PARQUET,
        TWITTER_COMPLAINTS_DATA_ARROW,
    ],
)
def test_run_causallm_ft_and_inference(dataset_path):
    """Check if we can bootstrap and finetune causallm models with different data formats"""
    with tempfile.TemporaryDirectory() as tempdir:
        data_args = copy.deepcopy(DATA_ARGS)
        data_args.training_data_path = dataset_path

        _test_run_causallm_ft(TRAIN_ARGS, MODEL_ARGS, data_args, tempdir)
        _test_run_inference(checkpoint_path=_get_checkpoint_path(tempdir))


def test_run_causallm_ft_save_with_save_model_dir_save_strategy_no():
    """Check if we can bootstrap and finetune causallm model with save_model_dir
    and save_strategy=no. Verify no checkpoints created and can save model.
    """
    with tempfile.TemporaryDirectory() as tempdir:
        save_model_args = copy.deepcopy(TRAIN_ARGS)
        save_model_args.save_strategy = "no"
        save_model_args.output_dir = tempdir

        trainer, _ = sft_trainer.train(MODEL_ARGS, DATA_ARGS, save_model_args, None)
        logs_path = os.path.join(
            tempdir, FileLoggingTrackerConfig.training_logs_filename
        )
        _validate_logfile(logs_path)
        # validate that no checkpoints created
        assert not any(x.startswith("checkpoint-") for x in os.listdir(tempdir))

        sft_trainer.save(tempdir, trainer, "debug")
        assert any(x.endswith(".safetensors") for x in os.listdir(tempdir))
        _test_run_inference(checkpoint_path=tempdir)


@pytest.mark.parametrize(
    "dataset_path, packing",
    [
        (TWITTER_COMPLAINTS_TOKENIZED_JSONL, True),
        (TWITTER_COMPLAINTS_TOKENIZED_PARQUET, True),
        (TWITTER_COMPLAINTS_TOKENIZED_JSON, False),
        (TWITTER_COMPLAINTS_TOKENIZED_ARROW, False),
    ],
)
def test_run_causallm_ft_pretokenized(dataset_path, packing):
    """Check if we can bootstrap and finetune causallm models using pretokenized data"""
    with tempfile.TemporaryDirectory() as tempdir:
        data_args = copy.deepcopy(DATA_ARGS)

        # below args not needed for pretokenized data
        data_args.data_formatter_template = None
        data_args.dataset_text_field = None
        data_args.response_template = None

        # update the training data path to tokenized data
        data_args.training_data_path = dataset_path

        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir
        train_args.packing = packing
        train_args.max_seq_length = 256

        sft_trainer.train(MODEL_ARGS, data_args, train_args)

        # validate full ft configs
        _validate_training(tempdir)
        checkpoint_path = _get_checkpoint_path(tempdir)

        # Load the model
        loaded_model = TunedCausalLM.load(checkpoint_path, MODEL_NAME)

        # Run inference on the text
        output_inference = loaded_model.run(
            "### Text: @NortonSupport Thanks much.\n\n### Label:", max_new_tokens=50
        )
        assert len(output_inference) > 0
        assert "### Text: @NortonSupport Thanks much.\n\n### Label:" in output_inference


@pytest.mark.parametrize(
    "datafiles, datasetconfigname",
    [
        (
            [TWITTER_COMPLAINTS_DATA_DIR_JSON],
            DATA_CONFIG_YAML_STREAMING_INPUT_OUTPUT,
        ),
        (
            [TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON],
            DATA_CONFIG_YAML_STREAMING_INPUT_OUTPUT,
        ),
        (
            [TWITTER_COMPLAINTS_TOKENIZED_JSON],
            DATA_CONFIG_YAML_STREAMING_PRETOKENIZED,
        ),
    ],
)
def test_run_causallm_ft_and_inference_streaming(datasetconfigname, datafiles):
    """Check if we can finetune causallm models using multiple datasets with multiple files"""
    with tempfile.TemporaryDirectory() as tempdir:
        data_formatting_args = copy.deepcopy(DATA_ARGS)

        # set training_data_path and response_template to none
        data_formatting_args.response_template = None
        data_formatting_args.training_data_path = None

        # add data_paths in data_config file
        with tempfile.NamedTemporaryFile(
            "w", delete=False, suffix=".yaml"
        ) as temp_yaml_file:
            with open(datasetconfigname, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                datasets = data["datasets"]
                for _, d in enumerate(datasets):
                    d["data_paths"] = datafiles
                yaml.dump(data, temp_yaml_file)
                data_formatting_args.data_config_path = temp_yaml_file.name

        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir
        train_args.max_steps = 1

        sft_trainer.train(MODEL_ARGS, data_formatting_args, train_args)

        # validate full ft configs
        _validate_training(tempdir)
        _, checkpoint_path = _get_latest_checkpoint_trainer_state(tempdir)

        # Load the model
        loaded_model = TunedCausalLM.load(checkpoint_path, MODEL_NAME)

        # Run inference on the text
        output_inference = loaded_model.run(
            "### Text: @NortonSupport Thanks much.\n\n### Label:", max_new_tokens=50
        )
        assert len(output_inference) > 0
        assert "### Text: @NortonSupport Thanks much.\n\n### Label:" in output_inference


@pytest.mark.parametrize(
    "datafiles, datasetconfigname",
    [
        (
            [TWITTER_COMPLAINTS_DATA_DIR_JSON],
            DATA_CONFIG_TOKENIZE_AND_APPLY_INPUT_MASKING_YAML,
        ),
        (
            [
                TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
                TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
            ],
            DATA_CONFIG_MULTIPLE_DATASETS_SAMPLING_YAML,
        ),
        (
            [
                TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
                TWITTER_COMPLAINTS_DATA_DIR_JSON,
            ],
            DATA_CONFIG_MULTIPLE_DATASETS_SAMPLING_YAML,
        ),
        (
            [
                TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSONL,
                TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSONL,
            ],
            DATA_CONFIG_MULTIPLE_DATASETS_SAMPLING_YAML,
        ),
        (
            [
                TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_ARROW,
                TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_ARROW,
            ],
            DATA_CONFIG_MULTIPLE_DATASETS_SAMPLING_YAML,
        ),
        (
            [
                TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_PARQUET,
                TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_PARQUET,
            ],
            DATA_CONFIG_MULTIPLE_DATASETS_SAMPLING_YAML,
        ),
        (
            [TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON],
            DATA_CONFIG_RENAME_SELECT_COLUMNS,
        ),
    ],
)
def test_run_causallm_ft_and_inference_with_multiple_dataset(
    datasetconfigname, datafiles
):
    """Check if we can finetune causallm models using multiple datasets with multiple files"""
    with tempfile.TemporaryDirectory() as tempdir:
        data_formatting_args = copy.deepcopy(DATA_ARGS)

        # set training_data_path and response_template to none
        data_formatting_args.response_template = None
        data_formatting_args.training_data_path = None

        # add data_paths in data_config file
        with tempfile.NamedTemporaryFile(
            "w", delete=False, suffix=".yaml"
        ) as temp_yaml_file:
            with open(datasetconfigname, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                datasets = data["datasets"]
                for _, d in enumerate(datasets):
                    d["data_paths"] = datafiles
                yaml.dump(data, temp_yaml_file)
                data_formatting_args.data_config_path = temp_yaml_file.name

        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir

        sft_trainer.train(MODEL_ARGS, data_formatting_args, train_args)

        # validate full ft configs
        _validate_training(tempdir)
        _, checkpoint_path = _get_latest_checkpoint_trainer_state(tempdir)

        # Load the model
        loaded_model = TunedCausalLM.load(checkpoint_path, MODEL_NAME)

        # Run inference on the text
        output_inference = loaded_model.run(
            "### Text: @NortonSupport Thanks much.\n\n### Label:", max_new_tokens=50
        )
        assert len(output_inference) > 0
        assert "### Text: @NortonSupport Thanks much.\n\n### Label:" in output_inference


def test_run_training_with_pretokenised_dataset_containing_input_ids():
    """Ensure that we can train on pretokenised dataset containing just input_ids by
    choosing duplicate_columns data handler via data config."""
    with tempfile.TemporaryDirectory() as tempdir:

        data_args = copy.deepcopy(DATA_ARGS)

        # set training_data_path and response_template to none
        data_args.response_template = None
        data_args.training_data_path = None

        dataconfigfile = DATA_CONFIG_DUPLICATE_COLUMNS
        datapath = TWITTER_COMPLAINTS_TOKENIZED_ONLY_INPUT_IDS_JSON

        # add data_paths in data_config file
        with tempfile.NamedTemporaryFile(
            "w", delete=False, suffix=".yaml"
        ) as temp_yaml_file:
            with open(dataconfigfile, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                datasets = data["datasets"]
                for _, d in enumerate(datasets):
                    d["data_paths"] = [datapath]
                yaml.dump(data, temp_yaml_file)
                data_args.data_config_path = temp_yaml_file.name

        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir

        sft_trainer.train(MODEL_ARGS, data_args, train_args)

        # validate full ft configs
        _validate_training(tempdir)
        checkpoint_path = _get_checkpoint_path(tempdir)

        # Load the model
        loaded_model = TunedCausalLM.load(checkpoint_path, MODEL_NAME)

        # Run inference on the text
        output_inference = loaded_model.run(
            "### Text: @NortonSupport Thanks much.\n\n### Label:", max_new_tokens=50
        )
        assert len(output_inference) > 0
        assert "### Text: @NortonSupport Thanks much.\n\n### Label:" in output_inference


def test_run_training_with_data_tokenized_using_tokenizer_handler():
    """Ensure that we can train on non tokenized dataset works by tokenizing using
    tokenizer data handler via data config."""
    with tempfile.TemporaryDirectory() as tempdir:

        data_args = copy.deepcopy(DATA_ARGS)

        # set training_data_path and response_template to none
        data_args.response_template = None
        data_args.training_data_path = None

        dataconfigfile = DATA_CONFIG_TOKENIZE_AND_TRAIN_WITH_HANDLER
        datapath = TWITTER_COMPLAINTS_DATA_JSONL

        # add data_paths in data_config file
        with tempfile.NamedTemporaryFile(
            "w", delete=False, suffix=".yaml"
        ) as temp_yaml_file:
            with open(dataconfigfile, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                datasets = data["datasets"]
                for _, d in enumerate(datasets):
                    d["data_paths"] = [datapath]
                yaml.dump(data, temp_yaml_file)
                data_args.data_config_path = temp_yaml_file.name

        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir

        sft_trainer.train(MODEL_ARGS, data_args, train_args)

        # validate full ft configs
        _validate_training(tempdir)
        checkpoint_path = _get_checkpoint_path(tempdir)

        # Load the model
        loaded_model = TunedCausalLM.load(checkpoint_path, MODEL_NAME)

        # Run inference on the text
        output_inference = loaded_model.run(
            "### Text: @NortonSupport Thanks much.\n\n### Label:", max_new_tokens=50
        )
        assert len(output_inference) > 0
        assert "### Text: @NortonSupport Thanks much.\n\n### Label:" in output_inference


def test_run_training_with_skip_large_column_handler():
    """Ensure that we can train succesfully after using skip large column handler."""
    with tempfile.TemporaryDirectory() as tempdir:

        data_args = copy.deepcopy(DATA_ARGS)

        # set training_data_path and response_template to none
        data_args.response_template = None
        data_args.training_data_path = None

        dataconfigfile = DATA_CONFIG_SKIP_LARGE_COLUMNS_HANDLER
        datapath = TWITTER_COMPLAINTS_DATA_JSONL

        # add data_paths in data_config file
        with tempfile.NamedTemporaryFile(
            "w", delete=False, suffix=".yaml"
        ) as temp_yaml_file:
            with open(dataconfigfile, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                datasets = data["datasets"]
                for _, d in enumerate(datasets):
                    d["data_paths"] = [datapath]
                yaml.dump(data, temp_yaml_file)
                data_args.data_config_path = temp_yaml_file.name

        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir

        sft_trainer.train(MODEL_ARGS, data_args, train_args)

        # validate full ft configs
        _validate_training(tempdir)
        checkpoint_path = _get_checkpoint_path(tempdir)

        # Load the model
        loaded_model = TunedCausalLM.load(checkpoint_path, MODEL_NAME)

        # Run inference on the text
        output_inference = loaded_model.run(
            "### Text: @NortonSupport Thanks much.\n\n### Label:", max_new_tokens=50
        )
        assert len(output_inference) > 0
        assert "### Text: @NortonSupport Thanks much.\n\n### Label:" in output_inference


@pytest.mark.parametrize(
    "dataset_path",
    [CHAT_DATA_SINGLE_TURN, CHAT_DATA_MULTI_TURN],
)
def test_run_chat_style_ft(dataset_path):
    """Check if we can perform an e2e run with chat template and multi turn chat training."""
    with tempfile.TemporaryDirectory() as tempdir:

        data_args = copy.deepcopy(DATA_ARGS)
        data_args.training_data_path = dataset_path
        data_args.chat_template = "{% for message in messages['messages'] %}\
            {% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + eos_token }}\
            {% elif message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + eos_token }}\
            {% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n'  + message['content'] + eos_token }}\
            {% endif %}\
            {% if loop.last and add_generation_prompt %}{{ '<|assistant|>' }}\
            {% endif %}\
            {% endfor %}"
        data_args.response_template = "<|assistant|>"
        data_args.instruction_template = "<|user|>"

        model_args = copy.deepcopy(MODEL_ARGS)
        model_args.tokenizer_name_or_path = CUSTOM_TOKENIZER_TINYLLAMA

        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir

        sft_trainer.train(model_args, data_args, train_args)

        # validate the configs
        _validate_training(tempdir)
        checkpoint_path = _get_checkpoint_path(tempdir)

        # Load the model
        loaded_model = TunedCausalLM.load(checkpoint_path, MODEL_NAME)

        # Run inference on the text
        output_inference = loaded_model.run(
            '<|user|>\nProvide two rhyming words for the word "love"\n\
            <nopace></s><|assistant|>',
            max_new_tokens=50,
        )
        assert len(output_inference) > 0
        assert 'Provide two rhyming words for the word "love"' in output_inference


def test_run_chat_style_add_special_tokens_ft():
    """Test to check an e2e multi turn chat training by adding special tokens via command line."""
    with tempfile.TemporaryDirectory() as tempdir:

        # sample hugging face dataset id
        data_args = configs.DataArguments(
            training_data_path="lhoestq/demo1",
            data_formatter_template="### Text:{{review}} \n\n### Stars: {{star}}",
            response_template="\n### Stars:",
            add_special_tokens=["<|assistant|>", "<|user|>"],
        )

        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir

        sft_trainer.train(MODEL_ARGS, data_args, train_args)

        # validate the configs
        _validate_training(tempdir)
        checkpoint_path = _get_checkpoint_path(tempdir)

        # Load the tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_path)

        # Check if all special tokens passed are in tokenizer
        for tok in data_args.add_special_tokens:
            assert tok in tokenizer.vocab


@pytest.mark.parametrize(
    "datafiles, dataconfigfile",
    [
        (
            [CHAT_DATA_SINGLE_TURN, CHAT_DATA_MULTI_TURN, CHAT_DATA_SINGLE_TURN],
            DATA_CONFIG_MULTIPLE_DATASETS_SAMPLING_YAML,
        )
    ],
)
def test_run_chat_style_ft_using_dataconfig(datafiles, dataconfigfile):
    """Check if we can perform an e2e run with chat template
    and multi turn chat training using data config."""
    with tempfile.TemporaryDirectory() as tempdir:

        data_args = copy.deepcopy(DATA_ARGS)
        data_args.chat_template = "{% for message in messages['messages'] %}\
            {% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + eos_token }}\
            {% elif message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + eos_token }}\
            {% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n'  + message['content'] + eos_token }}\
            {% endif %}\
            {% if loop.last and add_generation_prompt %}{{ '<|assistant|>' }}\
            {% endif %}\
            {% endfor %}"
        data_args.response_template = "<|assistant|>"
        data_args.instruction_template = "<|user|>"
        data_args.dataset_text_field = "new_formatted_field"

        handler_kwargs = {"formatted_text_column_name": data_args.dataset_text_field}
        kwargs = {
            "fn_kwargs": handler_kwargs,
            "batched": False,
            "remove_columns": "all",
        }

        handler_config = DataHandlerConfig(
            name="apply_tokenizer_chat_template", arguments=kwargs
        )

        model_args = copy.deepcopy(MODEL_ARGS)
        model_args.tokenizer_name_or_path = CUSTOM_TOKENIZER_TINYLLAMA

        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir

        with tempfile.NamedTemporaryFile(
            "w", delete=False, suffix=".yaml"
        ) as temp_yaml_file:
            with open(dataconfigfile, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            datasets = data["datasets"]
            for i, d in enumerate(datasets):
                d["data_paths"] = [datafiles[i]]
                # Basic chat datasets don't need data handling
                d["data_handlers"] = [asdict(handler_config)]
            yaml.dump(data, temp_yaml_file)
            data_args.data_config_path = temp_yaml_file.name

        sft_trainer.train(model_args, data_args, train_args)

        # validate the configs
        _validate_training(tempdir)
        checkpoint_path = _get_checkpoint_path(tempdir)

        # Load the model
        loaded_model = TunedCausalLM.load(checkpoint_path, MODEL_NAME)

        # Run inference on the text
        output_inference = loaded_model.run(
            '<|user|>\nProvide two rhyming words for the word "love"\n\
            <nopace></s><|assistant|>',
            max_new_tokens=50,
        )
        assert len(output_inference) > 0
        assert 'Provide two rhyming words for the word "love"' in output_inference


@pytest.mark.parametrize(
    "datafiles, dataconfigfile",
    [
        (
            [CHAT_DATA_SINGLE_TURN, CHAT_DATA_MULTI_TURN, CHAT_DATA_SINGLE_TURN],
            DATA_CONFIG_MULTITURN_DATA_YAML,
        ),
        (
            [
                CHAT_DATA_MULTI_TURN_GRANITE_3_1B,
                CHAT_DATA_MULTI_TURN_GRANITE_3_1B,
                CHAT_DATA_MULTI_TURN_GRANITE_3_1B,
            ],
            DATA_CONFIG_MULTITURN_GRANITE_3_1B_DATA_YAML,
        ),
    ],
)
def test_run_chat_style_ft_using_dataconfig_for_chat_template(
    datafiles, dataconfigfile
):
    """Check if we can perform an e2e run with chat template
    and multi turn chat training using data config."""
    with tempfile.TemporaryDirectory() as tempdir:

        data_args = copy.deepcopy(DATA_ARGS)
        if dataconfigfile == DATA_CONFIG_MULTITURN_GRANITE_3_1B_DATA_YAML:
            data_args.response_template = "<|start_of_role|>assistant<|end_of_role|>"
            data_args.instruction_template = "<|start_of_role|>user<|end_of_role|>"
            data_args.add_special_tokens = [
                "<|start_of_role|>assistant<|end_of_role|>",
                "<|start_of_role|>user<|end_of_role|>",
            ]
        elif dataconfigfile == DATA_CONFIG_MULTITURN_DATA_YAML:
            data_args.response_template = "<|assistant|>"
            data_args.instruction_template = "<|user|>"

        data_args.dataset_text_field = "formatted_chat_data"

        model_args = copy.deepcopy(MODEL_ARGS)
        model_args.tokenizer_name_or_path = CUSTOM_TOKENIZER_TINYLLAMA

        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir

        with tempfile.NamedTemporaryFile(
            "w", delete=False, suffix=".yaml"
        ) as temp_yaml_file:
            with open(dataconfigfile, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            datasets = data["datasets"]
            for i, d in enumerate(datasets):
                d["data_paths"] = [datafiles[i]]
            yaml.dump(data, temp_yaml_file)
            data_args.data_config_path = temp_yaml_file.name

        sft_trainer.train(model_args, data_args, train_args)

        # validate the configs
        _validate_training(tempdir)
        checkpoint_path = _get_checkpoint_path(tempdir)

        # Load the model
        loaded_model = TunedCausalLM.load(checkpoint_path, MODEL_NAME)

        # Run inference on the text
        output_inference = loaded_model.run(
            '<|user|>\nProvide two rhyming words for the word "love"\n\
            <nopace></s><|assistant|>',
            max_new_tokens=50,
        )
        assert len(output_inference) > 0
        assert 'Provide two rhyming words for the word "love"' in output_inference


@pytest.mark.parametrize(
    "data_args",
    [
        (
            # sample hugging face dataset id
            configs.DataArguments(
                training_data_path="lhoestq/demo1",
                data_formatter_template="### Text:{{review}} \n\n### Stars: {{star}}",
                response_template="\n### Stars:",
            )
        )
    ],
)
def test_run_e2e_with_hf_dataset_id(data_args):
    """
    Check if we can run an e2e test with a hf dataset id as training_data_path.
    """
    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir

        sft_trainer.train(MODEL_ARGS, data_args, train_args)

        # validate ft tuning configs
        _validate_training(tempdir)

        # validate inference
        _test_run_inference(checkpoint_path=_get_checkpoint_path(tempdir))


@pytest.mark.skipif(
    not is_fms_accelerate_available(plugins="moe"),
    reason="Only runs if fms-accelerate is installed along with accelerated-moe plugin",
)
@pytest.mark.parametrize(
    "dataset_path",
    [
        TWITTER_COMPLAINTS_DATA_JSONL,
    ],
)
def test_run_moe_ft_and_inference(dataset_path):
    """Check if we can finetune a moe model and check if hf checkpoint is created"""
    with tempfile.TemporaryDirectory() as tempdir:
        data_args = copy.deepcopy(DATA_ARGS)
        data_args.training_data_path = dataset_path
        model_args = copy.deepcopy(MODEL_ARGS)
        model_args.model_name_or_path = "Isotonic/TinyMixtral-4x248M-MoE"
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir
        fast_moe_config = FastMoeConfig(fast_moe=FastMoe(ep_degree=1))
        sft_trainer.train(
            model_args, data_args, train_args, fast_moe_config=fast_moe_config
        )
        _test_run_inference(
            checkpoint_path=os.path.join(
                _get_checkpoint_path(tempdir), "hf_converted_checkpoint"
            )
        )


@pytest.mark.skipif(
    not is_fms_accelerate_available(plugins="moe"),
    reason="Only runs if fms-accelerate is installed along with accelerated-moe plugin",
)
@pytest.mark.parametrize(
    "dataset_path",
    [
        TWITTER_COMPLAINTS_DATA_JSONL,
    ],
)
def test_run_moe_ft_with_save_model_dir(dataset_path):
    """Check if we can finetune a moe model and check if hf checkpoint is created"""
    with tempfile.TemporaryDirectory() as tempdir:
        save_model_dir = os.path.join(tempdir, "save_model")
        data_args = copy.deepcopy(DATA_ARGS)
        data_args.training_data_path = dataset_path
        model_args = copy.deepcopy(MODEL_ARGS)
        model_args.model_name_or_path = "Isotonic/TinyMixtral-4x248M-MoE"
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir
        train_args.save_model_dir = save_model_dir
        fast_moe_config = FastMoeConfig(fast_moe=FastMoe(ep_degree=1))
        sft_trainer.train(
            model_args, data_args, train_args, fast_moe_config=fast_moe_config
        )
        assert os.path.exists(os.path.join(save_model_dir, "hf_converted_checkpoint"))


############################# Helper functions #############################
def _test_run_causallm_ft(training_args, model_args, data_args, tempdir):
    train_args = copy.deepcopy(training_args)
    train_args.output_dir = tempdir
    sft_trainer.train(model_args, data_args, train_args, None)

    # validate ft tuning configs
    _validate_training(tempdir)


def _test_run_inference(checkpoint_path):
    # Load the model
    loaded_model = TunedCausalLM.load(checkpoint_path)

    # Run inference on the text
    output_inference = loaded_model.run(
        "### Text: @NortonSupport Thanks much.\n\n### Label:", max_new_tokens=50
    )
    assert len(output_inference) > 0
    assert "### Text: @NortonSupport Thanks much.\n\n### Label:" in output_inference


def _validate_training(
    tempdir,
    check_eval=False,
    train_logs_file="training_logs.jsonl",
    check_scanner_file=False,
):
    assert any(x.startswith("checkpoint-") for x in os.listdir(tempdir))
    train_logs_file_path = "{}/{}".format(tempdir, train_logs_file)
    _validate_logfile(train_logs_file_path, check_eval)

    if check_scanner_file:
        _validate_hf_resource_scanner_file(tempdir)


def _validate_logfile(log_file_path, check_eval=False):
    train_log_contents = ""
    with open(log_file_path, encoding="utf-8") as f:
        train_log_contents = f.read()

    assert os.path.exists(log_file_path) is True
    assert os.path.getsize(log_file_path) > 0
    assert "training_loss" in train_log_contents

    if check_eval:
        assert "validation_loss" in train_log_contents


def _validate_hf_resource_scanner_file(tempdir):
    scanner_file_path = os.path.join(tempdir, "scanner_output.json")
    assert os.path.exists(scanner_file_path) is True
    assert os.path.getsize(scanner_file_path) > 0

    with open(scanner_file_path, "r", encoding="utf-8") as f:
        scanner_contents = json.load(f)

    assert scanner_contents["time_data"] is not None
    assert scanner_contents["mem_data"] is not None


def _get_checkpoint_path(dir_path):
    checkpoint_dirs = [
        d
        for d in os.listdir(dir_path)
        if os.path.isdir(os.path.join(dir_path, d)) and re.match(r"^checkpoint-\d+$", d)
    ]
    checkpoint_dirs.sort(key=lambda name: int(name.split("-")[-1]))
    return os.path.join(dir_path, checkpoint_dirs[-1])


def _get_adapter_config(dir_path):
    with open(os.path.join(dir_path, "adapter_config.json"), encoding="utf-8") as f:
        return json.load(f)


def _validate_adapter_config(adapter_config, peft_type, tokenizer_name_or_path=None):
    assert adapter_config.get("task_type") == "CAUSAL_LM"
    assert adapter_config.get("peft_type") == peft_type
    assert (
        (adapter_config.get("tokenizer_name_or_path") == tokenizer_name_or_path)
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

    with pytest.raises((DatasetGenerationError, ValueError)):
        sft_trainer.train(MODEL_ARGS, data_args, TRAIN_ARGS, PEFT_PT_ARGS)


def test_empty_data():
    """Ensure that malformatted data explodes due to failure to generate the dataset."""
    data_args = copy.deepcopy(DATA_ARGS)
    data_args.training_data_path = EMPTY_DATA

    with pytest.raises((DatasetGenerationError, ValueError)):
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


def test_run_with_additional_callbacks():
    """Ensure that train() can work with additional_callbacks"""

    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir

        sft_trainer.train(
            MODEL_ARGS,
            DATA_ARGS,
            train_args,
            PEFT_PT_ARGS,
            additional_callbacks=[TrainerCallback()],
        )


def test_run_with_bad_additional_callbacks():
    """Ensure that train() raises error with bad additional_callbacks"""

    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir

        with pytest.raises(
            ValueError, match="additional callbacks should be of type TrainerCallback"
        ):
            sft_trainer.train(
                MODEL_ARGS,
                DATA_ARGS,
                train_args,
                PEFT_PT_ARGS,
                additional_callbacks=["NotSupposedToBeHere"],
            )


def test_run_with_bad_experimental_metadata():
    """Ensure that train() throws error with bad experimental metadata"""

    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir

        metadata = "deadbeef"

        with pytest.raises(
            ValueError, match="exp metadata passed should be a dict with valid json"
        ):
            sft_trainer.train(
                MODEL_ARGS,
                DATA_ARGS,
                train_args,
                PEFT_PT_ARGS,
                additional_callbacks=[TrainerCallback()],
                exp_metadata=metadata,
            )


def test_run_with_good_experimental_metadata():
    """Ensure that train() can work with good experimental metadata"""

    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir

        metadata = {"dead": "beef"}

        sft_trainer.train(
            MODEL_ARGS,
            DATA_ARGS,
            train_args,
            PEFT_PT_ARGS,
            additional_callbacks=[TrainerCallback()],
            exp_metadata=metadata,
        )


@pytest.mark.parametrize(
    "dataset_path",
    [
        TWITTER_COMPLAINTS_TOKENIZED_JSONL,
        TWITTER_COMPLAINTS_TOKENIZED_JSON,
    ],
)
### Tests for pretokenized data
def test_pretokenized_dataset(dataset_path):
    """Ensure that we can provide a pretokenized dataset with input/output format."""
    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir
        data_args = copy.deepcopy(DATA_ARGS)
        data_args.dataset_text_field = None
        data_args.response_template = None
        data_args.training_data_path = dataset_path
        sft_trainer.train(MODEL_ARGS, data_args, train_args, PEFT_PT_ARGS)
        _validate_training(tempdir)


@pytest.mark.parametrize(
    "dataset_text_field, response_template",
    [
        ("foo", None),
        (None, "bar"),
    ],
)
def test_pretokenized_dataset_bad_args(dataset_text_field, response_template):
    """Ensure that we can't provide only dataset text field / response template for pretok data."""
    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir

        data_args = copy.deepcopy(DATA_ARGS)
        data_args.dataset_text_field = dataset_text_field
        data_args.response_template = response_template
        data_args.training_data_path = TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSONL
        # We should raise an error since we should not have a dataset text
        # field or a response template if we have pretokenized data
        with pytest.raises(ValueError):
            sft_trainer.train(MODEL_ARGS, data_args, train_args, PEFT_PT_ARGS)


def test_pretokenized_dataset_wrong_format():
    """Ensure that we fail to generate data if the data is in the wrong format."""
    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir

        data_args = copy.deepcopy(DATA_ARGS)
        data_args.dataset_text_field = None
        data_args.response_template = None
        data_args.training_data_path = TWITTER_COMPLAINTS_DATA_JSONL

        # It would be best to handle this in a way that is more understandable; we might
        # need to directly add validation prior to the dataset generation since datasets
        # is essentially swallowing a KeyError here.
        with pytest.raises(ValueError):
            sft_trainer.train(MODEL_ARGS, data_args, train_args, PEFT_PT_ARGS)


###########################################################################
### Tests for checking different cases for the argument additional_handlers
### The argument `additional_handlers` in train::sft_trainer.py is used to pass
### extra data handlers which should be a Dict[str,callable]


@pytest.mark.parametrize(
    "additional_handlers",
    [
        "thisisnotokay",
        [],
        {lambda x: {"x": x}: "notokayeither"},
        {"thisisfine": "thisisnot"},
    ],
)
def test_run_with_bad_additional_data_handlers(additional_handlers):
    """Ensure that bad additional_handlers argument (which is not Dict[str,callable])
    throws an error"""
    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir

        with pytest.raises(
            ValueError,
            match="Handler should be of type tuning.data_handler.DataHandler, and name of str",
        ):
            sft_trainer.train(
                MODEL_ARGS,
                DATA_ARGS,
                train_args,
                PEFT_PT_ARGS,
                additional_data_handlers=additional_handlers,
            )


def test_run_with_additional_data_handlers_as_none():
    """Ensure that additional_handlers as None should work."""
    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir

        sft_trainer.train(
            MODEL_ARGS,
            DATA_ARGS,
            train_args,
            PEFT_PT_ARGS,
            additional_data_handlers=None,
        )
        _validate_training(tempdir)


def test_run_by_passing_additional_data_handlers():
    """Ensure that good additional_handlers argument can take a
    data handler and can successfully run a e2e training."""
    # This is my test handler
    TEST_HANDLER = "my_test_handler"

    def test_handler(element, tokenizer, **kwargs):
        return add_tokenizer_eos_token(element, tokenizer, "custom_formatted_field")

    # This data config calls for data handler to be applied to dataset
    preprocessor_config = DataPreProcessorConfig()
    handler_config = DataHandlerConfig(name="my_test_handler", arguments=None)
    dataaset_config = DataSetConfig(
        name="test_dataset",
        data_paths=TWITTER_COMPLAINTS_DATA_JSON,
        data_handlers=[handler_config],
    )
    data_config = DataConfig(
        dataprocessor=preprocessor_config, datasets=[dataaset_config]
    )

    # dump the data config to a file, also test if json data config works
    with tempfile.NamedTemporaryFile(
        "w", delete=False, suffix=".json"
    ) as temp_data_file:
        data_config_raw = json.dumps(asdict(data_config))
        temp_data_file.write(data_config_raw)
        data_config_path = temp_data_file.name

    # now launch sft trainer after registering data handler
    with tempfile.TemporaryDirectory() as tempdir:
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir
        data_args = copy.deepcopy(DATA_ARGS)
        data_args.data_config_path = data_config_path
        data_args.dataset_text_field = "custom_formatted_field"

        sft_trainer.train(
            MODEL_ARGS,
            DATA_ARGS,
            train_args,
            PEFT_PT_ARGS,
            additional_data_handlers={
                TEST_HANDLER: DataHandler(
                    op=test_handler,
                    handler_type=DataHandlerType.MAP,
                    allows_batching=False,
                )
            },
        )
        _validate_training(tempdir)
