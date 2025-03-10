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
import glob
import json
import os
import tempfile

# Third Party
from datasets import Dataset, IterableDataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from trl import DataCollatorForCompletionOnlyLM
import datasets
import pyarrow
import pytest
import yaml

# First Party
from scripts.offline_data_processing import get_processed_dataset, save_dataset_shards
from tests.artifacts.predefined_data_configs import (
    DATA_CONFIG_APPLY_CUSTOM_JINJA_TEMPLATE_YAML,
    DATA_CONFIG_APPLY_CUSTOM_TEMPLATE_YAML,
    DATA_CONFIG_MULTIPLE_DATASETS_SAMPLING_YAML,
    DATA_CONFIG_MULTITURN_DATA_YAML,
    DATA_CONFIG_PRETOKENIZE_JSON_DATA_YAML,
    DATA_CONFIG_RENAME_SELECT_COLUMNS,
    DATA_CONFIG_TOKENIZE_AND_APPLY_INPUT_MASKING_YAML,
    DATA_CONFIG_YAML_STREAMING_INPUT_OUTPUT,
    DATA_CONFIG_YAML_STREAMING_PRETOKENIZED,
)
from tests.artifacts.testdata import (
    CHAT_DATA_MULTI_TURN,
    CHAT_DATA_SINGLE_TURN,
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
    TWITTER_COMPLAINTS_TOKENIZED_PARQUET,
)

# Local
from tuning.config import configs
from tuning.data.data_config import (
    DataHandlerConfig,
    DataPreProcessorConfig,
    DataSetConfig,
)
from tuning.data.data_preprocessing_utils import get_data_collator
from tuning.data.data_processors import DataPreProcessor, get_datapreprocessor
from tuning.data.setup_dataprocessor import (
    _process_dataconfig_file,
    is_pretokenized_dataset,
    process_dataargs,
)


@pytest.mark.parametrize(
    "datafile, column_names",
    [
        (
            TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSONL,
            set(["ID", "Label", "input", "output"]),
        ),
        (
            TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_ARROW,
            set(["ID", "Label", "input", "output", "sequence"]),
        ),
        (
            TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_PARQUET,
            set(["ID", "Label", "input", "output"]),
        ),
        (
            TWITTER_COMPLAINTS_TOKENIZED_JSONL,
            set(
                [
                    "Tweet text",
                    "ID",
                    "Label",
                    "text_label",
                    "output",
                    "input_ids",
                    "labels",
                ]
            ),
        ),
        (
            TWITTER_COMPLAINTS_TOKENIZED_ARROW,
            set(
                [
                    "Tweet text",
                    "ID",
                    "Label",
                    "text_label",
                    "output",
                    "input_ids",
                    "labels",
                ]
            ),
        ),
        (
            TWITTER_COMPLAINTS_TOKENIZED_PARQUET,
            set(
                [
                    "Tweet text",
                    "ID",
                    "Label",
                    "text_label",
                    "output",
                    "input_ids",
                    "labels",
                ]
            ),
        ),
        (
            TWITTER_COMPLAINTS_DATA_JSONL,
            set(["Tweet text", "ID", "Label", "text_label", "output"]),
        ),
        (
            TWITTER_COMPLAINTS_DATA_ARROW,
            set(["Tweet text", "ID", "Label", "text_label", "output"]),
        ),
        (
            TWITTER_COMPLAINTS_DATA_PARQUET,
            set(["Tweet text", "ID", "Label", "text_label", "output"]),
        ),
    ],
)
def test_load_dataset_with_datafile(datafile, column_names):
    """Ensure that both dataset is loaded with datafile."""
    processor = get_datapreprocessor(
        processor_config=DataPreProcessorConfig(), tokenizer=None
    )
    load_dataset = processor.load_dataset(
        datasetconfig=None,
        streaming=processor.processor_config.streaming,
        splitName="train",
        datafile=datafile,
    )
    assert set(load_dataset.column_names) == column_names


@pytest.mark.parametrize("hf_dataset, splitName", [("squad", "validation")])
def test_load_dataset_with_hf_dataset(hf_dataset, splitName):
    """Ensure that hf dataset could be loaded."""
    datasetconfig = DataSetConfig(
        name="text_dataset_input_output_masking", data_paths=[hf_dataset]
    )
    processor = get_datapreprocessor(
        processor_config=DataPreProcessorConfig(), tokenizer=None
    )
    load_dataset = processor.load_dataset(
        datasetconfig=datasetconfig,
        streaming=processor.processor_config.streaming,
        splitName=splitName,
        datafile=None,
    )
    assert processor.processor_config.streaming is False
    assert isinstance(load_dataset, Dataset)


@pytest.mark.parametrize(
    "datafile, column_names, datasetconfigname, builder",
    [
        (
            TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSONL,
            set(["ID", "Label", "input", "output"]),
            "text_dataset_input_output_masking",
            None,
        ),
        (
            TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_ARROW,
            set(["ID", "Label", "input", "output", "sequence"]),
            "text_dataset_input_output_masking",
            None,
        ),
        (
            TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_PARQUET,
            set(["ID", "Label", "input", "output"]),
            "text_dataset_input_output_masking",
            None,
        ),
        (
            TWITTER_COMPLAINTS_TOKENIZED_JSONL,
            set(
                [
                    "Tweet text",
                    "ID",
                    "Label",
                    "text_label",
                    "output",
                    "input_ids",
                    "labels",
                ]
            ),
            "pretokenized_dataset",
            None,
        ),
        (
            TWITTER_COMPLAINTS_TOKENIZED_PARQUET,
            set(
                [
                    "Tweet text",
                    "ID",
                    "Label",
                    "text_label",
                    "output",
                    "input_ids",
                    "labels",
                ]
            ),
            "pretokenized_dataset",
            None,
        ),
        (
            TWITTER_COMPLAINTS_DATA_JSONL,
            set(["Tweet text", "ID", "Label", "text_label", "output"]),
            "apply_custom_data_template",
            None,
        ),
        (
            TWITTER_COMPLAINTS_DATA_ARROW,
            set(["Tweet text", "ID", "Label", "text_label", "output"]),
            "apply_custom_data_template",
            None,
        ),
        (
            TWITTER_COMPLAINTS_DATA_PARQUET,
            set(["Tweet text", "ID", "Label", "text_label", "output"]),
            "apply_custom_data_template",
            None,
        ),
        (
            TWITTER_COMPLAINTS_DATA_PARQUET,
            set(["Tweet text", "ID", "Label", "text_label", "output"]),
            "apply_custom_data_template",
            "parquet",
        ),
    ],
)
def test_load_dataset_with_datasetconfig(
    datafile, column_names, datasetconfigname, builder
):
    """Ensure that both dataset is loaded with datafile."""
    datasetconfig = DataSetConfig(
        name=datasetconfigname, data_paths=[datafile], builder=builder
    )
    processor = get_datapreprocessor(
        processor_config=DataPreProcessorConfig(), tokenizer=None
    )
    load_dataset = processor.load_dataset(
        datasetconfig=datasetconfig,
        streaming=processor.processor_config.streaming,
        splitName="train",
        datafile=None,
    )
    assert set(load_dataset.column_names) == column_names


@pytest.mark.parametrize(
    "data_paths, datasetconfigname",
    [
        (
            ["fake/path"],
            "apply_custom_data_template",
        ),
        (
            [
                TWITTER_COMPLAINTS_DATA_PARQUET.replace(
                    "twitter_complaints_small.parquet", "not_exist.parquet"
                )
            ],
            "apply_custom_data_template",
        ),
    ],
)
def test_load_dataset_with_non_exist_path(data_paths, datasetconfigname):
    """Ensure that load_dataset raises error for non-exist paths."""
    datasetconfig = DataSetConfig(name=datasetconfigname, data_paths=data_paths)
    processor = get_datapreprocessor(
        processor_config=DataPreProcessorConfig(), tokenizer=None
    )
    with pytest.raises((datasets.exceptions.DatasetNotFoundError, ValueError)):
        processor.load_dataset(
            datasetconfig=datasetconfig,
            streaming=processor.processor_config.streaming,
            splitName="train",
            datafile=None,
        )


@pytest.mark.parametrize(
    "datafile, datasetconfigname, builder",
    [
        (TWITTER_COMPLAINTS_DATA_PARQUET, "apply_custom_data_template", "arrow"),
    ],
)
def test_load_dataset_with_datasetconfig_incorrect_builder(
    datafile, datasetconfigname, builder
):
    """Ensure that directory with incorrect builder cannot be passed in datasetconfig."""
    datasetconfig = DataSetConfig(
        name=datasetconfigname, data_paths=[datafile], builder=builder
    )
    processor = get_datapreprocessor(
        processor_config=DataPreProcessorConfig(), tokenizer=None
    )
    with pytest.raises(pyarrow.lib.ArrowInvalid):
        processor.load_dataset(
            datasetconfig=datasetconfig,
            streaming=processor.processor_config.streaming,
            splitName="train",
            datafile=None,
        )


@pytest.mark.parametrize(
    "datafile, datasetconfigname",
    [
        (
            TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSONL,
            "text_dataset_input_output_masking",
        ),
        (
            TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_PARQUET,
            "text_dataset_input_output_masking",
        ),
        (TWITTER_COMPLAINTS_TOKENIZED_JSONL, "pretokenized_dataset"),
        (TWITTER_COMPLAINTS_TOKENIZED_PARQUET, "pretokenized_dataset"),
        (TWITTER_COMPLAINTS_DATA_JSONL, "apply_custom_data_template"),
        (TWITTER_COMPLAINTS_DATA_PARQUET, "apply_custom_data_template"),
    ],
)
def test_load_dataset_with_dataconfig_and_datafile(datafile, datasetconfigname):
    """Ensure that both datasetconfig and datafile cannot be passed."""
    datasetconfig = DataSetConfig(name=datasetconfigname, data_paths=[datafile])
    processor = get_datapreprocessor(
        processor_config=DataPreProcessorConfig(), tokenizer=None
    )
    with pytest.raises(ValueError):
        processor.load_dataset(
            datasetconfig=datasetconfig,
            streaming=processor.processor_config.streaming,
            splitName="train",
            datafile=datafile,
        )


@pytest.mark.parametrize(
    "datasetconfig, column_names",
    [
        (
            DataSetConfig(
                name="text_dataset_input_output_masking",
                data_paths=[TWITTER_COMPLAINTS_DATA_DIR_JSON],
            ),
            set(["ID", "Label", "input", "output"]),
        ),
        (
            DataSetConfig(
                name="text_dataset_input_output_masking",
                data_paths=[TWITTER_COMPLAINTS_DATA_DIR_JSON],
                builder="json",
            ),
            set(["ID", "Label", "input", "output"]),
        ),
    ],
)
def test_load_dataset_with_dataconfig_and_datafolder(datasetconfig, column_names):
    """Ensure that directory can be passed in datasetconfig with/without builder."""
    processor = get_datapreprocessor(
        processor_config=DataPreProcessorConfig(), tokenizer=None
    )
    load_dataset = processor.load_dataset(
        datasetconfig=datasetconfig,
        streaming=processor.processor_config.streaming,
        splitName="train",
        datafile=None,
    )
    assert set(load_dataset.column_names) == column_names


@pytest.mark.parametrize(
    "datasetconfig",
    [
        DataSetConfig(
            name="text_dataset_input_output_masking",
            data_paths=[TWITTER_COMPLAINTS_DATA_DIR_JSON],
            builder="arrow",
        ),
    ],
)
def test_load_dataset_with_dataconfig_and_datafolder_incorrect_builder(datasetconfig):
    """Ensure that directory with incorrect builder cannot be passed in datasetconfig."""
    processor = get_datapreprocessor(
        processor_config=DataPreProcessorConfig(), tokenizer=None
    )
    with pytest.raises(pyarrow.lib.ArrowInvalid):
        processor.load_dataset(
            datasetconfig=datasetconfig,
            streaming=processor.processor_config.streaming,
            splitName="train",
            datafile=None,
        )


def test_load_dataset_without_dataconfig_and_datafile():
    """Ensure that both datasetconfig and datafile cannot be None."""
    processor = get_datapreprocessor(
        processor_config=DataPreProcessorConfig(), tokenizer=None
    )
    with pytest.raises(ValueError):
        processor.load_dataset(
            datasetconfig=None,
            streaming=processor.processor_config.streaming,
            splitName="train",
            datafile=None,
        )


@pytest.mark.parametrize(
    "data_paths, column_names, datasetconfigname, builder",
    [
        (
            [
                TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
                TWITTER_COMPLAINTS_DATA_DIR_JSON,
            ],
            set(["ID", "Label", "input", "output"]),
            "text_dataset_input_output_masking",
            None,
        ),
        (
            [
                TWITTER_COMPLAINTS_DATA_DIR_JSON,
                TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_PARQUET,
            ],
            set(["ID", "Label", "input", "output"]),
            "text_dataset_input_output_masking",
            None,
        ),
    ],
)
def test_load_dataset_with_datasetconfig_files_folders(
    data_paths, column_names, datasetconfigname, builder
):
    """Ensure that load_dataset works with passing combination of files and folders."""
    datasetconfig = DataSetConfig(
        name=datasetconfigname, data_paths=data_paths, builder=builder
    )
    processor = get_datapreprocessor(
        processor_config=DataPreProcessorConfig(), tokenizer=None
    )
    load_dataset = processor.load_dataset(
        datasetconfig=datasetconfig,
        streaming=processor.processor_config.streaming,
        splitName="train",
        datafile=None,
    )
    assert set(load_dataset.column_names) == column_names


@pytest.mark.parametrize(
    "data_paths, datasetconfigname, builder",
    [
        (
            [
                TWITTER_COMPLAINTS_DATA_DIR_JSON,
                TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
            ],
            "text_dataset_input_output_masking",
            "arrow",
        ),
    ],
)
def test_load_dataset_with_datasetconfig_files_folders_incorrect_builder(
    data_paths, datasetconfigname, builder
):
    """Ensure that load_dataset with passing combination of files and folders does support mismatch in format"""
    datasetconfig = DataSetConfig(
        name=datasetconfigname, data_paths=data_paths, builder=builder
    )
    processor = get_datapreprocessor(
        processor_config=DataPreProcessorConfig(), tokenizer=None
    )
    with pytest.raises(ValueError):
        processor.load_dataset(
            datasetconfig=datasetconfig,
            streaming=processor.processor_config.streaming,
            splitName="train",
            datafile=None,
        )


@pytest.mark.parametrize(
    "data, result",
    [
        (TWITTER_COMPLAINTS_DATA_JSONL, False),
        (
            Dataset.from_list(
                [
                    {
                        "input_ids": [9437, 29, 210],
                        "attention_mask": [1, 1, 1],
                        "labels": [1, 20, 30],
                    }
                ]
            ),
            True,
        ),
    ],
)
def test_is_pretokenized_data(data, result):
    """Ensure that the correct collator type is fetched based on the data args"""
    assert is_pretokenized_dataset(data=data) == result


@pytest.mark.parametrize(
    "packing, response_template, formatted_train_dataset,\
     max_seq_length, instruction_template, is_padding_free, expected_collator",
    [
        (
            False,
            "\n### Label:",
            datasets.load_dataset(
                "json",
                data_files=TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
                split="train",
            ),
            1024,
            None,
            False,
            DataCollatorForCompletionOnlyLM,
        ),
        (
            False,
            None,
            Dataset.from_list(
                [
                    {
                        "input_ids": [9437, 29, 210],
                        "attention_mask": [1, 1, 1],
                        "labels": [1, 20, 30],
                    }
                ]
            ),
            1024,
            None,
            False,
            DataCollatorForSeq2Seq,
        ),
        (
            False,
            "\n### Label:",
            datasets.load_dataset(
                "json",
                data_files=TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
                split="train",
            ),
            1024,
            "\n### Text:",
            False,
            DataCollatorForCompletionOnlyLM,
        ),
        (
            False,
            None,
            Dataset.from_list(
                [
                    {
                        "input_ids": [9437, 29, 210],
                        "attention_mask": [1, 1, 1],
                        "labels": [1, 20, 30],
                    }
                ]
            ),
            1024,
            "\n### Text:",
            False,
            DataCollatorForSeq2Seq,
        ),
        (
            False,
            None,
            datasets.load_dataset(
                "json",
                data_files=TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
                split="train",
            ),
            1024,
            None,
            True,
            DataCollatorForSeq2Seq,
        ),
    ],
)
def test_get_data_collator(
    packing,
    response_template,
    formatted_train_dataset,
    max_seq_length,
    instruction_template,
    is_padding_free,
    expected_collator,
):
    """Ensure that the correct collator type is fetched based on the data args"""
    collator = get_data_collator(
        packing,
        response_template,
        AutoTokenizer.from_pretrained(MODEL_NAME),
        is_pretokenized_dataset(formatted_train_dataset),
        max_seq_length,
        instruction_template,
        is_padding_free,
    )
    assert isinstance(collator, expected_collator)


# Tests for validating data args
# Invalid args return ValueError
@pytest.mark.parametrize(
    "data_args, packing",
    [
        # dataset_text_field with no response_template
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_DATA_JSONL,
                dataset_text_field="output",
            ),
            False,
        ),
        # Data formatter with no response template
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_DATA_JSONL,
                data_formatter_template="### Input: {{input}} \n\n### Response: {{output}}",
            ),
            False,
        ),
        # Response template with no dataset_text_field or formatter
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_DATA_JSONL,
                response_template="\n### Label:",
            ),
            False,
        ),
        # JSONL without input / output for no single sequence arguments
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_DATA_JSONL,
            ),
            False,
        ),
        # Pretokenized dataset with dataset_text_field
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_TOKENIZED_JSONL,
                dataset_text_field="output",
            ),
            False,
        ),
        # Pretokenized dataset with data formatter
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_TOKENIZED_JSONL,
                data_formatter_template="### Input: {{input}} \n\n### Response: {{output}}",
            ),
            False,
        ),
        # Pretokenized dataset with response template
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_TOKENIZED_JSONL,
                response_template="\n### Label:",
            ),
            False,
        ),
        # Pretokenized training dataset with validation data not pretokenized
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_TOKENIZED_JSONL,
                validation_data_path=TWITTER_COMPLAINTS_DATA_JSONL,
            ),
            False,
        ),
        # Pretokenized data with dataset_text_field and response template
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_TOKENIZED_JSONL,
                response_template="\n### Label:",
                dataset_text_field="output",
            ),
            False,
        ),
    ],
)
def test_process_data_args_throws_error_where_needed(data_args, packing):
    """Ensure that respective errors are thrown for incorrect data arguments"""
    with pytest.raises(ValueError):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        TRAIN_ARGS = configs.TrainingArguments(
            packing=packing,
            max_seq_length=1024,
            output_dir="tmp",  # Not needed but positional
        )
        (_, _, _, _, _, _) = process_dataargs(data_args, tokenizer, TRAIN_ARGS)


@pytest.mark.parametrize(
    "data_config_path, data_path",
    [
        (
            DATA_CONFIG_YAML_STREAMING_INPUT_OUTPUT,
            TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
        ),
        (DATA_CONFIG_YAML_STREAMING_PRETOKENIZED, TWITTER_COMPLAINTS_TOKENIZED_JSON),
    ],
)
def test_process_dataconfig_file_with_streaming(data_config_path, data_path):
    """Ensure that datasets are formatted and validated correctly based on the arguments passed in config file."""
    with open(data_config_path, "r") as f:
        yaml_content = yaml.safe_load(f)
    yaml_content["datasets"][0]["data_paths"][0] = data_path
    datasets_name = yaml_content["datasets"][0]["name"]

    # Modify input_field_name and output_field_name according to dataset
    if datasets_name == "text_dataset_input_output_masking":
        yaml_content["datasets"][0]["data_handlers"][0]["arguments"]["fn_kwargs"] = {
            "input_field_name": "input",
            "output_field_name": "output",
        }

    # Modify dataset_text_field and template according to dataset
    formatted_dataset_field = "formatted_data_field"
    if datasets_name == "apply_custom_data_template":
        template = "### Input: {{Tweet text}} \n\n ### Response: {{text_label}}"
        yaml_content["datasets"][0]["data_handlers"][0]["arguments"]["fn_kwargs"] = {
            "dataset_text_field": formatted_dataset_field,
            "template": template,
        }

    with tempfile.NamedTemporaryFile(
        "w", delete=False, suffix=".yaml"
    ) as temp_yaml_file:
        yaml.dump(yaml_content, temp_yaml_file)
        temp_yaml_file_path = temp_yaml_file.name
        data_args = configs.DataArguments(data_config_path=temp_yaml_file_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    TRAIN_ARGS = configs.TrainingArguments(
        max_steps=1,
        output_dir="tmp",  # Not needed but positional
    )

    (train_set, _, _) = _process_dataconfig_file(data_args, TRAIN_ARGS, tokenizer)
    assert isinstance(train_set, IterableDataset)
    if datasets_name == "text_dataset_input_output_masking":
        column_names = set(["input_ids", "attention_mask", "labels"])
        assert set(train_set.column_names) == column_names
    elif datasets_name == "pretokenized_dataset":
        assert set(["input_ids", "labels"]).issubset(set(train_set.column_names))
    elif datasets_name == "apply_custom_data_template":
        assert formatted_dataset_field in set(train_set.column_names)


@pytest.mark.parametrize(
    "data_config_path, data_path",
    [
        (
            DATA_CONFIG_YAML_STREAMING_INPUT_OUTPUT,
            TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
        ),
    ],
)
def test_process_dataconfig_file_with_streaming_no_max_steps_errors(
    data_config_path, data_path
):
    """Ensure that if max steps aren't passed with streaming, error is raised"""
    with open(data_config_path, "r") as f:
        yaml_content = yaml.safe_load(f)
    yaml_content["datasets"][0]["data_paths"][0] = data_path
    datasets_name = yaml_content["datasets"][0]["name"]

    # Modify input_field_name and output_field_name according to dataset
    if datasets_name == "text_dataset_input_output_masking":
        yaml_content["datasets"][0]["data_handlers"][0]["arguments"]["fn_kwargs"] = {
            "input_field_name": "input",
            "output_field_name": "output",
        }

    # Modify dataset_text_field and template according to dataset
    formatted_dataset_field = "formatted_data_field"
    if datasets_name == "apply_custom_data_template":
        template = "### Input: {{Tweet text}} \n\n ### Response: {{text_label}}"
        yaml_content["datasets"][0]["data_handlers"][0]["arguments"]["fn_kwargs"] = {
            "dataset_text_field": formatted_dataset_field,
            "template": template,
        }

    with tempfile.NamedTemporaryFile(
        "w", delete=False, suffix=".yaml"
    ) as temp_yaml_file:
        yaml.dump(yaml_content, temp_yaml_file)
        temp_yaml_file_path = temp_yaml_file.name
        data_args = configs.DataArguments(data_config_path=temp_yaml_file_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    TRAIN_ARGS = configs.TrainingArguments(
        output_dir="tmp",  # Not needed but positional
    )

    with pytest.raises(ValueError):
        (train_set, _, _) = _process_dataconfig_file(data_args, TRAIN_ARGS, tokenizer)


@pytest.mark.parametrize(
    "data_config_path, data_path",
    [
        (DATA_CONFIG_APPLY_CUSTOM_TEMPLATE_YAML, TWITTER_COMPLAINTS_DATA_JSON),
        (DATA_CONFIG_APPLY_CUSTOM_TEMPLATE_YAML, TWITTER_COMPLAINTS_DATA_JSONL),
        (DATA_CONFIG_APPLY_CUSTOM_TEMPLATE_YAML, TWITTER_COMPLAINTS_DATA_PARQUET),
        (DATA_CONFIG_APPLY_CUSTOM_TEMPLATE_YAML, TWITTER_COMPLAINTS_DATA_ARROW),
        (DATA_CONFIG_APPLY_CUSTOM_JINJA_TEMPLATE_YAML, TWITTER_COMPLAINTS_DATA_JSON),
        (DATA_CONFIG_APPLY_CUSTOM_JINJA_TEMPLATE_YAML, TWITTER_COMPLAINTS_DATA_JSONL),
        (DATA_CONFIG_APPLY_CUSTOM_JINJA_TEMPLATE_YAML, TWITTER_COMPLAINTS_DATA_PARQUET),
        (DATA_CONFIG_APPLY_CUSTOM_JINJA_TEMPLATE_YAML, TWITTER_COMPLAINTS_DATA_ARROW),
        (DATA_CONFIG_PRETOKENIZE_JSON_DATA_YAML, TWITTER_COMPLAINTS_TOKENIZED_JSON),
        (DATA_CONFIG_PRETOKENIZE_JSON_DATA_YAML, TWITTER_COMPLAINTS_TOKENIZED_JSONL),
        (DATA_CONFIG_PRETOKENIZE_JSON_DATA_YAML, TWITTER_COMPLAINTS_TOKENIZED_PARQUET),
        (DATA_CONFIG_PRETOKENIZE_JSON_DATA_YAML, TWITTER_COMPLAINTS_TOKENIZED_ARROW),
        (
            DATA_CONFIG_TOKENIZE_AND_APPLY_INPUT_MASKING_YAML,
            TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
        ),
        (
            DATA_CONFIG_TOKENIZE_AND_APPLY_INPUT_MASKING_YAML,
            TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSONL,
        ),
        (
            DATA_CONFIG_TOKENIZE_AND_APPLY_INPUT_MASKING_YAML,
            TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_PARQUET,
        ),
        (
            DATA_CONFIG_TOKENIZE_AND_APPLY_INPUT_MASKING_YAML,
            TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_ARROW,
        ),
    ],
)
def test_process_dataconfig_file(data_config_path, data_path):
    """Ensure that datasets are formatted and validated correctly based on the arguments passed in config file."""
    with open(data_config_path, "r") as f:
        yaml_content = yaml.safe_load(f)
    yaml_content["datasets"][0]["data_paths"][0] = data_path
    datasets_name = yaml_content["datasets"][0]["name"]

    # Modify input_field_name and output_field_name according to dataset
    if datasets_name == "text_dataset_input_output_masking":
        yaml_content["datasets"][0]["data_handlers"][0]["arguments"]["fn_kwargs"] = {
            "input_field_name": "input",
            "output_field_name": "output",
        }

    # Modify dataset_text_field and template according to dataset
    formatted_dataset_field = "formatted_data_field"
    if datasets_name in (
        "apply_custom_data_template",
        "apply_custom_data_jinja_template",
    ):
        template = "### Input: {{Tweet text}} \n\n ### Response: {{text_label}}"
        yaml_content["datasets"][0]["data_handlers"][0]["arguments"]["fn_kwargs"] = {
            "dataset_text_field": formatted_dataset_field,
            "template": template,
        }

    with tempfile.NamedTemporaryFile(
        "w", delete=False, suffix=".yaml"
    ) as temp_yaml_file:
        yaml.dump(yaml_content, temp_yaml_file)
        temp_yaml_file_path = temp_yaml_file.name
        data_args = configs.DataArguments(data_config_path=temp_yaml_file_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    TRAIN_ARGS = configs.TrainingArguments(
        output_dir="tmp",  # Not needed but positional
    )

    (train_set, _, _) = _process_dataconfig_file(data_args, TRAIN_ARGS, tokenizer)
    assert isinstance(train_set, Dataset)
    if datasets_name == "text_dataset_input_output_masking":
        column_names = set(["input_ids", "attention_mask", "labels"])
        assert set(train_set.column_names) == column_names
    elif datasets_name == "pretokenized_dataset":
        assert set(["input_ids", "labels"]).issubset(set(train_set.column_names))
    elif datasets_name in (
        "apply_custom_data_template",
        "apply_custom_data_jinja_template",
    ):
        assert formatted_dataset_field in set(train_set.column_names)


@pytest.mark.parametrize(
    "data_config_path, data_path, add_eos_token",
    [
        (DATA_CONFIG_APPLY_CUSTOM_TEMPLATE_YAML, TWITTER_COMPLAINTS_DATA_JSON, True),
        (DATA_CONFIG_APPLY_CUSTOM_TEMPLATE_YAML, TWITTER_COMPLAINTS_DATA_JSON, False),
        (
            DATA_CONFIG_APPLY_CUSTOM_JINJA_TEMPLATE_YAML,
            TWITTER_COMPLAINTS_DATA_JSON,
            True,
        ),
        (
            DATA_CONFIG_APPLY_CUSTOM_JINJA_TEMPLATE_YAML,
            TWITTER_COMPLAINTS_DATA_JSON,
            False,
        ),
        (
            DATA_CONFIG_TOKENIZE_AND_APPLY_INPUT_MASKING_YAML,
            TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
            True,
        ),
        (
            DATA_CONFIG_TOKENIZE_AND_APPLY_INPUT_MASKING_YAML,
            TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
            False,
        ),
    ],
)
def test_process_datahandler_eos_token(data_config_path, data_path, add_eos_token):
    """Ensure that the data handlers correctly apply add_eos_token flag to append/remove eos_token."""
    with open(data_config_path, "r") as f:
        yaml_content = yaml.safe_load(f)
    yaml_content["datasets"][0]["data_paths"][0] = data_path
    datasets_name = yaml_content["datasets"][0]["name"]

    # Modify input_field_name and output_field_name according to dataset
    if datasets_name == "text_dataset_input_output_masking":
        yaml_content["datasets"][0]["data_handlers"][0]["arguments"]["fn_kwargs"][
            "input_field_name"
        ] = "input"
        yaml_content["datasets"][0]["data_handlers"][0]["arguments"]["fn_kwargs"][
            "output_field_name"
        ] = "output"
        yaml_content["datasets"][0]["data_handlers"][0]["arguments"]["fn_kwargs"][
            "add_eos_token"
        ] = add_eos_token

    # Modify dataset_text_field and template according to dataset
    formatted_dataset_field = "formatted_data_field"
    if datasets_name in (
        "apply_custom_data_template",
        "apply_custom_data_jinja_template",
    ):
        template = "### Input: {{Tweet text}} \n\n ### Response: {{text_label}}"
        yaml_content["datasets"][0]["data_handlers"][0]["arguments"]["fn_kwargs"][
            "dataset_text_field"
        ] = formatted_dataset_field
        yaml_content["datasets"][0]["data_handlers"][0]["arguments"]["fn_kwargs"][
            "template"
        ] = template
        yaml_content["datasets"][0]["data_handlers"][0]["arguments"]["fn_kwargs"][
            "add_eos_token"
        ] = add_eos_token

    with tempfile.NamedTemporaryFile(
        "w", delete=False, suffix=".yaml"
    ) as temp_yaml_file:
        yaml.dump(yaml_content, temp_yaml_file)
        temp_yaml_file_path = temp_yaml_file.name
        data_args = configs.DataArguments(data_config_path=temp_yaml_file_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({"eos_token": "</s>"})

    TRAIN_ARGS = configs.TrainingArguments(
        output_dir="tmp",  # Not needed but positional
    )

    (train_set, _, _) = _process_dataconfig_file(data_args, TRAIN_ARGS, tokenizer)
    assert isinstance(train_set, Dataset)
    if datasets_name == "text_dataset_input_output_masking":
        column_names = set(["input_ids", "attention_mask", "labels"])
        assert set(train_set.column_names) == column_names
        assert (
            train_set[0]["input_ids"][-1] == tokenizer.eos_token_id
            if add_eos_token
            else train_set[0]["input_ids"][-1] != tokenizer.eos_token_id
        )
    elif datasets_name == "pretokenized_dataset":
        assert set(["input_ids", "labels"]).issubset(set(train_set.column_names))
    elif datasets_name in (
        "apply_custom_data_template",
        "apply_custom_data_jinja_template",
    ):
        assert formatted_dataset_field in set(train_set.column_names)
        assert (
            train_set[0][formatted_dataset_field].endswith(tokenizer.eos_token)
            if add_eos_token
            else not train_set[0][formatted_dataset_field].endswith(tokenizer.eos_token)
        )


@pytest.mark.parametrize(
    "data_config_path, data_path_list",
    [
        (
            DATA_CONFIG_APPLY_CUSTOM_TEMPLATE_YAML,
            [TWITTER_COMPLAINTS_DATA_JSON, TWITTER_COMPLAINTS_DATA_JSON],
        ),
        (
            DATA_CONFIG_APPLY_CUSTOM_TEMPLATE_YAML,
            [
                TWITTER_COMPLAINTS_DATA_JSONL,
                TWITTER_COMPLAINTS_DATA_JSONL,
                TWITTER_COMPLAINTS_DATA_JSONL,
            ],
        ),
        (
            DATA_CONFIG_APPLY_CUSTOM_TEMPLATE_YAML,
            [TWITTER_COMPLAINTS_DATA_PARQUET, TWITTER_COMPLAINTS_DATA_PARQUET],
        ),
        (
            DATA_CONFIG_APPLY_CUSTOM_TEMPLATE_YAML,
            [TWITTER_COMPLAINTS_DATA_ARROW, TWITTER_COMPLAINTS_DATA_ARROW],
        ),
        (
            DATA_CONFIG_APPLY_CUSTOM_TEMPLATE_YAML,
            [TWITTER_COMPLAINTS_DATA_JSON, TWITTER_COMPLAINTS_DATA_PARQUET],
        ),
        (
            DATA_CONFIG_PRETOKENIZE_JSON_DATA_YAML,
            [TWITTER_COMPLAINTS_TOKENIZED_JSON, TWITTER_COMPLAINTS_TOKENIZED_JSON],
        ),
        (
            DATA_CONFIG_PRETOKENIZE_JSON_DATA_YAML,
            [TWITTER_COMPLAINTS_TOKENIZED_JSONL, TWITTER_COMPLAINTS_TOKENIZED_JSONL],
        ),
        (
            DATA_CONFIG_PRETOKENIZE_JSON_DATA_YAML,
            [
                TWITTER_COMPLAINTS_TOKENIZED_PARQUET,
                TWITTER_COMPLAINTS_TOKENIZED_PARQUET,
                TWITTER_COMPLAINTS_TOKENIZED_PARQUET,
            ],
        ),
        (
            DATA_CONFIG_PRETOKENIZE_JSON_DATA_YAML,
            [TWITTER_COMPLAINTS_TOKENIZED_ARROW, TWITTER_COMPLAINTS_TOKENIZED_ARROW],
        ),
        (
            DATA_CONFIG_TOKENIZE_AND_APPLY_INPUT_MASKING_YAML,
            [
                TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
                TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
            ],
        ),
        (
            DATA_CONFIG_TOKENIZE_AND_APPLY_INPUT_MASKING_YAML,
            [
                TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSONL,
                TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSONL,
            ],
        ),
        (
            DATA_CONFIG_TOKENIZE_AND_APPLY_INPUT_MASKING_YAML,
            [
                TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_PARQUET,
                TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_PARQUET,
            ],
        ),
        (
            DATA_CONFIG_TOKENIZE_AND_APPLY_INPUT_MASKING_YAML,
            [
                TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_ARROW,
                TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_ARROW,
                TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_ARROW,
            ],
        ),
        (
            DATA_CONFIG_TOKENIZE_AND_APPLY_INPUT_MASKING_YAML,
            [
                TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_PARQUET,
                TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSONL,
            ],
        ),
    ],
)
def test_process_dataconfig_multiple_files(data_config_path, data_path_list):
    """Ensure that datasets with multiple files are formatted and validated correctly based on the arguments passed in config file."""
    with open(data_config_path, "r") as f:
        yaml_content = yaml.safe_load(f)
    yaml_content["datasets"][0]["data_paths"] = data_path_list
    datasets_name = yaml_content["datasets"][0]["name"]

    # Modify input_field_name and output_field_name according to dataset
    if datasets_name == "text_dataset_input_output_masking":
        yaml_content["datasets"][0]["data_handlers"][0]["arguments"]["fn_kwargs"] = {
            "input_field_name": "input",
            "output_field_name": "output",
        }

    # Modify dataset_text_field and template according to dataset
    formatted_dataset_field = "formatted_data_field"
    if datasets_name == "apply_custom_data_template":
        template = "### Input: {{Tweet text}} \n\n ### Response: {{text_label}}"
        yaml_content["datasets"][0]["data_handlers"][0]["arguments"]["fn_kwargs"] = {
            "dataset_text_field": formatted_dataset_field,
            "template": template,
        }

    with tempfile.NamedTemporaryFile(
        "w", delete=False, suffix=".yaml"
    ) as temp_yaml_file:
        yaml.dump(yaml_content, temp_yaml_file)
        temp_yaml_file_path = temp_yaml_file.name
        data_args = configs.DataArguments(data_config_path=temp_yaml_file_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    TRAIN_ARGS = configs.TrainingArguments(
        output_dir="tmp",  # Not needed but positional
    )

    (train_set, _, _) = _process_dataconfig_file(data_args, TRAIN_ARGS, tokenizer)
    assert isinstance(train_set, Dataset)
    if datasets_name == "text_dataset_input_output_masking":
        column_names = set(["input_ids", "attention_mask", "labels"])
        assert set(train_set.column_names) == column_names
    elif datasets_name == "pretokenized_dataset":
        assert set(["input_ids", "labels"]).issubset(set(train_set.column_names))
    elif datasets_name == "apply_custom_data_template":
        assert formatted_dataset_field in set(train_set.column_names)


@pytest.mark.parametrize(
    "data_config_path, data_paths, builder",
    [
        (
            DATA_CONFIG_TOKENIZE_AND_APPLY_INPUT_MASKING_YAML,
            [os.path.join(TWITTER_COMPLAINTS_DATA_DIR_JSON, "*.json")],
            None,
        ),
        (
            DATA_CONFIG_TOKENIZE_AND_APPLY_INPUT_MASKING_YAML,
            [os.path.join(TWITTER_COMPLAINTS_DATA_DIR_JSON, "*.json")],
            "json",
        ),
        (
            DATA_CONFIG_TOKENIZE_AND_APPLY_INPUT_MASKING_YAML,
            [os.path.join(TWITTER_COMPLAINTS_DATA_DIR_JSON, "*")],
            "json",
        ),
        (
            DATA_CONFIG_TOKENIZE_AND_APPLY_INPUT_MASKING_YAML,
            [os.path.join(TWITTER_COMPLAINTS_DATA_DIR_JSON, "*complaints*")],
            "json",
        ),
        (
            DATA_CONFIG_TOKENIZE_AND_APPLY_INPUT_MASKING_YAML,
            [TWITTER_COMPLAINTS_DATA_DIR_JSON],
            None,
        ),
        (
            DATA_CONFIG_TOKENIZE_AND_APPLY_INPUT_MASKING_YAML,
            [TWITTER_COMPLAINTS_DATA_DIR_JSON],
            "json",
        ),
    ],
)
def test_process_dataconfig_multiple_files_folders_with_globbing(
    data_config_path, data_paths, builder
):
    """Ensure that datasets files matching globbing pattern are formatted and validated correctly based on the arguments passed in config file."""
    with open(data_config_path, "r") as f:
        yaml_content = yaml.safe_load(f)

    yaml_content["datasets"][0]["data_paths"] = data_paths
    yaml_content["datasets"][0]["builder"] = builder

    with tempfile.NamedTemporaryFile(
        "w", delete=False, suffix=".yaml"
    ) as temp_yaml_file:
        yaml.dump(yaml_content, temp_yaml_file)
        temp_yaml_file_path = temp_yaml_file.name
        data_args = configs.DataArguments(data_config_path=temp_yaml_file_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    TRAIN_ARGS = configs.TrainingArguments(
        output_dir="tmp",  # Not needed but positional
    )

    (train_set, _, _) = _process_dataconfig_file(data_args, TRAIN_ARGS, tokenizer)
    assert isinstance(train_set, Dataset)
    assert set(["input_ids", "attention_mask", "labels"]).issubset(
        set(train_set.column_names)
    )

    path_or_pattern = data_paths[0]
    if os.path.isdir(path_or_pattern):
        # Construct a pattern for JSON files in this directory
        pattern = os.path.join(path_or_pattern, "*.json")
    else:
        # Assume path_or_pattern is already a pattern
        pattern = path_or_pattern

    data_len = sum(len(json.load(open(file, "r"))) for file in glob.glob(pattern))
    assert len(train_set) == data_len


@pytest.mark.parametrize(
    "data_paths, datasetconfigname, builder",
    [
        (
            [os.path.join(TWITTER_COMPLAINTS_DATA_DIR_JSON, "*")],
            "tokenize_and_apply_input_masking",
            None,
        ),
        (
            [os.path.join(TWITTER_COMPLAINTS_DATA_DIR_JSON, "*complaints*")],
            "tokenize_and_apply_input_masking",
            None,
        ),
        (["*squad"], "tokenize_and_apply_input_masking", None),
        (
            [TWITTER_COMPLAINTS_DATA_DIR_JSON.replace("datafolder", "dataf*")],
            "tokenize_and_apply_input_masking",
            None,
        ),
        (
            [TWITTER_COMPLAINTS_DATA_DIR_JSON],
            DATA_CONFIG_TOKENIZE_AND_APPLY_INPUT_MASKING_YAML,
            "parquet",
        ),
    ],
)
def test_process_dataconfig_multiple_files_folders_without_builder(
    data_paths, datasetconfigname, builder
):
    """Ensure that datasets folders / files without ext and builder
    OR HF datasets passed via globbing pattern raises error."""
    datasetconfig = DataSetConfig(
        name=datasetconfigname, data_paths=data_paths, builder=builder
    )
    processor = get_datapreprocessor(
        processor_config=DataPreProcessorConfig(), tokenizer=None
    )
    with pytest.raises(
        (datasets.exceptions.DatasetNotFoundError, ValueError, pyarrow.lib.ArrowInvalid)
    ):
        processor.load_dataset(
            datasetconfig=datasetconfig,
            streaming=processor.processor_config.streaming,
            splitName="train",
            datafile=None,
        )


@pytest.mark.parametrize(
    "datafiles, datasetconfigname",
    [
        (
            [
                [
                    TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_PARQUET,
                    TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_PARQUET,
                ],
                [
                    TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
                    TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
                ],
                [
                    TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSONL,
                    TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSONL,
                ],
            ],
            DATA_CONFIG_MULTIPLE_DATASETS_SAMPLING_YAML,
        ),
    ],
)
def test_process_dataconfig_multiple_datasets_datafiles_sampling(
    datafiles, datasetconfigname
):
    """Ensure that multiple datasets with multiple files are formatted and validated correctly."""
    with open(datasetconfigname, "r") as f:
        yaml_content = yaml.safe_load(f)
    yaml_content["datasets"][0]["data_paths"] = datafiles[0]
    yaml_content["datasets"][1]["data_paths"] = datafiles[1]
    yaml_content["datasets"][2]["data_paths"] = datafiles[2]

    with tempfile.NamedTemporaryFile(
        "w", delete=False, suffix=".yaml"
    ) as temp_yaml_file:
        yaml.dump(yaml_content, temp_yaml_file)
        temp_yaml_file_path = temp_yaml_file.name
        data_args = configs.DataArguments(data_config_path=temp_yaml_file_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    TRAIN_ARGS = configs.TrainingArguments(
        packing=False,
        max_seq_length=1024,
        output_dir="tmp",
    )
    (train_set, eval_set, _, _, _, _) = process_dataargs(
        data_args=data_args, tokenizer=tokenizer, train_args=TRAIN_ARGS
    )

    assert isinstance(train_set, Dataset)
    if eval_set:
        assert isinstance(eval_set, Dataset)

    assert set(["input_ids", "attention_mask", "labels"]).issubset(
        set(train_set.column_names)
    )
    if eval_set:
        assert set(["input_ids", "attention_mask", "labels"]).issubset(
            set(eval_set.column_names)
        )


@pytest.mark.parametrize(
    "data_args, is_padding_free",
    [
        # single sequence JSON and response template
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_DATA_JSON,
                validation_data_path=TWITTER_COMPLAINTS_DATA_JSON,
                dataset_text_field="output",
                response_template="\n### Label:",
            ),
            False,
        ),
        # single sequence JSONL and response template
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_DATA_JSONL,
                validation_data_path=TWITTER_COMPLAINTS_DATA_JSONL,
                dataset_text_field="output",
                response_template="\n### Label:",
            ),
            False,
        ),
        # single sequence PARQUET and response template
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_DATA_PARQUET,
                validation_data_path=TWITTER_COMPLAINTS_DATA_PARQUET,
                dataset_text_field="output",
                response_template="\n### Label:",
            ),
            False,
        ),
        # data formatter template with input/output JSON
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
                validation_data_path=TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
                data_formatter_template="### Text:{{input}} \n\n### Label: {{output}}",
                response_template="\n### Label:",
            ),
            False,
        ),
        # data formatter template with input/output JSONL
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSONL,
                validation_data_path=TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSONL,
                data_formatter_template="### Text:{{input}} \n\n### Label: {{output}}",
                response_template="\n### Label:",
            ),
            False,
        ),
        # data formatter template with input/output PARQUET
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_PARQUET,
                validation_data_path=TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_PARQUET,
                data_formatter_template="### Text:{{input}} \n\n### Label: {{output}}",
                response_template="\n### Label:",
            ),
            False,
        ),
        # input/output JSON with masking on input
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
                validation_data_path=TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
            ),
            False,
        ),
        # input/output JSONL with masking on input
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSONL,
                validation_data_path=TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSONL,
            ),
            False,
        ),
        # input/output PARQUET with masking on input
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_PARQUET,
                validation_data_path=TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_PARQUET,
            ),
            False,
        ),
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_DATA_JSON,
                validation_data_path=TWITTER_COMPLAINTS_DATA_JSON,
                dataset_text_field="output",
            ),
            True,
        ),
    ],
)
def test_process_dataargs(data_args, is_padding_free):
    """Ensure that the train/eval data are properly formatted based on the data args / text field"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    max_seq_length = 5
    TRAIN_ARGS = configs.TrainingArguments(
        packing=False,
        max_seq_length=max_seq_length,
        output_dir="tmp",  # Not needed but positional
    )
    (train_set, eval_set, dataset_text_field, _, _, _) = process_dataargs(
        data_args, tokenizer, TRAIN_ARGS, is_padding_free=is_padding_free
    )
    assert isinstance(train_set, Dataset)
    assert isinstance(eval_set, Dataset)
    if dataset_text_field is None:
        column_names = set(["input_ids", "attention_mask", "labels"])
        assert set(eval_set.column_names) == column_names
        assert set(train_set.column_names) == column_names
        assert len(train_set[0]["input_ids"]) == max_seq_length
    else:
        assert dataset_text_field in train_set.column_names
        assert dataset_text_field in eval_set.column_names


@pytest.mark.parametrize(
    "data_args",
    [
        # JSON pretokenized train and validation datasets
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_TOKENIZED_JSON,
                validation_data_path=TWITTER_COMPLAINTS_TOKENIZED_JSON,
            )
        ),
        # JSONL pretokenized train and validation datasets
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_TOKENIZED_JSONL,
                validation_data_path=TWITTER_COMPLAINTS_TOKENIZED_JSONL,
            )
        ),
        # PARQUET pretokenized train and validation datasets
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_TOKENIZED_PARQUET,
                validation_data_path=TWITTER_COMPLAINTS_TOKENIZED_PARQUET,
            )
        ),
        # JSON pretokenized train datasets
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_TOKENIZED_JSON,
            )
        ),
        # JSONL pretokenized train datasets
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_TOKENIZED_JSONL,
            )
        ),
        # ARROW pretokenized train datasets
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_TOKENIZED_ARROW,
            )
        ),
        # PARQUET pretokenized train datasets
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_TOKENIZED_PARQUET,
            )
        ),
    ],
)
def test_process_dataargs_pretokenized(data_args):
    """Ensure that pretokenized datasets are loaded and returned as is"""
    TRAIN_ARGS = configs.TrainingArguments(
        packing=False,
        max_seq_length=1024,
        output_dir="tmp",  # Not needed but positional
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    (train_set, eval_set, _, _, _, _) = process_dataargs(
        data_args, tokenizer, TRAIN_ARGS
    )
    assert isinstance(train_set, Dataset)
    if eval_set:
        assert isinstance(eval_set, Dataset)

    assert set(["input_ids", "labels"]).issubset(set(train_set.column_names))
    if eval_set:
        assert set(["input_ids", "labels"]).issubset(set(eval_set.column_names))


@pytest.mark.parametrize(
    "datafile, column_names, datasetconfigname",
    [
        (
            TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
            set(["ID", "Label", "input", "output"]),
            "text_dataset_input_output_masking",
        ),
        (
            TWITTER_COMPLAINTS_TOKENIZED_JSON,
            set(
                [
                    "Tweet text",
                    "ID",
                    "Label",
                    "text_label",
                    "output",
                    "input_ids",
                    "labels",
                ]
            ),
            "pretokenized_dataset",
        ),
        (
            TWITTER_COMPLAINTS_DATA_JSON,
            set(["Tweet text", "ID", "Label", "text_label", "output"]),
            "apply_custom_data_template",
        ),
    ],
)
def test_process_dataset_configs(datafile, column_names, datasetconfigname):
    """Test process_dataset_configs for expected output."""
    dataprocessor_config = DataPreProcessorConfig()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    processor = DataPreProcessor(
        processor_config=dataprocessor_config,
        tokenizer=tokenizer,
    )
    datasetconfig = [DataSetConfig(name=datasetconfigname, data_paths=[datafile])]
    train_dataset = processor.process_dataset_configs(dataset_configs=datasetconfig)

    assert isinstance(train_dataset, Dataset)
    assert set(train_dataset.column_names) == column_names

    with open(datafile, "r") as file:
        data = json.load(file)
    assert len(train_dataset) == len(data)


@pytest.mark.parametrize(
    "datafiles, sampling, datasetconfigname",
    [
        (
            [
                TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_ARROW,
                TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSONL,
                TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_PARQUET,
            ],
            [0.3, None, 0.3],
            DATA_CONFIG_MULTIPLE_DATASETS_SAMPLING_YAML,
        ),
        (
            [
                TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_ARROW,
                TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSONL,
                TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_PARQUET,
            ],
            [0.3, 0.5, 0.3],
            DATA_CONFIG_MULTIPLE_DATASETS_SAMPLING_YAML,
        ),
    ],
)
def test_process_dataset_configs_with_sampling_error(
    datafiles, sampling, datasetconfigname
):
    """Ensure that if sampling ratios aren't correctly passed (don't add up to 1.0), error is raised"""
    data_args = configs.DataArguments()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    TRAIN_ARGS = configs.TrainingArguments(
        packing=False,
        max_seq_length=1024,
        output_dir="tmp",  # Not needed but positional
    )

    with tempfile.NamedTemporaryFile(
        "w", delete=False, suffix=".yaml"
    ) as temp_yaml_file:
        with open(datasetconfigname, "r") as f:
            data = yaml.safe_load(f)
            datasets = data["datasets"]
            for i in range(len(datasets)):
                d = datasets[i]
                d["data_paths"][0] = datafiles[i]
                d["sampling"] = sampling[i]
            yaml.dump(data, temp_yaml_file)
        data_args.data_config_path = temp_yaml_file.name

    with pytest.raises(ValueError):
        (_, _, _, _, _, _) = process_dataargs(
            data_args=data_args, tokenizer=tokenizer, train_args=TRAIN_ARGS
        )


@pytest.mark.parametrize(
    "datafile, rename, select, final, datasetconfigname",
    [
        (
            TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
            {"input": "instruction", "output": "response"},
            None,
            ["ID", "Label", "instruction", "response"],
            DATA_CONFIG_RENAME_SELECT_COLUMNS,
        ),
        (
            TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
            None,
            ["ID", "input", "output"],
            ["ID", "input", "output"],
            DATA_CONFIG_RENAME_SELECT_COLUMNS,
        ),
        (
            TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
            {"input": "instruction", "output": "response"},
            ["Label", "instruction", "response"],
            ["Label", "instruction", "response"],
            DATA_CONFIG_RENAME_SELECT_COLUMNS,
        ),
    ],
)
def test_rename_and_select_dataset_columns(
    datafile, rename, select, final, datasetconfigname
):
    """Test process_dataset_configs for expected output."""
    dataprocessor_config = DataPreProcessorConfig()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    processor = DataPreProcessor(
        processor_config=dataprocessor_config,
        tokenizer=tokenizer,
    )

    handlers = []
    if rename:
        handlers.append(
            DataHandlerConfig(
                name="rename_columns", arguments={"column_mapping": rename}
            )
        )
    if select:
        handlers.append(
            DataHandlerConfig(name="select_columns", arguments={"column_names": select})
        )
    data_paths = [datafile]

    datasetconfig = [
        DataSetConfig(
            name=datasetconfigname, data_paths=data_paths, data_handlers=handlers
        )
    ]
    train_dataset = processor.process_dataset_configs(dataset_configs=datasetconfig)

    assert isinstance(train_dataset, Dataset)
    assert set(train_dataset.column_names) == set(final)

    with open(datafile, "r") as file:
        data = json.load(file)
    assert len(train_dataset) == len(data)


@pytest.mark.parametrize(
    "datafile, datasetconfigname",
    [
        (
            CHAT_DATA_SINGLE_TURN,
            DATA_CONFIG_MULTITURN_DATA_YAML,
        ),
        (
            CHAT_DATA_MULTI_TURN,
            DATA_CONFIG_MULTITURN_DATA_YAML,
        ),
    ],
)
def test_get_processed_dataset(datafile, datasetconfigname):
    """
    Ensure functions in offline_data_preprocessing script,
    get_processed_dataset and save_dataset_shards process
    and saves the formatted dataset correctly.
    """

    DATA_ARGS = configs.DataArguments()
    DATA_ARGS.response_template = "<|assistant|>"
    DATA_ARGS.instruction_template = "<|user|>"
    DATA_ARGS.dataset_text_field = "formatted_chat_data"
    MODEL_ARGS = configs.ModelArguments(
        model_name_or_path=MODEL_NAME, use_flash_attn=False
    )
    columns = [DATA_ARGS.dataset_text_field]
    num_dataset_shards = 2

    with open(datasetconfigname, "r") as f:
        yaml_content = yaml.safe_load(f)
        datasets = [
            {
                "data_paths": [datafile],
                "data_handlers": [
                    {
                        "name": "apply_tokenizer_chat_template",
                        "arguments": {
                            "fn_kwargs": {
                                "dataset_text_field": DATA_ARGS.dataset_text_field
                            },
                            "batched": False,
                            "remove_columns": "all",
                        },
                    }
                ],
            }
        ]
        yaml_content["datasets"] = datasets

    with tempfile.NamedTemporaryFile(
        "w", delete=False, suffix=".yaml"
    ) as temp_yaml_file:
        yaml.dump(yaml_content, temp_yaml_file)
        temp_yaml_file_path = temp_yaml_file.name
        DATA_ARGS.data_config_path = temp_yaml_file_path

    with tempfile.TemporaryDirectory() as tmpdirname:
        TRAIN_ARGS = configs.TrainingArguments(
            output_dir=tmpdirname, max_seq_length=4096
        )
        formatted_train_dataset, _ = get_processed_dataset(
            model_args=MODEL_ARGS, data_args=DATA_ARGS, train_args=TRAIN_ARGS
        )

        assert isinstance(formatted_train_dataset, Dataset)
        assert set(formatted_train_dataset.column_names) == set(columns)
        assert len(formatted_train_dataset) == sum(1 for _ in open(datafile))

        train_dataset_dir = os.path.join(TRAIN_ARGS.output_dir, "train_dataset")
        save_dataset_shards(
            formatted_train_dataset,
            train_dataset_dir,
            num_dataset_shards,
            "train_dataset",
        )
        assert len(os.listdir(train_dataset_dir)) == num_dataset_shards
