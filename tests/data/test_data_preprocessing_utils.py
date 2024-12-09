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
import json
import tempfile

# Third Party
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from trl import DataCollatorForCompletionOnlyLM
import datasets
import pytest
import yaml

# First Party
from tests.artifacts.predefined_data_configs import (
    APPLY_CUSTOM_TEMPLATE_YAML,
    PRETOKENIZE_JSON_DATA_YAML,
    TOKENIZE_AND_APPLY_INPUT_MASKING_YAML,
)
from tests.artifacts.testdata import (
    MODEL_NAME,
    TWITTER_COMPLAINTS_DATA_ARROW,
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
from tuning.data.data_config import DataPreProcessorConfig, DataSetConfig
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
        datasetconfig=None, splitName="train", datafile=datafile
    )
    assert set(load_dataset.column_names) == column_names


@pytest.mark.parametrize(
    "datafile, column_names, datasetconfigname",
    [
        (
            TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSONL,
            set(["ID", "Label", "input", "output"]),
            "text_dataset_input_output_masking",
        ),
        (
            TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_ARROW,
            set(["ID", "Label", "input", "output", "sequence"]),
            "text_dataset_input_output_masking",
        ),
        (
            TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_PARQUET,
            set(["ID", "Label", "input", "output"]),
            "text_dataset_input_output_masking",
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
        ),
        (
            TWITTER_COMPLAINTS_DATA_JSONL,
            set(["Tweet text", "ID", "Label", "text_label", "output"]),
            "apply_custom_data_template",
        ),
        (
            TWITTER_COMPLAINTS_DATA_ARROW,
            set(["Tweet text", "ID", "Label", "text_label", "output"]),
            "apply_custom_data_template",
        ),
        (
            TWITTER_COMPLAINTS_DATA_PARQUET,
            set(["Tweet text", "ID", "Label", "text_label", "output"]),
            "apply_custom_data_template",
        ),
    ],
)
def test_load_dataset_with_datasetconfig(datafile, column_names, datasetconfigname):
    """Ensure that both dataset is loaded with datafile."""
    datasetconfig = DataSetConfig(name=datasetconfigname, data_paths=[datafile])
    processor = get_datapreprocessor(
        processor_config=DataPreProcessorConfig(), tokenizer=None
    )
    load_dataset = processor.load_dataset(
        datasetconfig=datasetconfig, splitName="train", datafile=None
    )
    assert set(load_dataset.column_names) == column_names


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
            datasetconfig=datasetconfig, splitName="train", datafile=datafile
        )


def test_load_dataset_without_dataconfig_and_datafile():
    """Ensure that both datasetconfig and datafile cannot be None."""
    processor = get_datapreprocessor(
        processor_config=DataPreProcessorConfig(), tokenizer=None
    )
    with pytest.raises(ValueError):
        processor.load_dataset(datasetconfig=None, splitName="train", datafile=None)


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
    "packing, response_template, formatted_train_dataset, max_seq_length, expected_collator",
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
            DataCollatorForSeq2Seq,
        ),
    ],
)
def test_get_data_collator(
    packing,
    response_template,
    formatted_train_dataset,
    max_seq_length,
    expected_collator,
):
    """Ensure that the correct collator type is fetched based on the data args"""
    collator = get_data_collator(
        packing,
        response_template,
        AutoTokenizer.from_pretrained(MODEL_NAME),
        is_pretokenized_dataset(formatted_train_dataset),
        max_seq_length,
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
        # Pretokenized data with packing to True
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_TOKENIZED_JSONL,
            ),
            True,
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
        (APPLY_CUSTOM_TEMPLATE_YAML, TWITTER_COMPLAINTS_DATA_JSON),
        (APPLY_CUSTOM_TEMPLATE_YAML, TWITTER_COMPLAINTS_DATA_JSONL),
        (APPLY_CUSTOM_TEMPLATE_YAML, TWITTER_COMPLAINTS_DATA_PARQUET),
        (PRETOKENIZE_JSON_DATA_YAML, TWITTER_COMPLAINTS_TOKENIZED_JSON),
        (PRETOKENIZE_JSON_DATA_YAML, TWITTER_COMPLAINTS_TOKENIZED_JSONL),
        (PRETOKENIZE_JSON_DATA_YAML, TWITTER_COMPLAINTS_TOKENIZED_PARQUET),
        (
            TOKENIZE_AND_APPLY_INPUT_MASKING_YAML,
            TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
        ),
        (
            TOKENIZE_AND_APPLY_INPUT_MASKING_YAML,
            TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSONL,
        ),
        (
            TOKENIZE_AND_APPLY_INPUT_MASKING_YAML,
            TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_PARQUET,
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
    (train_set, _, _) = _process_dataconfig_file(data_args, tokenizer)
    assert isinstance(train_set, Dataset)
    if datasets_name == "text_dataset_input_output_masking":
        column_names = set(["input_ids", "attention_mask", "labels"])
        assert set(train_set.column_names) == column_names
    elif datasets_name == "pretokenized_dataset":
        assert set(["input_ids", "labels"]).issubset(set(train_set.column_names))
    elif datasets_name == "apply_custom_data_template":
        assert formatted_dataset_field in set(train_set.column_names)


@pytest.mark.parametrize(
    "data_args",
    [
        # single sequence JSON and response template
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_DATA_JSON,
                validation_data_path=TWITTER_COMPLAINTS_DATA_JSON,
                dataset_text_field="output",
                response_template="\n### Label:",
            )
        ),
        # single sequence JSONL and response template
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_DATA_JSONL,
                validation_data_path=TWITTER_COMPLAINTS_DATA_JSONL,
                dataset_text_field="output",
                response_template="\n### Label:",
            )
        ),
        # single sequence PARQUET and response template
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_DATA_PARQUET,
                validation_data_path=TWITTER_COMPLAINTS_DATA_PARQUET,
                dataset_text_field="output",
                response_template="\n### Label:",
            )
        ),
        # data formatter template with input/output JSON
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
                validation_data_path=TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
                data_formatter_template="### Text:{{input}} \n\n### Label: {{output}}",
                response_template="\n### Label:",
            )
        ),
        # data formatter template with input/output JSONL
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSONL,
                validation_data_path=TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSONL,
                data_formatter_template="### Text:{{input}} \n\n### Label: {{output}}",
                response_template="\n### Label:",
            )
        ),
        # data formatter template with input/output PARQUET
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_PARQUET,
                validation_data_path=TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_PARQUET,
                data_formatter_template="### Text:{{input}} \n\n### Label: {{output}}",
                response_template="\n### Label:",
            )
        ),
        # input/output JSON with masking on input
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
                validation_data_path=TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
            )
        ),
        # input/output JSONL with masking on input
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSONL,
                validation_data_path=TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSONL,
            )
        ),
        # input/output PARQUET with masking on input
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_PARQUET,
                validation_data_path=TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_PARQUET,
            )
        ),
    ],
)
def test_process_dataargs(data_args):
    """Ensure that the train/eval data are properly formatted based on the data args / text field"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    TRAIN_ARGS = configs.TrainingArguments(
        packing=False,
        max_seq_length=1024,
        output_dir="tmp",  # Not needed but positional
    )
    (train_set, eval_set, dataset_text_field, _, _, _) = process_dataargs(
        data_args, tokenizer, TRAIN_ARGS
    )
    assert isinstance(train_set, Dataset)
    assert isinstance(eval_set, Dataset)
    if dataset_text_field is None:
        column_names = set(["input_ids", "attention_mask", "labels"])
        assert set(eval_set.column_names) == column_names
        assert set(train_set.column_names) == column_names
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
