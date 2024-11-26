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

# Third Party
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from trl import DataCollatorForCompletionOnlyLM
import datasets
import pytest

# First Party
from tests.artifacts.testdata import (
    MALFORMATTED_DATA,
    MODEL_NAME,
    TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
    TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSONL,
    TWITTER_COMPLAINTS_DATA_JSON,
    TWITTER_COMPLAINTS_DATA_JSONL,
    TWITTER_COMPLAINTS_TOKENIZED_JSON,
    TWITTER_COMPLAINTS_TOKENIZED_JSONL,
)

# Local
from tuning.config import configs
from tuning.data.data_config import DataLoaderConfig, DataSetConfig
from tuning.data.data_preprocessing_utils import (
    combine_sequence,
    get_data_collator,
    validate_data_args,
)
from tuning.data.data_processors import HFBasedDataPreProcessor
from tuning.data.setup_dataprocessor import is_pretokenized_dataset, process_dataargs


@pytest.mark.parametrize(
    "input_element,output_element,expected_res",
    [
        ("foo ", "bar", "foo bar"),
        ("foo\n", "bar", "foo\nbar"),
        ("foo\t", "bar", "foo\tbar"),
        ("foo", "bar", "foo bar"),
    ],
)
def test_combine_sequence(input_element, output_element, expected_res):
    """Ensure that input / output elements are combined with correct whitespace handling."""
    comb_seq = combine_sequence(input_element, output_element)
    assert isinstance(comb_seq, str)
    assert comb_seq == expected_res


@pytest.mark.parametrize(
    "input_element,output_element,expected_res",
    [
        ("foo ", "bar", "foo bar"),
        ("foo\n", "bar", "foo\nbar"),
        ("foo\t", "bar", "foo\tbar"),
        ("foo", "bar", "foo bar"),
    ],
)
def test_combine_sequence_adds_eos(input_element, output_element, expected_res):
    """Ensure that input / output elements are combined with correct whitespace handling."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    comb_seq = combine_sequence(input_element, output_element, tokenizer.eos_token)
    expected_res += tokenizer.eos_token
    assert isinstance(comb_seq, str)
    assert comb_seq == expected_res


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
def test_validate_args(data_args, packing):
    """Ensure that respective errors are thrown for incorrect data arguments"""
    with pytest.raises(ValueError):
        is_traindata_tokenized = is_pretokenized_dataset(data_args.training_data_path)
        is_evaldata_tokenized = is_pretokenized_dataset(data_args.validation_data_path)
        validate_data_args(
            data_args, packing, is_traindata_tokenized, is_evaldata_tokenized
        )


@pytest.mark.parametrize(
    "data_args, packing",
    [
        # pretokenized train dataset and no validation dataset passed
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_TOKENIZED_JSONL,
            ),
            False,
        ),
        # pretokenized train and validation datasets
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_TOKENIZED_JSONL,
                validation_data_path=TWITTER_COMPLAINTS_TOKENIZED_JSONL,
            ),
            False,
        ),
    ],
)
def test_validate_args_pretokenized(data_args, packing):
    """Ensure that supported data args do not error out when passing pretokenized datasets"""
    is_traindata_tokenized = is_pretokenized_dataset(data_args.training_data_path)
    is_evaldata_tokenized = is_pretokenized_dataset(data_args.validation_data_path)
    validate_data_args(
        data_args, packing, is_traindata_tokenized, is_evaldata_tokenized
    )


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
    dataloaderconfig = DataLoaderConfig()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    processor = HFBasedDataPreProcessor(
        dataloaderconfig=dataloaderconfig,
        tokenizer=tokenizer,
    )
    datasetconfig = [DataSetConfig(name=datasetconfigname, data_paths=[datafile])]
    train_dataset = processor.process_dataset_configs(dataset_configs=datasetconfig)

    assert isinstance(train_dataset, Dataset)
    assert set(train_dataset.column_names) == column_names

    with open(datafile, "r") as file:
        data = json.load(file)
    assert len(train_dataset) == len(data)
