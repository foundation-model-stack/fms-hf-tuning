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

# Third Party
from datasets import Dataset, IterableDatasetDict
from transformers import AutoTokenizer
import datasets
import pytest

# First Party
from tests.artifacts.testdata import (
    MODEL_NAME,
    TWITTER_COMPLAINTS_DATA_JSONL,
    TWITTER_COMPLAINTS_TOKENIZED_JSON,
    TWITTER_COMPLAINTS_TOKENIZED_ONLY_INPUT_IDS_JSON,
)

# Local
from tuning.data.data_handlers import (
    apply_custom_data_formatting_template,
    apply_custom_jinja_template,
    combine_sequence,
    duplicate_columns,
    skip_large_columns,
    tokenize,
)


def test_apply_custom_formatting_template():
    """Tests custom formatting data handler returns correct formatted response"""
    json_dataset = datasets.load_dataset(
        "json", data_files=TWITTER_COMPLAINTS_DATA_JSONL
    )
    template = "### Input: {{Tweet text}} \n\n ### Response: {{text_label}}"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    formatted_dataset_field = "formatted_data_field"
    formatted_dataset = json_dataset.map(
        apply_custom_data_formatting_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "dataset_text_field": formatted_dataset_field,
            "template": template,
        },
    )
    # First response from the data file that is read.
    expected_response = (
        "### Input: @HMRCcustomers No this is my first job"
        + " \n\n ### Response: no complaint"
        + tokenizer.eos_token
    )

    # a new dataset_text_field is created in Dataset
    assert formatted_dataset_field in formatted_dataset["train"][0]
    assert formatted_dataset["train"][0][formatted_dataset_field] == expected_response


def test_apply_custom_formatting_jinja_template():
    """Tests custom formatting data handler with jinja template dataset returns correct formatted response"""
    json_dataset = datasets.load_dataset(
        "json", data_files=TWITTER_COMPLAINTS_DATA_JSONL
    )
    template = "### Input: {{Tweet text}} \n\n ### Response: {{text_label}}"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    formatted_dataset_field = "formatted_data_field"
    formatted_dataset = json_dataset.map(
        apply_custom_jinja_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "dataset_text_field": formatted_dataset_field,
            "template": template,
        },
    )
    # First response from the data file that is read.
    expected_response = (
        "### Input: @HMRCcustomers No this is my first job"
        + " \n\n ### Response: no complaint"
        + tokenizer.eos_token
    )

    assert formatted_dataset_field in formatted_dataset["train"][0]
    assert formatted_dataset["train"][0][formatted_dataset_field] == expected_response


def test_apply_custom_formatting_template_iterable():
    """Tests custom formatting data handler with iterable dataset returns correct formatted response"""
    json_dataset = datasets.load_dataset(
        "json", data_files=TWITTER_COMPLAINTS_DATA_JSONL, streaming=True
    )
    template = "### Input: {{Tweet text}} \n\n ### Response: {{text_label}}"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    formatted_dataset_field = "formatted_data_field"
    formatted_dataset = json_dataset.map(
        apply_custom_data_formatting_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "dataset_text_field": formatted_dataset_field,
            "template": template,
        },
    )
    assert isinstance(formatted_dataset, IterableDatasetDict)

    # First response from the data file that is read.
    expected_response = (
        "### Input: @HMRCcustomers No this is my first job"
        + " \n\n ### Response: no complaint"
        + tokenizer.eos_token
    )

    first_sample = next(iter(formatted_dataset["train"]))

    # a new dataset_text_field is created in Dataset
    assert formatted_dataset_field in first_sample
    assert first_sample[formatted_dataset_field] == expected_response


def test_apply_custom_formatting_template_gives_error_with_wrong_keys():
    """Tests that the formatting function will throw error if wrong keys are passed to template"""
    json_dataset = datasets.load_dataset(
        "json", data_files=TWITTER_COMPLAINTS_DATA_JSONL
    )
    template = "### Input: {{not found}} \n\n ### Response: {{text_label}}"
    formatted_dataset_field = "formatted_data_field"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    with pytest.raises(KeyError):
        json_dataset.map(
            apply_custom_data_formatting_template,
            fn_kwargs={
                "tokenizer": tokenizer,
                "dataset_text_field": formatted_dataset_field,
                "template": template,
            },
        )


@pytest.mark.parametrize(
    "template",
    [
        "### Input: {{ not found }} \n\n ### Response: {{ text_label }}",
        "### Input: }} Tweet text {{ \n\n ### Response: {{ text_label }}",
        "### Input: {{ Tweet text }} \n\n ### Response: {{ ''.__class__ }}",
        "### Input: {{ Tweet text }} \n\n ### Response: {{ undefined_variable.split() }}",
    ],
)
def test_apply_custom_formatting_jinja_template_gives_error_with_wrong_keys(template):
    """Tests that the jinja formatting function will throw error if wrong keys are passed to template"""
    json_dataset = datasets.load_dataset(
        "json", data_files=TWITTER_COMPLAINTS_DATA_JSONL
    )
    formatted_dataset_field = "formatted_data_field"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    with pytest.raises((KeyError, ValueError)):
        json_dataset.map(
            apply_custom_jinja_template,
            fn_kwargs={
                "tokenizer": tokenizer,
                "dataset_text_field": formatted_dataset_field,
                "template": template,
            },
        )


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
    "dataset, old, new",
    [
        (TWITTER_COMPLAINTS_DATA_JSONL, "input_ids", "labels"),
        (TWITTER_COMPLAINTS_TOKENIZED_JSON, "input_ids", "labels"),
        (TWITTER_COMPLAINTS_DATA_JSONL, None, None),
        (TWITTER_COMPLAINTS_DATA_JSONL, "input_ids", None),
    ],
)
def test_duplicate_columns_throws_error_on_wrong_args(dataset, old, new):
    """Ensure that duplicate_columns data handler throws error if column names are wrong."""
    d = datasets.load_dataset("json", data_files=dataset)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    with pytest.raises(ValueError):
        d.map(
            duplicate_columns,
            fn_kwargs={
                "tokenizer": tokenizer,
                "old_column": old,
                "new_column": new,
            },
        )


def test_duplicate_columns_copies_columns():
    """Ensure that duplicate_columns data handler copies and maintains both columns."""
    old = "input_ids"
    new = "labels"
    d = datasets.load_dataset(
        "json", data_files=TWITTER_COMPLAINTS_TOKENIZED_ONLY_INPUT_IDS_JSON
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    updated_dataaset = d.map(
        duplicate_columns,
        fn_kwargs={
            "tokenizer": tokenizer,
            "old_column": old,
            "new_column": new,
        },
    )

    first_element = updated_dataaset["train"][0]
    assert new in first_element
    assert old in first_element
    assert first_element[new] == first_element[old]


def test_tokenizer_data_handler_tokenizes():
    "Ensure tokenizer data handler tokenizes the input properly with proper truncation"
    d = datasets.load_dataset("json", data_files=TWITTER_COMPLAINTS_DATA_JSONL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset_text_field = "output"
    truncation = True
    max_length = 10

    updated_dataaset = d.map(
        tokenize,
        fn_kwargs={
            "tokenizer": tokenizer,
            "dataset_text_field": dataset_text_field,
            "truncation": truncation,
            "max_length": max_length,
        },
    )

    assert "input_ids" in updated_dataaset["train"][0]
    for element in updated_dataaset["train"]:
        assert len(element["input_ids"]) <= max_length


@pytest.mark.parametrize(
    "column_name, max_length",
    [
        (None, None),
        ("input_ids", None),
        (1024, 1024),
        ("not_existing", "not_existing"),
    ],
)
def test_skip_large_columns_handler_throws_error_on_bad_args(column_name, max_length):
    "Ensure that skip large columns handler throws error on bad arguments"
    d = datasets.load_dataset("json", data_files=TWITTER_COMPLAINTS_DATA_JSONL)
    fn_kwargs = {}
    fn_kwargs["column_name"] = column_name
    fn_kwargs["max_length"] = max_length

    with pytest.raises(ValueError):
        filtered = d.filter(skip_large_columns, fn_kwargs=fn_kwargs)


def test_skip_large_columns_handler():
    "Ensure that skip large columns handler skips dataset as intended"

    def test_dataset_generator():
        for i in range(0, 100):
            yield {"input": list(range(0, i + 1))}

    d = Dataset.from_generator(test_dataset_generator)
    fn_kwargs = {}
    fn_kwargs["column_name"] = "input"
    fn_kwargs["max_length"] = 60

    filtered = d.filter(skip_large_columns, fn_kwargs=fn_kwargs)
    assert len(filtered) == 60
