# Third Party
from datasets import Dataset
from datasets.exceptions import DatasetGenerationError
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from trl import DataCollatorForCompletionOnlyLM
import pytest

# First Party
from tests.data import (
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
from tuning.utils.preprocessing_utils import (
    combine_sequence,
    format_dataset,
    get_data_collator,
    get_formatted_dataset_with_single_sequence,
    get_preprocessed_dataset,
    is_pretokenized_dataset,
    load_hf_dataset_from_file,
    validate_data_args,
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


# Tests for loading the dataset from disk
@pytest.mark.parametrize(
    "dataset_path",
    [TWITTER_COMPLAINTS_DATA_JSONL, TWITTER_COMPLAINTS_DATA_JSON],
)
def test_load_hf_dataset_from_file(dataset_path):
    input_field_name = "Tweet text"
    output_field_name = "text_label"
    data = load_hf_dataset_from_file(
        dataset_path,
        input_field_name=input_field_name,
        output_field_name=output_field_name,
    )
    # Our dataset should contain dicts that contain the input / output field name types
    next_data = next(iter(data))
    assert input_field_name in next_data
    assert output_field_name in next_data


def test_load_hf_dataset_from_jsonl_file_wrong_keys():
    """Ensure that we explode if the keys are not in the jsonl file."""
    with pytest.raises(DatasetGenerationError):
        load_hf_dataset_from_file(
            TWITTER_COMPLAINTS_DATA_JSONL,
            input_field_name="foo",
            output_field_name="bar",
        )


def test_load_hf_dataset_from_malformatted_data():
    """Ensure that we explode if the data is not properly formatted."""
    # NOTE: The actual keys don't matter here
    with pytest.raises(DatasetGenerationError):
        load_hf_dataset_from_file(
            MALFORMATTED_DATA, input_field_name="foo", output_field_name="bar"
        )


def test_load_hf_dataset_from_jsonl_file_duplicate_keys():
    """Ensure we cannot have the same key for input / output."""
    with pytest.raises(ValueError):
        load_hf_dataset_from_file(
            TWITTER_COMPLAINTS_DATA_JSONL,
            input_field_name="Tweet text",
            output_field_name="Tweet text",
        )


# Tests for custom masking / preprocessing logic
@pytest.mark.parametrize(
    "dataset_path, max_sequence_length",
    [
        (TWITTER_COMPLAINTS_DATA_JSONL, 1),
        (TWITTER_COMPLAINTS_DATA_JSONL, 10),
        (TWITTER_COMPLAINTS_DATA_JSONL, 100),
        (TWITTER_COMPLAINTS_DATA_JSONL, 1000),
        (TWITTER_COMPLAINTS_DATA_JSON, 1),
        (TWITTER_COMPLAINTS_DATA_JSON, 10),
        (TWITTER_COMPLAINTS_DATA_JSON, 100),
        (TWITTER_COMPLAINTS_DATA_JSON, 1000),
    ],
)
def test_get_preprocessed_dataset(dataset_path, max_sequence_length):
    """Ensure we can handle preprocessed datasets with different max_sequence_lengths
    to ensure proper tokenization and truncation.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    preprocessed_data = get_preprocessed_dataset(
        data_path=dataset_path,
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        input_field_name="Tweet text",
        output_field_name="text_label",
    )
    for tok_res in preprocessed_data:
        # Since the padding is left to the collator, there should be no 0s in the attention mask yet
        assert sum(tok_res["attention_mask"]) == len(tok_res["attention_mask"])
        # If the source text isn't empty, we start with masked inputs
        assert tok_res["labels"][0] == -100
        # All keys in the produced record must be the same length
        key_lengths = {len(tok_res[k]) for k in tok_res.keys()}
        assert len(key_lengths) == 1
        # And also that length should be less than or equal to the max length depending on if we
        # are going up to / over the max size and truncating - padding is handled separately
        assert key_lengths.pop() <= max_sequence_length


@pytest.mark.parametrize(
    "packing, response_template, formatted_train_dataset, max_seq_length, expected_collator",
    [
        (
            False,
            "\n### Label:",
            load_hf_dataset_from_file(
                TWITTER_COMPLAINTS_DATA_JSONL,
                input_field_name="Tweet text",
                output_field_name="text_label",
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
        formatted_train_dataset,
        max_seq_length,
    )
    assert isinstance(collator, expected_collator)


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
        validate_data_args(data_args, packing)


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
    validate_data_args(data_args, packing)


@pytest.mark.parametrize(
    "data_path, dataset_text_field, data_formatter_template",
    [
        (TWITTER_COMPLAINTS_DATA_JSON, "output", None),
        (TWITTER_COMPLAINTS_DATA_JSONL, "output", None),
        (
            TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON,
            "formatted_field",
            "### Text:{{input}} \n\n### Label: {{output}}",
        ),
        (
            TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSONL,
            "formatted_field",
            "### Text:{{input}} \n\n### Label: {{output}}",
        ),
    ],
)
def test_get_formatted_dataset_with_single_sequence(
    data_path, dataset_text_field, data_formatter_template
):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    formatted_dataset = get_formatted_dataset_with_single_sequence(
        data_path, dataset_text_field, tokenizer, data_formatter_template
    )
    assert isinstance(formatted_dataset, Dataset)
    assert dataset_text_field in formatted_dataset.column_names


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
                dataset_text_field="formatted_field",
                data_formatter_template="### Text:{{input}} \n\n### Label: {{output}}",
            )
        ),
        # data formatter template with input/output JSONL
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSONL,
                validation_data_path=TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSONL,
                dataset_text_field="formatted_field",
                data_formatter_template="### Text:{{input}} \n\n### Label: {{output}}",
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
def test_format_dataset(data_args):
    """Ensure that the train/eval data are properly formatted based on the data args / text field"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_set, eval_set, dataset_text_field = format_dataset(
        data_args, tokenizer, max_seq_length=1024
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
def test_format_dataset_pretokenized(data_args):
    """Ensure that pretokenized datasets are loaded and returned as is"""
    train_set, eval_set, _ = format_dataset(data_args, None, max_seq_length=1024)
    assert isinstance(train_set, Dataset)
    if eval_set:
        assert isinstance(eval_set, Dataset)

    assert set(["input_ids", "labels"]).issubset(set(train_set.column_names))
    if eval_set:
        assert set(["input_ids", "labels"]).issubset(set(eval_set.column_names))
