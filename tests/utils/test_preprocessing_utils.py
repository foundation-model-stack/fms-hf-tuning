# Third Party
from datasets import Dataset
from datasets.exceptions import DatasetGenerationError
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from trl import DataCollatorForCompletionOnlyLM
import pytest

# First Party
from tests.data import (
    MALFORMATTED_DATA,
    TWITTER_COMPLAINTS_DATA,
    TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT,
)

# Local
from tuning.config import configs
from tuning.utils.preprocessing_utils import (
    combine_sequence,
    get_data_trainer_kwargs,
    get_preprocessed_dataset,
    load_hf_dataset_from_jsonl_file,
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


# Tests for loading the dataset from disk
def test_load_hf_dataset_from_jsonl_file():
    input_field_name = "Tweet text"
    output_field_name = "text_label"
    data = load_hf_dataset_from_jsonl_file(
        TWITTER_COMPLAINTS_DATA,
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
        load_hf_dataset_from_jsonl_file(
            TWITTER_COMPLAINTS_DATA, input_field_name="foo", output_field_name="bar"
        )


def test_load_hf_dataset_from_malformatted_data():
    """Ensure that we explode if the data is not properly formatted."""
    # NOTE: The actual keys don't matter here
    with pytest.raises(DatasetGenerationError):
        load_hf_dataset_from_jsonl_file(
            MALFORMATTED_DATA, input_field_name="foo", output_field_name="bar"
        )


def test_load_hf_dataset_from_jsonl_file_duplicate_keys():
    """Ensure we cannot have the same key for input / output."""
    with pytest.raises(ValueError):
        load_hf_dataset_from_jsonl_file(
            TWITTER_COMPLAINTS_DATA,
            input_field_name="Tweet text",
            output_field_name="Tweet text",
        )


# Tests for custom masking / preprocessing logic
@pytest.mark.parametrize("max_sequence_length", [1, 10, 100, 1000])
def test_get_preprocessed_dataset(max_sequence_length):
    tokenizer = AutoTokenizer.from_pretrained("Maykeye/TinyLLama-v0")
    preprocessed_data = get_preprocessed_dataset(
        data_path=TWITTER_COMPLAINTS_DATA,
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


# Tests for fetching train args
@pytest.mark.parametrize(
    "use_validation_data, collator_type, packing",
    [
        (True, None, True),
        (False, None, True),
        (True, DataCollatorForCompletionOnlyLM, False),
        (False, DataCollatorForCompletionOnlyLM, False),
    ],
)
def test_get_trainer_kwargs_with_response_template_and_text_field(
    use_validation_data, collator_type, packing
):
    training_data_path = TWITTER_COMPLAINTS_DATA
    validation_data_path = training_data_path if use_validation_data else None
    # Expected columns in the raw loaded dataset for the twitter data
    column_names = set(["Tweet text", "ID", "Label", "text_label", "output"])
    trainer_kwargs = get_data_trainer_kwargs(
        training_data_path=training_data_path,
        validation_data_path=validation_data_path,
        packing=packing,
        response_template="\n### Label:",
        max_sequence_length=100,
        tokenizer=AutoTokenizer.from_pretrained("Maykeye/TinyLLama-v0"),
        dataset_text_field="output",
    )
    assert len(trainer_kwargs) == 3
    # If we are packing, we should not have a data collator
    if collator_type is None:
        assert trainer_kwargs["data_collator"] is None
    else:
        assert isinstance(trainer_kwargs["data_collator"], collator_type)

    # We should only have a validation dataset if one is present
    if validation_data_path is None:
        assert trainer_kwargs["eval_dataset"] is None
    else:
        assert isinstance(trainer_kwargs["eval_dataset"], Dataset)
        assert set(trainer_kwargs["eval_dataset"].column_names) == column_names

    assert isinstance(trainer_kwargs["train_dataset"], Dataset)
    assert set(trainer_kwargs["train_dataset"].column_names) == column_names


@pytest.mark.parametrize("use_validation_data", [True, False])
def test_get_trainer_kwargs_with_custom_masking(use_validation_data):
    training_data_path = TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT
    validation_data_path = training_data_path if use_validation_data else None
    # Expected columns in the raw loaded dataset for the twitter data
    column_names = set(["input_ids", "attention_mask", "labels"])
    trainer_kwargs = get_data_trainer_kwargs(
        training_data_path=training_data_path,
        validation_data_path=validation_data_path,
        packing=False,
        response_template=None,
        max_sequence_length=100,
        tokenizer=AutoTokenizer.from_pretrained("Maykeye/TinyLLama-v0"),
        dataset_text_field=None,
    )
    assert len(trainer_kwargs) == 4
    # If we are packing, we should not have a data collator
    assert isinstance(trainer_kwargs["data_collator"], DataCollatorForSeq2Seq)

    # We should only have a validation dataset if one is present
    if validation_data_path is None:
        assert trainer_kwargs["eval_dataset"] is None
    else:
        assert isinstance(trainer_kwargs["eval_dataset"], Dataset)
        assert set(trainer_kwargs["eval_dataset"].column_names) == column_names

    assert isinstance(trainer_kwargs["train_dataset"], Dataset)
    assert set(trainer_kwargs["train_dataset"].column_names) == column_names
    # Needed to sidestep TRL validation
    assert trainer_kwargs["formatting_func"] is not None


# Tests for validating data args
# Invalid args return ValueError
@pytest.mark.parametrize(
    "data_args, packing",
    [
        # dataset_text_field with no response_template
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_DATA,
                dataset_text_field="output",
            ),
            False,
        ),
        # response template with no dataset_text_field or formatter
        (
            configs.DataArguments(
                training_data_path=TWITTER_COMPLAINTS_DATA,
                response_template="\n### Label:",
            ),
            False,
        ),
    ],
)
def test_validate_args(data_args, packing):
    with pytest.raises(ValueError):
        validate_data_args(data_args, packing)
