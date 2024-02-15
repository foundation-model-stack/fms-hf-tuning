"""Tests validating the correctness of data formatting utilities.

IMPORTANT: Currently these tests are sensitive to dataset caching! If anything looks strange,
try running the test in isolation. In the future, we should patch the map to do things in memory
so that each test runs independently.
"""
import pytest
import json
import os
from tuning.utils.data_format_utils import preprocess_function, tokenize_function
from transformers import AutoTokenizer

SAMPLE_DATA = os.path.join(os.path.dirname(__file__), "sample_complaints.json")
TOKENIZER = AutoTokenizer.from_pretrained("bigscience/bloom-560m")


@pytest.mark.parametrize("use_iterable_dataset", [True, False])
def test_dataset_length(use_iterable_dataset):
    """Ensure that preprocessing a dataset yields a HF dataset of the same length."""
    with open(SAMPLE_DATA, "r") as data_ptr:
        data_records = [json.loads(line) for line in data_ptr.readlines() if line]

    hf_dataset = preprocess_function(
        data_path=SAMPLE_DATA,
        tokenizer=TOKENIZER,
        batch_size=1000,
        use_iterable_dataset=use_iterable_dataset,
    )
    assert len(list(hf_dataset)) == len(data_records)


@pytest.mark.parametrize("multiple_inputs", [True, False])
def test_preprocessing_same_length_example(multiple_inputs):
    """Test the correctness of processing a single example."""
    source = "@HMRCcustomers No this is my first job"
    target = "no complaint"
    if multiple_inputs:
        # Tests logic for processing as a batch with no padding issues
        input_dict = {"input": [source, source], "output": [target, target]}
    else:
        input_dict = {"input": [source], "output": [target]}

    hf_dataset = tokenize_function(
        input_dict,
        tokenizer=TOKENIZER,
    )
    # Every key in our produced HF dataset should have the same length value
    assert len(set([len(v) for v in hf_dataset.values()])) == 1
    # And the sequence should be the same length as the source + target + 1
    expected_len = (len(TOKENIZER(source).input_ids) + len(TOKENIZER(target).input_ids) + 1)
    assert len(hf_dataset.input_ids[0]) == expected_len


def test_preprocessing_differing_lengths():
    """Ensure that the longest sequence determines the padding behavior"""
    s1 = "@HMRCcustomers No this is my first job"
    t1 = "no complaint"
    s2 = "@KristaMariePark Thank you for your interest! If you decide to cancel, you can call Customer Care at 1-800-NYTIMES."
    t2 = "no complaint"
    input_dict = {"input": [s1, s2], "output": [t1, t2]}
    hf_dataset = tokenize_function(
        input_dict,
        tokenizer=TOKENIZER,
    )
    assert len(set([len(v) for v in hf_dataset.values()])) == 1
    # The expected sequence length is determined by the longer sample in this test
    r1_length = (len(TOKENIZER(s1).input_ids) + len(TOKENIZER(t1).input_ids) + 1)
    r2_length = (len(TOKENIZER(s2).input_ids) + len(TOKENIZER(t2).input_ids) + 1)
    # If they match, this test is redundant
    assert r1_length != r2_length
    expected_len = max(r1_length, r2_length)
    actual_input_ids_len = [len(input_ids) for input_ids in hf_dataset.input_ids].pop()
    assert expected_len == actual_input_ids_len


def test_max_seq_len_override():
    """Ensure that we can override the max sequence length."""
    max_sequence_length=1 # Creates one tok + FINAL_TOKEN_ID (EOS) sequences.
    hf_dataset = preprocess_function(
        data_path=SAMPLE_DATA,
        tokenizer=TOKENIZER,
        batch_size=1000,
        max_sequence_length=max_sequence_length,
    )
    for record in hf_dataset:
        value_lengths = [len(v) for v in record.values()]
        value_len_set = set(value_lengths)
        assert len(value_len_set) == 1
        # We should truncate, but still keep our EOS in the target
        assert value_len_set.pop() <= (max_sequence_length + 1)
        # Despite the trunction, we should still have a trailing EOS as the last token
        assert record["labels"][-1] == TOKENIZER.eos_token_id
        assert record["input_ids"][-1] == TOKENIZER.eos_token_id
        assert record["attention_mask"][-1] == 1


def test_formatting_with_divisible_per_batch_padding():
    """Ensure that if the dataset is smaller than the batch, the padding is done per batch."""
    max_sequence_length=100
    hf_dataset = preprocess_function(
        data_path=SAMPLE_DATA,
        tokenizer=TOKENIZER,
        batch_size=10, # NOTE: Tests data has 50 samples
        max_sequence_length=max_sequence_length,
    )
    sizes = set()
    for record in hf_dataset:
        value_lengths = [len(v) for v in record.values()]
        value_len_set = set(value_lengths)
        assert len(value_len_set) == 1
        sizes.add(value_len_set.pop())
    # Since we have 50 samples and a batch size of 10, we should have
    # 5 unique padding lengths [assuming there are no collisions].
    assert len(sizes) == 5


def test_formatting_with_non_divisible_per_batch_padding():
    """Ensure that if len(data)/batch_size is non integer, we have good behavior."""
    max_sequence_length=100
    hf_dataset = preprocess_function(
        data_path=SAMPLE_DATA,
        tokenizer=TOKENIZER,
        batch_size=11, # NOTE: Tests data has 50 samples
        max_sequence_length=max_sequence_length,
    )
    sizes = set()
    for record in hf_dataset:
        value_lengths = [len(v) for v in record.values()]
        value_len_set = set(value_lengths)
        assert len(value_len_set) == 1
        sizes.add(value_len_set.pop())
    # Same case as before (although in this case, one batch is smaller since 50/11
    # results in one smaller batch being formed at the end of the dataset).
    assert len(sizes) == 5
