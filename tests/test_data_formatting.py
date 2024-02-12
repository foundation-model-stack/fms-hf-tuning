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
        use_iterable_dataset=use_iterable_dataset
    )
    assert len(list(hf_dataset)) == len(data_records)


@pytest.mark.parametrize("use_iterable_dataset", [True, False])
def test_preprocessing_single_example(use_iterable_dataset):
    """Test the correctness of processing a single example."""
    source = "@HMRCcustomers No this is my first job"
    target = "no complaint"

    hf_dataset = tokenize_function(
        # NOTE: map to values here since the internal tokenization logic assumes batch processing.
        {"input": [source], "output": [target]},
        tokenizer=TOKENIZER,
    )
    # Every key in our produced HF dataset should have the same length value
    assert len(set([len(v) for v in hf_dataset.values()])) == 1
    # And the sequence should be the same length as the source + target + 1
    expected_len = (len(TOKENIZER(source).input_ids) + len(TOKENIZER(target).input_ids) + 1)
    assert len(hf_dataset.input_ids[0]) == expected_len
