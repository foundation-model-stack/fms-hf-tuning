import pytest
import json
import os
from tuning.utils.data_format_utils import preprocess_function
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
        max_seq_length=100,
        use_iterable_dataset=use_iterable_dataset
    )
    assert len(list(hf_dataset)) == len(data_records)
