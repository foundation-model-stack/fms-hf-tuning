# Third party
# Third Party
from transformers import AutoModelForCausalLM, AutoTokenizer

# First Party
from tests.data import MODEL_NAME

# Local
# First party
from tuning.data.tokenizer_data_utils import tokenizer_and_embedding_resize


def test_tokenizer_and_embedding_resize_return_values():
    """Test to ensure number of added tokens are returned correctly"""
    special_tokens_dict = {"pad_token": "<pad>"}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    metadata = tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)
    assert metadata["num_new_tokens"] == 1
    assert "new_embedding_size" in metadata
