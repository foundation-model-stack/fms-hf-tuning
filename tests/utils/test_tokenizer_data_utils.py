# Third party
from transformers import AutoModelForCausalLM, AutoTokenizer

# First Party
from tests.artifacts.testdata import MODEL_NAME
from tuning.config import configs

# Local
# First party
from tuning.utils.tokenizer_data_utils import (
    tokenizer_and_embedding_resize,
    set_special_tokens_dict,
)


def test_setting_special_tokens_with_LlamaTokenizerFast():
    # For LlamaTokenizerFast, Missing PAD Token
    tokenizer = AutoTokenizer.from_pretrained("Maykeye/TinyLLama-v0", legacy=True)
    model_args = configs.ModelArguments()
    special_tokens_dict = set_special_tokens_dict(model_args, tokenizer)
    print(tokenizer)
    print("Special Tokens", special_tokens_dict)
    assert special_tokens_dict != {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<PAD>",
    }


def test_setting_special_tokens_with_GPT2TokenizerFast():
    # For GPT2TokenizerFast, PAD token = EOS Token
    tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.1-8b-base")
    model_args = configs.ModelArguments()
    special_tokens_dict = set_special_tokens_dict(model_args, tokenizer)
    print(tokenizer)
    print("Special Tokens", special_tokens_dict)
    assert special_tokens_dict == {
        "pad_token": "<PAD>",
    }


def test_setting_special_tokens_with_GPTNeoXTokenizerFast():
    # For GPTNeoXTokenizerFast, Missing PAD Token
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model_args = configs.ModelArguments()
    special_tokens_dict = set_special_tokens_dict(model_args, tokenizer)
    print(tokenizer)
    print("Special Tokens", special_tokens_dict)
    assert special_tokens_dict == {
        "pad_token": "<PAD>",
    }


def test_tokenizer_and_embedding_resize_return_values():
    """Test to ensure number of added tokens are returned correctly"""
    special_tokens_dict = {"pad_token": "<pad>"}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    metadata = tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)
    assert metadata["num_new_tokens"] == 1
    assert "new_embedding_size" in metadata
