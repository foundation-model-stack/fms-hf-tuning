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
    special_tokens_dict = set_special_tokens_dict(
        tokenizer_name_or_path=model_args.tokenizer_name_or_path, tokenizer=tokenizer
    )
    assert special_tokens_dict == {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<PAD>",
    }


def test_setting_special_tokens_with_GPT2TokenizerFast():
    # For GPT2TokenizerFast, PAD token = EOS Token
    tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.1-8b-base")
    model_args = configs.ModelArguments()
    special_tokens_dict = set_special_tokens_dict(
        tokenizer_name_or_path=model_args.tokenizer_name_or_path, tokenizer=tokenizer
    )
    assert special_tokens_dict == {
        "pad_token": "<PAD>",
    }


def test_setting_special_tokens_with_GPTNeoXTokenizerFast():
    # For GPTNeoXTokenizerFast, Missing PAD Token
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model_args = configs.ModelArguments()
    special_tokens_dict = set_special_tokens_dict(
        tokenizer_name_or_path=model_args.tokenizer_name_or_path, tokenizer=tokenizer
    )
    assert special_tokens_dict == {
        "pad_token": "<PAD>",
    }


def test_setting_special_tokens_when_missing_all_special_tokens():
    # Missing all special tokens
    tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.1-8b-base")

    # Set all special tokens to None
    tokenizer.bos_token = None
    tokenizer.eos_token = None
    tokenizer.unk_token = None
    tokenizer.pad_token = None

    model_args = configs.ModelArguments()
    special_tokens_dict = set_special_tokens_dict(
        tokenizer_name_or_path=model_args.tokenizer_name_or_path, tokenizer=tokenizer
    )
    assert special_tokens_dict == {
        "pad_token": "<PAD>",
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>",
    }


def test_setting_special_tokens_when_path_is_not_none():
    # Test to ensure dictionary is empty when path is not none
    tokenizer = AutoTokenizer.from_pretrained("Maykeye/TinyLLama-v0", legacy=True)
    model_args = configs.ModelArguments(tokenizer_name_or_path="test_path")
    special_tokens_dict = set_special_tokens_dict(
        tokenizer_name_or_path=model_args.tokenizer_name_or_path, tokenizer=tokenizer
    )
    assert special_tokens_dict == {}


def test_tokenizer_and_embedding_resize_return_values_missing_one_token():
    """Test to ensure number of added tokens are returned correctly"""
    special_tokens_dict = {"pad_token": "<pad>"}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    metadata = tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)
    assert metadata["num_new_tokens"] == 1
    assert metadata["new_embedding_size"] == len(tokenizer)


def test_tokenizer_and_embedding_resize_return_values_missing_four_tokens():
    """Test to ensure number of added tokens are returned correctly"""
    special_tokens_dict = {
        "pad_token": "<PAD>",
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>",
    }
    tokenizer = AutoTokenizer.from_pretrained("Maykeye/TinyLLama-v0", legacy=True)
    model = AutoModelForCausalLM.from_pretrained("Maykeye/TinyLLama-v0")
    metadata = tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)
    assert metadata["num_new_tokens"] == 4
    assert metadata["new_embedding_size"] == len(tokenizer)


def test_tokenizer_and_embedding_resize_return_values_mutliple_of_two():
    """Test to ensure number of added tokens are returned correctly"""
    special_tokens_dict = {
        "pad_token": "<PAD>",
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>",
    }
    tokenizer = AutoTokenizer.from_pretrained("Maykeye/TinyLLama-v0", legacy=True)
    model = AutoModelForCausalLM.from_pretrained("Maykeye/TinyLLama-v0")
    metadata = tokenizer_and_embedding_resize(
        special_tokens_dict, tokenizer, model, multiple_of=2
    )
    assert metadata["num_new_tokens"] == 5
    assert metadata["new_embedding_size"] == len(tokenizer) + 1
