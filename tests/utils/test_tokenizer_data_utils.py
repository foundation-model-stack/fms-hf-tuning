# Third Party
from transformers import AutoModelForCausalLM, AutoTokenizer

# First Party
from tests.artifacts.testdata import MODEL_NAME

# Local
from tuning.config import configs
from tuning.utils.tokenizer_data_utils import (
    get_special_tokens_dict,
    tokenizer_and_embedding_resize,
)


def test_setting_special_tokens_with_LlamaTokenizerFast():
    """
    Unit test using a LlamaTokenizerFast tokenizer. This tokenizer is only missing a PAD token,
    however because it is a LlamaTokenizer, the function code automatically adds the BOS, EOS,
    UNK and PAD tokens to the special tokens dict. Then, the <pad> token is replaced with
    a <PAD> token, because the Llama tokenizer does not have a pad token specified.
    """
    tokenizer = AutoTokenizer.from_pretrained("Maykeye/TinyLLama-v0", legacy=True)
    model_args = configs.ModelArguments()
    special_tokens_dict = get_special_tokens_dict(
        tokenizer_name_or_path=model_args.tokenizer_name_or_path, tokenizer=tokenizer
    )
    assert special_tokens_dict == {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<PAD>",
    }


def test_setting_special_tokens_with_GPT2TokenizerFast():
    """
    Unit test using a GPT2TokenizerFast tokenizer. This tokenizer is the case where the
    EOS token = PAD token, both of them are <|endoftext|>. So, the pad token in the tokenizer is set
    to <PAD> and the "pad_token": "<PAD>" is also added to the special tokens dict.
    """
    tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.1-8b-base")
    model_args = configs.ModelArguments()
    special_tokens_dict = get_special_tokens_dict(
        tokenizer_name_or_path=model_args.tokenizer_name_or_path, tokenizer=tokenizer
    )
    assert special_tokens_dict == {
        "pad_token": "<PAD>",
    }


def test_setting_special_tokens_with_GPTNeoXTokenizerFast():
    """
    Unit test using a GPTNeoXTokenizerFast tokenizer. This tokenizer is another one that is
    hardcoded into the function to automatically add just a pad token to the special tokens dict.
    However, the tokenizer itself is also missing a pad token, so the function then replaces
    the <pad> token with the default <PAD> token.
    """
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model_args = configs.ModelArguments()
    special_tokens_dict = get_special_tokens_dict(
        tokenizer_name_or_path=model_args.tokenizer_name_or_path, tokenizer=tokenizer
    )
    assert special_tokens_dict == {
        "pad_token": "<PAD>",
    }


def test_setting_special_tokens_when_missing_all_special_tokens():
    """
    Unit test using the GPT2TokenizerFast tokenizer. All the special tokens have been
    removed from the tokenizer, so we expect all of them to appear in the special tokens dict.
    """
    tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.1-8b-base")

    # Set all special tokens to None
    tokenizer.bos_token = None
    tokenizer.eos_token = None
    tokenizer.unk_token = None
    tokenizer.pad_token = None

    model_args = configs.ModelArguments()
    special_tokens_dict = get_special_tokens_dict(
        tokenizer_name_or_path=model_args.tokenizer_name_or_path, tokenizer=tokenizer
    )
    assert special_tokens_dict == {
        "pad_token": "<PAD>",
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>",
    }


def test_setting_special_tokens_when_path_is_not_none():
    """
    A simple unit test that sets the `tokenizer_name_or_path` argument in
    `model_args` to a non None value. Since the argument is not None, almost
    the entire `get_special_tokens_dict` function is skipped and the
    special tokens dict is expected to be empty.
    """
    tokenizer = AutoTokenizer.from_pretrained("Maykeye/TinyLLama-v0", legacy=True)
    model_args = configs.ModelArguments(tokenizer_name_or_path="test_path")
    special_tokens_dict = get_special_tokens_dict(
        tokenizer_name_or_path=model_args.tokenizer_name_or_path, tokenizer=tokenizer
    )
    # Assert special_tokens_dict is empty
    assert not special_tokens_dict


def test_tokenizer_and_embedding_resize_return_values_missing_one_token():
    """
    Tests the resizing function when the special tokens dict contains a PAD token,
    which means the tokenizer is missing one special token.

    `mulitple_of` is set to 1.
    """
    special_tokens_dict = {"pad_token": "<pad>"}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    metadata = tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)
    assert metadata["num_new_tokens"] == 1
    assert metadata["new_embedding_size"] == len(tokenizer)


def test_tokenizer_and_embedding_resize_return_values_missing_four_tokens():
    """
    Tests the resizing when the special tokens dict contains a PAD, EOS, BOS and UNK token,
    which means the tokenizer is missing four special tokens.

    `mulitple_of` is set to 1.
    """
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
    """
    Tests the resizing when the special tokens dict contains a PAD, EOS, BOS and UNK token,
    which means the tokenizer is missing four special tokens.

    `mulitple_of` is set to 2; this add one to the count of num_new_tokens and adds
    one to the count of new_embedding_size.
    """
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
