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

# Standard
import copy

# Third Party
from transformers import (
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)
import torch

# First Party
from tests.artifacts.testdata import CUSTOM_TOKENIZER_TINYLLAMA
from tests.artifacts.vision_models import TINY_LLAMA_VISION_MODEL_NAME

# Local
from tuning.utils.tokenizer_data_utils import tokenizer_and_embedding_resize

MODEL_NAME = "Maykeye/TinyLLama-v0"
INPUT_TEXT = "### Text: @NortonSupport Thanks much.\n\n### Label:"


def _inference(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    input_text: str,
    max_new_tokens: int,
) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenized_input = tokenizer(input_text, return_tensors="pt").to(device)
    generated_output = model.generate(
        **tokenized_input,
        max_new_tokens=max_new_tokens,
    )
    return tokenizer.decode(generated_output[0], skip_special_tokens=True)


def test_output_unaltered_across_embedding_resizes():
    input_text = INPUT_TEXT
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_not_resized = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model_resized = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    tokenizer_and_embedding_resize(
        special_tokens_dict={}, tokenizer=tokenizer, model=model_resized, multiple_of=8
    )

    tokenizer_and_embedding_resize(
        special_tokens_dict={},
        tokenizer=tokenizer,
        model=model_not_resized,
        multiple_of=1,
    )

    # embedding size of the resized model should be a multiple of 8
    assert model_resized.get_output_embeddings().out_features % 8 == 0

    output_from_model_not_resized = _inference(
        model=model_not_resized,
        tokenizer=tokenizer,
        input_text=input_text,
        max_new_tokens=50,
    )
    output_from_model_resized = _inference(
        model=model_not_resized,
        tokenizer=tokenizer,
        input_text=input_text,
        max_new_tokens=50,
    )

    assert output_from_model_not_resized == output_from_model_resized


def test_resize_with_special_tokens():
    input_text = INPUT_TEXT
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    input_tokenizer_len = len(tokenizer.get_vocab())

    special_tokens = {"sep_token": "<SEP>", "pad_token": "<PAD>"}
    resize_result = tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens,
        tokenizer=tokenizer,
        model=model,
        multiple_of=1,
    )

    assert "<SEP>" in tokenizer.get_vocab()
    assert "<PAD>" in tokenizer.get_vocab()

    output_tokenizer_len = len(tokenizer.get_vocab())

    assert output_tokenizer_len == input_tokenizer_len + 2
    assert resize_result["num_new_tokens"] == output_tokenizer_len - input_tokenizer_len

    output = _inference(
        tokenizer=tokenizer, model=model, input_text=input_text, max_new_tokens=20
    )
    assert output is not None


def test_special_tokens_before_and_after():
    """Test if additional special tokens added do not replace existing tokens"""
    input_text = INPUT_TEXT
    tokenizer = AutoTokenizer.from_pretrained(CUSTOM_TOKENIZER_TINYLLAMA)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    input_tokenizer_len = len(tokenizer.get_vocab())
    addn_spl_tokens_before = tokenizer.special_tokens_map.get(
        "additional_special_tokens"
    )
    assert (
        len(addn_spl_tokens_before) > 0
    ), "this test needs tokenizer special tokens to not be empty before testing"

    special_tokens_dict = {"sep_token": "<SEP>", "pad_token": "<PAD>"}
    addn_spl_tokens_added = ["<NotSeenTokenA>", "<NotSeenTokenB>", "<NotSeenTokenC>"]
    special_tokens_dict["additional_special_tokens"] = addn_spl_tokens_added

    resize_result = tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
        multiple_of=1,
    )

    output_tokenizer_len = len(tokenizer.get_vocab())
    addn_spl_tokens_before.extend(addn_spl_tokens_added)
    expected_addn_special_tokens = addn_spl_tokens_before
    expected_embedding_size = input_tokenizer_len + len(addn_spl_tokens_added) + 2
    addn_spl_tokens_after = tokenizer.special_tokens_map.get(
        "additional_special_tokens"
    )

    assert "<SEP>" in tokenizer.get_vocab()
    assert "<PAD>" in tokenizer.get_vocab()
    assert output_tokenizer_len == expected_embedding_size
    assert resize_result["num_new_tokens"] == output_tokenizer_len - input_tokenizer_len
    assert resize_result["new_embedding_size"] == expected_embedding_size

    assert len(addn_spl_tokens_after) == len(
        expected_addn_special_tokens
    ), "length of the additional special tokens after must equal length before plus added tokens"

    for tok in expected_addn_special_tokens:
        assert (
            tok in addn_spl_tokens_after
        ), "additional special tokens added are not in tokenizer"

    output = _inference(
        tokenizer=tokenizer, model=model, input_text=input_text, max_new_tokens=20
    )
    assert output is not None


def test_no_resize_when_no_special_tokens():
    input_text = INPUT_TEXT
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    input_tokenizer_len = len(tokenizer.get_vocab())

    resize_result = tokenizer_and_embedding_resize(
        special_tokens_dict={}, tokenizer=tokenizer, model=model, multiple_of=1
    )

    output_tokenizer_len = len(tokenizer.get_vocab())

    assert input_tokenizer_len == output_tokenizer_len
    assert resize_result["num_new_tokens"] == 0

    output = _inference(
        tokenizer=tokenizer, model=model, input_text=input_text, max_new_tokens=20
    )

    assert output is not None


def test_resize_with_multiple_of():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    resize_result = tokenizer_and_embedding_resize(
        special_tokens_dict={}, tokenizer=tokenizer, model=model, multiple_of=8
    )

    assert model.get_input_embeddings().embedding_dim % 8 == 0
    assert resize_result["new_embedding_size"] % 8 == 0
    assert model.get_output_embeddings().out_features % 8 == 0


def test_resize_llama_vision_model():
    model = AutoModelForVision2Seq.from_pretrained(TINY_LLAMA_VISION_MODEL_NAME)
    processor = AutoProcessor.from_pretrained(TINY_LLAMA_VISION_MODEL_NAME)
    tokenizer = processor.tokenizer

    current_input_embeddings = model.get_input_embeddings()
    current_input_embeddings = copy.deepcopy(current_input_embeddings)
    current_output_embeddings = model.get_output_embeddings()
    current_output_embeddings = copy.deepcopy(current_output_embeddings)

    current_tokenizer_len = len(tokenizer.get_vocab())

    resize_result = tokenizer_and_embedding_resize(
        special_tokens_dict={"unk_token": "<unk>"},
        tokenizer=tokenizer,
        model=model,
        multiple_of=1,
    )

    resized_input_embeddings = model.get_input_embeddings()
    resized_output_embeddings = model.get_output_embeddings()
    resized_tokenizer_len = len(tokenizer.get_vocab())

    assert resized_tokenizer_len == current_tokenizer_len + 1
    assert "<unk>" in tokenizer.get_vocab()
    assert resize_result["num_new_tokens"] == 1

    # For Llama vision models, resizing adds 2 tokens (<unk> and <image>) because the
    # tokenizer vocabulary size (128257) is one more than the output embedding size (128256),
    # i.e., len(tokenizer) == model.get_output_embeddings().weight.shape[0] + 1.

    # When special_tokens_dict contains only <unk>, the embedding size is increased from
    # 128256 to 128258 (adding both <unk> and <image> tokens). As a result, the model's input
    # embeddings are also resized by 2 tokens.

    # This behavior is not observed in Granite or Llava vision models, where
    # len(tokenizer) == model.get_output_embeddings().weight.shape[0].

    assert (
        resized_output_embeddings.weight.shape[0]
        == current_output_embeddings.weight.shape[0] + 2
    )
    assert (
        resized_input_embeddings.weight.shape[0]
        == current_input_embeddings.weight.shape[0] + 2
    )
