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

# Third Party
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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
