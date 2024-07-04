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
from tuning.data import tokenizer_data_utils

MODEL_NAME = "Maykeye/TinyLLama-v0"


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
    input_text = "### Text: @NortonSupport Thanks much.\n\n### Label:"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_not_resized = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model_resized = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    tokenizer_data_utils.tokenizer_and_embedding_resize(
        special_tokens_dict={}, tokenizer=tokenizer, model=model_resized, multiple_of=8
    )

    tokenizer_data_utils.tokenizer_and_embedding_resize(
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
