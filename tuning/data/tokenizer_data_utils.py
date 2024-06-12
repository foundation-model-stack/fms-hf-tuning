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

# Standard
from typing import Dict, List
import math

# Third Party
import transformers


def tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    TODO: In the future, make sure we can have vocab size divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def create_attention_mask(labels: List[List], input_ids: List[List]) -> List[List]:
    """create an attention mask with all 1s

    Args:
        labels (List[List]): labels
        input_ids (List[List]): input ids

    Raises:
        ValueError: if batch size of labels and input ids mismatch raises error

    Returns:
        List[List]: attention mask for the batch
    """
    attention_mask = []
    if len(labels) != len(input_ids):
        raise ValueError("Number of labels and input sequences must be equal.")
    for i in range(len(labels)):
        attention_mask.append([1] * len(input_ids[i]))
    return attention_mask


def embedding_resize(
    model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer
) -> None:
    """resize embedding layer of the model to closest multiple of 8 based on the
    given tokenizer

    Args:
        model (transformers.PreTrainedModel): model object
        tokenizer (transformers.PreTrainedTokenizer): tokenizer object
    """
    model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))
