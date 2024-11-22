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

# Definition of some predefined data preprocessing functions that we need.

# Standard
from typing import Dict

# Third Party
from transformers import AutoTokenizer

# Local
from tuning.data.data_preprocessing_utils import combine_sequence, custom_data_formatter


def tokenize_and_apply_input_masking(
    element: Dict[str, str],
    tokenizer: AutoTokenizer,
    input_field_name: str,
    output_field_name: str,
    **tokenizer_kwargs,
):
    input = element[input_field_name]
    output = element[output_field_name]

    # TODO: Eventually move the code here
    combined = combine_sequence(input, output, eos_token=tokenizer.eos_token)

    fn_kwargs = tokenizer_kwargs.get("fn_kwargs", {})
    tokenizer_inner_kwargs = fn_kwargs.get("tokenizer_kwargs", {})

    tokenized_comb_seqs = tokenizer(combined, **tokenizer_inner_kwargs)
    tokenized_input = tokenizer(input, **tokenizer_inner_kwargs)

    masked_labels = [-100] * len(
        tokenized_input.input_ids
    ) + tokenized_comb_seqs.input_ids[len(tokenized_input.input_ids) :]

    # Any benefit of retaining the old columns?
    return {
        "input_ids": tokenized_comb_seqs.input_ids,
        "labels": masked_labels,
        "attention_mask": tokenized_comb_seqs.attention_mask,
    }


def apply_dataset_formatting(
    element: Dict[str, str], tokenizer: AutoTokenizer, dataset_text_field: str, **kwargs
):
    return {
        f"{dataset_text_field}": element[f"{dataset_text_field}"] + tokenizer.eos_token
    }


def apply_custom_data_formatting_template(
    element: Dict[str, str],
    tokenizer: AutoTokenizer,
    dataset_text_field: str,
    template: str,
    **kwargs,
):
    """Function to format datasets with Alpaca style / other templates.
       Expects to be run as a HF Map API function.
    Args:
        element: the HF Dataset element loaded from a JSON or DatasetDict object.
        template: Template to format data with. Features of Dataset
            should be referred to by {{key}}
        formatted_dataset_field: Dataset_text_field
        eos_token: string EOS token to be appended while formatting data to a single sequence.
            Defaults to empty
    Returns:
        Formatted HF Dataset
    """

    template += tokenizer.eos_token

    # TODO: Eventually move the code here.
    return custom_data_formatter(element, template, dataset_text_field)


AVAILABLE_DATA_HANDLERS = {
    "tokenize_and_apply_instruction_masking": tokenize_and_apply_input_masking,
    "apply_dataset_formatting": apply_dataset_formatting,
    "apply_custom_data_formatting_template": apply_custom_data_formatting_template,
}
