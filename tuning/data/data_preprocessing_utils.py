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
from typing import Callable, Optional
import re

# Third Party
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from trl import DataCollatorForCompletionOnlyLM
import datasets

# Local
from tuning.config import configs

# In future we may make the fields configurable
DEFAULT_JSON_INPUT_KEY = "input"
DEFAULT_JSON_OUTPUT_KEY = "output"


def validate_data_args(
    data_args: configs.DataArguments,
    packing: bool,
    is_traindataset_tokenized: bool,
    is_evaldataset_tokenized: bool,
):

    assert isinstance(
        data_args.training_data_path, str
    ), "Training data path has to be set and str"

    ### Data format 1
    # if the provided train dataset is pretokenized
    # however user provides formatting flags, error out
    if is_traindataset_tokenized:
        if (
            data_args.response_template
            or data_args.data_formatter_template
            or data_args.dataset_text_field
        ):
            raise ValueError(
                "fields response_template, data_formatter_template, and dataset_text_field \
                                are not applicable for pretokenized datasets"
            )

        # if the train dataset is pretokenized
        # ensure validation dataset is pretokenized otherwise error out
        if data_args.validation_data_path and not is_evaldataset_tokenized:
            raise ValueError(
                "validation data should be pretokenized to be used \
                along with pretokenized train data"
            )

        # packing wont be available for pretokenized datasets in trl library
        # see: https://github.com/huggingface/trl/issues/1848
        if packing:
            raise ValueError("packing will not be used when datasets are pretokenized")
        return

    ### Data format 2
    # Dataset containing single sequence needs a response template for masking
    if data_args.dataset_text_field or data_args.data_formatter_template:
        if data_args.response_template is None:
            if packing is False:
                raise ValueError(
                    "Since dataset_text_field or data_formatter_template \
                       is provided and packing is disabled, \
                       needs a corresponding response template for masking"
                )

    if data_args.response_template:
        # To use Response template, pass datasets with single sequence instances \
        # or a formatter template to create single sequence on the fly.
        if not (data_args.dataset_text_field or data_args.data_formatter_template):
            raise ValueError(
                "dataset_text_field and data_formatter_template are None. \
                            One of them needs to be set to use response_template"
            )
        # Only one of dataset_text_field or data_formatter_template should be set.
        if data_args.dataset_text_field and data_args.data_formatter_template:
            raise ValueError(
                "dataset_text_field and data_formatter_template are both set,\
                but are mutually exclusive options"
            )

    ### Data format 3
    # If not single sequence, JSON should contain input/output fields
    if not (data_args.dataset_text_field or data_args.data_formatter_template):
        json_dataset = datasets.load_dataset(
            "json", data_files=data_args.training_data_path
        )
        if DEFAULT_JSON_INPUT_KEY not in json_dataset["train"].column_names:
            raise ValueError(
                "JSON should contain input field if no dataset_text_field or \
                     data_formatter_template specified"
            )
        if DEFAULT_JSON_OUTPUT_KEY not in json_dataset["train"].column_names:
            raise ValueError(
                "JSON should contain output field if no dataset_text_field or \
                    data_formatter_template specified"
            )


### Utils for custom masking / manipulating input / output strs, etc
def combine_sequence(input_element: str, output_element: str, eos_token: str = ""):
    """Combines / concatenates input & output element.

    Args:
        input_element: str
            Input component of the combined sequence.
        output_element: str
            Output component of the combined sequence.
        eos_token: str
            EOS token associated with the tokenizer. \
            If passed, it will be concatenated at end

    Returns:
        str
            Sequence combined with whitespace.
    """
    if not input_element.endswith((" ", "\n", "\t")) and not output_element.startswith(
        (" ", "\n", "\t")
    ):
        return input_element + " " + output_element + eos_token
    return input_element + output_element + eos_token


def get_data_collator(
    packing: bool,
    response_template: Optional[str],
    tokenizer: AutoTokenizer,
    is_traindata_tokenized: bool,
    max_seq_length: int,
) -> Callable:
    """Create and return the the appropriate collator type based on the configuration for packing,
    response_template, and dataset_text_field.

    Args:
        packing: bool
            Whether or not we should apply packing or not.
        response_template: Optional[str]
            Response template to be used for formatting by TRL.
        tokenizer: AutoTokenizer
            Loaded tokenizer object to be used by the collator.
        is_traindata_tokenized: bool
            Whether train Dataset is tokenized or not
        max_seq_length: int
            Max sequence length expected

    Returns:
        Callable
            Callable collator to be leveraged by the trainer.
    """

    if not packing:
        # TODO: near term - how response template ids are parsed out needs to be cleaned.
        # The [2:] here applies if response template has \n prefix, it is needed to strip \n,
        # otherwise template is not found. We will create issue to clean this out after we discuss
        # data formats and collators we will support.
        if response_template:
            response_template_ids = tokenizer.encode(
                response_template, add_special_tokens=False
            )[2:]
            return DataCollatorForCompletionOnlyLM(
                response_template=response_template_ids,
                tokenizer=tokenizer,
                ignore_index=configs.IGNORE_INDEX,
            )
        # Note that this automatically pads labels with -100
        # TODO check if this is sufficient for preprocessed
        if is_traindata_tokenized:
            return DataCollatorForSeq2Seq(
                tokenizer=tokenizer, padding=True, max_length=max_seq_length
            )
        raise ValueError(
            "Could not pick a data collator. Please refer to supported data formats"
        )


def custom_data_formatter(element, template, formatted_dataset_field):
    def replace_text(match_obj):
        captured_groups = match_obj.groups()
        if len(captured_groups) != 1:
            raise ValueError(
                "Unexpectedly captured multiple groups in template formatting"
            )

        index_object = captured_groups[0]
        if index_object not in element:
            raise KeyError("Requested template string is not a valid key in dict")

        return element[index_object]

    return {
        formatted_dataset_field: re.sub(
            r"{{([\s0-9a-zA-Z_\-\.]+)}}", replace_text, template
        )
    }
