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
from typing import Any, Callable, Dict, Optional, Union
import json
import logging

# Third Party
from datasets import Dataset, IterableDataset
from datasets.exceptions import DatasetGenerationError
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from trl import DataCollatorForCompletionOnlyLM
import datasets

# Local
from tuning.config import configs
from tuning.utils.data_utils import apply_custom_formatting_template

# In future we may make the fields configurable
JSON_INPUT_KEY = "input"
JSON_OUTPUT_KEY = "output"


# check if the provided dataset is pretokenized or not
# the check is taken from trl
# https://github.com/huggingface/trl/blob/ddf4c8dc3ecf6d9ee2b24f94c62182ffd682c808/trl/trainer/sft_trainer.py#L498-L509
def is_pretokenized_dataset(data: Union[str, Dataset, IterableDataset]):
    if not data:
        return False
    if isinstance(data, str):
        try:
            data = datasets.load_dataset("json", data_files=data, split="train[:1]")
        except DatasetGenerationError as e:
            raise DatasetGenerationError("failed to load the provided dataset") from e

    return ("input_ids" in data.column_names) and ("labels" in data.column_names)


def validate_data_args(data_args: configs.DataArguments, packing: bool):

    assert isinstance(
        data_args.training_data_path, str
    ), "Training data path has to be set and str"

    is_train_data_pretokenized = is_pretokenized_dataset(data_args.training_data_path)
    is_eval_data_pretokenized = is_pretokenized_dataset(data_args.validation_data_path)

    ### Data format 1
    # if the provided train dataset is pretokenized
    # however user provides formatting flags, error out
    if is_train_data_pretokenized:
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
        if data_args.validation_data_path and not is_eval_data_pretokenized:
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
        if JSON_INPUT_KEY not in json_dataset["train"].column_names:
            raise ValueError(
                "JSON should contain input field if no dataset_text_field or \
                     data_formatter_template specified"
            )
        if JSON_OUTPUT_KEY not in json_dataset["train"].column_names:
            raise ValueError(
                "JSON should contain output field if no dataset_text_field or \
                    data_formatter_template specified"
            )


def get_data_collator(
    packing: bool,
    response_template: Optional[str],
    tokenizer: AutoTokenizer,
    formatted_train_dataset: Dataset,
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
        formatted_train_dataset: Dataset
            Train Dataset formatted for tuning
        max_seq_length: int
            Max sequence length expected

    Returns:
        Callable
            Callable collator to be leveraged by the trainer.
    """
    is_train_data_pretokenized = is_pretokenized_dataset(formatted_train_dataset)

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
        if is_train_data_pretokenized:
            return DataCollatorForSeq2Seq(
                tokenizer=tokenizer, padding=True, max_length=max_seq_length
            )
        raise ValueError(
            "Could not pick a data collator. Please refer to supported data formats"
        )


def format_dataset(
    data_args: configs.DataArguments, tokenizer: AutoTokenizer, max_seq_length: int
):
    """
    Args:
        data_args: tuning.config.configs.DataArguments
        tokenizer: AutoTokenizer
        max_seq_length: int
            Max sequence length expected
    Returns:
        Tuple(Dataset, Dataset, str)
            tuple containing train_dataset, eval_dataset and dataset_text_field
    """
    eval_dataset = None
    is_train_data_pretokenized = is_pretokenized_dataset(data_args.training_data_path)

    if is_train_data_pretokenized:
        train_dataset = datasets.load_dataset(
            "json", data_files=data_args.training_data_path, split="train"
        )
        if data_args.validation_data_path:
            eval_dataset = datasets.load_dataset(
                "json", data_files=data_args.validation_data_path, split="train"
            )
        # dataset_text_field is irrelevant to pretokenized datasets
        return train_dataset, eval_dataset, None

    dataset_text_field = data_args.dataset_text_field
    if data_args.data_formatter_template or dataset_text_field:
        if dataset_text_field is None:
            dataset_text_field = "new_formatted_field"
        train_dataset = get_formatted_dataset_with_single_sequence(
            data_args.training_data_path,
            dataset_text_field,
            tokenizer,
            data_args.data_formatter_template,
        )
        logging.info("Training dataset length is %s", len(train_dataset))
        if data_args.validation_data_path:
            (eval_dataset) = get_formatted_dataset_with_single_sequence(
                data_args.validation_data_path,
                dataset_text_field,
                tokenizer,
                data_args.data_formatter_template,
            )
            logging.info("Validation dataset length is %s", len(eval_dataset))
    else:
        # This is for JSON containing input/output fields
        train_dataset = get_preprocessed_dataset(
            data_args.training_data_path,
            tokenizer,
            max_seq_length,
            input_field_name=JSON_INPUT_KEY,
            output_field_name=JSON_OUTPUT_KEY,
        )
        if data_args.validation_data_path:
            eval_dataset = get_preprocessed_dataset(
                data_args.validation_data_path,
                tokenizer,
                max_seq_length,
                input_field_name=JSON_INPUT_KEY,
                output_field_name=JSON_OUTPUT_KEY,
            )

    return train_dataset, eval_dataset, dataset_text_field


def get_formatted_dataset_with_single_sequence(
    data_path: str,
    dataset_text_field: str,
    tokenizer: AutoTokenizer,
    data_formatter_template: Optional[str] = None,
) -> Dataset:
    """Applies formatting to the loaded dataset instance; does NOT pretokenize data.

    Args:
        data_path: str
            Path to the file to be loaded.
        dataset_text_field: str
            Dataset text field to be used for formatting.
            If data_formatter_template specified, \
                this will be the new field creating single sequence.
        tokenizer: AutoTokenizer
            Loaded tokenizer object to be used by the collator.
        data_formatter_template: str
            Template to apply to create single sequence and store it in dataset_text_field

    Returns:
        Dataset
            HF Dataset with formatted [str] data.
    """

    json_dataset = datasets.load_dataset("json", data_files=data_path)
    format_dataset_EOS = (
        lambda example: {  # pylint: disable=unnecessary-lambda-assignment
            f"{dataset_text_field}": example[f"{dataset_text_field}"]
            + tokenizer.eos_token
        }
    )
    if data_formatter_template:
        formatted_train_dataset = apply_custom_formatting_template(
            json_dataset["train"],
            data_formatter_template,
            dataset_text_field,
            tokenizer.eos_token,
        )
    else:
        formatted_train_dataset = json_dataset.map(format_dataset_EOS)[
            "train"
        ]  # HACK - for now, we just do both datasets separately; train is the default split
    return formatted_train_dataset


def get_preprocessed_dataset(
    data_path: str,
    tokenizer: AutoTokenizer,
    max_sequence_length: int,
    input_field_name: str,
    output_field_name: str,
) -> Dataset:
    """Loads the dataset and applies the tokenizer + custom masking logic.

    Args:
        data_path: str
            Path to the file to be loaded.
        tokenizer: AutoTokenizer
            Loaded tokenizer object to be used by the collator.
        max_sequence_length: int
            Max sequence length to be used for sequence tokenization.
        input_field_name: str
            Name of the input field in the data.
        output_field_name: str
            Name of the output field in the data.

    Returns:
        Dataset
            HF Dataset with the pretokenized data.
    """
    dataset = load_hf_dataset_from_jsonl_file(
        data_path, input_field_name, output_field_name
    )
    return dataset.map(
        preprocess_and_tokenize,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_seq_length": max_sequence_length,
            "input_field_name": input_field_name,
            "output_field_name": output_field_name,
        },
        remove_columns=[input_field_name, output_field_name],
    )


### Utils for loading the data from disk in supported formats [currently only jsonl]
def load_hf_dataset_from_jsonl_file(
    data_path: str, input_field_name: str, output_field_name: str
) -> Dataset:
    """Loads the huggingface datase as a generator.

    Args:
        data_path: str
            Path to the file to be loaded.
        input_field_name: str
            Name of the input field in the data.
        output_field_name: str
            Name of the output field in the data.

    Returns:
        Dataset
            HF Dataset with the data to be tokenized.
    """
    if input_field_name == output_field_name:
        raise ValueError("Input field name and output field name should not match!")

    def get_jsonl_object():
        with open(data_path, "r", encoding="utf-8") as jsonl_file:
            data_stream = [json.loads(line) for line in jsonl_file]
            for data in data_stream:
                yield {
                    input_field_name: data[input_field_name],
                    output_field_name: data[output_field_name],
                }

    return Dataset.from_generator(get_jsonl_object)


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


def preprocess_and_tokenize(
    element: Dict[str, str],
    tokenizer: AutoTokenizer,
    max_seq_length: int,
    input_field_name: str,
    output_field_name: str,
) -> Dict[str, Any]:
    """Loads the dataset and applies the tokenizer + custom masking logic.
    NOTE: Truncation is done in this step, but padding is not, and generally
    handled by the collator.

    Args:
        element: Dict[str, str]
            A single element of the raw Dataset of strings, whose data we would like to apply
            custom masking + tokenization logic to.
        tokenizer: AutoTokenizer
            Loaded tokenizer object to be used by the collator.
        max_sequence_length: int
            Max sequence length to be used for sequence tokenization.
        input_field_name: str
            Name of the input field in the data.
        output_field_name: str
            Name of the output field in the data.

    Returns:
        Dict[str, Any]
            Dictionary containing the input IDs/labels/attention mask for this record.
    """
    combined_seq = combine_sequence(
        element[input_field_name], element[output_field_name], tokenizer.eos_token
    )

    tokenized_comb_seqs = tokenizer(
        combined_seq, max_length=max_seq_length, truncation=True, padding=False
    )
    tokenized_input = tokenizer(
        element[input_field_name],
        max_length=max_seq_length,
        truncation=True,
        padding=False,
    )

    # mask the prompt part for avoiding loss
    masked_labels = [-100] * len(
        tokenized_input.input_ids
    ) + tokenized_comb_seqs.input_ids[len(tokenized_input.input_ids) :]

    return {
        "input_ids": tokenized_comb_seqs.input_ids,
        "labels": masked_labels,
        "attention_mask": tokenized_comb_seqs.attention_mask,
    }
