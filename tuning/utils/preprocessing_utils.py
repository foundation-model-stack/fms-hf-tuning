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
from typing import Any, Callable, Dict, Optional
import json

# Third Party
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from trl import DataCollatorForCompletionOnlyLM
import datasets

# Local
from tuning.config import configs


def validate_data_args(
    dataset_text_field: Optional[str],
    response_template: Optional[str],
):
    # Dataset containing single sequence needs a single sequence and a response template
    if dataset_text_field is None and response_template is not None:
        raise ValueError(
            "Needs a corresponding dataset_text_feld \
                in which to look for response_template"
        )
    if response_template is None and dataset_text_field is not None:
        raise ValueError(
            "Since dataset_text_field is provided, \
                needs a corresponding response template for masking"
        )
    # Dataset containing JSON with fields and a formatter template
    # TO DO load JSON and check input/output field is present

    # in future : pretokenized Dataset may be added.


def get_data_trainer_kwargs(
    training_data_path: str,
    validation_data_path: str,
    packing: bool,
    response_template: Optional[str],
    max_sequence_length: int,
    tokenizer: AutoTokenizer,
    dataset_text_field: Optional[str],
) -> Dict[str, Any]:
    """Get trainer args related to data / processing. At the moment, this consists of:
        - the training dataset
        - the evaluation dataset
        - the data collator
        - Maybe a formatting a function [only for a special case for validation]
    The result can be kwarg expanded into the trainer initialization.

    Args:
        training_data_path: str
            Path to the training data.
        validation_data_path: str
            Path to the validation data.
        packing: bool
            Whether or not we should apply packing or not.
        response_template: Optional[str]
            Response template to be used for formatting by TRL.
        max_sequence_length: int
            Max sequence length to be used for sequence tokenization.
        tokenizer: AutoTokenizer
            Loaded tokenizer object to be used by the collator.
        dataset_text_field: Optional[str]
            Dataset text field fto be used for formatting by TRL.

    Returns:
        Dict[str, Any]
            Data related kwargs to be used by the SFT Trainer.
    """
    data_collator = get_data_collator(
        packing, dataset_text_field, response_template, max_sequence_length, tokenizer
    )
    eval_dataset = None
    data_kwargs = {}
    if isinstance(data_collator, DataCollatorForSeq2Seq):
        # HACK: This function is never called, but is needed to sidestep TRL's internal validation.
        data_kwargs["formatting_func"] = lambda x: x
        train_dataset = get_preprocessed_dataset(
            training_data_path,
            tokenizer,
            max_sequence_length,
            input_field_name="input",
            output_field_name="output",
        )
        if validation_data_path:
            eval_dataset = get_preprocessed_dataset(
                validation_data_path,
                tokenizer,
                max_sequence_length,
                input_field_name="input",
                output_field_name="output",
            )
    else:
        # Collator is a DataCollatorForCompletionOnlyLM or None;
        # Load it as JSON and apply our normal preprocessing logic
        train_dataset = get_formatted_dataset(
            training_data_path, dataset_text_field, tokenizer
        )
        if validation_data_path:
            eval_dataset = get_formatted_dataset(
                validation_data_path, dataset_text_field, tokenizer
            )

    data_kwargs["data_collator"] = data_collator
    data_kwargs["train_dataset"] = train_dataset
    data_kwargs["eval_dataset"] = eval_dataset
    return data_kwargs


def get_data_collator(
    packing: bool,
    dataset_text_field: Optional[str],
    response_template: Optional[str],
    max_sequence_length: int,
    tokenizer: AutoTokenizer,
) -> Callable:
    """Create and return the the appropriate collator type based on the configuration for packing,
    response_template, and dataset_text_field.

    Args:
        packing: bool
            Whether or not we should apply packing or not.
        dataset_text_field: Optional[str]
            Dataset text field fto be used for formatting by TRL.
        response_template: Optional[str]
            Response template to be used for formatting by TRL.
        max_sequence_length: int
            Max sequence length to be used for sequence tokenization.
        tokenizer: AutoTokenizer
            Loaded tokenizer object to be used by the collator.

    Returns:
        Callable
            Callable collator to be leveraged by the trainer.
    """
    if not packing:
        if dataset_text_field is None and response_template is None:
            # Use the seq2seq data collator; note that this automatically pads labels with -100
            return DataCollatorForSeq2Seq(
                tokenizer=tokenizer, padding=True, max_length=max_sequence_length
            )
        # TODO: near term - how response template ids are parsed out needs to be cleaned.
        # The [2:] here applies if response template has \n prefix, it is needed to strip \n,
        # otherwise template is not found. We will create issue to clean this out after we discuss
        # data formats and collators we will support.
        response_template_ids = tokenizer.encode(
            response_template, add_special_tokens=False
        )[2:]
        return DataCollatorForCompletionOnlyLM(
            response_template=response_template_ids,
            tokenizer=tokenizer,
            ignore_index=configs.IGNORE_INDEX,
        )


def get_formatted_dataset(
    data_path: str, dataset_text_field: str, tokenizer: AutoTokenizer
) -> Dataset:
    """Applies formatting to the loaded dataset instance; does NOT pretokenize data.

    Args:
        data_path: str
            Path to the file to be loaded.
        dataset_text_field: str
            Dataset text field fto be used for formatting by TRL.
        tokenizer: AutoTokenizer
            Loaded tokenizer object to be used by the collator.

    Returns:
        Dataset
            HF Dataset with formatted [str] data.
    """
    format_dataset = lambda example: {  # pylint: disable=unnecessary-lambda-assignment
        f"{dataset_text_field}": example[f"{dataset_text_field}"] + tokenizer.eos_token
    }
    json_dataset = datasets.load_dataset("json", data_files=data_path)
    return json_dataset.map(format_dataset)[
        "train"
    ]  # HACK - for now, we just do both datasets separately; train is the default split


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
def combine_sequence(input_element: str, output_element: str):
    """Combines / concatenates input & output element.

    Args:
        input_element: str
            Input component of the combined sequence.
        output_element: str
            Output component of the combined sequence.

    Returns:
        str
            Sequence combined with whitespace.
    """
    if not input_element.endswith((" ", "\n", "\t")) and not output_element.startswith(
        (" ", "\n", "\t")
    ):
        return input_element + " " + output_element
    return input_element + output_element


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
        element[input_field_name], element[output_field_name]
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
