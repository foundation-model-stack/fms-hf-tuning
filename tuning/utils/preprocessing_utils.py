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
import json

# Third Party
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from trl import DataCollatorForCompletionOnlyLM
import datasets

# Local
from tuning.config import configs


def get_data_trainer_kwargs(
    training_data_path: str,
    validation_data_path: str,
    packing: bool,
    response_template,
    max_sequence_length,
    tokenizer,
    dataset_text_field,
):
    """Get trainer args related to data / processing. At the moment, this consists of:
        - the training dataset
        - the evaluation dataset
        - the data collator
    The result can be kwarg expanded into the trainer initialization.
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
    packing, dataset_text_field, response_template, max_sequence_length, tokenizer
):
    if not packing:
        if dataset_text_field is None and response_template is None:
            # Use the seq2seq data collator; note that this automatically pads labels with -100
            return DataCollatorForSeq2Seq(
                tokenizer=tokenizer, padding=True, max_length=max_sequence_length
            )
        if dataset_text_field is None and response_template is not None:
            raise ValueError(
                "Packing is disabled, but no dataset_text_field is provided"
            )
        if response_template is None and dataset_text_field is not None:
            raise ValueError(
                "Packing is disabled, but no response template is provided"
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


def get_formatted_dataset(data_path: str, dataset_text_field, tokenizer):
    format_dataset = lambda example: {  # pylint: disable=unnecessary-lambda-assignment
        f"{dataset_text_field}": example[f"{dataset_text_field}"] + tokenizer.eos_token
    }
    json_dataset = datasets.load_dataset("json", data_files=data_path)
    return json_dataset.map(format_dataset)[
        "train"
    ]  # HACK - just do both datasets separately


def get_preprocessed_dataset(
    data_path: str,
    tokenizer: AutoTokenizer,
    max_sequence_length: int,
    input_field_name: str,
    output_field_name: str,
):
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
    data_path, input_field_name: str, output_field_name: str
):
    if input_field_name == output_field_name:
        raise ValueError("Input field name and output field name should not match!")

    def get_jsonl_object():
        jsonl_file = open(data_path, "r")
        data_stream = [json.loads(line) for line in jsonl_file]
        for data in data_stream:
            yield {
                input_field_name: data[input_field_name],
                output_field_name: data[output_field_name],
            }

    return Dataset.from_generator(get_jsonl_object)


### Utils for custom masking / manipulating input / output strs, etc
def combine_sequence(input_element: str, output_element: str):
    if not input_element.endswith((" ", "\n", "\t")) and not output_element.startswith(
        (" ", "\n", "\t")
    ):
        return input_element + " " + output_element
    return input_element + output_element


def preprocess_and_tokenize(
    element, tokenizer, max_seq_length, input_field_name, output_field_name
):
    """Custom preprocesssing logic to pre-tokenize the HF dataset."""
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
