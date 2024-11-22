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
from typing import Union
import logging

# Third Party
from datasets import Dataset, IterableDataset

# Third
from transformers import AutoTokenizer

# Local
from tuning.config.configs import DataArguments, TrainingArguments
from tuning.data.data_config import (
    DataHandlerConfig,
    DataLoaderConfig,
    DataSetConfig,
    load_and_validate_data_config,
)
from tuning.data.data_preprocessing_utils import (
    DEFAULT_JSON_INPUT_KEY,
    DEFAULT_JSON_OUTPUT_KEY,
    get_data_collator,
    validate_data_args,
)
from tuning.data.data_processors import get_dataprocessor


# check if the provided dataset is pretokenized or not
# the check is taken from trl
# https://github.com/huggingface/trl/blob/ddf4c8dc3ecf6d9ee2b24f94c62182ffd682c808/trl/trainer/sft_trainer.py#L498-L509
def is_pretokenized_dataset(data: Union[str, Dataset, IterableDataset]):
    if not data:
        return False
    if isinstance(data, str):
        # Create a data processor with default loader config
        processor = get_dataprocessor(
            dataloaderconfig=DataLoaderConfig(), tokenizer=None
        )
        data = processor.load_dataset(None, splitName="train[:1]", datafile=data)

    return ("input_ids" in data.column_names) and ("labels" in data.column_names)


# For now assume only training dataset is passed via data config file.
# This is very limited but is done to keep first implementation minimal
def _process_dataconfig_file(
    data_args: DataArguments, tokenizer: AutoTokenizer, packing: bool, max_seq_len: int
):
    data_config = load_and_validate_data_config(data_args.data_config_path)
    processor = get_dataprocessor(
        dataloaderconfig=data_config.dataloader, tokenizer=tokenizer
    )
    train_dataset = processor.process_dataset_configs(data_config.datasets)

    data_collator = get_data_collator(
        packing,
        data_args.response_template,
        tokenizer,
        # Note: Its important to recompute this post handling to
        #       check if we already tokenized the dataset or not.
        is_pretokenized_dataset(train_dataset),
        max_seq_len,
    )

    dataset_kwargs = {}
    if is_pretokenized_dataset(train_dataset):
        dataset_kwargs["skip_prepare_dataset"] = True

    ## HACK: For now just assume we take train_dataset via data config
    return (
        train_dataset,
        None,
        data_args.dataset_text_field,
        data_collator,
        max_seq_len,
        dataset_kwargs,
    )


def process_dataargs(
    data_args: DataArguments, tokenizer: AutoTokenizer, train_args: TrainingArguments
):
    """
    Args:
        data_args: tuning.config.configs.DataArguments
        tokenizer: AutoTokenizer
        train_args: TrainingArguments
            Training arguments passed to the library
            Used for packing and max_seq_length
    Returns:
        Tuple(Dataset, Dataset, str, DataCollator, int, Dict)
            tuple containing train_dataset, eval_dataset, dataset_text_field,
                data_collator, max_seq_length and dataset_kwargs
    """

    max_seq_length = min(train_args.max_seq_length, tokenizer.model_max_length)
    logging.info("Max sequence length is %s", max_seq_length)
    if train_args.max_seq_length > tokenizer.model_max_length:
        logging.warning(
            "max_seq_length %s exceeds tokenizer.model_max_length \
            %s, using tokenizer.model_max_length %s",
            train_args.max_seq_length,
            tokenizer.model_max_length,
            tokenizer.model_max_length,
        )

    if data_args.data_config_path:
        # Data config is specified so our processing path is diverging
        return _process_dataconfig_file(
            data_args, tokenizer, train_args.packing, max_seq_length
        )

    # Create a data processor with default loader config
    default_loader_config = DataLoaderConfig()
    data_processor = get_dataprocessor(
        dataloaderconfig=default_loader_config, tokenizer=tokenizer
    )

    # TODO: This check loads first slice of the dataset to view its columns
    # Since this load is not done via processor it is redundant
    is_traindata_tokenized = is_pretokenized_dataset(data_args.training_data_path)
    is_evaldata_tokenized = is_pretokenized_dataset(data_args.validation_data_path)

    # Validate if data args are set properly
    validate_data_args(
        data_args, train_args.packing, is_traindata_tokenized, is_evaldata_tokenized
    )

    is_eval_dataset_present = False
    if data_args.validation_data_path:
        is_eval_dataset_present = True

    train_dataset_config = DataSetConfig(
        name="training_data",
        data_paths=[data_args.training_data_path],
        data_handlers=None,
    )
    if is_eval_dataset_present:
        eval_dataset_config = DataSetConfig(
            name="validation_data",
            data_paths=[data_args.validation_data_path],
            data_handlers=None,
        )

    fn_kwargs = {}
    handlers = None

    # Setup some tokenizer kwargs for when we need a tokenizer
    # TODO: Figure out a way to not hardcode this.
    tokenizer_kwargs = {}
    tokenizer_kwargs["max_length"] = max_seq_length
    tokenizer_kwargs["truncation"] = True
    tokenizer_kwargs["padding"] = False

    dataset_text_field = data_args.dataset_text_field

    # Use case specific handlers
    if is_traindata_tokenized:
        # dataset_text_field is irrelevant to pretokenized datasets
        dataset_text_field = None
    elif data_args.data_formatter_template or dataset_text_field:
        if dataset_text_field is None:
            dataset_text_field = "new_formatted_field"

        if data_args.data_formatter_template is None:
            fn_kwargs["dataset_text_field"] = dataset_text_field
            handler = DataHandlerConfig(
                "apply_dataset_formatting",
                arguments={"fn_kwargs": fn_kwargs, "batched": False},
            )
            handlers = [handler]
        else:
            fn_kwargs["dataset_text_field"] = dataset_text_field
            fn_kwargs["template"] = data_args.data_formatter_template
            handler = DataHandlerConfig(
                "apply_custom_data_formatting_template",
                arguments={"fn_kwargs": fn_kwargs, "batched": False},
            )
            handlers = [handler]
    else:
        # TODO: These should be called DEFAULT in the name as they are hardcoded.
        fn_kwargs["input_field_name"] = DEFAULT_JSON_INPUT_KEY
        fn_kwargs["output_field_name"] = DEFAULT_JSON_OUTPUT_KEY

        fn_kwargs["tokenizer_kwargs"] = tokenizer_kwargs

        kwargs = {
            "fn_kwargs": fn_kwargs,
            "batched": False,
            "remove_columns": "all",
        }

        handler = DataHandlerConfig(
            "tokenize_and_apply_instruction_masking", arguments=kwargs
        )
        handlers = [handler]

    # Now set handlers in the dataset configs
    train_dataset_config.data_handlers = handlers
    if is_eval_dataset_present:
        eval_dataset_config.data_handlers = handlers

    # And let processor handle the logic
    train_dataset = data_processor.process_dataset_configs([train_dataset_config])

    logging.info("Training dataset length is %s", len(train_dataset))

    eval_dataset = None
    if is_eval_dataset_present:
        eval_dataset = data_processor.process_dataset_configs([eval_dataset_config])
        logging.info("Validation dataset length is %s", len(eval_dataset))

    data_collator = get_data_collator(
        train_args.packing,
        data_args.response_template,
        tokenizer,
        # Note: Its important to recompute this post handling to
        #       check if we already tokenized the dataset or not.
        is_pretokenized_dataset(train_dataset),
        max_seq_length,
    )

    dataset_kwargs = {}
    if is_pretokenized_dataset(train_dataset or eval_dataset):
        dataset_kwargs["skip_prepare_dataset"] = True

    return (
        train_dataset,
        eval_dataset,
        dataset_text_field,
        data_collator,
        max_seq_length,
        dataset_kwargs,
    )
