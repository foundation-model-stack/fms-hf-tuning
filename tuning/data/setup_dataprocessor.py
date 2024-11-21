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
from typing import Optional
import logging

# Third Party
from transformers import AutoTokenizer

# Local
from tuning.config.configs import DataArguments
from tuning.data.data_config import (
    DataHandlerConfig,
    DataLoaderConfig,
    DataSetConfig,
    load_and_validate_data_config,
)
from tuning.data.data_processors import get_dataprocessor
from tuning.utils.preprocessing_utils import (
    JSON_INPUT_KEY,
    JSON_OUTPUT_KEY,
    is_pretokenized_dataset,
)


def process_dataargs(
    data_args: DataArguments, tokenizer: AutoTokenizer, max_seq_length: int
):

    if data_args.data_config_path:
        data_config = load_and_validate_data_config(data_args.data_config_path)
        processor = get_dataprocessor(
            dataloaderconfig=data_config.dataloader, tokenizer=tokenizer
        )
        train_dataset = processor.process_dataset_configs(data_config.datasets)
        ## HACK: For now just assume we take train_dataset via data config
        return train_dataset, None, data_args.dataset_text_field

    validation_dataset = False
    if data_args.validation_data_path:
        validation_dataset = True

    # Create a data processor with default loader config
    default_loader_config = DataLoaderConfig()
    data_processor = get_dataprocessor(
        dataloaderconfig=default_loader_config, tokenizer=tokenizer
    )

    train_dataset_config = DataSetConfig(
        name="training_data",
        data_paths=[data_args.training_data_path],
        data_handlers=None,
    )
    if validation_dataset:
        eval_dataset_config = DataSetConfig(
            name="validation_data",
            data_paths=[data_args.validation_data_path],
            data_handlers=None,
        )

    # TODO: This check is something which is not appreciated as it loads dataset again.
    # This should be done somehow post the loading of dataset or removed altogether
    is_train_data_pretokenized = is_pretokenized_dataset(data_args.training_data_path)

    fn_kwargs = {}
    handlers = None

    dataset_text_field = data_args.dataset_text_field

    # Use case specific handlers
    if is_train_data_pretokenized:
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
        fn_kwargs["input_field_name"] = JSON_INPUT_KEY
        fn_kwargs["output_field_name"] = JSON_OUTPUT_KEY

        # TODO: This is a bad hardcode, move this up in the code.
        tokenizer_kwargs = {}
        tokenizer_kwargs["max_length"] = max_seq_length
        tokenizer_kwargs["truncation"] = True
        tokenizer_kwargs["padding"] = False

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
    if validation_dataset:
        eval_dataset_config.data_handlers = handlers

    # Now let process handle the logic
    train_dataset = data_processor.process_dataset_configs([train_dataset_config])

    logging.info("Training dataset length is %s", len(train_dataset))

    eval_dataset = None
    if validation_dataset:
        eval_dataset = data_processor.process_dataset_configs([eval_dataset_config])
        logging.info("Validation dataset length is %s", len(eval_dataset))

    return train_dataset, eval_dataset, dataset_text_field


# For now assume 2 differnet arguments for training and validation dataset config files.
# This is very limited but is done to keep first implementation minimal
def process_dataconfig_file(dataconfigfile: str, tokenizer: AutoTokenizer):
    dataconfig = load_and_validate_data_config(data_config_file=dataconfigfile)
    loader = dataconfig.dataloader
    processor = get_dataprocessor(dataloaderconfig=loader, tokenizer=tokenizer)
    dataset = processor.process_dataset_configs(dataset_cofigs=dataconfig.datasets)
    return dataset
