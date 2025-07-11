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
from pathlib import Path
from typing import Dict, Union
import logging

# Third Party
from datasets import Dataset, IterableDataset

# Third
from transformers import AutoProcessor, AutoTokenizer

# Local
from tuning.config.configs import DataArguments, TrainingArguments
from tuning.data.data_config import (
    DataHandlerConfig,
    DataPreProcessorConfig,
    DataSetConfig,
    load_and_validate_data_config,
)
from tuning.data.data_handlers import DataHandler
from tuning.data.data_preprocessing_utils import get_data_collator
from tuning.data.data_processors import get_datapreprocessor

logger = logging.getLogger(__name__)

# In future we may make the fields configurable
DEFAULT_INPUT_COLUMN = "input"
DEFAULT_OUTPUT_COLUMN = "output"

# check if the provided dataset is pretokenized or not
# the check is taken from trl
# https://github.com/huggingface/trl/blob/ddf4c8dc3ecf6d9ee2b24f94c62182ffd682c808/trl/trainer/sft_trainer.py#L498-L509
def is_pretokenized_dataset(data: Union[str, Dataset, IterableDataset]):
    if not data:
        return False

    if isinstance(data, str):
        # Create a data processor with default processor config
        data_processor = get_datapreprocessor(
            processor_config=DataPreProcessorConfig(), tokenizer=None
        )
        data = data_processor.load_dataset(
            None,
            streaming=False,
            splitName="train[:1]",
            datafile=data,
        )

    return ("input_ids" in data.column_names) and ("labels" in data.column_names)


def process_dataconfig_file(
    data_args: DataArguments,
    train_args: TrainingArguments,
    tokenizer: AutoTokenizer,
    additional_data_handlers: Dict[str, DataHandler] = None,
    processor: AutoProcessor = None,
    is_multipack: bool = False,
    is_padding_free: bool = False,
):
    """
    Args:
        data_args: tuning.config.configs.DataArguments
        train_args: TrainingArguments
            Training arguments passed to the library
            Used for max_steps if streaming is set.
        tokenizer: AutoTokenizer
        additional_data_handlers: A Dict of [str, DataHandler] data handlers
            which need to be registered with the data preprocessor
        processor:
            Model processor to combine text and image data if using
            multi-modal model. Defaults to None.
        is_multipack: A bool representing is Multipack plugin is enabled.
                         Defauts to False.
    Returns:
        Tuple(Dataset, Dataset, str)
            tuple containing
            train_dataset (Dataset/IterableDataset),
            eval_dataset (Dataset/IterableDataset),
            dataset_text_field (str),
    """

    data_config = load_and_validate_data_config(data_args.data_config_path)

    if not data_config:
        raise ValueError("Data config is not provided. Please check data args.")
    if (
        data_args.training_data_path is not None
        or data_args.validation_data_path is not None
    ):
        raise ValueError(
            "Both training_data_path and validation_data_path must be None when using "
            "data_config. Please provide paths in data_config instead."
        )

    data_processor = get_datapreprocessor(
        processor_config=data_config.dataprocessor,
        tokenizer=tokenizer,
        processor=processor,
        additional_data_handlers=additional_data_handlers,
    )

    if data_processor.processor_config.streaming:
        if is_padding_free:
            logging.error(
                "`padding_free` is not supported when streaming is enabled.",
            )
            raise ValueError(
                "`--padding_free` is not allowed when `streaming=True`. "
                "Please remove the `padding_free` argument from your configuration."
            )
        if train_args.max_steps < 1:
            logging.error(
                "ValueError: `--max_steps` must be set when streaming is set in data "
                "preprocessor config"
            )
            raise ValueError(
                "`--max_steps` must be set when streaming is set in data preprocessor config"
            )
        if is_multipack:
            logging.error(
                "Multipack is not compatible with streaming=true please set streaming=false "
                "or disable multipack sampler"
            )

            raise ValueError(
                "Multipack is not compatible with streaming=true please set streaming=false "
                "or disable multipack sampler"
            )

    if data_processor.processor_config.chat_template is not None:
        if tokenizer.chat_template:
            logger.warning(
                "replacing existing chat_template %s with data config's chat_template %s",
                tokenizer.chat_template,
                data_processor.processor_config.chat_template,
            )
        tokenizer.chat_template = data_processor.processor_config.chat_template

    train_dataset, eval_dataset = data_processor.process_dataset_configs(
        data_config.datasets
    )

    return (train_dataset, eval_dataset, data_args.dataset_text_field)


# Data Format 1: Pretokenized Data
def _get_pretokenized_dataset_handlers(data_args, is_eval_tokenized):

    # if the provided train dataset is pretokenized
    # however user provides formatting flags, error out
    if (
        data_args.response_template
        or data_args.data_formatter_template
        or data_args.dataset_text_field
        or data_args.instruction_template
    ):
        raise ValueError(
            "fields response_template, data_formatter_template,"
            " dataset_text_field and instruction_template \
                            are not applicable for pretokenized datasets"
        )

    # if the train dataset is pretokenized
    # ensure validation dataset is pretokenized otherwise error out
    if is_eval_tokenized:
        raise ValueError(
            "validation data should be pretokenized to be used \
            along with pretokenized train data"
        )

    # We do not need a handler here as this is tokenized dataset
    return [], None


### Data format 2
# pylint: disable=unused-argument
def _get_dataset_formatting_handlers(data_args, packing, is_padding_free=False):

    if data_args.response_template is None:
        if packing is False:
            if is_padding_free:
                logger.debug(
                    "Assuming pretraining scenario (loss over all tokens) "
                    + "because, packing is false,"
                    + " padding_free plugin is used and no response template was provided."
                )
            else:
                raise ValueError(
                    "Since response_template is not provided for masking, \
                    either use packing or padding_free to enable \
                    pretraining scenario (loss over all tokens)."
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
                "dataset_text_field and data_formatter_template are both set, "
                "but are mutually exclusive options"
            )

    fn_kwargs = {}

    if data_args.dataset_text_field:
        # First create a template using old dataset text field
        fn_kwargs[
            "template"
        ] = f"{{{{ element['{data_args.dataset_text_field}'] }}}}{{{{ eos_token }}}}"
        # Then create a formatted text column name and
        # set it to new text field which is picked while training.
        fn_kwargs["formatted_text_column_name"] = data_args.dataset_text_field
    else:
        data_args.dataset_text_field = "text"
        fn_kwargs["formatted_text_column_name"] = data_args.dataset_text_field
        fn_kwargs["template"] = data_args.data_formatter_template

    handler = DataHandlerConfig(
        "apply_custom_jinja_template",
        arguments={"fn_kwargs": fn_kwargs, "batched": False, "remove_columns": "all"},
    )
    return [handler], data_args.dataset_text_field


### Default Format 3
def _get_chat_dataset_handlers(data_args, tokenizer_kwargs):

    if data_args.dataset_text_field is None:
        data_args.dataset_text_field = "text"

    fn_kwargs = {}
    fn_kwargs["formatted_text_column_name"] = data_args.dataset_text_field
    fn_kwargs["tokenizer_kwargs"] = tokenizer_kwargs
    if data_args.dataset_conversation_field is not None:
        fn_kwargs["conversation_column"] = data_args.dataset_conversation_field

    kwargs = {"fn_kwargs": fn_kwargs, "batched": False, "remove_columns": "all"}

    handlers = [
        DataHandlerConfig("apply_tokenizer_chat_template", arguments=kwargs),
    ]

    return handlers, data_args.dataset_text_field


### Default Data format
def _get_default_dataset_handlers(data_args, tokenizer_kwargs):

    fn_kwargs = {}
    fn_kwargs["input_column_name"] = DEFAULT_INPUT_COLUMN
    fn_kwargs["output_column_name"] = DEFAULT_OUTPUT_COLUMN
    fn_kwargs["tokenizer_kwargs"] = tokenizer_kwargs

    kwargs = {
        "fn_kwargs": fn_kwargs,
        "batched": False,
        "remove_columns": "all",
    }

    handler = DataHandlerConfig("tokenize_and_apply_input_masking", arguments=kwargs)
    # This dataset is now tokenized
    return [handler], None


### Vsion Data Format
def _get_vision_dataset_handlers(data_args, processor_kwargs):

    handlers = []

    # First data handler configuration
    handler_fn_kwargs1 = {
        "dataset_text_field": data_args.dataset_text_field,
        "conversation_column": data_args.dataset_text_field,
    }
    handler_kwargs1 = {
        "fn_kwargs": handler_fn_kwargs1,
        "remove_columns": None,
    }
    handlers.append(
        DataHandlerConfig("apply_tokenizer_chat_template", arguments=handler_kwargs1)
    )

    # Second data handler configuration
    handler_fn_kwargs2 = {
        "fields_name": {
            "dataset_text_field": data_args.dataset_text_field,
            "dataset_image_field": data_args.dataset_image_field,
        },
        "processor_kwargs": processor_kwargs,
    }
    kwargs2 = {
        "fn_kwargs": handler_fn_kwargs2,
    }
    handlers.append(
        DataHandlerConfig("prepare_multimodal_data_processor", arguments=kwargs2)
    )

    return handlers, data_args.dataset_text_field


# Process raw dataargs for various usecases.
# Data Format 1: Pretokenized Data
#   Use pretokenized data as-is without preprocessing.
#   No handlers are needed for this format.
# Data Format 2: Single Sequence Dataset
#   If a text field is specified, append the tokenizer's EOS token to it.
#   If a formatter template is provided, apply it and save the result.
#   Data remains un-tokenized.
# Data Format 3: Chat datasets
#   User provides response_template and instruction_template.
# Default Data Format: Dataset with Input/Output Fields
#   Combine input and output fields, tokenize the data, and apply input attention masking.
#   Requires both input and output fields; throws an error if missing.
def _process_raw_data_args(
    data_args: DataArguments,
    tokenizer: AutoTokenizer,
    packing: bool,
    max_seq_length: int,
    additional_data_handlers: Dict[str, DataHandler] = None,
    is_padding_free: bool = False,
    processor: AutoProcessor = None,
):

    if data_args.data_config_path is not None:
        raise ValueError(
            "Both training_data_path and validation_data_path must be None when using "
            "data_config. Please provide paths in data_config instead."
        )
    # Create a data processor with default processor config
    default_processor_config = DataPreProcessorConfig()
    data_processor = get_datapreprocessor(
        processor_config=default_processor_config,
        tokenizer=tokenizer,
        processor=processor,
        additional_data_handlers=additional_data_handlers,
    )
    assert isinstance(
        data_args.training_data_path, str
    ), "Training data path has to be set and str"

    is_eval_dataset_present = False
    if data_args.validation_data_path:
        is_eval_dataset_present = True

    # TODO: This check loads first slice of the dataset to view its columns
    # Since this load is not done via processor it is redundant
    is_traindata_tokenized = is_pretokenized_dataset(data_args.training_data_path)
    is_evaldata_tokenized = is_pretokenized_dataset(data_args.validation_data_path)

    train_dataset_config = DataSetConfig(
        name="training_data",
        data_paths=[data_args.training_data_path],
        data_handlers=None,
        split={"train": 1.0, "validation": 0.0},
    )
    if is_eval_dataset_present:
        eval_dataset_config = DataSetConfig(
            name="validation_data",
            data_paths=[data_args.validation_data_path],
            data_handlers=None,
            split={"train": 0.0, "validation": 1.0},
        )

    # Setup some tokenizer kwargs for when we need a tokenizer
    # TODO: Figure out a way to not hardcode this.
    tokenizer_kwargs = {}
    tokenizer_kwargs["max_length"] = max_seq_length
    tokenizer_kwargs["truncation"] = True
    # As of peft=0.14, this must be True in batched settings
    if tokenizer.pad_token is not None:
        tokenizer_kwargs["padding"] = True
    else:
        tokenizer_kwargs["padding"] = False  # can be handled by collator

    processor_kwargs = {}
    processor_kwargs["return_tensors"] = "pt"
    processor_kwargs["padding"] = not is_padding_free

    handlers = None
    dataset_text_field = None

    if is_traindata_tokenized:
        # Data Format 1: Pretokenized Data
        handlers, dataset_text_field = _get_pretokenized_dataset_handlers(
            data_args, (is_eval_dataset_present and not is_evaldata_tokenized)
        )
    elif processor and data_args.dataset_text_field and data_args.dataset_image_field:

        handlers, dataset_text_field = _get_vision_dataset_handlers(
            data_args, processor_kwargs
        )
    elif data_args.instruction_template and data_args.response_template:
        # Data Format 2: Chat dataset with instruction and response template
        # We don't do processing for chat dataset
        handlers, dataset_text_field = _get_chat_dataset_handlers(
            data_args, tokenizer_kwargs
        )
    elif data_args.data_formatter_template or data_args.dataset_text_field:
        # Data Format 3: Single Sequence Dataset
        handlers, dataset_text_field = _get_dataset_formatting_handlers(
            data_args, packing, is_padding_free
        )
    else:
        # Default Data Format: Dataset with Input/Output Fields
        handlers, dataset_text_field = _get_default_dataset_handlers(
            data_args, tokenizer_kwargs
        )

    # Now set handlers in the dataset configs
    train_dataset_config.data_handlers = handlers
    if is_eval_dataset_present:
        eval_dataset_config.data_handlers = handlers

    eval_dataset = None
    dataset_configs = [train_dataset_config]

    if is_eval_dataset_present:
        dataset_configs.append(eval_dataset_config)

    train_dataset, eval_dataset = data_processor.process_dataset_configs(
        dataset_configs
    )

    return (train_dataset, eval_dataset, dataset_text_field)


def dump_dataset(
    output_dir: Path, nshards: int, d: Union[Dataset, IterableDataset], name: str
):
    """
    Saves the given dataset in the specified number of shards.
    Currently supports only parquet shards

    Args:
        output_dir: output directory where dataset is dumped
        shards (int): Num of shards to split the dataset into
        d (Dataset, IterableDataset): The dataset to dump
        name (str): Name of the dataset (used for logging).
    """

    if d is None:
        return
    try:
        if isinstance(d, IterableDataset):
            d = Dataset(d)
        d = d.flatten_indices()

        # Save dataset shards
        output_dir = output_dir / name
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(
            "Dumping processesd dataaset %s at %s in %d shards",
            name,
            output_dir,
            nshards,
        )
        for shard_idx in range(nshards):
            shard = d.shard(index=shard_idx, num_shards=nshards)
            shard_path = output_dir / f"ds_{shard_idx:05d}.parquet"
            shard.to_parquet(shard_path)
        logging.info("Dumped %d shards of %s", nshards, name)

        logging.info("Saved dataset %s to %s", name, output_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to dump dataset due to error {e}") from e


# If a data config file is provided, load it to get the training dataset.
# - Assumes only the training dataset is specified in the config file.
# - Expects a complete and valid data config file from the user.
#
# If no data config file is specified, process the remaining data arguments
# to determine the use case based on their presence, as explained in _process_raw_data_args.
def process_dataargs(
    data_args: DataArguments,
    tokenizer: AutoTokenizer,
    train_args: TrainingArguments,
    additional_data_handlers: Dict[str, DataHandler] = None,
    is_padding_free: bool = False,
    processor: AutoProcessor = None,
    is_multipack: bool = False,
):
    """
    Args:
        data_args: tuning.config.configs.DataArguments
        tokenizer: AutoTokenizer
        train_args: TrainingArguments
            Training arguments passed to the library
            Used for packing and max_seq_length
        additional_data_handlers: A Dict of [str, DataHandler] data handlers
            which need to be registered with the data preprocessor
        is_padding_free: A bool representing if Padding free plugin is enabled.
                         Defaults to False.
        processor:
            Model processor to combine text and image data if using
            multi-modal model. Defaults to None.
        is_multipack: A bool representing is Multipack plugin is enabled.
                         Defauts to False.
    Returns:
        Tuple(Dataset, Dataset, str, DataCollator, int, Dict)
            tuple containing
            train_dataset (Dataset/IterableDataset),
            eval_dataset (Dataset/IterableDataset),
            dataset_text_field (str),
            data_collator (DataCollator)
            max_seq_length(int) and
            dataset_kwargs (Dict)
    """

    max_seq_length = min(train_args.max_seq_length, tokenizer.model_max_length)
    logger.info("Max sequence length is %s", max_seq_length)
    if train_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            "max_seq_length %s exceeds tokenizer.model_max_length \
            %s, using tokenizer.model_max_length %s",
            train_args.max_seq_length,
            tokenizer.model_max_length,
            tokenizer.model_max_length,
        )

    train_dataset = eval_dataset = dataset_text_field = None

    if processor and not (
        data_args.dataset_text_field or data_args.dataset_image_field
    ):
        raise ValueError(
            f"When running a vision model you must provide the dataset_text_field and \
            dataset_image_field for the columns in the dataset. Values should be from \
            column names: {train_dataset.column_names}",
        )

    if data_args.data_config_path:
        train_dataset, eval_dataset, dataset_text_field = process_dataconfig_file(
            data_args,
            train_args,
            tokenizer,
            additional_data_handlers,
            processor,
            is_multipack,
            is_padding_free,
        )
    else:
        train_dataset, eval_dataset, dataset_text_field = _process_raw_data_args(
            data_args,
            tokenizer,
            train_args.packing,
            max_seq_length,
            additional_data_handlers,
            is_padding_free,
            processor,
        )

    if train_args.eval_strategy != "no" and eval_dataset is None:
        raise ValueError(
            f"`eval_strategy` is set to '{train_args.eval_strategy}' but no evaluation "
            f"dataset was provided. Please ensure that an evaluation dataset is specified "
            f"or set `eval_strategy='no'` to disable evaluation."
        )
    if train_dataset is None:
        raise ValueError(
            "Training dataset could not be created! Training Dataset is None."
            "Check your data config or ensure split sizes are valid."
        )
    if data_args.do_dataprocessing_only:
        dump_dir = Path(train_args.output_dir)
        if not dump_dir.is_absolute():
            dump_dir = dump_dir.absolute()
        dump_dataset(
            dump_dir,
            data_args.num_train_dataset_shards,
            train_dataset,
            "train_dataset",
        )
        if eval_dataset:
            dump_dataset(
                dump_dir,
                data_args.num_eval_dataset_shards,
                eval_dataset,
                "validataion_dataset",
            )
        logger.info(
            "Data Processing execution completed. Datasets saved in %s directory.",
            train_args.output_dir,
        )
        return (train_dataset, eval_dataset, None, None, None, None)

    # Note: This check should not be removed.
    #       Its important to recompute this post handling to
    #       check if we already tokenized the dataset or not.
    is_tokenized_dataset = is_pretokenized_dataset(train_dataset or eval_dataset)

    data_collator = get_data_collator(
        train_args.packing,
        data_args.response_template,
        tokenizer,
        is_tokenized_dataset,
        max_seq_length,
        data_args.instruction_template,
        is_padding_free=is_padding_free,
        processor=processor,
    )

    dataset_kwargs = {}
    # For vision model tuning prepare_dataset is skipped.
    if processor is not None:
        dataset_kwargs["skip_prepare_dataset"] = True

    if isinstance(train_dataset, IterableDataset):
        train_args.accelerator_config = {"split_batches": True}
        logger.info(
            "Setting `split_batches` to true - splitting batches among devices \
                    `per_device_train_batch_size` is now the global batch size, and \
                    should be treated as such. The main process will fetch a full \
                    batch and slice it into `num_processes` batches for each process."
        )

    return (
        train_dataset,
        eval_dataset,
        dataset_text_field,
        data_collator,
        max_seq_length,
        dataset_kwargs,
    )
