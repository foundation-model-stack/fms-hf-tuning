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
from dataclasses import dataclass, field
from typing import List, Optional, Union
import os

# Third Party
from datasets import Dataset, IterableDataset, interleave_datasets
from pyarrow.lib import ArrowInvalid
from tqdm import tqdm
from transformers.utils import logging
import datasets
import torch
import transformers
import yaml

# Local
from tuning.trackers.tracker_factory import FILE_LOGGING_TRACKER
from tuning.utils.data_loaders import ConstantLengthHybridDataset
from tuning.utils.preprocessing_utils import is_pretokenized_dataset

DEFAULT_CONTEXT_LENGTH = 4096
DEFAULT_OPTIMIZER = "adamw_torch"

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<PAD>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


logger = logging.get_logger("sft_trainer")


def _load_data(data_path, split, streaming, config_kwargs):
    if os.path.isdir(data_path):
        return datasets.load_dataset(
            path=data_path,
            split=split,
            streaming=streaming,
            **config_kwargs,
        )
    return datasets.load_dataset(
        path=os.path.dirname(data_path),
        data_files=data_path,
        split=split,
        streaming=streaming,
        **config_kwargs,
    )


def load_dataset(
    data_path: str, split="train", streaming=False, column_name_options: list = []
) -> Union[Dataset, IterableDataset, None]:
    """loads datasets given as either a file or a directory

    Args:
        data_path (str): path to the dataset file or directory
        split (str): data split
        column_names (list): list of column names to try to load.
        loads only one column at this point.
        Applicable only for columnar datasets
    Returns:
        datasets.Dataset: loaded dataset
    """
    if not data_path:
        return None
    config_kwargs = None
    for option in column_name_options:
        config_kwargs = {"columns": [option]}
        try:
            return _load_data(
                data_path=data_path,
                split=split,
                streaming=streaming,
                config_kwargs=config_kwargs,
            )
        except Exception:
            logger.warning(
                "column name {} is not availble in the given data file. or \
                    provided data format does not support columnar operations \
                    Trying other options".format(
                    option
                )
            )
    logger.warning(
        "None of the given column name options worked, loading the entire dataset"
    )
    config_kwargs = {}
    return _load_data(
        data_path=data_path,
        split=split,
        streaming=streaming,
        config_kwargs=config_kwargs,
    )


def load_multi_dataset_with_sampling(data_config, column_name_options):
    train_datasets = []
    train_probabilities = []
    # load all train dataset and collect corresponding sampling probablities
    logger.warning("loading train datasets")
    for data_path in tqdm(
        data_config.train_datasets, total=len(data_config.train_datasets)
    ):
        train_datasets.append(
            load_dataset(
                data_path["path"],
                streaming=data_config.streaming,
                column_name_options=column_name_options,
            )
        )
        train_probabilities.append(data_path["prob"])
    validation_datasets = []
    validation_probabilities = [1 / len(train_datasets)] * len(train_datasets)
    updated_traindatasets = []
    if data_config.test_split != 0:
        for train_dataset in train_datasets:
            train_dataset = train_dataset.train_test_split(
                test_size=data_config.test_split
            )
            updated_traindatasets.append(train_dataset["train"])
            validation_datasets.append(train_dataset["test"])
        return (
            updated_traindatasets,
            train_probabilities,
            validation_datasets,
            validation_probabilities,
        )

    validation_datasets = []
    validation_probabilities = []
    # load all validation datasets and collect sampling probabilities
    if data_config.validation_datasets:
        for data_path in data_config.validation_datasets:
            validation_datasets.append(
                load_dataset(
                    data_path["path"],
                    streaming=data_config.streaming,
                    column_name_options=column_name_options,
                )
            )
            validation_probabilities.append(data_path["prob"])

    return (
        train_datasets,
        train_probabilities,
        validation_datasets,
        validation_probabilities,
    )


@dataclass
class ModelDataArguments(
    transformers.TrainingArguments
):  # pylint: disable=too-many-instance-attributes
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Use Flash attention v2 from transformers, default is True"},
    )
    torch_dtype: Optional[Union[torch.dtype, str]] = torch.bfloat16
    embedding_size_multiple_of: Optional[int] = field(
        default=1,
        metadata={
            "help": "Resize model embedding layer to the nearest multiple of \
                the given number after tokenizer modifications. \
                    NOTE: This involves extending \
                    the embedding layer without any corresponding real tokens."
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to custom tokenizer. \
                If not provided it defaults to model_name_or_path \
                and special tokens will be added as needed for specific tokenizer classes. \
                For prompt tuning, if tokenizer_name_or_path provided, special tokens are not added, \
                otherwise, it defaults to model_name_or_path with special tokens for specific \
                tokenizer classes."
        },
    )
    training_data_path: str = field(
        default=None,
        metadata={"help": "Path to the training data in JSON/JSONL format."},
    )
    response_template: str = field(
        default=None,
        metadata={"help": "Response template, separator to train on completions only"},
    )
    dataset_text_field: str = field(
        default=None,
        metadata={
            "help": "Training dataset text field containing single sequence. \
                    Either the dataset_text_field \
                    or data_formatter_template need to be supplied."
        },
    )
    validation_data_path: str = field(
        default=None,
        metadata={"help": "Path to the validation data in JSON/JSONL format."},
    )
    data_formatter_template: str = field(
        default=None,
        metadata={
            "help": "formatter template to format a single sequence \
                         from each instance in JSONL files. \
                         Keys of JSON can be referred to as {{key}} in template. \
                         Either the dataset_text_field \
                         or data_formatter_template needs to be supplied."
        },
    )
    data_config_path: str = field(
        default=None,
        metadata={
            "help": "supports advanced data usecases such as sampling \
            ratios, data splits, streaming. Should be a .yaml / .yml file"
        },
    )
    train_dataset: Dataset = field(
        default=None,
        metadata={"help": "pass HuggingFace IterableDataset or Dataset"},
    )
    validation_dataset: Dataset = field(
        default=None,
        metadata={"help": "pass HuggingFace IterableDataset or Dataset"},
    )
    input_feature: str = field(default=None)
    output_feature: str = field(default=None)
    tokens_field: str = field(default=None)
    seq_length: int = field(default=None)
    add_bos_token: bool = field(default=None)
    add_eos_token: bool = field(default=None)

    cache_dir: Optional[str] = field(default=None)
    # optim: str = field(default=DEFAULT_OPTIMIZER)
    max_seq_length: int = field(
        default=DEFAULT_CONTEXT_LENGTH,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded \
            (and possibly truncated)."
        },
    )
    packing: bool = field(
        default=False,
        metadata={"help": "Packing to be enabled in SFT Trainer, default is False"},
    )
    save_strategy: str = field(
        default="epoch",
        metadata={
            "help": "The checkpoint save strategy to adopt during training. \
            Possible values are 'no'(no save is done during training), \
            'epoch' (save is done at the end of each epoch), \
            'steps' (save is done every `save_steps`)"
        },
    )
    save_model_dir: str = field(
        default=None,
        metadata={
            "help": "Directory where tuned model will be saved to \
                  using SFTTrainer.save_model()."
        },
    )
    logging_strategy: str = field(
        default="epoch",
        metadata={
            "help": "The logging strategy to adopt during training. \
            Possible values are 'no'(no logging is done during training), \
            'epoch' (logging is done at the end of each epoch), \
            'steps' (logging is done every `logging_steps`)"
        },
    )
    trackers: Optional[List[str.lower]] = field(
        default_factory=lambda: [FILE_LOGGING_TRACKER],
        metadata={
            "help": "Experiment trackers to use.\n"
            + "Available trackers are - file_logger(default), aim, none\n"
            + "Requires additional configs, see tuning.configs/tracker_configs.py"
        },
    )
    log_level: str = field(
        default="passive",
        metadata={
            "help": "The log level to adopt during training. \
            By default, 'passive' level is set which keeps the \
            current log level for the Transformers library (which will be 'warning` by default) \
            Other possible values are 'debug', 'info', 'warning', 'error' and 'critical'"
        },
    )
    dataset_kwargs = None

    def __post_init__(self):
        # loading of the data is handled at the data class so that only
        # the object flows to the remaining code
        if not (self.data_config_path or self.training_data_path):
            raise FileNotFoundError(
                "either data_config_path or training_data_path should be provided"
            )
        if self.data_config_path and self.training_data_path:
            raise ValueError(
                "both data_config_path and training_data_path can't be provided"
            )

        if self.data_config_path:
            data_config = None
            with open(self.data_config_path, "r", encoding="utf-8") as f:
                data_config: AdvDataConfig = AdvDataConfig(**yaml.safe_load(f))
            logger.warning(
                "using load_multi_dataset_with_sampling to load the datasets"
            )
            logger.warning("populating column names options to load")
            column_name_options = []
            if not self.input_feature:
                self.input_feature = data_config.input_feature
                if self.input_feature:
                    column_name_options.append(self.input_feature)
            if not self.output_feature:
                self.output_feature = data_config.output_feature
                if self.output_feature:
                    column_name_options.append(self.output_feature)
            if not self.tokens_field:
                self.tokens_field = data_config.tokens_field
                if self.tokens_field:
                    column_name_options.append(self.tokens_field)
            if self.dataset_text_field:
                column_name_options.append(self.dataset_text_field)
            if not self.seq_length:
                self.seq_length = data_config.seq_length
            if not self.add_bos_token:
                self.add_bos_token = data_config.add_bos_token
            if not self.add_eos_token:
                self.add_eos_token = data_config.add_eos_token
            (
                train_datasets,
                train_probs,
                validation_datasets,
                validation_probs,
            ) = load_multi_dataset_with_sampling(
                data_config=data_config, column_name_options=column_name_options
            )
            if data_config.data_sampler == "tokens_based":
                if not self.packing:
                    raise ValueError(
                        "tokens based data sampler can only be used when packing is set to true"
                    )
            if (
                is_pretokenized_dataset(train_datasets[0])
                or self.tokens_field
                or data_config.data_sampler == "tokens_based"
            ):
                if self.packing:
                    logger.warning("using ConstantLengthHybridDataset")
                    self.dataset_kwargs = {"skip_prepare_dataset": True}
                    if validation_datasets and not is_pretokenized_dataset(
                        validation_datasets[0]
                    ):
                        raise ValueError(
                            "validation data has to be pretokenized when \
                                training data is pretokenized"
                        )
                    tokenizer = transformers.AutoTokenizer.from_pretrained(
                        (
                            self.tokenizer_name_or_path
                            if self.tokenizer_name_or_path
                            else self.model_name_or_path
                        ),
                        cache_dir=self.cache_dir,
                        use_fast=True,
                    )
                    self.train_dataset = ConstantLengthHybridDataset(
                        train_datasets,
                        train_probs,
                        self.seq_length,
                        1024,
                        tokenizer,
                        self.tokens_field,
                        self.dataset_text_field,
                        self.add_bos_token,
                        self.add_eos_token,
                    )
                    if validation_datasets:
                        self.validation_dataset = ConstantLengthHybridDataset(
                            validation_datasets,
                            validation_probs,
                            self.seq_length,
                            1024,
                            tokenizer,
                            self.tokens_field,
                            self.dataset_text_field,
                            self.add_bos_token,
                            self.add_eos_token,
                        )
                    if not data_config.streaming:

                        def data_generator(constant_length_iterator):
                            yield from constant_length_iterator

                        self.train_dataset = Dataset.from_generator(
                            data_generator,
                            gen_kwargs={"constant_length_iterator": self.train_dataset},
                        )
                        if self.validation_dataset:
                            self.validation_dataset = Dataset.from_generator(
                                data_generator,
                                gen_kwargs={
                                    "constant_length_iterator": self.validation_dataset
                                },
                            )
                    return

            self.train_dataset = interleave_datasets(
                train_datasets,
                seed=data_config.sampling_seed,
                stopping_strategy=data_config.dataset_stopping_strategy,
                probabilities=train_probs,
            )
            if validation_datasets:
                self.validation_dataset = interleave_datasets(
                    validation_datasets,
                    seed=data_config.sampling_seed,
                    stopping_strategy=data_config.dataset_stopping_strategy,
                    probabilities=validation_probs,
                )
            return
        if self.training_data_path:
            self.train_dataset = load_dataset(self.training_data_path)
            if self.validation_data_path:
                self.validation_dataset = load_dataset(self.validation_data_path)


@dataclass
class TrainerControllerArguments:
    trainer_controller_config_file: str = field(
        default=None,
        metadata={
            "help": (
                "Trainer controller configuration file (e.g trainercontroller_config.yaml) \
                    in YAML format."
            )
        },
    )


@dataclass
class AdvDataConfig:  # pylint: disable=too-many-instance-attributes
    train_datasets: list = field(default=None)
    validation_datasets: list = field(default=None)
    test_split: float = field(default=0)
    streaming: bool = field(default=True)
    dataset_stopping_strategy: str = field(default="all_exhausted")
    sampling_seed: int = field(default=42)
    input_feature: str = field(default=None)
    output_feature: str = field(default=None)
    tokens_field: str = field(default=None)
    seq_length: int = field(default=2048)
    add_bos_token: bool = field(default=True)
    add_eos_token: bool = field(default=True)
    # two options
    # tokens_based
    # interleave_datasets
    data_sampler: str = field(default="tokens_based")

    def __post_init__(self):
        if not self.train_datasets:
            raise ValueError("train_datasets have to be provided")
        if self.test_split != 0 and self.validation_datasets:
            raise ValueError(
                "both test_split and validation_datasets cannot be provided"
            )
        if self.test_split < 0 or self.test_split > 1:
            raise ValueError("test split should be between 0 and 1")
        if self.streaming and self.test_split > 0:
            raise ValueError("test split is not supported when streaming is true")
