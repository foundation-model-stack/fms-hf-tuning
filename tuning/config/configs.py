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

# Third Party
import torch
import transformers

# Local
from tuning.trackers.tracker_factory import FILE_LOGGING_TRACKER

DEFAULT_CONTEXT_LENGTH = 4096
DEFAULT_OPTIMIZER = "adamw_torch"

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<PAD>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Use Flash attention v2 from transformers, default is True"},
    )
    torch_dtype: Optional[Union[torch.dtype, str]] = torch.bfloat16
    embedding_size_multiple_of: Optional[int] = field(
        default=8,
        metadata={
            "help": "Resize model embedding layer to the nearest multiple of \
                the given number after tokenizer modifications."
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to custom tokenizer.\
                    If not provided it defaults to model_name_or_path"
        },
    )

    def __post_init__(self):
        if not self.tokenizer_name_or_path:
            self.tokenizer_name_or_path = self.model_name_or_path


@dataclass
class DataArguments:
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


@dataclass
class TrainingArguments(transformers.TrainingArguments):
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
