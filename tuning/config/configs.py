from dataclasses import dataclass, field
from typing import Dict, Optional, Union
import torch
import transformers

DEFAULT_CONTEXT_LENGTH=4096
DEFAULT_OPTIMIZER="adamw_torch"

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
        metadata={"help": "Use Flash attention v2 from transformers, default is True"}
    )
    torch_dtype: Optional[torch.dtype | str] = torch.bfloat16

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data in JSONL format."})
    response_template: str = field(default=None, metadata={"help": "Response template, separator to train on completions only"})
    dataset_text_field: str = field(default=None, metadata={"help": "Training dataset text field"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    peft_method: Optional[str] = field(
        default="lora",
        metadata={"help": "pt, lora or None. PEFT method to use while tuning. \
                  Either pt for prompt tuning or lora; or None for fine tuning "},
    )
    cache_dir: Optional[str] = field(default=None)
    # optim: str = field(default=DEFAULT_OPTIMIZER)
    model_max_length: int = field(
        default=DEFAULT_CONTEXT_LENGTH,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    packing: bool = field(
        default=False,
        metadata={"help": "Packing to be enabled in SFT Trainer, default is False"},
    )
