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
from typing import Callable, Optional, Union
import logging

# Third Party
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    LlavaProcessor,
)
from trl import DataCollatorForCompletionOnlyLM

# Local
from tuning.config import configs
from tuning.data.collators import VisionDataCollator

logger = logging.getLogger(__name__)


def get_data_collator(
    packing: bool,
    response_template: Optional[str],
    tokenizer: AutoTokenizer,
    is_traindata_tokenized: bool,
    max_seq_length: int,
    instruction_template: Optional[str],
    is_padding_free: bool = False,
    processor: Optional[Union[AutoProcessor, LlavaProcessor]] = None,
) -> Callable:
    """Create and return the the appropriate collator type based on the configuration
    for packing, padding_free, chat style training, instruction tuning.

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
        instruction_template: str
            str representing the human response in a chat template
        is_padding_free: bool
            if padding free plugin is used or not
        processor:
            Model processor to combine text and image data if using
            multi-modal vision model.

    Returns:
        Callable
            Callable collator to be leveraged by the trainer.
    """

    if processor:
        if is_padding_free or packing:
            raise ValueError(
                "Vision model tuning does not support packing or padding_free tuning."
                "Please set packing=False and is_padding_free=False."
            )
        return VisionDataCollator(processor)

    if packing:
        # With SFTTrainer, packing for both tokenized and non tokenized dataset use
        # default collator, DataCollatorForLanguageModeling, and we do not need to
        # pass any explicit collator in that case.
        return None

    if is_padding_free:
        # when packing is false but padding_free is used and
        # no response template is used then its a pretrained scenario.
        # Current plugin in fms-acceleration is compatible with
        # `DataCollatorForSeq2Seq` collator hence we use this.
        return DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=False, max_length=max_seq_length
        )

    if is_traindata_tokenized:
        # Note that this automatically pads labels with -100
        # TODO check if this is sufficient for preprocessed
        # TODO with the release of trl v0.17.0, DataCollatorForSeq2Seq
        # was removed from tokenized data processing, should eventually
        # be added back in with support directly in fms-hf-tuning, not
        # dependent on trl.
        # return DataCollatorForSeq2Seq(
        #     tokenizer=tokenizer, padding=True, max_length=max_seq_length
        # )
        return None

    # TODO: near term - how response template ids are parsed out needs to be cleaned.
    # The [2:] here applies if response template has \n prefix, it is needed to strip \n,
    # otherwise template is not found. We will create issue to clean this out after we discuss
    # data formats and collators we will support.
    if response_template and instruction_template:
        # Pass both instruction and response template for chat style training.
        return DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            instruction_template=instruction_template,
            tokenizer=tokenizer,
            ignore_index=configs.IGNORE_INDEX,
        )

    if response_template:
        response_template_ids = tokenizer.encode(
            response_template, add_special_tokens=False
        )[2:]
        return DataCollatorForCompletionOnlyLM(
            response_template=response_template_ids,
            tokenizer=tokenizer,
            ignore_index=configs.IGNORE_INDEX,
        )

    raise ValueError(
        "Could not pick a data collator. Please refer to supported data formats"
    )
