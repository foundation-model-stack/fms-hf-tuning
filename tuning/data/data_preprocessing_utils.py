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
from typing import Callable, Optional

# Third Party
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from trl import DataCollatorForCompletionOnlyLM

# Local
from tuning.config import configs


def get_data_collator(
    packing: bool,
    response_template: Optional[str],
    tokenizer: AutoTokenizer,
    is_traindata_tokenized: bool,
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
        is_traindata_tokenized: bool
            Whether train Dataset is tokenized or not
        max_seq_length: int
            Max sequence length expected

    Returns:
        Callable
            Callable collator to be leveraged by the trainer.
    """

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
        if is_traindata_tokenized:
            return DataCollatorForSeq2Seq(
                tokenizer=tokenizer, padding=True, max_length=max_seq_length
            )
        raise ValueError(
            "Could not pick a data collator. Please refer to supported data formats"
        )
