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
import logging

# Third Party
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, LlavaProcessor
from trl import DataCollatorForCompletionOnlyLM

# Local
from tuning.config import configs

logger = logging.getLogger(__name__)


def get_data_collator(
    packing: bool,
    response_template: Optional[str],
    tokenizer: AutoTokenizer,
    is_traindata_tokenized: bool,
    max_seq_length: int,
    instruction_template: Optional[str],
    text_field_name: Optional[str],
    image_field_name: Optional[str],
    processor=None,
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
        instruction_template: str
            str representing the human response in a chat template
        text_field_name: str
            Field name for the text used in multi-modal dataset.
        image_field_name: str
            Field name for the images used in multi-modal dataset.
        processor:
            Model processor to combine text and image data if using
            multi-modal vision model.

    Returns:
        Callable
            Callable collator to be leveraged by the trainer.
    """

    if processor:
        if not (text_field_name or image_field_name):
            logger.error(
                "When training a vision model, you must pass in the \
                text_field_name and image_field_name of the dataset being used."
            )
        return VisionDataCollator(processor, text_field_name, image_field_name)

    if response_template and instruction_template:
        return DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            instruction_template=instruction_template,
            tokenizer=tokenizer,
            ignore_index=configs.IGNORE_INDEX,
        )

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

class VisionDataCollator:
    def __init__(self, processor, text_field_name, image_field_name):
        self.processor = processor
        self.text_field_name = text_field_name
        self.image_field_name = image_field_name

    def __call__(self, examples):
        """
        Processes both the text and images by applying the chat template
        and tokenizing the data.
        This collator takes a list of examples as input and
        returns a batch of processed data
        """
        # Get the texts and images, and apply the chat template
        texts = [
            self.processor.apply_chat_template(
                example[self.text_field_name], tokenize=False
            )
            for example in examples
        ]
        images = [example[self.image_field_name] for example in examples]

        # LLava1.5 does not support multiple images
        if isinstance(self.processor, LlavaProcessor):
            images = [image[0] for image in images]

        # Tokenize the texts and process the images
        batch = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True
        )

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        # TOOD: should we be ensuring EOS tokens is set?
        labels = batch["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        # Ignore the image token index in the loss computation (model specific)
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.image_token
        )
        labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch
