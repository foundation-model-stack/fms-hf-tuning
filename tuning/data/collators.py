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
from typing import Any, Optional, Union
import logging

# Third Party
from transformers import DataCollatorForLanguageModeling
import numpy as np
import torch

# Local
from tuning.data.utils import try_convert_bytes_dict_to_pil


class VisionDataCollator:
    """
    A data collator specialized for vision model (text + image) inputs.
    It uses a processor (e.g., LlavaProcessor or MllamaProcessor) to
    combine text and images into model-ready tensors.

    Padding-free tuning is not supported.
    Args:
        processor: A processor (like `LlavaProcessor`, `MllamaProcessor`, etc.).
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        """
        Collate function that:
            1. Extracts text and image data from each example in `features`.
            2. Uses `self.processor` to tokenize/encode them into a single batch.
            3. Creates `labels` by cloning `input_ids` and masking out:
                - Padding tokens (if `pad_token_id` is defined) with `-100`.
                - Special image tokens with `-100`.

        Args:
            features (List[Dict[str, Any]]):
                A list of examples (dicts). Each dict must have:
                - "processor_kwargs": Additional arguments passed to the processor.
                - "fields_name": A dict with "dataset_text_field" and "dataset_image_field"
                that identify the text and image keys inside each example.

        Returns:
            batch (Dict[str, torch.Tensor]):
                A dict where each key corresponds to a batched tensor created from the
                respective feature key in `features`. A new key `"labels"` is added,
                which is a clone of `"input_ids"` with padding and image tokens masked as `-100`.
        """

        # Pull out the keys that point to text & image fields, along with processor kwargs
        processor_kwargs = features[0]["processor_kwargs"]
        fields_name = features[0]["fields_name"]
        text_field = fields_name["dataset_text_field"]
        image_field = fields_name["dataset_image_field"]

        # Extract lists of text and images across all examples in the batch
        batch_text = [feature[text_field] for feature in features]
        batch_image = [
            feature[image_field]
            if isinstance(feature[image_field], list)
            else [feature[image_field]]
            for feature in features
        ]

        # Convert any byte-based image data to PIL images (if needed)
        batch_image = try_convert_bytes_dict_to_pil(batch_image)

        # Let the processor tokenize/combine text & images into a batch
        batch = self.processor(text=batch_text, images=batch_image, **processor_kwargs)

        # Clone input_ids to create labels
        labels = batch["input_ids"].clone()

        # Mask out pad tokens if the processor's tokenizer defines a pad_token_id
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Mask out special image tokens so they're ignored for the language loss
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.image_token
        )
        labels[labels == image_token_id] = -100

        # Include these masked labels in the batch
        batch["labels"] = labels

        return batch


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks.
    It ensures that all the tokens of the labels
    are set to an 'ignore_index'
    when they do not come from the assistant.
    This ensure that the loss is only calculated on the completion made by
    the assistant.

    Args:
        response_template (`Union[str, list[int]]`):
            the template form that indicates the
            start of the response, typically
            something like '### Response:\n'. It
            can also be passed as tokenized ids,
            which can be useful when using a tokenizer
            that encodes the response
            differently if it does not have proper context.
        instruction_template (`Union[str, list[int]]`):
            the template form that indicates the start
            of the human instruction, typically
            something like '###
            Human:\n'. Useful for assistant-style
            conversation datasets. It can also be passed
            as tokenized ids.
        mlm (`bool`, *optional*, defaults to `False`): Whether
        to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class.
            Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        *args,
        response_template: Union[str, list[int]],
        instruction_template: Optional[Union[str, list[int]]] = None,
        mlm: bool = False,
        ignore_index: int = -100,
        padding_free: bool = False,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            # The user provides a string, must tokenize
            self.instruction_token_ids = self.tokenizer.encode(
                self.instruction_template, add_special_tokens=False
            )
        else:
            # The user already provides the token ids
            self.instruction_token_ids = instruction_template

        self.response_template = response_template
        if isinstance(response_template, str):
            # The user provides a string, must tokenize
            self.response_token_ids = self.tokenizer.encode(
                self.response_template, add_special_tokens=False
            )
        else:
            # The user already provides the token ids
            self.response_token_ids = response_template

        if (
            not self.mlm
            and self.instruction_template
            and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id
        ):
            logging.warning(
                "The pad_token_id and eos_token_id values "
                "of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating "
                "questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value.",
            )

        self.ignore_index = ignore_index
        self.padding_free = padding_free

    def torch_call(
        self, examples: list[Union[list[int], Any, dict[str, Any]]]
    ) -> dict[str, Any]:
        batch = super().torch_call(examples)

        if self.instruction_template is None:
            for i in range(len(examples)):
                response_token_ids_start_idx = None

                for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[
                    0
                ]:
                    # `response_token_ids` is
                    # `'### Response:\n'`, here we are just making sure
                    # that the token IDs match
                    if (
                        self.response_token_ids
                        == batch["labels"][i][
                            idx : idx + len(self.response_token_ids)
                        ].tolist()
                    ):
                        response_token_ids_start_idx = idx

                if response_token_ids_start_idx is None:
                    logging.warning(
                        "Could not find response key %s in the following instance: "
                        "%s. This instance will be ignored in loss "
                        "calculation. Note, if this happens often, "
                        "consider increasing the `max_length`.",
                        self.response_template,
                        self.tokenizer.decode(batch["input_ids"][i]),
                    )
                    batch["labels"][i, :] = self.ignore_index
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(
                        self.response_token_ids
                    )

                    # Make pytorch loss function ignore all
                    # tokens up through the end of the response key
                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        else:
            for i in range(len(examples)):
                response_token_ids_idxs = []
                human_token_ids_idxs = []

                for assistant_idx in np.where(
                    batch["labels"][i] == self.response_token_ids[0]
                )[0]:
                    # find the indexes of the start of a response.
                    if (
                        self.response_token_ids
                        == batch["labels"][i][
                            assistant_idx : assistant_idx + len(self.response_token_ids)
                        ].tolist()
                    ):
                        response_token_ids_idxs.append(
                            assistant_idx + len(self.response_token_ids)
                        )

                if len(response_token_ids_idxs) == 0:
                    logging.warning(
                        "Could not find response key %s in the following instance: "
                        "%s. This instance will be ignored in loss "
                        "calculation. Note, if this happens often, "
                        "consider increasing the `max_length`.",
                        self.response_template,
                        self.tokenizer.decode(batch["input_ids"][i]),
                    )
                    batch["labels"][i, :] = self.ignore_index

                human_token_ids = self.instruction_token_ids
                for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:
                    # find the indexes of the start of a human answer.
                    if (
                        human_token_ids
                        == batch["labels"][i][
                            human_idx : human_idx + len(human_token_ids)
                        ].tolist()
                    ):
                        human_token_ids_idxs.append(human_idx)

                if len(human_token_ids_idxs) == 0:
                    logging.warning(
                        "Could not find instruction key `%s` in the following instance: "
                        "%s. This instance will be ignored in loss "
                        "calculation. Note, if this happens often, "
                        "consider increasing the `max_length`.",
                        self.instruction_template,
                        self.tokenizer.decode(batch["input_ids"][i]),
                    )
                    batch["labels"][i, :] = self.ignore_index

                if (
                    len(human_token_ids_idxs) > 0
                    and len(response_token_ids_idxs) > 0
                    and human_token_ids_idxs[0] > response_token_ids_idxs[0]
                ):
                    human_token_ids_idxs = [0] + human_token_ids_idxs

                for idx, (start, end) in enumerate(
                    zip(human_token_ids_idxs, response_token_ids_idxs)
                ):
                    # Make pytorch loss function ignore all non response tokens
                    if idx != 0:
                        batch["labels"][i, start:end] = self.ignore_index
                    else:
                        batch["labels"][i, :end] = self.ignore_index

                if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                    batch["labels"][i, human_token_ids_idxs[-1] :] = self.ignore_index

        if self.padding_free:
            # remove padding, `attention_mask` and add `position_ids`
            attn_mask = batch.pop("attention_mask")
            batch["input_ids"] = batch["input_ids"][attn_mask.bool()].unsqueeze(0)
            batch["position_ids"] = (
                attn_mask.cumsum(1)[attn_mask.bool()].unsqueeze(0) - 1
            )
            batch["labels"] = batch["labels"][attn_mask.bool()].unsqueeze(0)
            batch["labels"][batch["position_ids"] == 0] = self.ignore_index

            # Calculate cumulative sequence lengths for queries and
            # keys to prevent graph breaks during further computations.
            flattened_position_ids = batch["position_ids"].flatten()
            indices_q = torch.arange(
                flattened_position_ids.size(0),
                device=flattened_position_ids.device,
                dtype=torch.int32,
            )
            batch["cu_seq_lens_q"] = torch.cat(
                (
                    indices_q[flattened_position_ids == 0],
                    torch.tensor(
                        flattened_position_ids.size(),
                        device=flattened_position_ids.device,
                        dtype=torch.int32,
                    ),
                )
            ).unsqueeze(0)
            batch["cu_seq_lens_k"] = batch["cu_seq_lens_q"]

            # Determine maximum sequence lengths to
            # prevent graph breaks during further computations.
            batch["max_length_k"] = torch.tensor(
                [flattened_position_ids.max().item() + 1]
            )
            batch["max_length_q"] = batch["max_length_k"]

        return batch
