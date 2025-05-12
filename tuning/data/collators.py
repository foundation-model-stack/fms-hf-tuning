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
