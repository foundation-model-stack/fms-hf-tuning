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
# Third Party
import torch


class VisionDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        """
        Collator function for batching already padded inputs.

        Args:
            features (List[Dict[str, List[Any]]]): A list of dict, where each dict
            represents a single example in the batch. The dict contains key as
            input_ids, attention_mask, pixel_values etc.

        Returns:
            batch (Dict[str, torch.Tensor]): A dict where each key corresponds to a batched tensor
            created from the respective feature key in `features`. A new key `"labels"` is added,
            which is a clone of `"input_ids"` with padding and image tokens masked as `-100`.
        """

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        # As chat template is applied so it should be set.
        batch = {}
        for key in features[0].keys():
            values = [feature[key] for feature in features]
            batch[key] = torch.tensor(values)

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
