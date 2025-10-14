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
from dataclasses import dataclass
from types import MethodType
import warnings

# Third Party
from transformers import DefaultDataCollator, default_data_collator
import numpy as np


@dataclass
class DataCollatorWithFlattening(DefaultDataCollator):
    """
    Data collator used for padding free approach. Does the following:
    - concatate the entire mini batch into single long sequence [1, total_tokens]
    - no padding will be added, returns `input_ids`, `labels` and `position_ids`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            "Using `DataCollatorWithFlattening` will flatten the entire mini batch "
            "into single long sequence."
            "Make sure your attention computation is able to handle it!"
        )

    def __call__(self, features, return_tensors=None):
        """
        This implementation assumes that only 3 arguments, input_ids, position_ids and labels
        are needed by the model, anything else is dropped by the collator
        """
        if return_tensors is None:
            return_tensors = self.return_tensors

        # Preserve the the original collate behaviour to cater to all use cases
        is_labels_provided = "labels" in features[0]
        ret = {"input_ids": [], "labels": [], "position_ids": []}
        for feature in features:
            ret["input_ids"] += feature["input_ids"]
            ret["position_ids"] += list(range(len(feature["input_ids"])))
            if is_labels_provided:
                ret["labels"] += [-100] + feature["labels"][1:]
            else:
                ret["labels"] += [-100] + feature["input_ids"][1:]
        return default_data_collator([ret], return_tensors)


# from https://github.com/huggingface/trl/pull/1887
def patch_torch_call_remove_padding(collate_fn):
    _old_collate_torch_call = collate_fn.torch_call

    def _torch_call_with_remove_pad(self, examples):
        batch = _old_collate_torch_call(examples)

        # logic for removing padding as found in later TRL releases
        attn_mask = batch.pop("attention_mask")
        batch["input_ids"] = batch["input_ids"][attn_mask.bool()].unsqueeze(0)
        batch["position_ids"] = attn_mask.cumsum(1)[attn_mask.bool()].unsqueeze(0) - 1
        batch["labels"] = batch["labels"][attn_mask.bool()].unsqueeze(0)
        batch["labels"][batch["position_ids"] == 0] = self.ignore_index

        return batch

    collate_fn.torch_call = MethodType(_torch_call_with_remove_pad, collate_fn)
    return collate_fn


def calculate_token_lengths(dataset, num_processes):
    return np.array(
        dataset.map(
            lambda x: {"len": len(x["input_ids"])},
            num_proc=num_processes,
            load_from_cache_file=True,
        )["len"]
    )
