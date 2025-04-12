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
from .checkpoint_utils import (
    patch_huggingface_save_and_load_for_dtensors,
    recover_safetensors_from_dcp,
)
from .scattermoe_prepare import prepare_scattermoe

# this is a special patch function to disable foreach for
# dtensors, which has been introduced since torch 2.4.
# The reason is because this will cause problems in the optimizer
# RuntimeError: aten._foreach_mul_.Scalar: got mixed torch.Tensor and DTensor,
# need to convert all torch.Tensor to DTensor before calling distributed operators!


# - this function patches torch
def patch_torch_optim_foreach_to_not_apply_to_dtensors():
    # guarded.
    # this is an array of supported types, we will remove
    # dtensor from it, so the optimizer will faillback to per
    # parameter
    # Third Party
    # pylint: disable=import-outside-toplevel
    from torch.optim.optimizer import _foreach_supported_types

    i = 0  # list index
    while i < len(_foreach_supported_types):
        x = _foreach_supported_types[i]
        if x.__name__ == "DTensor":
            # pop from list
            _foreach_supported_types.pop(i)
        else:
            i += 1
