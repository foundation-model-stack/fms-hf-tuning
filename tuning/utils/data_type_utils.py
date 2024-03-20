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
from typing import Union

# Third Party
from transformers.utils import logging
import torch

logger = logging.get_logger("data_utils")


def str_to_torch_dtype(dtype_str: str) -> torch.dtype:
    """Given a string representation of a Torch data type, convert it to the actual torch dtype.

    Args:
        dtype_str: String representation of Torch dtype to be used; this should be an attr
        of the torch library whose value is a dtype.

    Returns:
        torch.dtype
            Data type of the Torch class being used.
    """
    dt = getattr(torch, dtype_str, None)
    if not isinstance(dt, torch.dtype):
        logger.error(" ValueError: Unrecognized data type of a torch.Tensor")
        raise ValueError("Unrecognized data type of a torch.Tensor")
    return dt


def get_torch_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    """Get the Torch data type to be used for interacting with a model.

    Args:
        dtype: Union[str, torch.dtype]
            If dtype is a torch.dtype, returns it; if it's a string, grab it from the Torch lib.

    Returns:
        torch.dtype
            Torch data type to be used.
    """
    # If a Torch dtype is passed, nothing to do
    if isinstance(dtype, torch.dtype):
        return dtype
    # TODO - If None/empty str was provided, read it from model config?
    # Otherwise convert it from a string
    return str_to_torch_dtype(dtype)
