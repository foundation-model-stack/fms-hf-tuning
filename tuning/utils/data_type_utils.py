# Standard
from typing import Union
import sys

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
