# SPDX-License-Identifier: Apache-2.0
# https://spdx.dev/learn/handling-license-info/

# Third Party
import pytest
import torch

# Local
from tuning.utils import data_type_utils

dtype_dict = {
    "bool": torch.bool,
    "double": torch.double,
    "float32": torch.float32,
    "int64": torch.int64,
    "long": torch.long,
}


def test_str_to_torch_dtype():
    for t in dtype_dict.keys():
        assert data_type_utils.str_to_torch_dtype(t) == dtype_dict.get(t)


def test_str_to_torch_dtype_exit():
    with pytest.raises(SystemExit):
        data_type_utils.str_to_torch_dtype("foo")


def test_get_torch_dtype():
    for t in dtype_dict.keys():
        assert data_type_utils.get_torch_dtype(t) == dtype_dict.get(t)
        assert data_type_utils.get_torch_dtype(dtype_dict.get(t)) == dtype_dict.get(t)
