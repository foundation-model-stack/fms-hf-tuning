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
    for t in dtype_dict:
        assert data_type_utils.str_to_torch_dtype(t) == dtype_dict.get(t)


def test_str_to_torch_dtype_exit():
    with pytest.raises(ValueError):
        data_type_utils.str_to_torch_dtype("foo")


def test_get_torch_dtype():
    for t in dtype_dict:
        # When passed a string, it gets converted to torch.dtype
        assert data_type_utils.get_torch_dtype(t) == dtype_dict.get(t)
        # When passed a torch.dtype, we get the same torch.dtype returned
        assert data_type_utils.get_torch_dtype(dtype_dict.get(t)) == dtype_dict.get(t)
