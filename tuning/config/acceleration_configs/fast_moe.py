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
from dataclasses import dataclass, field
from typing import Union
import argparse

# Local
from .utils import ensure_nested_dataclasses_initialized, parsable_dataclass


@parsable_dataclass
@dataclass
class FastMoe:
    ep_degree: Union[int, bool] = 1
    disable_distributed: bool = field(
        default=False, metadata={"help": argparse.SUPPRESS}
    )

    def __post_init__(self):
        if isinstance(self.ep_degree, bool):
            self.disable_distributed = self.ep_degree
            self.ep_degree = 1


@dataclass
class FastMoeConfig:
    fast_moe: FastMoe = None

    def __post_init__(self):
        # ensure nested dataclasses initialized
        ensure_nested_dataclasses_initialized(self)
