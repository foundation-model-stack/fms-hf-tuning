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

# Local
from .utils import ensure_nested_dataclasses_initialized, parsable_dataclass


@parsable_dataclass
@dataclass
class PaddingFree:

    method: str = "huggingface"

    def __post_init__(self):
        if self.method != "huggingface":
            raise ValueError("only 'huggingface' method currently supported.")


@parsable_dataclass
@dataclass
class MultiPack:

    num_processes: int = 16


@dataclass
class AttentionAndDistributedPackingConfig:

    padding_free: PaddingFree = None

    multipack: MultiPack = None

    def __post_init__(self):
        # ensure nested dataclasses initialized
        ensure_nested_dataclasses_initialized(self)

    @property
    def is_padding_free(self):
        return self.padding_free is not None
