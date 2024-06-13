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
from dataclasses import fields
from typing import Dict, Type, get_type_hints

# Third Party
from transformers.hf_argparser import DataClass, string_to_bool


def ensure_nested_dataclasses_initialized(dataclass: DataClass):
    type_hints: Dict[str, type] = get_type_hints(dataclass)
    for f in fields(dataclass):
        nested_type = type_hints[f.name]
        values = getattr(dataclass, f.name)
        if values is not None:
            values = nested_type(*values)
        setattr(dataclass, f.name, values)


class EnsureTypes:
    def __init__(self, *types: Type):
        _map = {bool: string_to_bool}
        self.types = [_map.get(t, t) for t in types]
        self.reset()

    def reset(self):
        self.cnt = 0

    def __call__(self, val):
        if self.cnt >= len(self.types):
            raise ValueError("EnsureTypes require 'reset' to be called to be re-used.")

        t = self.types[self.cnt]
        self.cnt += 1
        return t(val)
