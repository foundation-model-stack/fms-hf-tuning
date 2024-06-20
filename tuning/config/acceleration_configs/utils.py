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
from dataclasses import fields, is_dataclass
from typing import Dict, List, Type, get_type_hints

# Third Party
from transformers.hf_argparser import DataClass, string_to_bool


def ensure_nested_dataclasses_initialized(dataclass: DataClass):
    """HfArgumentParser will think of the dataclass as a List with
    multiple inputs, but it will not call the constructor, so
    this is to be called at the top-level class to init all the
    nested dataclasses.
    """
    type_hints: Dict[str, type] = get_type_hints(dataclass)
    for f in fields(dataclass):
        nested_type = type_hints[f.name]
        values = getattr(dataclass, f.name)
        if values is not None and not is_dataclass(values):
            values = nested_type(*values)
        setattr(dataclass, f.name, values)


class EnsureTypes:
    """EnsureTypes is a caster with an internal state to memorize the
    the casting order, so that we can apply the correct casting type.

    e.g., EnsureTypes(int, str) will cast [x1, x2] as [int(x1), str(x2)]
    """

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


def parsable_dataclass(cls):
    """dataset decorator to masquarade as a list type, so that
    HfArgumentParser will take in multiple arguments after the
    --key arg1 arg2, ...,

    * when we override __args__, we can ensure the parseds
      - arg1 arg2 .. will get casted to the correct type

    """

    if not is_dataclass(cls):
        raise ValueError("parsable only works with dataclass")

    types = [fi.type for fi in fields(cls)]

    class ParsableDataclass(cls, List):

        # to help the HfArgumentParser arrive at correct types
        __args__ = [EnsureTypes(*types)]

        def __post_init__(self):
            # reset for another parse
            ParsableDataclass.__args__[0].reset()

            super().__post_init__()

    return ParsableDataclass
