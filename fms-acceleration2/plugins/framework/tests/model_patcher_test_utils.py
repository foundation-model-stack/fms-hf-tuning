# Copyright The IBM Tuning Team
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
from contextlib import contextmanager
from typing import Any, Dict, Type
import importlib
import os
import sys

# Third Party
import torch

ROOT = "tests.model_patcher_fixtures"
MODULE_PATHS = []
for root, dirs, files in os.walk(ROOT.replace(".", os.path.sep)):
    for f in files:
        filename, ext = os.path.splitext(f)
        if ext != ".py":
            continue
        if filename != "__init__":
            p = os.path.join(root, filename)
        else:
            p = root

        MODULE_PATHS.append(p.replace(os.path.sep, "."))


@contextmanager
def isolate_test_module_fixtures():
    old_mod = {k: sys.modules[k] for k in MODULE_PATHS if k in sys.modules}
    yield

    # Reload only reloads the speicified module, but makes not attempt to reload
    # the imports of that module.
    # - i.e., This moeans that if and import had been changed
    #         then the reload will take the changed import.
    # - i.e., This also means that the individuals must be reloaded seperatedly
    #            for a complete reset.
    #
    # Therefore, we need to reload ALL Modules in opposite tree order, meaning that
    # the children must be reloaded before their parent

    for key in sorted(old_mod.keys(), key=len, reverse=True):
        # Unclear why but needs a reload, to be investigated later
        importlib.reload(old_mod[key])


def create_module_class(
    class_name: str,
    namespaces: Dict[str, Any] = None,
    parent_class: Type = torch.nn.Module,
):
    if namespaces is None:
        namespaces = {}
    return type(class_name, (parent_class,), namespaces)
