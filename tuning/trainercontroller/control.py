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

# SPDX-License-Identifier: Apache-2.0
# https://spdx.dev/learn/handling-license-info/

# Standard
from dataclasses import dataclass
from typing import Dict, List, Optional
import ast

# Local
from tuning.trainercontroller.operations import Operation
from tuning.trainercontroller.patience import PatienceControl


@dataclass
class OperationAction:
    """Stores the operation handler instance and corresponding action"""

    instance: Operation
    action: str


@dataclass
class Rule:
    """Stores the rule and its configuration"""

    rule: str
    rule_ast: Optional[
        ast.AST
    ] = None  # stores the abstract syntax tree of the parsed rule


@dataclass
class Control:
    """Stores the name of control, rule byte-code corresponding actions"""

    name: str
    rule: Rule
    patience: Optional[PatienceControl] = None
    operation_actions: Optional[List[OperationAction]] = None
    config: Optional[Dict] = None
