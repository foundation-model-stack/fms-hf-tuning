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
from typing import Tuple

# Third Party
import numpy as np
import pytest

# Local
from tuning.utils.evaluator import RuleEvaluator


def test_mailicious_inputs_to_eval():
    """Tests the malicious rules

    Each test case has the format:
    (validation_error: str, expected_rule_is_true: bool, rule: str)
    """
    rules: list[Tuple[str, bool, str]] = [
        # Valid rules
        ("", False, "flags['is_training'] == False"),
        ("", False, "not flags['is_training']"),
        ("", True, "-10 < loss"),
        ("", True, "+1000 > loss"),
        ("", True, "~1000 < loss"),
        ("", True, "(10 + 10) < loss"),
        ("", True, "(20 - 10) < loss"),
        ("", True, "(20/10) < loss"),
        ("", True, "(20 % 10) < loss"),
        ("", False, "loss < 1.0"),
        ("", False, "(loss < 1.0)"),
        ("", False, "loss*loss < 1.0"),
        ("", False, "loss*loss*loss < 1.0"),
        ("", False, "(loss*loss)*loss < 1.0"),
        ("", True, "int(''.join(['3', '4'])) < loss"),
        ("", True, "loss < 9**9"),
        ("", False, "loss < math_sqrt(xs[0]*xs[0] + xs[1]*xs[1])"),
        ("", True, "len(xs) > 2"),
        ("", True, "loss < abs(-100)"),
        ("", True, "loss == flags.aaa.bbb[0].ccc"),
        ("", True, "array3d[0][1][1] == 4"),
        ("", True, "numpyarray[0][1][1] == 4"),
        ("", True, "unavailablemetric == None"),
        ("", False, "unavailablemetric != None"),
        ("", False, "loss < 2.0 if unavailablemetric == None else loss > 0.0"),
        ("", True, "loss < 2.0 if unavailablemetric != None else loss > 0.0"),
        ("", True, "False if loss == None else loss > 0.0"),
        (
            "",
            True,
            "len(xs) == 4 and xs[0] == 1 and (xs[1] == 0 or xs[2] == 0) and xs[3] == 2",
        ),
        # Invalid rules
        (
            "'aaa' is not defined for expression 'loss == aaa.bbb[0].ccc'",
            False,
            "loss == aaa.bbb[0].ccc",
        ),
        ("0", False, "loss == flags[0].ccc"),  # KeyError
        (
            "Attribute 'ddd' does not exist in expression 'loss == flags.ddd[0].ccc'",
            False,
            "loss == flags.ddd[0].ccc",
        ),
        (
            "Sorry, access to __attributes  or func_ attributes is not available. (__class__)",
            False,
            "'x'.__class__",
        ),
        (
            "Lambda Functions not implemented",
            False,
            # Try to instantiate and call Quitter
            "().__class__.__base__.__subclasses__()[141]('', '')()",
        ),
        (
            "Lambda Functions not implemented",
            False,
            # pylint: disable=line-too-long
            "[x for x in ().__class__.__base__.__subclasses__() if x.__name__ == 'Quitter'][0]('', '')()",
        ),
        (
            "Function 'getattr' not defined, for expression 'getattr((), '__class__')'.",
            False,
            "getattr((), '__class__')",
        ),
        (
            "Function 'getattr' not defined, for expression 'getattr((), '_' '_class_' '_')'.",
            False,
            "getattr((), '_' '_class_' '_')",
        ),
        (
            "Sorry, I will not evaluate something that long.",
            False,
            '["hello"]*10000000000',
        ),
        (
            "Sorry, I will not evaluate something that long.",
            False,
            "'i want to break free'.split() * 9999999999",
        ),
        (
            "Lambda Functions not implemented",
            False,
            "(lambda x='i want to break free'.split(): x * 9999999999)()",
        ),
        (
            "Sorry, NamedExpr is not available in this evaluator",
            False,
            "(x := 'i want to break free'.split()) and (x * 9999999999)",
        ),
        ("Sorry! I don't want to evaluate 9 ** 387420489", False, "9**9**9**9"),
        (
            "Function 'mymetric1' not defined, for expression 'mymetric1() > loss'.",
            True,
            "mymetric1() > loss",
        ),
        (
            "Function 'mymetric2' not defined, for expression 'mymetric2(loss) > loss'.",
            True,
            "mymetric2(loss) > loss",
        ),
        (
            "'<' not supported between instances of 'NoneType' and 'float'",
            True,
            "None < 2.0",
        ),
        (
            "'nonexistentmetric' is not defined for expression 'nonexistentmetric < 3.0'",
            True,
            "nonexistentmetric < 3.0",
        ),
        (
            "The metric 'unavailablemetric' is not available",
            True,
            "unavailablemetric < 2.0",
        ),
        (
            "The metric 'unavailablemetric' is not available",
            True,
            "unavailablemetric <= 2.0",
        ),
        (
            "The metric 'unavailablemetric' is not available",
            True,
            "unavailablemetric == 2.0",
        ),
        (
            "The metric 'unavailablemetric' is not available",
            True,
            "unavailablemetric != 2.0",
        ),
        (
            "The metric 'unavailablemetric' is not available",
            True,
            "unavailablemetric > 2.0",
        ),
        (
            "The metric 'unavailablemetric' is not available",
            True,
            "unavailablemetric >= 2.0",
        ),
        (
            "The metric 'unavailablemetric' is not available",
            True,
            "(unavailablemetric + unavailablemetric) < 2.0",
        ),
        (
            "The metric 'unavailablemetric' is not available",
            True,
            "(unavailablemetric - unavailablemetric) < 2.0",
        ),
        (
            "The metric 'unavailablemetric' is not available",
            True,
            "(unavailablemetric * unavailablemetric) < 2.0",
        ),
        (
            "The metric 'unavailablemetric' is not available",
            True,
            "(unavailablemetric / unavailablemetric) < 2.0",
        ),
        (
            "The metric 'unavailablemetric' is not available",
            True,
            "(unavailablemetric // unavailablemetric) < 2.0",
        ),
        (
            "The metric 'unavailablemetric' is not available",
            True,
            r"(unavailablemetric % unavailablemetric) < 2.0",
        ),
        (
            "The metric 'unavailablemetric' is not available",
            True,
            "(unavailablemetric ** unavailablemetric) < 2.0",
        ),
        (
            "The metric 'unavailablemetric' is not available",
            True,
            "(unavailablemetric << unavailablemetric) < 2.0",
        ),
        (
            "The metric 'unavailablemetric' is not available",
            True,
            "(unavailablemetric >> unavailablemetric) < 2.0",
        ),
        (
            "The metric 'unavailablemetric' is not available",
            True,
            "(unavailablemetric & unavailablemetric) < 2.0",
        ),
        (
            "The metric 'unavailablemetric' is not available",
            True,
            "(unavailablemetric ^ unavailablemetric) < 2.0",
        ),
        (
            "The metric 'unavailablemetric' is not available",
            True,
            "(unavailablemetric | unavailablemetric) < 2.0",
        ),
        # https://docs.python.org/3/reference/datamodel.html#object.__radd__
        (
            "unsupported operand type(s) for +: 'NoneType' and 'UnavailableMetric'",
            True,
            "(None + unavailablemetric) < 2.0",
        ),
        (
            "unsupported operand type(s) for -: 'NoneType' and 'UnavailableMetric'",
            True,
            "(None - unavailablemetric) < 2.0",
        ),
        (
            "The metric 'unavailablemetric' is not available",
            True,
            "xs[unavailablemetric] < 3.0",
        ),
        (
            "The metric 'unavailablemetric' is not available",
            True,
            "unavailablemetric[0] < 3.0",
        ),
        (
            "The metric 'unavailablemetric' is not available",
            True,
            "int(unavailablemetric) < 3.0",
        ),
        (
            "The metric 'unavailablemetric' is not available",
            True,
            "float(unavailablemetric) < 3.0",
        ),
        (
            "The metric 'unavailablemetric' is not available",
            True,
            "-unavailablemetric < 3.0",
        ),
        (
            "The metric 'unavailablemetric' is not available",
            True,
            "+unavailablemetric < 3.0",
        ),
        (
            "The metric 'unavailablemetric' is not available",
            True,
            "abs(unavailablemetric) < 3.0",
        ),
        (
            "The metric 'unavailablemetric' is not available",
            True,
            "(~unavailablemetric) < 3.0",
        ),
        (
            "The metric 'unavailablemetric' is not available",
            True,
            "round(unavailablemetric) < 3.0",
        ),
        (
            "The metric 'unavailablemetric' is not available",
            True,
            "math_trunc(unavailablemetric) < 3.0",
        ),
        (
            "The metric 'unavailablemetric' is not available",
            True,
            "math_floor(unavailablemetric) < 3.0",
        ),
        (
            "The metric 'unavailablemetric' is not available",
            True,
            "math_ceil(unavailablemetric) < 3.0",
        ),
    ]
    metrics = {
        "loss": 42.0,
        "flags": {"is_training": True, "aaa": {"bbb": [{"ccc": 42.0}]}},
        "xs": [1, 0, 0, 2],
        "array3d": [
            [
                [1, 2],
                [3, 4],
            ],
            [
                [5, 6],
                [7, 8],
            ],
        ],
        "numpyarray": (np.arange(8).reshape((2, 2, 2)) + 1),
        "unavailablemetric": None,
    }

    evaluator = RuleEvaluator(metrics=metrics)

    for validation_error, expected_rule_is_true, rule in rules:
        rule_parsed = evaluator.parse(expr=rule)
        if validation_error == "":
            actual_rule_is_true = evaluator.eval(
                expr=rule,
                previously_parsed=rule_parsed,
            )
            assert (
                actual_rule_is_true == expected_rule_is_true
            ), f"failed to execute the rule: '{rule}'"
        else:
            with pytest.raises(Exception) as exception_handler:
                evaluator.eval(
                    expr=rule,
                    previously_parsed=rule_parsed,
                )
            assert str(exception_handler.value) == validation_error
