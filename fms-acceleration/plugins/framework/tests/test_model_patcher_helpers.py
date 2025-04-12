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

# Third Party
import pytest  # pylint: disable=(import-error
import torch

# First Party
from fms_acceleration.model_patcher import (
    ModelPatcherRule,
    ModelPatcherTrigger,
    ModelPatcherTriggerType,
    combine_triggers,
    patch_target_module,
)

# Local
from .model_patcher_fixtures import module1
from .model_patcher_test_utils import create_module_class, isolate_test_module_fixtures

MOD_CLS_A = create_module_class("MOD_CLS_A")
MOD_SUBCLS_A = create_module_class("MOD_SUBCLS_A", parent_class=MOD_CLS_A)
MOD_CLS_B = create_module_class("MOD_CLS_B")


def returns_false(*args, **kwargs):
    "falsy function"
    return False


def returns_true(*args, **kwargs):
    "truthy function"
    return True


DUMMY_RULE_ID = "test_patch"

# | ------------------ Test ModelPatcherTrigger ----------------------- |


def test_mp_trigger_constructs_with_check_arg_only():
    "Test construction of trigger with check argument"
    # Test that error is raised when check is not of accepted type
    with pytest.raises(
        TypeError, match="check argument needs to be torch.nn.Module or Callable"
    ):
        ModelPatcherTrigger(check=None)

    # Test module trigger type is correctly inferred from check
    trigger = ModelPatcherTrigger(check=torch.nn.Module)
    assert trigger.type == ModelPatcherTriggerType.module

    # Test callable trigger type is correctly inferred from check
    trigger = ModelPatcherTrigger(check=returns_true)
    assert trigger.type == ModelPatcherTriggerType.callable


def test_mp_trigger_constructs_with_check_and_trigger_type_args():
    "Test construction of trigger with check and type arguments"
    # check that trigger constructs successfully as check conforms to specified type
    ModelPatcherTrigger(
        check=torch.nn.Module,
        type=ModelPatcherTriggerType.module,
    )

    ModelPatcherTrigger(
        check=returns_true,
        type=ModelPatcherTriggerType.callable,
    )

    # Ensure an error is raised when check is callable but type is module
    with pytest.raises(
        AssertionError,
        match="type argument passed but `check` argument does not match type specified",
    ):
        ModelPatcherTrigger(
            check=returns_true,
            type=ModelPatcherTriggerType.module,
        )

    # Ensure an error is raised when check is module but type is callable
    with pytest.raises(
        AssertionError,
        match="type argument passed but `check` argument does not match type specified",
    ):
        ModelPatcherTrigger(
            check=torch.nn.Module,
            type=ModelPatcherTriggerType.callable,
        )


def test_mp_trigger_correctly_triggers():
    "Test for correctnness of trigger behaviour"

    ModClassA = create_module_class(
        "ModClassA",
        namespaces={"attr_1": None},
    )

    ModClassB = create_module_class(
        "ModClassB",
    )

    ModSubClassA = create_module_class(
        "ModSubClassA",
        parent_class=ModClassA,
    )

    # Scenario 1:
    # if check is a Callable, is_triggered result must be equal to the boolean output of check
    # 1. create function to check that returns true if module has attribute `attr_1`,
    # otherwise return False
    # 2. create trigger that checks the above function
    # 3. create a subclass of module_A and ensure is_triggered returns True
    # 4. create a module_B and ensure is_triggered returns False
    def check_module(module):
        if hasattr(module, "attr_1"):
            return True
        return False

    assert (
        ModelPatcherTrigger(check=check_module).is_triggered(
            ModClassA(),
        )
        is True
    )

    assert (
        ModelPatcherTrigger(check=check_module).is_triggered(
            ModClassB(),
        )
        is False
    )

    # Scenario 2:
    # Ensure return True, if is not an instance of ModelPatcherTrigger.check
    # 1. create trigger that checks for ModClassA
    # 2. create an instance of ModClassA and check is_triggered returns True
    # 3. create a subclass instance of ModClassA and check is_triggered returns True
    # 4. create an instance of ModClassB and check is_triggered returns False
    assert (
        ModelPatcherTrigger(check=ModClassA).is_triggered(
            ModClassA(),
        )
        is True
    )

    assert (
        ModelPatcherTrigger(check=ModClassA).is_triggered(
            ModSubClassA(),
        )
        is True
    )

    # Ensure returns False, if is not an instance of ModelPatcherTrigger.check
    assert (
        ModelPatcherTrigger(check=ModClassA).is_triggered(
            ModClassB(),
        )
        is False
    )

    # Scenario 3:
    # Static check to ensure additional constraint is checked
    # 1. create an instance of ModClassA as model
    # 2. register 2 submodules instances of ModClassB, Submodule_1 and SubModule_2
    # 3. create a trigger that checks for an instance of module_B and `submodule_1` module name
    # 4. for each module in model, ensure returns true if trigger detects module,
    # otherwise it should return false

    # Create model
    model = ModClassA()
    # register submodules
    model.add_module("submodule_1", ModClassB())
    model.add_module("submodule_2", ModClassB())
    # create trigger with search criteria
    trigger = ModelPatcherTrigger(check=ModClassB, module_name="submodule_1")
    # iterate through modules in model
    for name, module in model.named_modules():
        if name == "submodule_1":
            # assert that is_triggered returns true when module is found
            assert trigger.is_triggered(module, name) is True
        else:
            # assert that is_triggered otherwise returns false
            assert trigger.is_triggered(module, name) is False


# Each test instance has
#  - target_module,
#  - tuple of trigger check arguments
#  - a logic operator string
#  - expected result as either a boolean or an error tuple
# 1. Instantiate list of triggers from tuple of trigger check arguments
# 2. construct a combined trigger given list of triggers and logic
# 3. if expected_result is a tuple, ensure an error is raised upon constructing the trigger
# 4. Otherwise, ensure that the combined_trigger returns the expected result on the target module
@pytest.mark.parametrize(
    "target_module,trigger_checks,logic,expected_result",
    [
        [MOD_SUBCLS_A(), (returns_true, MOD_CLS_B), "OR", True],
        [MOD_SUBCLS_A(), (MOD_CLS_B, returns_false), "OR", False],
        [MOD_SUBCLS_A(), (MOD_CLS_A, returns_true), "OR", True],
        [MOD_CLS_B(), (returns_false, MOD_CLS_A), "AND", False],
        [MOD_CLS_B(), (MOD_CLS_B, returns_false), "AND", False],
        [MOD_CLS_B(), (MOD_CLS_B, returns_true), "AND", True],
        [
            MOD_SUBCLS_A(),
            (MOD_CLS_B, MOD_CLS_A),
            "NOR",
            (
                AssertionError,
                "Only `AND`, `OR` logic implemented for combining triggers",
            ),
        ],
    ],
)
def test_combine_mp_triggers_produces_correct_output(
    target_module, trigger_checks, logic, expected_result
):
    triggers = [ModelPatcherTrigger(check=check) for check in trigger_checks]

    # if expected_result is a tuple of (Exception, Exception_message)
    if isinstance(expected_result, tuple):
        with pytest.raises(
            expected_result[0],
            match=expected_result[1],
        ):
            combine_triggers(
                *triggers,
                logic=logic,
            )
    else:  # otherwise ensure is_triggered output returns the expected_result
        assert (
            combine_triggers(
                *triggers,
                logic=logic,
            ).is_triggered(target_module)
            is expected_result
        )


def test_mp_rule_raises_error_when_arguments_incorrectly_configured():
    "Ensure MP rule throws appropriate error when wrong argument combinations are passed"
    # Test mp rule construction raises with multiple arguments
    with pytest.raises(
        ValueError,
        match="must only have at most one of forward, "
        "forward builder, or import_and_maybe_reload, specified.",
    ):
        ModelPatcherRule(
            rule_id=DUMMY_RULE_ID,
            forward=lambda self, X: X,
            import_and_maybe_reload=(),
            forward_builder=lambda self, X: X,
        )

    # Test mp rule construction raises with trigger and import_and_reload
    with pytest.raises(
        ValueError,
        match="has import_and_maybe_reload specified, " "and trigger must be None.",
    ):
        ModelPatcherRule(
            rule_id=DUMMY_RULE_ID,
            trigger=ModelPatcherTrigger(check=torch.nn.Module),
            import_and_maybe_reload=(),
        )

    # Test that rule construction raises forward_builder_args are provided
    # without a forward_builder, this can be the case when user passes in a
    # forward instead of forward_builder
    with pytest.raises(
        ValueError, match="has forward_builder_args but no " "forward_builder."
    ):
        ModelPatcherRule(
            rule_id=DUMMY_RULE_ID, forward=lambda self, X: X, forward_builder_args=[]
        )


def test_patch_target_module_replaces_module_or_function_correctly():
    """
    Test patching of standalone file functions

    Fixtures Class Structure

    model_patcher_fixtures:
        - module1:
            - module1_1:
                - Module2Class:
                    - attribute: Module2Class
                - mod_1_function
            - module3:
                - module3_1
                    - Module3Class:
                        - attribute: mod_1_function
        - module2:
            - Module2Class:

        - module4:
            - Module4Class:
                - attribute: mod_1_function


    """

    PatchedModuleClass = create_module_class(
        "PatchedModClass",
    )

    def patched_mod_function():
        return "patched_mod_function"

    # S1 - module1_1 has function mod_1_function
    # 1. Replace module1_1.mod_1_function with new function
    # 2. Ensure patch_target_module replaces with a new function
    with isolate_test_module_fixtures():
        patch_target_module(
            "tests.model_patcher_fixtures.module1.module1_1.mod_1_function",
            patched_mod_function,
            "tests.model_patcher_fixtures.module1",
        )
        assert module1.mod_1_function() == "patched_mod_function"

    # # test patches are reset outside the context manager
    assert module1.mod_1_function() == "unpatched_mod_function"

    # S2 - module1_1.Module1Class has an attribute module2.Module2Class
    # 1. Replace Module2Class with new class and reload module1_1
    # 2. Ensure patch_target_module replaces the attribute with a new attr class
    with isolate_test_module_fixtures():
        patch_target_module(
            "tests.model_patcher_fixtures.module2.Module2Class",
            PatchedModuleClass,
            "tests.model_patcher_fixtures.module1.module1_1",
        )
        assert isinstance(module1.Module1Class().attribute, PatchedModuleClass)

    # check the the fixture isolation works
    assert not isinstance(module1.Module1Class().attribute, PatchedModuleClass)

    # S3.1 - module1.module3.module3_1 is a submodule of module1
    # 1. Replace module1.module3.module3_1.Module3Class with a new class
    # 2. No target reploading
    # - this test shows that a replacement only affects the EXACT PATH that was patched
    with isolate_test_module_fixtures():
        patch_target_module(
            "tests.model_patcher_fixtures.module1.module3.module3_1.Module3Class",
            PatchedModuleClass,
        )

        # - this is the exact module path that was patched, so it will reflect the patched class
        assert isinstance(module1.module3.module3_1.Module3Class(), PatchedModuleClass)

        # - this is the top-level module path, and shows that upper level paths will be
        #   be affected
        assert not isinstance(module1.module3.Module3Class(), PatchedModuleClass)

    # S3.2 - module1.module3.module3_1 is a submodule of module1
    # 1. Replace module1.module3.module3_1.Module3Class with a new class
    # 2. reload the top-level module path module1
    # -> NOTE: in general, we should avoid targeting any parent paths
    #    for reload
    with isolate_test_module_fixtures():
        patch_target_module(
            "tests.model_patcher_fixtures.module1.module3.module3_1.Module3Class",
            PatchedModuleClass,
            "tests.model_patcher_fixtures.module1",
        )

        # - the reload of the top level module path module1, will NOT replace module1.module3
        #   with the original version
        # - reloading top-level paths is tricky due to caching of the modules
        # - the reload of a top-level module does not cascade down to children modules.
        assert not isinstance(
            module1.module3.module3_1.Module3Class(), PatchedModuleClass
        )

    # S3.3 - module1.module3 is a submodule of module1
    # 1. Replace module1.module3.module3_1.Module3Class with a new class
    # 2. reload the top-level module path module1
    # -> NOTE: in general, we should avoid targeting any parent paths
    #    for reload
    with isolate_test_module_fixtures():
        patch_target_module(
            "tests.model_patcher_fixtures.module1.module3.Module3Class",
            PatchedModuleClass,
            "tests.model_patcher_fixtures.module1",
        )

        # - the reload of the top level module path module1, will replace module1.module3
        #   with the original version
        assert not isinstance(
            module1.module3.module3_1.Module3Class(), PatchedModuleClass
        )

    # S4 - module1.module3 submodule has a dependency on
    #      module1.module1_1.mod_1_function
    # 1. Replace the module1.module1_1.mod_1_function with a new function
    # 2. Ensure the target reloading of module1.module3 picks up the patched function
    with isolate_test_module_fixtures():
        patch_target_module(
            "tests.model_patcher_fixtures.module1.module1_1.mod_1_function",
            patched_mod_function,
            "tests.model_patcher_fixtures.module1.module3.module3_1",
        )
        assert module1.module3.Module3Class().attribute() == "patched_mod_function"
