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

# Third Party
import pytest  # pylint: disable=(import-error

# First Party
from fms_acceleration.model_patcher import (
    ModelPatcher,
    ModelPatcherRule,
    ModelPatcherTrigger,
    patch_target_module,
)
from fms_acceleration.utils.test_utils import instantiate_model_patcher

# Local
from .model_patcher_fixtures import module4
from .model_patcher_test_utils import create_module_class, isolate_test_module_fixtures
from .test_model_patcher_helpers import DUMMY_RULE_ID


# Test patching of model attribute
def test_simple_forward_rule_with_mp_replaces_old_forward():
    """
    Ensure that a child submodule forward function
    is patched with a new forward function

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
            - Module4Class(torch.nn.Module):
                - attribute: Module5Class
            - module4_1
                - mod_4_function
            - module5:
                - module5_1
                    - Module5Class
                    - module_5_function

    """

    def patched_forward_function(X):
        return "patched_forward_function"

    # 1. Create an instance of Module4Class as model
    # 2. Add a submodule to Module4Class
    # 3. Create and register rule to patch forward of submodule class
    # 4. Patch model
    # 5. Ensure that model's submodule forward is replaced
    with isolate_test_module_fixtures():
        with instantiate_model_patcher():
            model = module4.Module4Class()
            SubModule1 = create_module_class(
                "SubModule1",
                namespaces={"forward": lambda self: "unpatched_forward_function"},
            )
            model.add_module("submodule_1", SubModule1())
            rule = ModelPatcherRule(
                rule_id=DUMMY_RULE_ID,
                trigger=ModelPatcherTrigger(check=SubModule1),
                forward=patched_forward_function,
            )
            ModelPatcher.register(rule)
            ModelPatcher.patch(model)

            assert model.submodule_1.forward() == "patched_forward_function"


def test_import_and_maybe_reload_rule_with_mp_replaces_old_attribute():
    """
    Module4Class has an attribute from Module5Class,
    ensure that patching Module5Class with a PatchedModuleClass,
    replaces the old attribute in Module4Class

    Module4Class(torch.nn.Module):
        - attribute: Module5Class

    """
    # 1. Register rule replacing module5.module5_1.Module5Class with a patched_mod_function
    #    reload_target is test.model_patcher.fixtures.module4
    # 2. Patch module4.Module4Class with ModelPatcher
    # 3. check patched module exist in module4.Module4Class.attribute
    PatchedModuleClass = create_module_class(
        "PatchedModClass",
    )

    with isolate_test_module_fixtures():
        with instantiate_model_patcher():
            model = module4.Module4Class()
            ModelPatcher.register(
                ModelPatcherRule(
                    rule_id=DUMMY_RULE_ID,
                    import_and_maybe_reload=(
                        "tests.model_patcher_fixtures.module4.module5.Module5Class",
                        PatchedModuleClass,
                        "tests.model_patcher_fixtures.module4",
                    ),
                )
            )
            ModelPatcher.patch(model)
            assert isinstance(module4.Module4Class().attribute, PatchedModuleClass)


def test_mp_multiple_reloads_on_same_target():
    """
    Simulate a case where two rules attempt to reload on the same target prefix

    example:
        - Rule 1 target path 1: x.y.z
        - Rule 2 target path 2: x.y

    this MIGHT reverse the patch on Rule 1 and needs to be prevented

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
            - Module4Class(torch.nn.Module):
                - attribute: Module5Class
            - module4_1
                - mod_4_function
            - module5:
                - module5_1
                    - Module5Class
                    - module_5_function

    """

    PatchedModuleClass = create_module_class(
        "PatchedModuleClass",
    )

    def patched_mod_function():
        return "patched_mod_function"

    # Demonstrate how the 2nd patch overwrites the 1st patch if the reload module paths are the same
    with isolate_test_module_fixtures():
        # 1st patch on a function
        patch_target_module(
            "tests.model_patcher_fixtures.module4.module5.module5_1.mod_5_function",
            patched_mod_function,
            "tests.model_patcher_fixtures.module4.module5",
        )

        assert module4.module5.mod_5_function() == "patched_mod_function"

        # 2nd patch on a class that has a target path that reloads module5 as well
        patch_target_module(
            "tests.model_patcher_fixtures.module4.module5.module5_1.Module5Class",
            PatchedModuleClass,
            "tests.model_patcher_fixtures.module4.module5",
        )

        assert isinstance(module4.module5.Module5Class(), PatchedModuleClass)
        assert module4.module5.mod_5_function() == "unpatched_mod_function"

    # Ensure that an assertion is raised if target paths
    # are a prefixes of another longer target path
    with pytest.raises(
        AssertionError,
    ):
        with isolate_test_module_fixtures():
            with instantiate_model_patcher():
                # 1. Initialize a model with module path tests.model_patcher_fixtures.module4
                model = module4.Module4Class()

                # 2. Simulate patching a function in module4.module5
                ModelPatcher.register(
                    ModelPatcherRule(
                        rule_id=f"{DUMMY_RULE_ID}.2",
                        import_and_maybe_reload=(
                            "tests.model_patcher_fixtures.module4.module5.module5_1.mod_5_function",
                            patched_mod_function,
                            "tests.model_patcher_fixtures.module4.module5",
                        ),
                    )
                )

                # 3. Simulate patching a class in module4 (an upstream path)
                ModelPatcher.register(
                    ModelPatcherRule(
                        rule_id=f"{DUMMY_RULE_ID}.1",
                        import_and_maybe_reload=(
                            "tests.model_patcher_fixtures.module4.module5.module5_1.Module5Class",
                            PatchedModuleClass,
                            "tests.model_patcher_fixtures.module4",
                        ),
                    )
                )

                # while there are occasions repeated reloads along the same target path prefix work,
                # the model patch will only call a reload once on the path.
                # - this is because reloading on upstream paths may intefere with downstream
                # - reload on tests.model_patcher_fixtures.module4 (shorter) will be skipped
                # - reload on tests.model_patcher_fixtures.module4.module5 (longer) will be called
                ModelPatcher.patch(model)

    # However the patch_target_module will be surreptiously called to prevent
    # the overwrites demonstrated above if targets paths are
    # are a prefixes of another longer target path
    with isolate_test_module_fixtures():
        with instantiate_model_patcher():
            # 1. Initialize a model with module path tests.model_patcher_fixtures.module4
            model = module4.Module4Class()

            # 2. Simulate patching a function in module4.module5
            ModelPatcher.register(
                ModelPatcherRule(
                    rule_id=f"{DUMMY_RULE_ID}.2",
                    import_and_maybe_reload=(
                        "tests.model_patcher_fixtures.module4.module5.module5_1.mod_5_function",
                        patched_mod_function,
                        "tests.model_patcher_fixtures.module4.module5",
                    ),
                )
            )

            # 3. Simulate patching a class in module4 (an upstream path)
            ModelPatcher.register(
                ModelPatcherRule(
                    rule_id=f"{DUMMY_RULE_ID}.1",
                    import_and_maybe_reload=(
                        "tests.model_patcher_fixtures.module4.module5.module5_1.Module5Class",
                        PatchedModuleClass,
                        "tests.model_patcher_fixtures.module4.module5",
                    ),
                )
            )

            # while there are occasions repeated reloads along the same target path prefix work,
            # the model patch will only call a reload once on the path.
            ModelPatcher.patch(model)

            # check that patching is applied to both
            assert isinstance(module4.module5.Module5Class(), PatchedModuleClass)
            assert module4.module5.mod_5_function() == "patched_mod_function"


def test_mp_throws_warning_with_multiple_patches():
    """
    Ensure for each module, only one forward patch is implemented on it.
    The patch implementation checks if there are multiple forward patch rules
    that are applied to the module, only the 1st forward patch rule is applied,
    the others will be ignored and a warning will be raised

    In the case of a list of new rules generated by `forward_builder`, it will be
    handled similarly since it decomposes to multiple single forward patch rules downstream.
    """
    with pytest.warns(
        UserWarning,
    ):
        with isolate_test_module_fixtures():
            with instantiate_model_patcher():
                # 1. Create a model
                # 2. Create a submodule to patch on
                # 3. Create 1st rule to patch submodule forward function
                # 4. Create 2nd rule to patch submodule forward function again
                # 5. Throws warning that any subsequent forward patches after
                #    the 1st patch is ignored

                model = module4.Module4Class()
                SubModule1 = create_module_class(
                    "SubModule1",
                    namespaces={"forward": lambda self: "unpatched_forward_function"},
                )
                model.add_module("submodule_1", SubModule1())

                ModelPatcher.register(
                    ModelPatcherRule(
                        rule_id=DUMMY_RULE_ID + ".1",
                        trigger=ModelPatcherTrigger(check=SubModule1),
                        forward=lambda self: "patched_forward_function",
                    )
                )
                ModelPatcher.register(
                    ModelPatcherRule(
                        rule_id=DUMMY_RULE_ID + ".2",
                        trigger=ModelPatcherTrigger(check=SubModule1),
                        forward=lambda self: "patched_forward_function_2",
                    )
                )
                ModelPatcher.patch(model)


def test_forward_builder_rule_with_mp_replaces_old_forward():
    """
    Ensure that patching a model with a rule using forward_builder argument will
    replace the children module forwards
    """

    def is_module_type_B(module):
        if hasattr(module, "B"):
            return True
        return False

    def is_module_type_C(module):
        if hasattr(module, "C"):
            return True
        return False

    def patched_forward_function(X):
        return "patched_forward_function"

    with isolate_test_module_fixtures():
        with instantiate_model_patcher():
            # 1. Create Model and 3 different child submodules
            # 2. Create the forward builder function to produce a list of
            # (trigger obj, patched forwards) for each child module in model
            # 3. Create rule on model class to patch the submodules using a forward_builder function
            # 4. Ensure all submodule forwards are patched

            SubModule1 = create_module_class(
                "SubModule1",
                namespaces={"forward": lambda X: "unpatched_forward_function"},
            )
            SubModule1A = create_module_class(
                "SubModule1A", parent_class=SubModule1, namespaces={"A": "attributeA"}
            )
            SubModule1B = create_module_class(
                "SubModule1B", parent_class=SubModule1, namespaces={"B": "attributeB"}
            )
            SubModule2 = create_module_class(
                "SubModule2",
                namespaces={
                    "C": "attributeC",
                    "forward": lambda X: "unpatched_forward_function",
                },
            )

            model = module4.module5.Module5Class()
            model.add_module("submodule_1A", SubModule1A())
            model.add_module("submodule_1B", SubModule1B())
            model.add_module("submodule_2", SubModule2())

            # Function to create different triggers for different submodules
            def build_list_of_triggers(
                module,
            ):
                return [
                    (ModelPatcherTrigger(check=SubModule1A), patched_forward_function),
                    (
                        ModelPatcherTrigger(check=is_module_type_B),
                        patched_forward_function,
                    ),
                    (
                        ModelPatcherTrigger(check=is_module_type_C),
                        patched_forward_function,
                    ),
                ]

            ModelPatcher.register(
                ModelPatcherRule(
                    rule_id=DUMMY_RULE_ID,
                    trigger=ModelPatcherTrigger(check=module4.module5.Module5Class),
                    forward_builder=build_list_of_triggers,
                )
            )

            ModelPatcher.patch(model)

            for _, mod in model.named_children():
                if hasattr(mod, "forward"):
                    assert mod.forward() == "patched_forward_function"
