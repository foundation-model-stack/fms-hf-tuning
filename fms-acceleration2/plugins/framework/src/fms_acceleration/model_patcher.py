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
from dataclasses import asdict, dataclass
from enum import Enum
from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import importlib
import inspect
import warnings

# Third Party
import pandas as pd
import torch

# ------------------------ helpers -----------------------


def patch_target_module(
    to_patch: str,
    replace_with: Any,
    target_module: str = None,
):
    to_patch = to_patch.split(".")
    assert len(to_patch) > 1, "must have an object to patch"

    to_patch, obj_name_to_patch = to_patch[:-1], to_patch[-1]
    to_patch = ".".join(to_patch)
    source = importlib.import_module(to_patch)
    original_obj = getattr(source, obj_name_to_patch)
    setattr(source, obj_name_to_patch, replace_with)

    if target_module is not None:
        # reload and this should get the patched object
        target_module = importlib.import_module(target_module)
        importlib.reload(target_module)

        # replace it
        setattr(source, obj_name_to_patch, original_obj)


# ------------------------ classes -----------------------

# Rules will trigger on either
# - module class, which triggers on isinstance
# - callable, which will be useful to trigger on custom checks
# - (consider): adding a regex will will apply on the name
# ModelPatcherTrigger = Union[
#     torch.nn.Module, # trigger on isinstance
#     Callable[[torch.nn.Module], bool] # trigger on callable
# ]
# NOTE: triggering on instance checks will not be robust to reloading


class ModelPatcherTriggerType(Enum):
    module = 1
    callable = 2


@dataclass
class ModelPatcherTrigger:
    "Holds the triggering logic for the model patcher rule."

    # the trigger operation
    check: Union[
        Type[torch.nn.Module],  # trigger on isinstance
        Callable[[torch.nn.Module], bool],  # trigger on callable
    ]

    # holds the type of the trigger
    # - type is None that it will be a single call
    type: ModelPatcherTriggerType = None

    # if the trigger is specific to model name
    module_name: str = None

    def is_triggered(
        self,
        module: torch.nn.Module,
        module_name: str = None,
    ):
        "Check if trigger returns truthful."

        if self.module_name is not None and module_name != self.module_name:
            return False

        if self.type == ModelPatcherTriggerType.module and isinstance(
            module, self.check
        ):
            return True

        try:
            # the function call may raise
            if self.type == ModelPatcherTriggerType.callable and self.check(module):
                return True
        except Exception:  # pylint: disable=broad-exception-caught
            # NOTE: not sure if its good idea to let the exception pass through
            pass

        return False

    def __post_init__(self):
        # if check is a module
        if inspect.isclass(self.check) and issubclass(self.check, torch.nn.Module):
            if self.type is None:
                self.type = ModelPatcherTriggerType.module
            else:
                # ensure check conforms with self.type
                assert (
                    self.type == ModelPatcherTriggerType.module
                ), "type argument passed but `check` argument does not match type specified"
        # if check is a callable
        elif callable(self.check):
            if self.type is None:
                self.type = ModelPatcherTriggerType.callable
            else:
                # ensure check conforms with self.type
                assert (
                    self.type == ModelPatcherTriggerType.callable
                ), "type argument passed but `check` argument does not match type specified"
        else:
            raise TypeError("check argument needs to be torch.nn.Module or Callable")


# type for model forward
ModelForward = Callable


@dataclass
class ModelPatcherRule:
    # id, must be unique
    rule_id: str

    # trigger
    # - if trigger is none, then it will be a model file patching
    trigger: ModelPatcherTrigger = None

    # takes in the torch module to build the forward.
    # will be helpful to
    # - do any pre-modification on the torch module

    # this is mutually exclusive from forward_builder
    forward: ModelForward = None

    # returns either
    # - a callable, which will be patched on the triggered module
    # - a list of trigger-forward tuples
    forward_builder: Callable[
        [torch.nn.Module],
        Union[ModelForward, List[Tuple[ModelPatcherTrigger, ModelForward]]],
    ] = None

    # if specified, these will be passed on frrom ModelPatcher.patch
    # (if they exist)
    forward_builder_args: List[str] = None

    # this is mutually exclusive from forward and forward builder
    import_and_maybe_reload: Tuple[
        str,  # path to the object to be patched (e.g., 'torch.nn.CrossEntropyLoss')
        Type,  # replacement object (e.g., FastCrossEntropyLoss)
        Optional[
            str
        ],  # path to module to be reloaded (e.g., transformers.models.llama.modeling_llama)
    ] = None

    def __post_init__(self):
        if (
            sum(
                [
                    self.forward is not None,
                    self.forward_builder is not None,
                    self.import_and_maybe_reload is not None,
                ]
            )
            > 1
        ):
            raise ValueError(
                f"Rule '{self.rule_id}' must only have at most one of forward, "
                "forward builder, or import_and_maybe_reload, specified."
            )

        if self.import_and_maybe_reload is not None and self.trigger is not None:
            raise ValueError(
                f"Rule '{self.rule_id}' has import_and_maybe_reload specified, "
                "and trigger must be None."
            )

        if self.forward_builder_args is not None and self.forward_builder is None:
            raise ValueError(
                f"Rule '{self.rule_id}' has forward_builder_args but no "
                "forward_builder."
            )


# helpful to keep a history of all patching that has been done
@dataclass
class ModelPatcherHistory:
    # instance id of the class that was patched
    instance: int

    # class of the torch.nn.Module that was patched
    cls: str

    # parent class of the torch.nn.Module that was patched
    parent_cls: str

    # module name
    module_name: str

    # parent
    parent_module_name: str

    # name of the rule that was applied
    rule_id: str


# singleton class for patching models
class ModelPatcher:

    # singleton history of patches
    history: List[ModelPatcherHistory] = []

    # singleton list of rules that have been registered
    rules: Dict[str, ModelPatcherRule] = {}

    @staticmethod
    def load_patches(module_names: List[str], reload: bool = False):
        # each patch should be in a module that calls
        # ModelPatcher.register. So these will search
        # and load all the modules it can find

        # reload will trigger the register in that module
        for plugin_name in module_names:
            if importlib.util.find_spec(plugin_name):
                m = importlib.import_module(plugin_name)

                # attempt a reload of imported patch modules if requested
                # NOTE: but this is brittle as triggering on instance types is
                # not robust to reloading
                if reload:
                    try:
                        importlib.reload(m)
                    except AssertionError:
                        # this is if it was loaded already
                        pass

    @staticmethod
    def register(rule: ModelPatcherRule):
        # raise if added rule in duplicity
        assert (
            rule.rule_id not in ModelPatcher.rules
        ), f"patch rule '{rule.rule_id}' already exists"

        ModelPatcher.rules[rule.rule_id] = rule

    @staticmethod
    def did_rule_trigger(module: torch.nn.Module, module_name: str):

        active_rule_name, active_rule = None, None
        for name, rule in ModelPatcher.rules.items():

            # if there is no trigger
            if rule.trigger is None:
                continue

            if rule.trigger.is_triggered(module, module_name):
                # if active rule, assign the the current rule as active
                if active_rule is None:
                    active_rule_name = name
                    active_rule = rule
                # otherwise, if there is already an active rule, raise warning
                # that subsequent compatible forward rules will be ignored
                # for simple forward patches. forward_builder args are handled
                # when they are decomposed into new simple forward rules
                elif rule.forward is not None:
                    warnings.warn(
                        f"rule {rule.rule_id} is ignored on {module_name} as an \
                        earlier rule {active_rule.rule_id} has been applied"
                    )

        return active_rule_name, active_rule

    @staticmethod
    def _import_and_reload(model: torch.nn.Module):
        # each rule.import_and_maybe_reload is a triple
        # - path to be patched
        # - replacement object
        # - path to be reloaded

        # USE CASE 1:
        # from a import A # <- want to replace A by A_patched
        # def func():
        #   obj = A()

        # USE CASE 2:
        # from a import
        # def A(): # <- want to replace A by A_patched
        #   ...

        # for 1: requires a reload of the func def.
        # - the patch of A does not need to be perm
        # for 2: just requires a patch of a.A.
        # - the patch of a.A needs to be perm
        # - once a.A has been patched, 'a' cannot be reloaded

        # so for simplicity:
        # - only allow a single reload
        # - this is to allow the reload to happen first
        # - any forward patches that happen after / before
        # this import and reload should not be affected

        # (a more advanced version could be considered)
        #   targets that have a reload path as a prefix, then
        #   the reload path happens first

        # this will be the path to the module
        module_path = model.__module__

        # activate the one time rules (i.e. those with no trigger)
        _with_reload = []
        _no_reload = []
        for rule in ModelPatcher.rules.values():
            if rule.import_and_maybe_reload is not None:
                _target, _, _reload = rule.import_and_maybe_reload
                if _reload and _reload.startswith(module_path):
                    _with_reload.append(rule)
                elif _target.startswith(module_path):
                    _no_reload.append(rule)

        # If there are multiple reload targets,
        # ensure that their paths do not conflict as reloading same module might reset patches
        if len(_with_reload) > 1:
            # sort ascending target path length
            _with_reload = sorted(
                _with_reload,
                key=lambda _rule: len(_rule.import_and_maybe_reload[2]),
                reverse=False,
            )

            for i_s, rule_s in enumerate(_with_reload[:-1]):
                for rule_l in _with_reload[i_s + 1 :]:
                    # if target paths in rule s is a prefix of rule l, raise an error
                    _name_s, _obj_s, _path_s = rule_s.import_and_maybe_reload
                    _, _, _path_l = rule_l.import_and_maybe_reload

                    if _path_s == _path_l:
                        # - in the even the target is exactly the same, we will
                        # only reload once
                        rule_s.import_and_maybe_reload = (_name_s, _obj_s, None)
                        continue

                    # - otherwise, we do not consider the cases where the target
                    # is a subpath since this results in unpredictablity.
                    assert not _path_l.startswith(
                        _path_s
                    ), f"Attempting to reload a subpath`{_path_s}` multiple times in \
                            {rule_s.rule_id} and {rule_l.rule_id}"

        # handle those with reload first
        for rule in _with_reload + _no_reload:
            _target, _object, _reload = rule.import_and_maybe_reload
            patch_target_module(_target, _object, _reload)
            ModelPatcher.history.append(
                ModelPatcherHistory(
                    instance=id(model),
                    cls=model.__class__.__name__,
                    parent_cls="",
                    module_name="",
                    parent_module_name="",
                    rule_id=rule.rule_id,
                )
            )

    @staticmethod
    def _patch_forwards(
        model: torch.nn.Module,
        patch_kwargs: Dict = None,
        visited: Set = None,
        parent_prefix: str = None,
        parent_mcn: str = None,
    ):
        # NOTE: should we avoid repatching of the forwards

        if patch_kwargs is None:
            patch_kwargs = {}

        if visited is None:
            visited = set()

        for name, mod in model.named_modules():

            # some stats
            mod_id = id(mod)
            mod_class_name = mod.__class__.__name__
            name = name.split(".")
            if len(name) > 2:
                parent_module_name, module_name = ".".join(name[:-1]), name[-1]
                parent_mod = model.get_submodule(parent_module_name)
                parent_mod_class_name = parent_mod.__class__.__name__
            else:
                # patching on model itself
                module_name = name[0]
                parent_mod_class_name = parent_module_name = ""
                if parent_prefix is not None:
                    parent_module_name = parent_prefix + "." + parent_module_name
                if parent_mcn is not None:
                    parent_mod_class_name = parent_mcn

            rule_id, rule = ModelPatcher.did_rule_trigger(mod, module_name)
            if rule_id is None:
                continue

            # otherwise triggered
            if rule.forward is not None:
                forward = rule.forward
            elif rule.forward_builder is not None:
                fba = {}
                if rule.forward_builder_args is not None:
                    fba = {
                        k: w
                        for k, w in patch_kwargs.items()
                        if rule.forward_builder_args
                    }
                forward = rule.forward_builder(mod, **fba)
            else:
                # trigger-only case
                forward = None

            if isinstance(forward, list):
                # this will be list of tuples case

                # will descend down but
                # - clear old rules
                # - replace new rules
                old_rules = ModelPatcher.rules
                ModelPatcher.rules = {}
                for i, (trig, forw) in enumerate(forward):
                    ModelPatcher.register(
                        ModelPatcherRule(
                            rule_id=f"{rule_id}-{i+1}",
                            trigger=trig,
                            forward=forw,
                        )
                    )

                # this is an isolated patch
                ModelPatcher.patch(
                    mod,
                    patch_kwargs=patch_kwargs,
                    visited=visited,
                    parent_prefix=parent_module_name,
                    parent_mcn=parent_mod_class_name,
                )

                # replace the rules
                ModelPatcher.rules = old_rules

                # done
                continue

            # otherwise
            if forward is not None:
                mod.forward = MethodType(forward, mod)
            ModelPatcher.history.append(
                ModelPatcherHistory(
                    instance=mod_id,
                    cls=mod_class_name,
                    parent_cls=parent_mod_class_name,
                    module_name=module_name,
                    parent_module_name=parent_module_name,
                    rule_id=rule_id,
                )
            )

    @staticmethod
    def patch(model: torch.nn.Module, **kwargs):
        # NOTE: for a set of rules, this patch function should be called
        # only once. We do not have any checks for this at the moment

        try:
            ModelPatcher._import_and_reload(model.get_base_model())
        except AttributeError:
            ModelPatcher._import_and_reload(model)

        # this will patch the forwards
        ModelPatcher._patch_forwards(model, patch_kwargs=kwargs)

    @staticmethod
    def summary(raw: bool = False):
        df = pd.DataFrame([asdict(entry) for entry in ModelPatcher.history])
        if raw:
            return df

        if len(df) == 0:
            return ""

        # summarize and return string
        df = (
            df.groupby(["rule_id", "module_name", "cls"])["instance"]
            .count()
            .reset_index()
        )
        result = []
        result.append("***************** Module Forwards Patching *************")
        for x in df.to_dict("records"):
            result.append(
                "Rule: {0:15s} Module: {1:25s} Class: {2:15s} Num: {3:2d}".format(
                    x["rule_id"], x["module_name"], x["cls"], x["instance"]
                )
            )

        return "\n".join(result)


# ------------------------ function -----------------------


def patch_model(model: torch.nn.Module, **kwargs):
    ModelPatcher.patch(model, **kwargs)
    return model


def patch_model_summary():
    return ModelPatcher.summary()


def combine_triggers(*triggers: ModelPatcherTrigger, logic: str = "OR"):
    assert logic in [
        "AND",
        "OR",
    ], "Only `AND`, `OR` logic implemented for combining triggers"

    # NOTE: this can be probably simplified
    def _or_logic(*args, **kwargs):
        for trig in triggers:
            if trig.is_triggered(*args, **kwargs):
                return True
        return False

    def _and_logic(*args, **kwargs):
        for trig in triggers:
            if not trig.is_triggered(*args, **kwargs):
                return False
        return True

    _logic = _or_logic
    if logic == "AND":
        _logic = _and_logic

    return ModelPatcherTrigger(check=_logic)


def combine_functions(*funcs: Callable, logic: str = "APPEND"):
    assert logic == "APPEND", "only APPEND logic implemented for combining functions"

    def _append(*args, **kwargs):
        results = []
        for f in funcs:
            results += f(*args, **kwargs)
        return results

    return _append
