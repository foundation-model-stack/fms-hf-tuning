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
from enum import Enum
from types import MethodType
from typing import Any, Callable, Dict, List

# Third Party
from accelerate import Accelerator
from torch.utils.data import DataLoader

# AcceleratorPatcher allows various modifications to the accelerator object:
# - includes replacements of various components, and other things (e.g., inserting)
#   additional metrics in the outputs of a model forward
# - AcceleratorPatcherComponent regards the components that can be replaced
# - the AcceleratorRule abstracts logic for modifying AcceleratorPatcherComponent
# NOTE: currently only AcceleratorRuleReplace is implemented

# ---------------------------------- CLASSES -----------------------------------


# Components that can be modified / replaced via the patching of the accelerator
class AcceleratorPatcherComponent(Enum):

    # The dataloader can be replaced
    data_loader = 1

    # The data collator within the dataloader can be replaced
    data_collator = 2


# Components that are replaceable
# DataCollator is a typing.NewType and does not work well with isinstance
# - so we type data_collator as a Callable
REPLACEABLE_COMPONENTS = {
    AcceleratorPatcherComponent.data_loader.value: DataLoader,
    AcceleratorPatcherComponent.data_collator.value: Callable,
}


# History of all the patching that has been performed
@dataclass
class AcceleratorPatcherHistory:

    # component that is patched
    component: AcceleratorPatcherComponent

    # type of rule (see RULE below)
    kind: str

    # id of the rule that was applied
    rule_id: str


# ---------------------------------- RULE -----------------------------------

RULE_KIND_REPLACEMENT = "replacement"

# List of special kwargs that may affect behavior of specific rules
RULE_SPECIAL_KWARGS = {RULE_KIND_REPLACEMENT: {"skip_prepare"}}


@dataclass
class AcceleratorRuleReplace:

    # id, must be unique
    rule_id: str

    # component that is patched
    component: AcceleratorPatcherComponent

    # replacement:
    replacement: Any = None

    # replacement builder
    replacement_builder: Callable[[Any, Accelerator], Any] = None

    # pre-req check on the object to be replaced
    pre_req: Callable = None

    # additional kwargs that can be used for special behaviors based on component
    # e.g, skip_repare for dataloader will skip running the old prepare
    kwargs: Dict = None

    def __post_init__(self):

        assert (self.replacement is None and self.replacement_builder is not None) or (
            self.replacement is not None and self.replacement_builder is None
        ), "either replacement or replacement_builder should be specified"

        if self.kwargs is None:
            self.kwargs = {}

        assert all(
            k in RULE_SPECIAL_KWARGS[RULE_KIND_REPLACEMENT] for k in self.kwargs
        ), f"Invalid special behavior kwargs in '{self.kwargs.keys()}'"

    def pre_req_check(self, to_be_replaced: Any):
        if self.pre_req is None:
            return

        assert self.pre_req(
            to_be_replaced
        ), f"Rule '{self.rule_id}' failed pre-requisite check for type '{type(to_be_replaced)}'."


# Sigleton AcceleratorPatcher
class AcceleratorPatcher:

    # singleton history of patches
    history: List[AcceleratorPatcherHistory] = []

    # singleton collection of replacements
    replacement_rules: Dict[str, AcceleratorRuleReplace] = {}

    @staticmethod
    def replace(
        rule_id: str,
        component: AcceleratorPatcherComponent,
        replacement: Any = None,
        replacement_builder: Callable = None,
        pre_requisite_check: Callable = None,
        **kwargs,
    ):
        """replace a component. Note that once this method is called, the replacement
        is expected to occur, that is there is no fallback behavior
        - if the pre_requisite_check fails will raise.
        - if there are two replace calls on the same component will raise.

        replacement: the replacement object, if not specified, then replacement builder
            must be specified.
        replacement_builder (callable): the replacement builder object. If not specified,
            then replacement must be specified.
        pre_requisite_check (callable): the component to be replaced is expected to
            pass this check, otherwise raises.
        kwargs (dict): These control special behaviors of the replacement rules, see
            RULE_SPECIAL_KWARGS above.
        """

        # - ensure that rule has not been added before
        assert not any(
            h.rule_id == rule_id for h in AcceleratorPatcher.history
        ), f"Rule '{rule_id}' has already been added"

        assert (
            component.value not in AcceleratorPatcher.replacement_rules
        ), f"replace has already been called once on component '{component.name}'"

        # handle the replacement
        # - if replacement is not None, ensure replacement object is of the correct type
        if replacement is not None:
            comp_cls = REPLACEABLE_COMPONENTS.get(component.value)
            if comp_cls:
                assert isinstance(replacement, comp_cls), (
                    f"Rule '{rule_id}' replacing component '{component}' with wrong ",
                    f"type '{type(replacement)}'",
                )
        elif replacement_builder is not None:
            # NOTE: there is no class check for the replacement builder pattern
            pass

        # - register the replacement rule
        AcceleratorPatcher.replacement_rules[component.value] = AcceleratorRuleReplace(
            rule_id,
            component,
            replacement,
            replacement_builder,
            pre_requisite_check,
            kwargs=kwargs,
        )

        # - record the history. This is done in advance for replacements even
        #   the pre-req check has not been run.
        # - This advanced registration simplifies logic in the patch, and its ok
        #   because we will raise in the pre-req if fails, as the semantics for
        #   replace is that it is expected to occur once called.
        AcceleratorPatcher.history.append(
            AcceleratorPatcherHistory(component, RULE_KIND_REPLACEMENT, rule_id)
        )

    @staticmethod
    def patch(accelerator: Accelerator):

        # some rules will require patching the prepare function
        # - e.g., if replacements are required.
        if any(
            key
            in (
                AcceleratorPatcherComponent.data_collator.value,
                AcceleratorPatcherComponent.data_loader.value,
            )
            for key in AcceleratorPatcher.replacement_rules
        ):
            AcceleratorPatcher._patch_prepare(accelerator)

    # function to patch the accelerator prepare
    @staticmethod
    def _patch_prepare(accelerator: Accelerator):

        # hijack the dataloader in accelerator.prepare to replace the collate_fn
        _old_prepare = accelerator.prepare

        def prepare(self, *args, device_placement=None):
            if len(args) > 1 or not isinstance(args[0], DataLoader):
                return _old_prepare(*args, device_placement=device_placement)

            # if there is dataloader replacment
            dataloader_replacement_rule = AcceleratorPatcher.replacement_rules.get(
                AcceleratorPatcherComponent.data_loader.value
            )
            # the original dataloader
            dataloader = args[0]

            if dataloader_replacement_rule:
                dataloader_replacement_rule.pre_req_check(dataloader)
                if dataloader_replacement_rule.replacement is not None:
                    dataloader = dataloader_replacement_rule.replacement
                else:
                    dataloader = dataloader_replacement_rule.replacement_builder(
                        dataloader, accelerator
                    )

            # if there is dataloader replacment
            collator_replacement_rule = AcceleratorPatcher.replacement_rules.get(
                AcceleratorPatcherComponent.data_collator.value
            )

            if collator_replacement_rule:
                # - first we run the check on the rule (if any)
                # - then we replace
                collator_replacement_rule.pre_req_check(dataloader.collate_fn)

                # Replace the collate_fn in dataloader
                if collator_replacement_rule.replacement is not None:
                    dataloader.collate_fn = collator_replacement_rule.replacement
                else:
                    dataloader.collate_fn = (
                        collator_replacement_rule.replacement_builder(
                            dataloader.collate_fn
                        )
                    )

            # - special behavior for dataloader replacements
            # - need to know if we run the original prepare
            if (
                dataloader_replacement_rule is not None
                and dataloader_replacement_rule.kwargs.get("skip_prepare", False)
            ):
                return dataloader

            return _old_prepare(dataloader)

        accelerator.prepare = MethodType(prepare, accelerator)

    @staticmethod
    def summary():
        result = []
        result.append("***************** Accelerator Patching *************")
        for x in AcceleratorPatcher.history:
            result.append(
                "Rule: {0:25s} Kind: {1:10s} Component: {2:20s}".format(
                    x.rule_id, x.kind, x.component.name
                )
            )

        return "\n".join(result)
