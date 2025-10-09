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
from accelerate import Accelerator
import pytest  # pylint: disable=import-error
import torch

# First Party
from fms_acceleration.accelerator_patcher import (
    AcceleratorPatcher,
    AcceleratorPatcherComponent,
    AcceleratorRuleReplace,
)
from fms_acceleration.utils.test_utils import instantiate_accel_patcher


def test_AP_rule_raises_correct_errors():
    # not specifying any replacement objects will throw an error
    with pytest.raises(
        AssertionError,
        match="either replacement or replacement_builder should be specified",
    ):
        AcceleratorRuleReplace(
            rule_id="bad-rule-empty-builders",
            component=AcceleratorPatcherComponent.data_loader,
            replacement=None,
            replacement_builder=None,
        )

    # Ensure that rule registration throws error when attempting
    # to specify an unknown flag for an unsupported behaviour
    # handling of the component by AP
    with pytest.raises(
        AssertionError,
        match=r"Invalid special behavior kwargs in '.*'",
    ):
        AcceleratorRuleReplace(
            rule_id="invalid-special-kwargs",
            component=AcceleratorPatcherComponent.data_loader,
            replacement=torch.utils.data.DataLoader(
                torch.utils.data.Dataset(),
            ),
            kwargs={"unsupported_kwarg": True},
        )


def test_AP_failing_prereq_check_raises_error():
    # 1. register AP rule
    # 2. instantiate accelerator
    # 3. attempt to patch accelerator prepare function w a pre-req check
    # 4. call accelerator prepare
    # 5. ensure that pre-req check raises error when condition not satisfied
    pre_req_error_message = "pre-requisite check failed"

    def pre_req_check(dataloader):
        raise ValueError(pre_req_error_message)

    with pytest.raises(
        ValueError,
        match=pre_req_error_message,
    ):
        with instantiate_accel_patcher():
            dummy_dataloader = torch.utils.data.DataLoader(torch.utils.data.Dataset())

            # register the replacement rule
            AcceleratorPatcher.replace(
                rule_id="pre-req-check-raises-error",
                component=AcceleratorPatcherComponent.data_loader,
                replacement=dummy_dataloader,
                pre_requisite_check=pre_req_check,
            )
            # instantiate an accelerator object
            accelerator = Accelerator()
            # patch the prepare function
            AcceleratorPatcher.patch(accelerator)
            # call accelerator prepare
            accelerator.prepare(dummy_dataloader)


def test_AP_patch_correctly_with_simple_replacement():
    # 1. register rule to replace collate fn
    # 2. patch the accelerator
    # 3. call accelerator.prepare with a dataloader
    # 4. verify that the dataloader's collate fn behaviour has updated
    message = "replacement successful"

    def replaced_collater():
        return message

    with instantiate_accel_patcher():
        dataloader = torch.utils.data.DataLoader(torch.utils.data.Dataset())
        # register the replacement rule for new collate fn
        AcceleratorPatcher.replace(
            rule_id="simple-replacement-successful",
            component=AcceleratorPatcherComponent.data_collator,
            replacement=replaced_collater,
        )
        # instantiate an accelerator object
        accelerator = Accelerator()
        # patch the prepare function
        AcceleratorPatcher.patch(accelerator)
        # call accelerator prepare
        dataloader = accelerator.prepare(dataloader)
        assert dataloader.collate_fn() == "replacement successful"


def test_AP_patch_correctly_with_replacement_builder():
    # 1. Create a builder function for a new dataloader class
    # 2. Register a replacement rule to take in the builder function
    # 3. Instantiate and patch accelerator
    # 4. call accelerator.prepare on a standard dataloader
    # 5. verify that the dataloader has been replaced
    class NewDataLoader(torch.utils.data.DataLoader):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

    def build_new_dataloader(
        dataloader: torch.utils.data.DataLoader, accelerator: Accelerator
    ):
        return NewDataLoader(
            torch.utils.data.Dataset(),
        )

    with instantiate_accel_patcher():
        original_dataloader = torch.utils.data.DataLoader(torch.utils.data.Dataset())
        # register the replacement rule
        AcceleratorPatcher.replace(
            rule_id="replacement-builder-successful",
            component=AcceleratorPatcherComponent.data_loader,
            replacement_builder=build_new_dataloader,
            skip_prepare=True,
        )
        # instantiate an accelerator object
        accelerator = Accelerator()
        # patch the prepare function
        AcceleratorPatcher.patch(accelerator)
        # call accelerator prepare
        replaced_dataloader = accelerator.prepare(original_dataloader)
        assert isinstance(replaced_dataloader, NewDataLoader)
