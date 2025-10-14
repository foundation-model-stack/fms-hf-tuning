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

# Third Party
from fms_acceleration.model_patcher import ModelPatcherRule

# Local
from ..kernels.unsloth.cross_entropy_loss import FastCrossEntropyLoss


def get_mp_rules(base_type: str):
    """
    Function to access all patch rules in this module.
    If it is a forward_builder rule with `base_type` in
    its forward builder argument, wrap the forward_builder
    function as a partial function with the base_type argument
    """
    return [
        # TODO: have a generic version of this rule
        # - get the module_name and reload on that
        ModelPatcherRule(
            rule_id="gpt-bigcode-cross-ent",
            import_and_maybe_reload=(
                "torch.nn.CrossEntropyLoss",
                FastCrossEntropyLoss,
                "transformers.models.gpt_bigcode.modeling_gpt_bigcode",
            ),
        ),
    ]
