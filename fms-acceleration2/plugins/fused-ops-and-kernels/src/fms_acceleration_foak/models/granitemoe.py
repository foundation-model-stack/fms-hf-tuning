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
from functools import partial

# Third Party
from fms_acceleration.model_patcher import (
    ModelPatcherRule,
    ModelPatcherTrigger,
    combine_functions,
    combine_triggers,
)

# Local
from ..kernels.unsloth.cross_entropy_loss import FastCrossEntropyLoss
from ..kernels.unsloth.rms_layernorm import fast_rms_layernorm
from ..kernels.unsloth.rope_embedding import fast_rope_embedding
from .utils import KEY_O, KEY_QKV, build_lora_fused_ops, trigger_fused_ops


def get_mp_rules(base_type: str):
    """
    Function to access all patch rules in this module.
    If it is a forward_builder rule with `base_type` in
    its forward builder argument, wrap the forward_builder
    function as a partial function with the base_type argument
    """
    try:
        # Third Party
        from transformers.models.granitemoe.modeling_granitemoe import (  # pylint: disable=import-outside-toplevel
            GraniteMoeAttention,
            GraniteMoeRMSNorm,
        )
    except ImportError:
        return []

    return [
        # TODO: have a generic version of this rule
        # - do regex on RMSNorm class name
        # - check on the tensors required for fast_rms_layernorm
        ModelPatcherRule(
            rule_id="granitemoe-rms",
            trigger=ModelPatcherTrigger(check=GraniteMoeRMSNorm),
            forward=fast_rms_layernorm,
        ),
        # TODO: have a generic version of this rule
        # - do regex on Attention class name
        # - have a set of qkv / o module names and check on that
        ModelPatcherRule(
            rule_id="granitemoe-qkvo",
            trigger=combine_triggers(
                ModelPatcherTrigger(
                    check=partial(
                        trigger_fused_ops,
                        attn_cls=GraniteMoeAttention,
                        submodule_names=["q_proj", "k_proj", "v_proj"],
                    )
                ),
                ModelPatcherTrigger(
                    check=partial(
                        trigger_fused_ops,
                        attn_cls=GraniteMoeAttention,
                        submodule_names=["o_proj"],
                    )
                ),
                logic="OR",
            ),
            forward_builder=combine_functions(
                partial(
                    build_lora_fused_ops,
                    submodule_names=["q_proj", "k_proj", "v_proj"],
                    fused_op=KEY_QKV,
                    base_type=base_type,
                ),
                partial(
                    build_lora_fused_ops,
                    submodule_names=["o_proj"],
                    fused_op=KEY_O,
                    base_type=base_type,
                ),
                logic="APPEND",
            ),
        ),
        ModelPatcherRule(
            rule_id="granitemoe-cross-ent",
            import_and_maybe_reload=(
                "torch.nn.CrossEntropyLoss",
                FastCrossEntropyLoss,
                "transformers.models.granitemoe.modeling_granitemoe",
            ),
        ),
        # TODO: have a generic version of this rule
        # - get the module name
        # - check if "apply_rotary_pos_emb" exists
        # - patch
        ModelPatcherRule(
            rule_id="granitemoe-rope",
            import_and_maybe_reload=(
                "transformers.models.granitemoe.modeling_granitemoe.apply_rotary_pos_emb",
                fast_rope_embedding,
                None,
            ),
        ),
    ]
