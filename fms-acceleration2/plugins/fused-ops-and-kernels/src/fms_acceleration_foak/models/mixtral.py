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
from transformers.models.mixtral.modeling_mixtral import (
    MixtralAttention,
    MixtralForCausalLM,
    MixtralRMSNorm,
)

# Local
from ..kernels.unsloth.cross_entropy_loss import (
    FastCrossEntropyLoss,
    replace_custom_loss_when_triggered,
)
from ..kernels.unsloth.rms_layernorm import fast_rms_layernorm
from ..kernels.unsloth.rope_embedding import fast_rope_embedding
from .utils import (
    KEY_O,
    KEY_QKV,
    build_lora_fused_ops,
    get_transformers_version,
    trigger_fused_ops,
)


def get_mp_rules(base_type):
    """
    Function to access all patch rules in this module.
    If it is a forward_builder rule with `base_type` in
    its forward builder argument, wrap the forward_builder
    function as a partial function with the base_type argument
    """

    # - do regex on RMSNorm class name
    # - check on the tensors required for fast_rms_layernorm
    return [
        ModelPatcherRule(
            rule_id="mixtral-rms",
            trigger=ModelPatcherTrigger(check=MixtralRMSNorm),
            forward=fast_rms_layernorm,
        ),
        ModelPatcherRule(
            rule_id="mixtral-qkvo",
            trigger=combine_triggers(
                ModelPatcherTrigger(
                    check=partial(
                        trigger_fused_ops,
                        attn_cls=MixtralAttention,
                        submodule_names=["q_proj", "k_proj", "v_proj"],
                    )
                ),
                ModelPatcherTrigger(
                    check=partial(
                        trigger_fused_ops,
                        attn_cls=MixtralAttention,
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
        *[
            (
                ModelPatcherRule(
                    rule_id="mixtral-custom-loss",
                    trigger=ModelPatcherTrigger(
                        check=replace_custom_loss_when_triggered(
                            MixtralForCausalLM, custom_loss_type="mixtral-custom-loss"
                        )
                    ),
                )
                if get_transformers_version() >= "4.46"
                else ModelPatcherRule(
                    rule_id="mixtral-cross-ent",
                    import_and_maybe_reload=(
                        "torch.nn.CrossEntropyLoss",
                        FastCrossEntropyLoss,
                        "transformers.models.mixtral.modeling_mixtral",
                    ),
                )
            )
        ],
        ModelPatcherRule(
            rule_id="mixtral-rope",
            import_and_maybe_reload=(
                "transformers.models.mixtral.modeling_mixtral.apply_rotary_pos_emb",
                fast_rope_embedding,
                None,
            ),
        ),
    ]
