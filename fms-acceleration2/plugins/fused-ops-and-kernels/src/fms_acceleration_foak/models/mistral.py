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
import warnings

# Third Party
from fms_acceleration.model_patcher import (
    ModelPatcherRule,
    ModelPatcherTrigger,
    combine_functions,
    combine_triggers,
)
from transformers import PretrainedConfig
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralForCausalLM,
    MistralMLP,
    MistralRMSNorm,
)

# Local
from ..fused_ops.liger_ce.fused_linear_cross_entropy_loss import lce_forward
from ..kernels.unsloth.cross_entropy_loss import (
    FastCrossEntropyLoss,
    replace_custom_loss_when_triggered,
)
from ..kernels.unsloth.rms_layernorm import fast_rms_layernorm
from ..kernels.unsloth.rope_embedding import fast_rope_embedding
from ..utils import filter_mp_rules
from .utils import (
    KEY_MLP,
    KEY_O,
    KEY_QKV,
    build_lora_fused_ops,
    get_hidden_activation_fn_key,
    get_transformers_version,
    trigger_fused_ops,
)


def get_mp_rules(base_type: str, config: PretrainedConfig = None):
    """
    Function to access all patch rules in this module.
    If it is a forward_builder rule with `base_type` in
    its forward builder argument, wrap the forward_builder
    function as a partial function with the base_type argument
    """

    rules = [
        # - do regex on RMSNorm class name
        # - check on the tensors required for fast_rms_layernorm
        ModelPatcherRule(
            rule_id="mistral-rms",
            trigger=ModelPatcherTrigger(check=MistralRMSNorm),
            forward=fast_rms_layernorm,
        ),
        ModelPatcherRule(
            rule_id="mistral-qkvo",
            trigger=combine_triggers(
                ModelPatcherTrigger(
                    check=partial(
                        trigger_fused_ops,
                        attn_cls=MistralAttention,
                        submodule_names=["q_proj", "k_proj", "v_proj"],
                    )
                ),
                ModelPatcherTrigger(
                    check=partial(
                        trigger_fused_ops,
                        attn_cls=MistralAttention,
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
            rule_id="mistral-mlp",
            trigger=ModelPatcherTrigger(
                check=partial(
                    trigger_fused_ops,
                    attn_cls=MistralMLP,
                    submodule_names=["up_proj", "down_proj", "gate_proj"],
                )
            ),
            forward_builder=partial(
                build_lora_fused_ops,
                submodule_names=["up_proj", "down_proj", "gate_proj"],
                fused_op=KEY_MLP,
                base_type=base_type,
            ),
        ),
        *[
            (
                ModelPatcherRule(
                    rule_id="mistral-custom-loss",
                    trigger=ModelPatcherTrigger(
                        check=replace_custom_loss_when_triggered(
                            MistralForCausalLM, custom_loss_type="mistral-custom-loss"
                        )
                    ),
                )
                if get_transformers_version() >= "4.46"
                else ModelPatcherRule(
                    rule_id="mistral-cross-ent",
                    import_and_maybe_reload=(
                        "torch.nn.CrossEntropyLoss",
                        FastCrossEntropyLoss,
                        "transformers.models.mistral.modeling_mistral",
                    ),
                )
            )
        ],
        ModelPatcherRule(
            rule_id="mistral-fused-lce",
            trigger=ModelPatcherTrigger(check=MistralForCausalLM),
            forward=lce_forward,
        ),
        ModelPatcherRule(
            rule_id="mistral-rope",
            import_and_maybe_reload=(
                "transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb",
                fast_rope_embedding,
                None,
            ),
        ),
    ]

    # perform model specific filtering
    act = get_hidden_activation_fn_key(config)
    if config and act != "silu":
        warnings.warn(
            f"Mistral rules: activation is {act}, "
            "thus disabling LoRA fused-op for MLP, since only SwiGLU "
            "is supported. This only affects quantized-peft."
        )
        rules = filter_mp_rules(rules, {"mlp"}, drop=True)

    return rules
