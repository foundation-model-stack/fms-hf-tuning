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
from accelerate.logging import get_logger



logger = get_logger(__name__)

# function to monkey patch activation checkpointing function
def patch_activation_checkpointing_fsdp():
    from fms_acceleration.model_patcher import patch_target_module

    patch_target_module("accelerate.utils.fsdp_utils.fsdp2_apply_ac", fsdp2_apply_ac)
    patch_target_module("torch.distributed.algorithms._checkpoint.checkpoint_wrapper.apply_activation_checkpointing", apply_activation_checkpointing)

# /Users/kmehant/accelerate/src/accelerate/utils/fsdp_utils.py
# fsdp2_apply_ac
# patch torch path or can we do it on import
# apply_activation_checkpointing

def fsdp2_apply_ac(accelerator=None, model=None):
    from torch.utils.checkpoint import create_selective_checkpoint_contexts, CheckpointPolicy
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
    )
    # def policy_fn(ctx, op, *args, **kwargs):
    #     if op in ops_to_save:
    #         return CheckpointPolicy.MUST_SAVE
    #     else:
    #         return CheckpointPolicy.PREFER_RECOMPUTE
    level = model._ac_level
    print("level", level)
    def chpk(l, module):
        if l==0:
            return
        for nm, mod in module.named_children():
            chpk(l-1, mod)
            module.register_module(nm, checkpoint_wrapper(mod, preserve_rng_state=False))

    for layer in model.layers:
        chpk(level-1, layer)
        if level>0:
            layer = checkpoint_wrapper(layer, preserve_rng_state=False)
    return model

def apply_activation_checkpointing(model, checkpoint_wrapper_fn=None, check_fn=None, auto_wrap_policy=None):
    from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts
    from torch._functorch.partitioners import get_default_op_list
    compute_intensive_ops = get_default_op_list().compute_intensive_ops
    def policy_fn(ctx, op, *args, **kwargs):
        if op in compute_intensive_ops:
            return CheckpointPolicy.MUST_SAVE
        else:
            return CheckpointPolicy.MUST_RECOMPUTE
    from functools import partial
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
    )
    level = model._ac_level
    print("level", level)
    def chpk(l, module):
        if l==0:
            return
        for nm, mod in module.named_children():
            chpk(l-1, mod)
            module.register_module(nm, checkpoint_wrapper(mod, preserve_rng_state=False, context_fn=partial(create_selective_checkpoint_contexts, policy_fn)))

    for layer in model.model.layers:
        chpk(level-1, layer)
        if level>0:
            layer = checkpoint_wrapper(layer, preserve_rng_state=False,context_fn=partial(create_selective_checkpoint_contexts, policy_fn))

# def apply_activation_checkpointing(model):
#     from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
#         checkpoint_wrapper,
#     )
#     level = model._ac_level
#     #dfs
#     def chpk(l, module):
#         if l==0:
#             return
#         for nm, mod in module.named_children():
#             chpk(l-1, mod)
#             module.register_module(nm, checkpoint_wrapper(mod, preserve_rng_state=False))
#     for layer in model.model.layers:
#         chpk(level, layer.self_attn)
#         if level>0:
#             layer = checkpoint_wrapper(layer, preserve_rng_state=False)
