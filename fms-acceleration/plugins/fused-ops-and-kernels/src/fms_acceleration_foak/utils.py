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
from typing import List, Set

# Third Party
from accelerate.utils import set_module_tensor_to_device
from fms_acceleration.model_patcher import ModelPatcherRule
from transformers.modeling_utils import is_fsdp_enabled
import torch
import torch.distributed as dist


# consider moving this somewhere else later
def lora_adapters_switch_ddp_from_fsdp(modules, fsdp_plugin):
    """
    This function installs hooks on the target adapter parameters and
    reduces the accumulated gradients across devices
    """

    # NOTE: assuming lora has no bias
    fsdp_plugin.ignored_modules = []
    for mod in modules:
        fsdp_plugin.ignored_modules.append(mod.lora_A)
        fsdp_plugin.ignored_modules.append(mod.lora_B)

    def _all_reduce_hook(grad):
        if grad is not None:
            grad = grad.contiguous()
            dist.all_reduce(grad, op=dist.ReduceOp.AVG, group=None)
        return grad

    for mod in modules:
        A = mod.lora_A.default
        B = mod.lora_B.default

        # because we will ignore these from FSDP, we need to manually
        # move them to gpu if they are already not on them
        # - if the adapters are on meta, we assume that this is for FSDP
        #   low_cpu_mem_mode purposes, and that the values will be synced over
        # - So just initialize them to empty.
        if not A.weight.is_cuda:
            value = A.weight

            if is_fsdp_enabled() and value.device == torch.device("meta"):
                # if low_cpu_mem_mode
                value = torch.empty(*value.size(), dtype=value.dtype)

            set_module_tensor_to_device(A, "weight", "cuda", value)

            if is_fsdp_enabled():
                dist.broadcast(A.weight, src=0)

        if not B.weight.is_cuda:
            value = B.weight

            if is_fsdp_enabled() and value.device == torch.device("meta"):
                value = torch.empty(*value.size(), dtype=value.dtype)

            set_module_tensor_to_device(B, "weight", "cuda", value)

            if is_fsdp_enabled():
                dist.broadcast(B.weight, src=0)

        # install hooks on the adapters
        # - this has to be done after all weight replacement happens
        A.weight.register_hook(_all_reduce_hook)
        B.weight.register_hook(_all_reduce_hook)


# helper function to filter rules
def filter_mp_rules(
    rules: List[ModelPatcherRule],
    filter_endswith: Set[str],
    drop: bool = False,
):
    if drop:
        # this means if any of the filter terms appear, we drop
        return [
            r for r in rules if not any(r.rule_id.endswith(x) for x in filter_endswith)
        ]

    # this means if any if the filter terms appear, we keep
    return [r for r in rules if any(r.rule_id.endswith(x) for x in filter_endswith)]
