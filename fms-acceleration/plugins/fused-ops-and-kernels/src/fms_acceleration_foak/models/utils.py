# Standard
from functools import partial
from typing import Callable, List, Type
import os

# Third Party
from fms_acceleration.model_patcher import ModelPatcherTrigger
from transformers import PretrainedConfig
from transformers.utils.import_utils import _is_package_available
import torch

# Local
# NOTE: the default activation is swiglu in both cases
from ..fused_ops.unsloth_lora.bnb.fast_lora import (
    apply_lora_mlp_swiglu as fused_op_mlp_bnb,
)
from ..fused_ops.unsloth_lora.bnb.fast_lora import apply_lora_o_v2 as fused_op_o_bnb
from ..fused_ops.unsloth_lora.bnb.fast_lora import apply_lora_qkv as fused_op_qkv_bnb
from ..fused_ops.unsloth_lora.gptq.fast_lora import apply_lora_mlp as fused_op_mlp_gptq
from ..fused_ops.unsloth_lora.gptq.fast_lora import apply_lora_o_v2 as fused_op_o_gptq
from ..fused_ops.unsloth_lora.gptq.fast_lora import apply_lora_qkv as fused_op_qkv_gptq

KEY_QKV = "qkv"
KEY_O = "o"
KEY_MLP = "mlp"

# - need to update this for models
# - activation keys are non-standard
KEY_HIDDEN_ACTIVATIONS = ["hidden_act", "activation_function"]

FUSED_OPS = {
    "auto_gptq": {
        KEY_QKV: fused_op_qkv_gptq,
        KEY_O: fused_op_o_gptq,
        KEY_MLP: fused_op_mlp_gptq,
    },
    "bitsandbytes": {
        KEY_QKV: fused_op_qkv_bnb,
        KEY_O: fused_op_o_bnb,
        KEY_MLP: fused_op_mlp_bnb,
    },
}


# simple utility function to guess if its lora layer
def _is_loralayer(module: torch.nn.Module, names: List[str] = None):
    if names is None:
        names = ["lora_A", "lora_B", "base_layer"]
    return all(hasattr(module, x) for x in names)


# builds a triple of forward functions, that each can be attached
# on a series of QKV's, where if the first one is called, will call the
# fused op
# NOTE: this is not thread-safe (issue warning?)
# NOTE: the unsloth fused_operation "apply_lora_qkv" assumes that the
#       modules are called q_proj, k_proj, and v_proj, respectively.
# the fused operation can be changed, depending on what the base layer is
# i.e. gptq or bnb
def _build_fused_forwards(
    attn: torch.nn.Module,
    fused_operation: Callable = fused_op_qkv_gptq,
    submodule_names: List[str] = None,
):
    # fused opts expected to produce singular or multiple results
    # module names must be passed in order of what the fused

    outs = {}

    # the fused operation will be called on first one that passes in the
    # input X.
    # - populates the triple Q, K, V
    # - subsequent calls will be a no-op until ALL Q, K, V get reset to None
    def _fused_op(X):

        # if all of the outs are not yet populated
        if all(x not in outs for x in submodule_names):
            fused_outs = fused_operation(attn, X)
            try:
                fused_outs = list(fused_outs)  # not sure if this is correct
            except TypeError:
                # if fused_outs is not iterable
                fused_outs = [fused_outs]
            for n, x in zip(submodule_names, fused_outs):
                outs[n] = x

    # each of these functions
    # - calls the fused op
    # -

    def _forward(self, X, name: str):
        _fused_op(X)
        assert (
            name in outs
        ), "Fused_op needs to be first reset with sequential calls to each of them"
        V = outs[name]
        del outs[name]
        return V

    return zip(submodule_names, [partial(_forward, name=n) for n in submodule_names])


def build_lora_fused_ops(
    attn: torch.nn.Module,
    base_type: str = "auto_gptq",
    submodule_names: List[str] = None,
    fused_op: str = KEY_QKV,
):

    assert (
        len(submodule_names) > 0
    ), "When building lora fused ops requires more than one submodule."

    if submodule_names is None:
        submodule_names = ["q_proj", "k_proj", "v_proj"]

    # get the fused op
    fused_operation = FUSED_OPS[base_type][fused_op]

    # handle casting issues
    if base_type == "auto_gptq":

        # this is required due to this FSDP fix
        # https://github.com/foundation-model-stack/fms-acceleration/pull/15
        try:
            world_size = torch.distributed.get_world_size()
        except ValueError:
            world_size = 1  # pg not init

        if (
            world_size > 1
            and os.environ.get("ACCELERATE_USE_FSDP", "false").lower() == "true"
        ):

            # guarded import
            # pylint: disable=import-outside-toplevel,import-error
            # Third Party
            from fms_acceleration_peft.autogptq_utils import (
                PATCH_FOR_FSDP_TRITON_V2,
                patch_forward_to_view_attributes_before_call,
            )

            # patch each of the fused ops to view the attributes
            # back into torch.int32
            # - if there are multiple submodules, then we assume that
            #   'fused_operation' will be called on module that has
            #   submodules specified in 'submodule_names'.
            # - otherwise if there is only a single 'submodule_name', then
            #   assume that 'fused_operation' called on the submodule specified
            #   by 'submodule_name' itself
            if len(submodule_names) > 1:
                patched_submodule_names = [n + ".base_layer" for n in submodule_names]
            else:
                # otherwise assume calling on the 'submodule_name' itself
                # so its just the base layer.
                patched_submodule_names = "base_layer"

            fused_operation = patch_forward_to_view_attributes_before_call(
                fused_operation,
                PATCH_FOR_FSDP_TRITON_V2,
                torch.int32,
                submodule_names=patched_submodule_names,
                is_method_forward=False,
            )

    if fused_op == KEY_QKV:
        return [
            (ModelPatcherTrigger(check=_is_loralayer, module_name=name), forward)
            for name, forward in _build_fused_forwards(
                attn,
                fused_operation=fused_operation,
                submodule_names=submodule_names,
            )
        ]
    if fused_op == KEY_O:
        # otherwise its just a single op
        submodule_names = submodule_names[0]
        return [
            (
                ModelPatcherTrigger(check=_is_loralayer, module_name=submodule_names),
                fused_operation,
            )
        ]
    if fused_op == KEY_MLP:
        # otherwise just return the fused_op that should be attached at the
        # top MLP level
        return fused_operation

    raise NotImplementedError(f"Unknown fused op '{fused_op}'")


# trigger if either of the conditions are met
# 1. qkv all have LoRA adapters for a fused op
# 2. o has a lora adapter for the fused op
def trigger_fused_ops(
    module: torch.nn.Module,
    attn_cls: Type,
    submodule_names: List[str],
):

    # trigger if the module meets the attn class and the submodules
    # are all loralayers
    _mods = [getattr(module, x) for x in submodule_names]
    return isinstance(module, attn_cls) and all(_is_loralayer(x) for x in _mods)


# helper function to get the hidden activation function str
def get_hidden_activation_fn_key(config: PretrainedConfig):
    for key in KEY_HIDDEN_ACTIVATIONS:
        value = getattr(config, key, None)
        if value:
            return value

    raise ValueError(
        "Unable to determine activation function key for "
        f"architecture {config.architectures}."
    )


def get_transformers_version():
    _, _transformers_version = _is_package_available(
        "transformers", return_version=True
    )
    return _transformers_version
