# Standard
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
from typing import Dict

# Third Party
from fms_acceleration.model_patcher import ModelPatcher, patch_model
from fms_acceleration.utils.test_utils import instantiate_model_patcher
from peft import LoraConfig
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.utils.import_utils import _is_package_available
import pytest  # pylint: disable=import-error
import torch

BNB = "bitsandbytes"
GPTQ = "auto_gptq"

LORA_QUANTIZED_CLASSES = {}
if _is_package_available("bitsandbytes"):
    # pylint: disable=ungrouped-imports
    # Third Party
    from peft.tuners.lora.bnb import Linear4bit as LoraBNBLinear4bit

    LORA_QUANTIZED_CLASSES[BNB] = LoraBNBLinear4bit

if _is_package_available("auto_gptq"):
    # pylint: disable=ungrouped-imports
    # Third Party
    from peft.tuners.lora.gptq import QuantLinear as LoraGPTQLinear4bit

    LORA_QUANTIZED_CLASSES[GPTQ] = LoraGPTQLinear4bit

TEST_MODELS = {
    BNB: (
        "TinyLlama/TinyLlama-1.1B-Chat-v0.3",
        LlamaAttention,
        ["q_proj", "k_proj", "v_proj", "o_proj"],
    ),
    GPTQ: (
        "TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ",
        LlamaAttention,
        ["q_proj", "k_proj", "v_proj", "o_proj"],
    ),
}


ADAPTER_NAME = "default"

LOSS_TOL = 1e-3
ALLCLOSE_RTOL = 1e-3
ALLCLOSE_ATOL = 1e-4

FLOAT16 = "float16"
DTYPES = [FLOAT16]

DROPOUTS = [0, 0.1, 0.5]

# r, lora_alpha
LORA_PARAMS = [(8, 1.0)]

# bs, seqlen, hiddim
SIZES = [(1, 128, 2048), (2, 256, 2048)]


# set a fixed dropout to match outputs between runs
class DummyDropout(torch.nn.Module):

    dropout_mask: torch.tensor = None

    def __init__(self):
        super().__init__()

    def forward(self, X):
        # return X * self.dropout_mask.repeat(0, sel)
        if len(X.shape) == 2:
            sl = self.dropout_mask.shape[0]
            bs_sl = X.shape[0]
            # this is needed since fast-rope does this:
            # dY.reshape(batch*seq_len, n_heads*head_dim)
            return X * self.dropout_mask.repeat(bs_sl // sl, 1)
        return X * self.dropout_mask


@pytest.fixture()
def attention_inputs(seed: int = 42, device: torch.device = "cuda"):
    torch.manual_seed(seed)

    # see
    inputs = {}
    for dtype in DTYPES:
        for bs, seq_len, dim_size in SIZES:
            inputs[(bs, seq_len, dim_size, dtype)] = (
                torch.rand(
                    (bs, seq_len, dim_size),
                    dtype=getattr(torch, dtype),
                    device=device,
                ),
                torch.arange(seq_len).unsqueeze(0).to(device),  # only can handle
            )
    yield inputs


@pytest.fixture()
def model_inputs(seed: int = 42, device: torch.device = "cuda"):
    torch.manual_seed(seed)

    inputs = {}
    for dtype in DTYPES:
        for bs, seq_len, dim_size in SIZES:
            inputs[(bs, seq_len, dim_size, dtype)] = (
                torch.randint(
                    0,
                    10000,  # most models should have more than 10K
                    (bs, seq_len),
                    dtype=torch.int,
                    device=device,
                ),
                torch.randint(
                    0,
                    10000,  # most models should have more than 10K
                    (bs, seq_len),
                    dtype=torch.long,
                    device=device,
                ),
                None,  # dont pass in position ids for now
            )
    yield inputs


@pytest.fixture()
def dropout_masks(seed: int = 42, device: torch.device = "cuda"):
    torch.manual_seed(seed)

    masks = {}
    for d in DROPOUTS:
        binomial = torch.distributions.binomial.Binomial(probs=1 - d)
        for _, sl, hid in SIZES:
            if (sl, hid) not in masks:
                masks[(sl, hid, d)] = binomial.sample((sl, hid)).to(device)

    yield masks


@pytest.fixture()
def attention_layers(device: torch.device = "cuda"):
    # pylint: disable=import-error,import-outside-toplevel
    # Third Party
    from bitsandbytes.nn.modules import Linear4bit

    # from auto_gptq.nn_modules.qlinear.qlinear_tritonv2 import QuantLinear

    QUANTIZED_BASE_LAYER_CLASSES = {BNB: Linear4bit}

    # this is only done for BNB
    layers = {}
    base_type = BNB
    for dtype in DTYPES:
        for r, lora_alpha in LORA_PARAMS:
            model_name, attn_cls, target_modules = TEST_MODELS[base_type]
            quant_cls = QUANTIZED_BASE_LAYER_CLASSES[base_type]
            peft_cls = LORA_QUANTIZED_CLASSES[base_type]

            if base_type == BNB:
                base_type_kwargs = {
                    "compute_dtype": getattr(torch, dtype),
                    "quant_type": "nf4",  # NOTE: fp4 is not supported by atm
                    "quant_storage": getattr(torch, dtype),
                }
            elif base_type == GPTQ:
                base_type_kwargs = {"bits": 4, "group_size": -1}

            # use the llama model
            config = AutoConfig.from_pretrained(model_name)
            attn_module = attn_cls(config, layer_idx=0)
            for tm in target_modules:
                mod = getattr(attn_module, tm)
                quant_base_layer = quant_cls(
                    input_features=mod.in_features,
                    output_features=mod.out_features,
                    bias=mod.bias is not None,
                    **base_type_kwargs,
                )

                if base_type == BNB:
                    # need to cast it so that the quant_type will be correct
                    # because it takes the quantype from the weights dtype
                    quant_base_layer.to(getattr(torch, dtype))

                # bring to device
                quant_base_layer = quant_base_layer.to(device)

                lora_linear_layer = peft_cls(
                    quant_base_layer,
                    ADAPTER_NAME,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=0.0,  # will override the dropout anyway
                )

                # this means all target modules get the same dropout
                lora_linear_layer.lora_dropout = torch.nn.ModuleDict(
                    [[ADAPTER_NAME, DummyDropout()]]
                )
                setattr(attn_module, tm, lora_linear_layer)

            layers[(base_type, r, lora_alpha, dtype)] = attn_module.to(device)

    yield layers


@pytest.fixture()
def loaded_models(device: torch.device = "cuda"):

    # pylint: disable=import-outside-toplevel
    # Third Party
    from fms_acceleration_peft.framework_plugin_autogptq import (
        AutoGPTQAccelerationPlugin,
    )
    from fms_acceleration_peft.framework_plugin_bnb import BNBAccelerationPlugin

    plugins = {
        BNB: BNBAccelerationPlugin(
            {
                "peft": {
                    "quantization": {
                        "bitsandbytes": {"quant_type": "nf4", "no_peft_model": False}
                    }
                }
            }
        ),
        GPTQ: AutoGPTQAccelerationPlugin(
            {
                "peft": {
                    "quantization": {
                        "auto_gptq": {"kernel": "triton_v2", "from_quantized": True}
                    }
                }
            }
        ),
    }

    @dataclass
    class TrainArgs:
        gradient_checkpointing: bool = False
        gradient_checkpointing_kwargs: Dict = field(default_factory=dict)
        fp16: bool = False
        bf16: bool = False

    all_models = {}
    for dtype in DTYPES:

        args = TrainArgs(fp16=dtype == FLOAT16)

        for base_type in [BNB, GPTQ]:

            for r, lora_alpha in LORA_PARAMS:
                model_name, _, target_modules = TEST_MODELS[base_type]
                peft_config = LoraConfig(
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=0.0,  # anyway we are going to override it
                    target_modules=target_modules,
                )

                _plugin = plugins[base_type]
                model = _plugin.model_loader(
                    model_name, torch_dtype=getattr(torch, dtype)
                )
                model, _ = _plugin.augmentation(model, args, (peft_config,))
                all_models[(base_type, r, lora_alpha, dtype)] = model

                lora_mods = []
                for mod in model.modules():
                    if hasattr(mod, "lora_dropout"):
                        lora_mods.append(mod)

                for mod in lora_mods:
                    mod.lora_dropout = torch.nn.ModuleDict(
                        [[ADAPTER_NAME, DummyDropout()]]
                    )

    return all_models


def run_model(
    model,
    dtype,
    X,
    device: torch.device = "cuda",
    **kwargs,
):
    with torch.autocast(dtype=getattr(torch, dtype), device_type=device):
        outputs = model(X, **kwargs)

    if hasattr(outputs, "loss"):
        out = outputs.loss
    else:
        out = outputs[0]
    loss = out.norm()
    loss.backward()

    return loss


def get_modules_with_class(model, cls):
    modules = []
    name = cls.__name__
    for mod in model.modules():
        # check for the class name in the MRO
        mro = mod.__class__.mro()
        if any(name == x.__name__ for x in mro):
            modules.append(mod)

    if len(modules) == 0:
        raise ValueError(f"cannot find modules with class '{cls}'")
    return modules


def get_attention_lora_grads(model, target_modules):
    # comparing the grads on the adapters
    adapter_grads = []
    for tm in target_modules:
        for n in ["lora_A", "lora_B"]:
            mod = model.get_submodule(f"{tm}.{n}.{ADAPTER_NAME}")
            adapter_grads.append(mod.weight.grad)
    if len(adapter_grads) == 0:
        raise ValueError("cannot find adapter grads")
    return adapter_grads


# helper function to register the fused ops
def register_fused_ops_rules(base_type: str):

    # First Party
    # add more models if needed later
    from fms_acceleration_foak.models import (  # pylint: disable=import-outside-toplevel
        llama,
    )

    for r in [*llama.get_mp_rules(base_type)]:
        if any(r.rule_id.endswith(x) for x in ["qkvo", "mlp"]):
            ModelPatcher.register(r)


# -------------------------- TESTS ----------------------------------


# small
@pytest.mark.skipif(
    not _is_package_available("bitsandbytes"),
    reason="Only runs if bitsandbytes is installed",
)
def test_adapter_gradients_match_with_attention_layer(
    attention_inputs,  # pylint: disable=redefined-outer-name
    attention_layers,  # pylint: disable=redefined-outer-name
    dropout_masks,  # pylint: disable=redefined-outer-name
):
    """
    Construct and test equivalence on a single constructed attention
    module.
    - For GPTQ it seems to be troublesome to initialize an insolated
      layer, thus this test is only done for BNB.
    """
    # pylint: disable=too-many-nested-blocks
    for base_type in [BNB]:
        model_name, _, target_modules = TEST_MODELS[base_type]
        config = AutoConfig.from_pretrained(model_name)

        # find compatible sizes
        sizes = [(bs, sl, hd) for bs, sl, hd in SIZES if hd == config.hidden_size]

        for (bs, sl, hd), dtype in product(sizes, DTYPES):
            X, position_ids = attention_inputs[(bs, sl, hd, dtype)]
            _kwargs = {"position_ids": position_ids}

            for r, lora_alpha in LORA_PARAMS:
                for d in DROPOUTS:

                    # attn layer + mask
                    attn = attention_layers[(base_type, r, lora_alpha, dtype)]
                    DummyDropout.dropout_mask = dropout_masks[(sl, hd, d)]

                    # because we want to check the input gradients, we need to have
                    # the lora_B be initialized to non-zero
                    for name, param in attn.named_parameters():
                        if "lora_B" in name:
                            torch.nn.init.normal_(param)

                    X_without = X.clone().detach().requires_grad_()
                    X_with = X.clone().detach().requires_grad_()

                    # instantiate the sigleton model patcher
                    with instantiate_model_patcher():

                        # register the fused op rules
                        register_fused_ops_rules(base_type)

                        # prepare models
                        without_foak = deepcopy(attn)
                        with_foak = patch_model(deepcopy(attn))

                        # run the models and get the loss and gradients
                        loss_unpatched = run_model(
                            without_foak, dtype, X_without, **_kwargs
                        )
                        loss_patched = run_model(with_foak, dtype, X_with, **_kwargs)

                        # check the model has been patched
                        assert (
                            len(ModelPatcher.history) > 0
                        ), "Fused ops did not correctly patch"

                        # compute outputs
                        assert (
                            loss_unpatched - loss_patched
                        ).abs() < LOSS_TOL, "Loss after foak patch do not match"

                        # check input gradients
                        torch.allclose(
                            X_without.grad,
                            X_with.grad,
                            atol=ALLCLOSE_ATOL,
                            rtol=ALLCLOSE_RTOL,
                        )

                        grads_unpatched = get_attention_lora_grads(
                            without_foak, target_modules
                        )
                        grads_patched = get_attention_lora_grads(
                            with_foak, target_modules
                        )

                        for x, y in zip(grads_unpatched, grads_patched):
                            assert torch.allclose(
                                x, y, atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL
                            ), "Gradients don't match after foak patch"


@pytest.mark.skipif(
    not _is_package_available("bitsandbytes"),
    reason="Only runs if both bitsandbytes",
)
def test_adapter_gradients_match_with_model(
    model_inputs, loaded_models, dropout_masks  # pylint: disable=redefined-outer-name
):
    """
    Using full models loaded by plugins, test equivalence on random inputs.
    """

    for base_type in [BNB, GPTQ]:
        model_name, attn_cls, target_modules = TEST_MODELS[base_type]
        config = AutoConfig.from_pretrained(model_name)

        # find compatible sizes
        sizes = [(bs, sl, hd) for bs, sl, hd in SIZES if hd == config.hidden_size]

        for (bs, sl, hd), dtype in product(sizes, DTYPES):
            input_ids, labels, position_ids = model_inputs[(bs, sl, hd, dtype)]
            _kwargs = {"position_ids": position_ids, "labels": labels}

            for r, lora_alpha in LORA_PARAMS:
                for d in DROPOUTS:

                    # attn layer + mask
                    model = loaded_models[(base_type, r, lora_alpha, dtype)]
                    DummyDropout.dropout_mask = dropout_masks[(sl, hd, d)]

                    # instantiate the sigleton model patcher
                    with instantiate_model_patcher():

                        # register the fused op rules
                        register_fused_ops_rules(base_type)

                        # prepare models
                        without_foak = deepcopy(model)
                        with_foak = patch_model(deepcopy(model))

                        # check the model has been patched
                        assert (
                            len(ModelPatcher.history) > 0
                        ), "Fused ops did not correctly patch"

                        # run the models and get the loss and gradients
                        loss_unpatched = run_model(
                            without_foak, dtype, input_ids, **_kwargs
                        )
                        loss_patched = run_model(with_foak, dtype, input_ids, **_kwargs)

                        # compute outputs
                        assert (
                            loss_unpatched - loss_patched
                        ).abs() < LOSS_TOL, "Loss after foak patch do not match"

                        for _without, _with in zip(
                            get_modules_with_class(without_foak, attn_cls),
                            get_modules_with_class(with_foak, attn_cls),
                        ):
                            grads_unpatched = get_attention_lora_grads(
                                _without, target_modules
                            )
                            grads_patched = get_attention_lora_grads(
                                _with, target_modules
                            )

                        for x, y in zip(grads_unpatched, grads_patched):
                            assert torch.allclose(
                                x, y, atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL
                            ), "Gradients don't match after foak patch"
