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

# SPDX-License-Identifier: Apache-2.0
# https://spdx.dev/learn/handling-license-info/

# Standard
from typing import List

# Third Party
from peft import LoraConfig
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from transformers.utils.import_utils import _is_package_available
import pytest  # pylint: disable=import-error
import torch

GPTQ = "gptq"
# r, lora_alpha
FLOAT16 = "float16"
LORA_r = 8
LORA_alpha = 1.0
BS = 1
SEQLEN = 128

LOSS_TOLERANCE = 1e-3
ALLCLOSE_RTOL = 1e-3
ALLCLOSE_ATOL = 1e-4

VANILLA_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v0.3"
QUANTIZED_MODEL_NAME = "TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


# Model loading function for quantized models
def load_autogptq_plugin_model(
    model_name: str,
    target_modules: List,
    torch_dtype: str,
    use_external_lib: bool = False,
):
    # First Party
    from fms_acceleration_peft.framework_plugin_autogptq import (
        AutoGPTQAccelerationPlugin,
    )

    _plugin = AutoGPTQAccelerationPlugin(
        {
            "peft": {
                "quantization": {
                    "auto_gptq": {
                        "kernel": "triton_v2",
                        "from_quantized": True,
                        "use_external_lib": use_external_lib,
                    }
                }
            }
        },
    )

    class TrainArgs:
        gradient_checkpointing = False
        gradient_checkpointing_kwargs = {}

    args = TrainArgs()
    peft_config = LoraConfig(
        r=LORA_r,
        lora_alpha=LORA_alpha,
        lora_dropout=0.0,  # anyway we are going to override it
        target_modules=target_modules,
    )

    model = _plugin.model_loader(model_name, torch_dtype=getattr(torch, torch_dtype))
    model, _ = _plugin.augmentation(model, args, (peft_config,))
    model.eval()
    return model


# quantization function to manage the loading and quantizing of pretrained model
# using external or local autogptq
def quantize_model(
    model_name,
    config,
    calibration_dataset,
    quant_config_kwargs,
    device,
    torch_dtype,
    use_external_lib=False,
):
    if use_external_lib:
        # Third Party
        from auto_gptq import AutoGPTQForCausalLM as GPTQModel
        from auto_gptq import BaseQuantizeConfig as QuantizeConfig

        quantize_kwargs = {"use_triton": True}
    else:
        # First Party
        from fms_acceleration_peft.gptqmodel import GPTQModel, QuantizeConfig

        quantize_kwargs = {}

    quantize_config = QuantizeConfig(**quant_config_kwargs)
    # load un-quantized model, by default, the model will always be loaded into CPU memory
    model = GPTQModel.from_pretrained(
        model_name,
        quantize_config=quantize_config,
        config=config,
        torch_dtype=getattr(torch, torch_dtype),
    ).to(device)
    # quantize model, the examples should be list of dict whose keys can only be "input_ids"
    model.quantize(calibration_dataset, **quantize_kwargs)
    model.eval()
    return model


def get_wikitext2(tokenizer, num_samples=128, seqlen=128):
    # Standard
    import random

    # Third Party
    from datasets import load_dataset
    import numpy as np
    import torch

    wikidata = load_dataset("wikitext", "wikitext-2-v1", split="test")
    wikilist = [" \n" if s == "" else s for s in wikidata["text"]]

    text = "".join(wikilist)
    trainenc = tokenizer(text, return_tensors="pt")

    random.seed(0)
    np.random.seed(0)
    torch.random.manual_seed(0)

    traindataset = []

    for _ in range(num_samples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return traindataset


@pytest.fixture()
def input_ids(seed: int = 42, device: torch.device = "cuda"):
    torch.manual_seed(seed)
    yield torch.randint(0, 10000, (BS, SEQLEN), device=device)


@pytest.mark.skipif(
    not _is_package_available("auto_gptq"),
    reason="Only runs if auto_gptq is installed",
)
def test_pre_quantized_model_outputs_match(
    input_ids,
    seed: int = 42,
):
    """
    Test for output equivalence when loading quantized models between
    extracted gptq library against original autogptq library
    """
    torch.manual_seed(seed)
    original_model = load_autogptq_plugin_model(
        QUANTIZED_MODEL_NAME, TARGET_MODULES, FLOAT16, use_external_lib=True
    )
    refactored_model = load_autogptq_plugin_model(
        QUANTIZED_MODEL_NAME, TARGET_MODULES, FLOAT16
    )
    with torch.autocast(device_type="cuda", dtype=torch.float32):
        with torch.no_grad():
            original_logits = original_model(input_ids.to(original_model.device)).logits
            refactored_logits = refactored_model(
                input_ids.to(refactored_model.device)
            ).logits

    assert torch.allclose(
        original_logits, refactored_logits, atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL
    ), "Pre-quantized model logits don't match between extracted and external autogptq library"


@pytest.mark.skipif(
    not _is_package_available("auto_gptq"),
    reason="Only runs if auto_gptq is installed",
)
def test_quantizing_pretrained_model_outputs_match(
    input_ids,
    seed: int = 42,
):
    """
    Test for regression of quantizing pretrained models
    with refactored gptq library against original autogptq library
    by calculating KL loss on the output logits of both variants
    """
    torch.manual_seed(seed)
    # Initialize common arguments
    device = input_ids.device
    tokenizer = AutoTokenizer.from_pretrained(VANILLA_MODEL_NAME, use_fast=True)
    config = AutoConfig.from_pretrained(VANILLA_MODEL_NAME)
    config.num_hidden_layers = 2
    # calibration_dataset = [
    #     tokenizer(
    #         "The world is a wonderful place full of beauty and love."
    #     )
    # ]
    calibration_dataset = get_wikitext2(tokenizer, num_samples=128, seqlen=128)
    quant_config_kwargs = {
        "bits": 4,
        "group_size": 64,
        "desc_act": True,
        "damp_percent": 0.1,
        "static_groups": False,
        "sym": True,
        "true_sequential": True,
    }

    # quantize models for external autogptq lib and extracted gptq lib
    original_model = quantize_model(
        VANILLA_MODEL_NAME,
        config,
        calibration_dataset,
        quant_config_kwargs,
        device,
        FLOAT16,
        use_external_lib=True,
    )
    refactored_model = quantize_model(
        VANILLA_MODEL_NAME,
        config,
        calibration_dataset,
        quant_config_kwargs,
        device,
        FLOAT16,
        use_external_lib=False,
    )

    # compare generated tokens between
    # unquantized, original library and refactored gptqmodel library
    unquantized_model = AutoModelForCausalLM.from_pretrained(
        VANILLA_MODEL_NAME, config=config
    ).to(device)
    gen_config = GenerationConfig.from_pretrained(VANILLA_MODEL_NAME)
    gen_config.max_new_tokens = 5
    _inputs = torch.tensor(
        [tokenizer("auto-gptq is an easy to use")["input_ids"]], device="cuda"
    )
    output1 = tokenizer.decode(
        original_model.generate(inputs=_inputs, generation_config=gen_config).view(-1),
        skip_special_tokens=True,
    )
    output2 = tokenizer.decode(
        refactored_model.generate(inputs=_inputs, generation_config=gen_config).view(
            -1
        ),
        skip_special_tokens=True,
    )
    output3 = tokenizer.decode(
        unquantized_model.generate(inputs=_inputs, generation_config=gen_config).view(
            -1
        ),
        skip_special_tokens=True,
    )
    assert (
        output1 == output2 == output3
    ), f"generated tokens ({output1}, {output2}, {output3}) \
        don't match between both libraries after quantization"

    # compare prob. distributions between original library and refactored gptqmodel library
    with torch.no_grad():
        original_logits = original_model(input_ids).logits
        refactored_logits = refactored_model(input_ids).logits

    # Measure the distribution error with KD Loss
    # flatten as a single batch bs*seqlen
    # since batchmean sums the loss and averages on dim=0
    loss_fn = torch.nn.KLDivLoss(reduction="sum")
    # input should be a distribution in the log space
    input = torch.nn.functional.log_softmax(refactored_logits, dim=-1)
    input = input.view(BS * SEQLEN, -1)
    # target must be prob distribution
    target = torch.nn.functional.softmax(original_logits, dim=-1)
    target = target.view(BS * SEQLEN, -1)
    error = loss_fn(input, target)
    assert error.lt(
        LOSS_TOLERANCE
    ), "Model logits don't match between both libraries \
        after quantization"
