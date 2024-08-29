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

# Third Party
import pytest
import transformers

# Local
from tuning.config.acceleration_configs import (
    FusedOpsAndKernelsConfig,
    QuantizedLoraConfig,
)
from tuning.config.acceleration_configs.attention_and_distributed_packing import (
    AttentionAndDistributedPackingConfig,
    MultiPack,
    PaddingFree,
)
from tuning.config.acceleration_configs.fused_ops_and_kernels import (
    FastKernelsConfig,
    FusedLoraConfig,
)
from tuning.config.acceleration_configs.quantized_lora_config import (
    AutoGPTQLoraConfig,
    BNBQLoraConfig,
)


def test_dataclass_parse_successfully():
    parser = transformers.HfArgumentParser(dataclass_types=QuantizedLoraConfig)

    # if nothing is specified then it will parse into the null class
    (cfg, _) = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    assert cfg.auto_gptq is None
    assert cfg.bnb_qlora is None

    # 1.1 specifying "--auto_gptq" with the first item of AutoGPTQLoraConfig
    #    will parse
    (cfg,) = parser.parse_args_into_dataclasses(
        ["--auto_gptq", "triton_v2"],
    )
    assert isinstance(cfg.auto_gptq, AutoGPTQLoraConfig)
    assert cfg.bnb_qlora is None

    # 1.2 specifying "--auto_gptq" with the two items of AutoGPTQLoraConfig
    #    will parse
    (cfg,) = parser.parse_args_into_dataclasses(
        ["--auto_gptq", "triton_v2", "true"],
    )
    assert isinstance(cfg.auto_gptq, AutoGPTQLoraConfig)
    assert cfg.bnb_qlora is None

    # 2. specifying "--bnb_qlora" with the first item of BNBQLoraConfig
    #    will parse
    (cfg,) = parser.parse_args_into_dataclasses(
        ["--bnb_qlora", "nf4"],
    )
    assert cfg.auto_gptq is None
    assert isinstance(cfg.bnb_qlora, BNBQLoraConfig)

    # 3. Specifing "--padding_free" will parse a PaddingFree class
    parser = transformers.HfArgumentParser(
        dataclass_types=AttentionAndDistributedPackingConfig
    )
    (cfg,) = parser.parse_args_into_dataclasses(
        ["--padding_free", "huggingface"],
    )
    assert isinstance(cfg.padding_free, PaddingFree)

    # 4. Specifing "--multipack" will parse a MultiPack class
    parser = transformers.HfArgumentParser(
        dataclass_types=AttentionAndDistributedPackingConfig
    )
    (cfg,) = parser.parse_args_into_dataclasses(
        ["--multipack", "16"],
    )
    assert isinstance(cfg.multipack, MultiPack)


def test_two_dataclasses_parse_successfully_together():
    """Ensure that the two dataclasses can parse arguments successfully
    together.
    """
    parser = transformers.HfArgumentParser(
        dataclass_types=(QuantizedLoraConfig, FusedOpsAndKernelsConfig)
    )

    # 1. specifying "--auto_gptq" together with "--fused_lora" and
    #    "--fast_kernels" will parse.
    cfg, cfg2 = parser.parse_args_into_dataclasses(
        [
            "--auto_gptq",
            "triton_v2",
            "--fused_lora",
            "auto_gptq",
            "true",
            "--fast_kernels",
            "true",
            "true",
            "true",
        ],
    )
    assert isinstance(cfg.auto_gptq, AutoGPTQLoraConfig)
    assert cfg.bnb_qlora is None
    assert isinstance(cfg2.fused_lora, FusedLoraConfig)
    assert isinstance(cfg2.fast_kernels, FastKernelsConfig)


def test_dataclass_will_fail_to_parse_with_no_args():
    """Ensure that the dataclass arg parser will refuse to parse if
    only the key is specified without any following arguments.
    """
    parser = transformers.HfArgumentParser(dataclass_types=QuantizedLoraConfig)

    # 1. passing only the key without any body will fail
    #   - at least the first argument of the dataclass will be expected.
    with pytest.raises(
        SystemExit,  # argparse will exit
    ):
        (_,) = parser.parse_args_into_dataclasses(
            ["--auto_gptq"],
        )


def test_dataclass_will_fail_to_accept_illegal_args():
    """Ensure that some basic rules that are put in the dataclasses will
    fail at initialization of the class.
    """

    # 1. auto_gptq does not support from_quantized at the moment.
    with pytest.raises(
        ValueError, match="only 'from_quantized' == True currently supported."
    ):
        AutoGPTQLoraConfig(from_quantized=False)

    # 1.1 auto_gptq only supports triton_v2 at the moment
    with pytest.raises(
        ValueError, match="only 'triton_v2' kernel currently supported."
    ):
        AutoGPTQLoraConfig(kernel="fake-kernel")

    # 2 bnb only supports two quant types
    with pytest.raises(
        ValueError, match="quant_type can only be either 'nf4' or 'fp4."
    ):
        BNBQLoraConfig(quant_type="fake-quant-type")

    # 3 padding-free plugin only supports huggingface models
    with pytest.raises(
        ValueError, match="only 'huggingface' method currently supported."
    ):
        PaddingFree(method="invalid-method")
