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
from dataclasses import dataclass, replace
from typing import Annotated
from unittest.mock import patch
import copy
import os
import tempfile

# Third Party
import pytest
import torch

# First Party
from tests.test_sft_trainer import DATA_ARGS, MODEL_ARGS, PEFT_LORA_ARGS, TRAIN_ARGS

# Local
from .spying_utils import create_mock_plugin_class_and_spy
from tuning import sft_trainer
from tuning.config.acceleration_configs import (
    AccelerationFrameworkConfig,
    FusedOpsAndKernelsConfig,
    QuantizedLoraConfig,
)
from tuning.config.acceleration_configs.acceleration_framework_config import (
    ConfigAnnotation,
)
from tuning.config.acceleration_configs.attention_and_distributed_packing import (
    AttentionAndDistributedPackingConfig,
    MultiPack,
    PaddingFree,
)
from tuning.config.acceleration_configs.fast_moe import FastMoe, FastMoeConfig
from tuning.config.acceleration_configs.fused_ops_and_kernels import (
    FastKernelsConfig,
    FusedLoraConfig,
)
from tuning.config.acceleration_configs.quantized_lora_config import (
    AutoGPTQLoraConfig,
    BNBQLoraConfig,
)
from tuning.utils.import_utils import is_fms_accelerate_available

# for some reason the CI will raise an import error if we try to import
# these from tests.artifacts.testdata
TWITTER_COMPLAINTS_JSON_FORMAT = os.path.join(
    os.path.dirname(__file__),
    "../artifacts/testdata/json/twitter_complaints_small.json",
)
TWITTER_COMPLAINTS_TOKENIZED = os.path.join(
    os.path.dirname(__file__),
    "../artifacts/testdata/twitter_complaints_tokenized_with_maykeye_tinyllama_v0.json",
)

# pylint: disable=import-error
if is_fms_accelerate_available():

    # Third Party
    from fms_acceleration.utils.test_utils import (
        build_framework_and_maybe_instantiate,
        instantiate_model_patcher,
    )

    if is_fms_accelerate_available(plugins="peft"):
        # Third Party
        from fms_acceleration_peft import (
            AutoGPTQAccelerationPlugin,
            BNBAccelerationPlugin,
        )

    if is_fms_accelerate_available(plugins="foak"):
        # Third Party
        from fms_acceleration_foak import FastKernelsAccelerationPlugin

    if is_fms_accelerate_available(plugins="aadp"):
        # Third Party
        from fms_acceleration_aadp import PaddingFreeAccelerationPlugin

    if is_fms_accelerate_available(plugins="moe"):
        # Third Party
        from fms_acceleration_moe import ScatterMoEAccelerationPlugin


# There are more extensive unit tests in the
# https://github.com/foundation-model-stack/fms-acceleration
# repository.
# - see plugins/framework/tests/test_framework.py


@pytest.mark.skipif(
    not is_fms_accelerate_available(plugins="peft"),
    reason="Only runs if fms-accelerate is installed along with accelerated-peft plugin",
)
def test_acceleration_framework_fail_construction():
    """Ensure that construct of the framework will fail if rules regarding
    the dataclasess are violated.
    """

    # 1. Rule 1: No two standalone dataclasses can exist at the same path
    # - Test that the framework will fail to construct if there are multiple
    #    standalone plugins under the same path that are simultaneously requested.
    invalid_quantized_lora_config = QuantizedLoraConfig(
        auto_gptq=AutoGPTQLoraConfig(), bnb_qlora=BNBQLoraConfig()
    )
    with pytest.raises(
        ValueError,
        match="Configuration path 'peft.quantization' already has one standalone config.",
    ):
        AccelerationFrameworkConfig.from_dataclasses(
            invalid_quantized_lora_config
        ).get_framework()

    def peft_unavailable(plugin=None):
        if plugin == "peft":
            return False
        return True

    quantized_lora_config = QuantizedLoraConfig(auto_gptq=AutoGPTQLoraConfig())

    # 2. Rule 2: Dataclass cannot request a plugin that is not yet installed.
    # - Test that framework will fail to construct if trying to activate a plugin
    #   that is not yet installed
    with pytest.raises(
        ValueError,
        match="An acceleration feature is requested by specifying the '--auto_gptq' argument, "
        "but the this requires acceleration packages to be installed.",
    ):
        with patch(
            "tuning.config.acceleration_configs.acceleration_framework_config."
            "is_fms_accelerate_available",
            peft_unavailable,
        ):
            AccelerationFrameworkConfig.from_dataclasses(
                quantized_lora_config
            ).get_framework()

    # 3. Rule 3: Dataclass that corresponds to experimental plugin will
    #            give user a warning.
    # - Test that if a plugin is experimental the user will be warned

    # - create a dataclas with an experimental annotation that to be
    #   used for mocking
    # - mocked auto_gptq here to be experimental
    @dataclass
    class DataClassWithExperimental:
        auto_gptq: Annotated[
            AutoGPTQLoraConfig,
            ConfigAnnotation(path="peft.quantization", experimental=True),
        ] = None

    with pytest.warns(
        UserWarning,
        match="An experimental acceleration feature is requested by specifying the "
        "'--auto_gptq' argument. Please note this feature may not support certain "
        "edge cases at this juncture. When the feature matures this "
        "message will be turned off.",
    ):
        with patch.dict(
            "tuning.config.acceleration_configs.acceleration_framework_config."
            "AccelerationFrameworkConfig.__dataclass_fields__",
            DataClassWithExperimental.__dataclass_fields__,
        ):

            AccelerationFrameworkConfig.from_dataclasses(
                quantized_lora_config
            ).get_framework()


def test_acceleration_framework_pass_construction_with_no_active_configs():
    """Ensure framework is properly constructed in the null pattern where
    no configs are active
    """

    # for the fallback, if the dataclasses
    AccelerationFrameworkConfig.from_dataclasses(QuantizedLoraConfig)
    assert QuantizedLoraConfig.auto_gptq is None
    assert QuantizedLoraConfig.bnb_qlora is None


@pytest.mark.skip(
    """ NOTE: this scenario will actually never happen, since in the code we always
    provide at least one dataclass (can consider to remove this test).
    """
)
def test_construct_framework_config_raise_if_constructing_with_no_dataclassess():
    """Ensure that framework configuration config will refused to construct
    if no dataclasses are provided.
    """

    with pytest.raises(
        ValueError,
        match="AccelerationFrameworkConfig construction requires at least one dataclass",
    ):
        AccelerationFrameworkConfig.from_dataclasses()


@pytest.mark.skipif(
    not is_fms_accelerate_available(plugins="peft"),
    reason="Only runs if fms-accelerate is installed along with accelerated-peft plugin",
)
def test_construct_framework_with_auto_gptq_peft_successfully():
    "Ensure that framework object is correctly configured."

    # 1. correctly initialize a set of quantized lora config dataclass
    #    with auto-gptq
    quantized_lora_config = QuantizedLoraConfig(auto_gptq=AutoGPTQLoraConfig())

    # - instantiate the acceleration config
    acceleration_config = AccelerationFrameworkConfig.from_dataclasses(
        quantized_lora_config
    )

    # build the framework by
    # - passing acceleration configuration contents (via .to_dict()).
    # - NOTE: we skip the required packages check in the framework since it is
    #         not necessary for this test (e.g., we do not need auto_gptq installed)
    # - check that the plugin is correctly activated
    with build_framework_and_maybe_instantiate(
        [],
        acceleration_config.to_dict(),  # pass in contents
        reset_registrations=False,
        require_packages_check=False,  # not required
    ) as framework:

        # plugin activated!
        assert len(framework.active_plugins) == 1


@pytest.mark.skipif(
    not is_fms_accelerate_available(),
    reason="Only runs if fms-accelerate is installed",
)
def test_framework_raises_if_used_with_missing_package():
    """Ensure that trying the use the framework, without first installing fms_acceleration
    will raise.
    """
    train_args = copy.deepcopy(TRAIN_ARGS)
    train_args.output_dir = None

    quantized_lora_config = QuantizedLoraConfig(auto_gptq=AutoGPTQLoraConfig())

    # patch is_fms_accelerate_available to return False inside sft_trainer
    # to simulate fms_acceleration not installed
    with patch(
        "tuning.config.acceleration_configs.acceleration_framework_config."
        "is_fms_accelerate_available",
        return_value=False,
    ):
        with pytest.raises(
            ValueError, match="No acceleration framework package found."
        ):
            sft_trainer.train(
                MODEL_ARGS,
                DATA_ARGS,
                TRAIN_ARGS,
                PEFT_LORA_ARGS,
                quantized_lora_config=quantized_lora_config,
            )


invalid_kwargs_map = [
    (
        {
            "model_name_or_path": "TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ",
        },
        PEFT_LORA_ARGS,
        AssertionError,
        "need to run in fp16 mixed precision or load model in fp16",
    ),
    (
        {
            "model_name_or_path": "TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ",
            "torch_dtype": torch.float16,
        },
        None,
        AssertionError,
        "need peft_config to install PEFT adapters",
    ),
]


@pytest.mark.skipif(
    not is_fms_accelerate_available(plugins="peft"),
    reason="Only runs if fms-accelerate is installed along with accelerated-peft plugin",
)
@pytest.mark.parametrize(
    "bad_kwargs,peft_config,exception,exception_msg",
    invalid_kwargs_map,
    ids=["triton_v2 requires fp16", "accelerated peft requires peft config"],
)
def test_framework_raises_due_to_invalid_arguments(
    bad_kwargs, peft_config, exception, exception_msg
):
    """Ensure that invalid arguments will be checked by the activated framework
    plugin.
    """
    with tempfile.TemporaryDirectory() as tempdir:
        model_args = copy.deepcopy(MODEL_ARGS)
        model_args = replace(model_args, **bad_kwargs)
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir

        quantized_lora_config = QuantizedLoraConfig(auto_gptq=AutoGPTQLoraConfig())

        # 1. activate the accelerated peft plugin
        # 2. demonstrate that the invalid arguments will be checked
        with pytest.raises(exception, match=exception_msg):
            sft_trainer.train(
                model_args,
                DATA_ARGS,
                train_args,
                peft_config,
                quantized_lora_config=quantized_lora_config,
            )


acceleration_configs_map = [
    (
        QuantizedLoraConfig(bnb_qlora=BNBQLoraConfig()),
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        (
            "peft.quantization.bitsandbytes",
            create_mock_plugin_class_and_spy(
                "PluginMock",
                BNBAccelerationPlugin
                if is_fms_accelerate_available(plugins="peft")
                else object,
            ),
        ),
    ),
    (
        QuantizedLoraConfig(auto_gptq=AutoGPTQLoraConfig()),
        "TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ",
        (
            "peft.quantization.auto_gptq",
            create_mock_plugin_class_and_spy(
                "PluginMock",
                AutoGPTQAccelerationPlugin
                if is_fms_accelerate_available(plugins="peft")
                else object,
            ),
        ),
    ),
]


@pytest.mark.skipif(
    not is_fms_accelerate_available(plugins="peft"),
    reason="Only runs if fms-accelerate is installed along with accelerated-peft plugin",
)
@pytest.mark.parametrize(
    "quantized_lora_config,model_name_or_path,mock_and_spy",
    acceleration_configs_map,
    ids=["bitsandbytes", "auto_gptq"],
)
def test_framework_initialized_properly_peft(
    quantized_lora_config, model_name_or_path, mock_and_spy
):
    """Ensure that specifying a properly configured acceleration dataclass
    properly activates the framework plugin and runs the train sucessfully.
    """
    with tempfile.TemporaryDirectory() as tempdir:
        model_args = copy.deepcopy(MODEL_ARGS)
        model_args.model_name_or_path = model_name_or_path
        model_args.torch_dtype = torch.float16
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir
        train_args.save_strategy = "no"
        train_args.fp16 = True
        peft_args = copy.deepcopy(PEFT_LORA_ARGS)
        peft_args.target_modules = ["q_proj", "k_proj"]

        installation_path, (MockedPlugin, spy) = mock_and_spy

        # 1. mock a plugin class
        # 2. register the mocked plugin
        # 3. call sft_trainer.train
        with build_framework_and_maybe_instantiate(
            [([installation_path], MockedPlugin)],
            instantiate=False,
        ):
            with instantiate_model_patcher():
                sft_trainer.train(
                    model_args,
                    DATA_ARGS,
                    train_args,
                    peft_args,
                    quantized_lora_config=quantized_lora_config,
                )

        # spy inside the train to ensure that the acceleration plugin
        # was called. In the context of the AutoGPTQ plugin
        # 1. Will sucessfully load the model (to load AutoGPTQ 4-bit linear)
        # 2. Will successfully agument the model (to install PEFT)
        # 3. Will succesfully call get_ready_for_train
        assert spy["model_loader_calls"] == 1
        assert spy["augmentation_calls"] == 1
        assert spy["get_ready_for_train_calls"] == 1


@pytest.mark.skipif(
    not is_fms_accelerate_available(plugins=["peft", "foak"]),
    reason=(
        "Only runs if fms-accelerate is installed along with accelerated-peft "
        "and foak plugins"
    ),
)
def test_framework_initialized_properly_foak():
    """Ensure that specifying a properly configured acceleration dataclass
    properly activates the framework plugin and runs the train sucessfully.
    """
    with tempfile.TemporaryDirectory() as tempdir:

        model_args = copy.deepcopy(MODEL_ARGS)
        model_args.model_name_or_path = "TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ"
        model_args.torch_dtype = torch.float16
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir
        train_args.save_strategy = "no"
        train_args.fp16 = True
        peft_args = copy.deepcopy(PEFT_LORA_ARGS)
        peft_args.target_modules = ["q_proj", "k_proj"]

        # setup default quantized lora args dataclass
        # - with auth gptq as the quantized method
        quantized_lora_config = QuantizedLoraConfig(auto_gptq=AutoGPTQLoraConfig())
        fusedops_kernels_config = FusedOpsAndKernelsConfig(
            fused_lora=FusedLoraConfig(base_layer="auto_gptq", fused_lora=True),
            fast_kernels=FastKernelsConfig(
                fast_loss=True, fast_rms_layernorm=True, fast_rope_embeddings=True
            ),
        )

        # create mocked plugin class for spying
        MockedPlugin1, spy = create_mock_plugin_class_and_spy(
            "AutoGPTQMock", AutoGPTQAccelerationPlugin
        )
        MockedPlugin2, spy2 = create_mock_plugin_class_and_spy(
            "FastPeftMock", FastKernelsAccelerationPlugin
        )

        # 1. mock a plugin class
        # 2. register the mocked plugins
        # 3. call sft_trainer.train
        with build_framework_and_maybe_instantiate(
            [
                (["peft.quantization.auto_gptq"], MockedPlugin1),
                (["peft.quantization.fused_ops_and_kernels"], MockedPlugin2),
            ],
            instantiate=False,
        ):
            with instantiate_model_patcher():
                sft_trainer.train(
                    model_args,
                    DATA_ARGS,
                    train_args,
                    peft_args,
                    quantized_lora_config=quantized_lora_config,
                    fusedops_kernels_config=fusedops_kernels_config,
                )

        # spy inside the train to ensure that the AutoGPTQ plugin is called
        assert spy["model_loader_calls"] == 1
        assert spy["augmentation_calls"] == 1
        assert spy["get_ready_for_train_calls"] == 1

        # also test that the FusedOpsPlugin is called
        assert spy2["model_loader_calls"] == 0
        assert spy2["augmentation_calls"] == 1
        assert spy2["get_ready_for_train_calls"] == 1


@pytest.mark.skipif(
    not is_fms_accelerate_available(plugins="moe"),
    reason="Only runs if fms-accelerate is installed along with accelerated-moe plugin",
)
def test_framework_initialized_properly_moe():
    """Ensure that specifying a properly configured acceleration dataclass
    properly activates the framework plugin and runs the train sucessfully.
    """

    with tempfile.TemporaryDirectory() as tempdir:

        model_args = copy.deepcopy(MODEL_ARGS)
        model_args.model_name_or_path = "Isotonic/TinyMixtral-4x248M-MoE"
        model_args.torch_dtype = torch.bfloat16
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir
        train_args.save_strategy = "no"
        train_args.bf16 = True
        data_args = copy.deepcopy(DATA_ARGS)
        data_args.training_data_path = TWITTER_COMPLAINTS_JSON_FORMAT
        data_args.response_template = "\n\n### Label:"
        data_args.dataset_text_field = "output"

        # initialize a config
        moe_config = FastMoeConfig(fast_moe=FastMoe(ep_degree=1))

        # create mocked plugin class for spying
        MockedPlugin1, spy = create_mock_plugin_class_and_spy(
            "FastMoeMock", ScatterMoEAccelerationPlugin
        )

        # 1. mock a plugin class
        # 2. register the mocked plugins
        # 3. call sft_trainer.train
        with build_framework_and_maybe_instantiate(
            [
                (["training.moe.scattermoe"], MockedPlugin1),
            ],
            instantiate=False,
        ):
            with instantiate_model_patcher():
                sft_trainer.train(
                    model_args,
                    data_args,
                    train_args,
                    fast_moe_config=moe_config,
                )

        # spy inside the train to ensure that the ilab plugin is called
        assert spy["model_loader_calls"] == 1
        assert spy["augmentation_calls"] == 0
        assert spy["get_ready_for_train_calls"] == 1


@pytest.mark.skipif(
    not is_fms_accelerate_available(plugins="aadp"),
    reason="Only runs if fms-accelerate is installed along with \
        attention_and_distributed_packing plugin",
)
def test_framework_initialize_and_trains_with_aadp():
    """
    Ensure that a properly configured aadp dataclass is
    correctly activated in train.
    """

    with tempfile.TemporaryDirectory() as tempdir:

        model_args = copy.deepcopy(MODEL_ARGS)
        model_args.model_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v0.3"
        model_args.use_flash_attn = True
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir
        train_args.save_strategy = "no"
        data_args = copy.deepcopy(DATA_ARGS)
        data_args.training_data_path = TWITTER_COMPLAINTS_JSON_FORMAT
        data_args.response_template = "\n\n### Label:"
        data_args.dataset_text_field = "output"

        # initialize a config
        aadp_config = AttentionAndDistributedPackingConfig(
            padding_free=PaddingFree(method="huggingface")
        )

        # create mocked plugin class for spying
        MockedPlugin1, spy = create_mock_plugin_class_and_spy(
            "PaddingFreeMock", PaddingFreeAccelerationPlugin
        )

        # 1. mock a plugin class
        # 2. register the mocked plugins
        # 3. call sft_trainer.train
        with build_framework_and_maybe_instantiate(
            [
                (["training.attention.padding_free"], MockedPlugin1),
            ],
            instantiate=False,
        ):
            with instantiate_model_patcher():
                sft_trainer.train(
                    model_args,
                    data_args,
                    train_args,
                    attention_and_distributed_packing_config=aadp_config,
                )

        # spy inside the train to ensure that the ilab plugin is called
        assert spy["model_loader_calls"] == 0
        assert spy["augmentation_calls"] == 1
        assert spy["get_ready_for_train_calls"] == 1


@pytest.mark.skipif(
    not is_fms_accelerate_available(plugins="aadp"),
    reason="Only runs if fms-accelerate is installed along with \
        attention_and_distributed_packing plugin",
)
def test_error_raised_with_paddingfree_and_flash_attn_disabled():
    """Ensure error raised when padding-free is not used with flash attention"""
    with pytest.raises(
        ValueError,
        match="`--padding_free` argument was called without enabling "
        "flash attention, ensure `use_flash_attn = True` to use padding-free flash attention",
    ):
        attention_and_distributed_packing_config = AttentionAndDistributedPackingConfig(
            padding_free=PaddingFree(method="huggingface")
        )
        model_args = copy.deepcopy(MODEL_ARGS)
        model_args.use_flash_attn = False
        sft_trainer.train(
            model_args,
            DATA_ARGS,
            TRAIN_ARGS,
            attention_and_distributed_packing_config=attention_and_distributed_packing_config,
        )


@pytest.mark.skipif(
    not is_fms_accelerate_available(plugins="aadp"),
    reason="Only runs if fms-accelerate is installed along with \
        attention_and_distributed_packing plugin",
)
def test_error_raised_with_multipack_and_paddingfree_disabled():
    """Ensure error raised when padding-free is not used with multipack"""
    with pytest.raises(
        ValueError,
        match="`--multipack` is currently only supported with `--padding_free`",
    ):
        attention_and_distributed_packing_config = AttentionAndDistributedPackingConfig(
            multipack=MultiPack(num_processes=16),
            padding_free=None,
        )
        model_args = copy.deepcopy(MODEL_ARGS)
        sft_trainer.train(
            model_args,
            DATA_ARGS,
            TRAIN_ARGS,
            attention_and_distributed_packing_config=attention_and_distributed_packing_config,
        )


@pytest.mark.skipif(
    not is_fms_accelerate_available(plugins="aadp"),
    reason="Only runs if fms-accelerate is installed along with \
        attention_and_distributed_packing plugin",
)
def test_error_raised_with_packing_and_paddingfree_enabled():
    """Ensure error raised when padding-free is used with packing"""
    with pytest.raises(
        ValueError,
        match="`--padding_free` argument was called with `packing=True`, "
        "Trainer should not perform packing when using `--padding_free`",
    ):
        attention_and_distributed_packing_config = AttentionAndDistributedPackingConfig(
            padding_free=PaddingFree(method="huggingface")
        )
        model_args = copy.deepcopy(MODEL_ARGS)
        model_args.use_flash_attn = True
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.packing = True
        sft_trainer.train(
            model_args,
            DATA_ARGS,
            train_args,
            attention_and_distributed_packing_config=attention_and_distributed_packing_config,
        )


@pytest.mark.skipif(
    not is_fms_accelerate_available(plugins="foak"),
    reason="Only runs if fms-accelerate is installed along with \
        fused_ops_and_kernels plugin",
)
def test_error_raised_with_fused_lora_enabled_without_quantized_argument():
    """
    Ensure error is thrown when `--fused_lora` is passed without
    `--auto_gptq` or `bitsandbytes`
    """
    with pytest.raises(
        ValueError,
        match="`--fused_lora` must be accompanied by a quantized base layer "
        "`--auto_gptq` or `--bitsandbytes`.",
    ):
        with tempfile.TemporaryDirectory() as tempdir:
            # instantiate the arguments
            model_args = copy.deepcopy(MODEL_ARGS)
            model_args.model_name_or_path = "TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ"
            model_args.torch_dtype = torch.float16
            train_args = copy.deepcopy(TRAIN_ARGS)
            train_args.output_dir = tempdir
            train_args.save_strategy = "no"
            train_args.fp16 = True
            peft_args = copy.deepcopy(PEFT_LORA_ARGS)
            peft_args.target_modules = ["q_proj", "k_proj"]

            # setup FOAK config with fused lora
            fusedops_kernels_config = FusedOpsAndKernelsConfig(
                fused_lora=FusedLoraConfig(base_layer="auto_gptq", fused_lora=True),
            )

            # pass FOAK config but don't specify quantized base layer to sft_trainer
            # expect error in framework instantiation
            with build_framework_and_maybe_instantiate(
                [
                    (["training.fused_ops_and_kernels"], fusedops_kernels_config),
                ],
                instantiate=False,
            ):
                with instantiate_model_patcher():
                    sft_trainer.train(
                        model_args,
                        DATA_ARGS,
                        train_args,
                        peft_args,
                        quantized_lora_config=None,
                        fusedops_kernels_config=fusedops_kernels_config,
                    )


@pytest.mark.skipif(
    not is_fms_accelerate_available(plugins="moe"),
    reason="Only runs if fms-accelerate is installed along with accelerated-moe plugin",
)
def test_error_raised_with_undividable_fastmoe_argument():
    """
    Ensure error is thrown when `--fast_moe` is passed and world_size
    is not divisible by ep_degree
    """
    with pytest.raises(
        AssertionError, match="world size \\(1\\) not divisible by ep_size \\(3\\)"
    ):
        with tempfile.TemporaryDirectory() as tempdir:

            model_args = copy.deepcopy(MODEL_ARGS)
            model_args.model_name_or_path = "Isotonic/TinyMixtral-4x248M-MoE"
            model_args.torch_dtype = torch.bfloat16
            train_args = copy.deepcopy(TRAIN_ARGS)
            train_args.output_dir = tempdir
            train_args.save_strategy = "no"
            train_args.bf16 = True
            data_args = copy.deepcopy(DATA_ARGS)
            data_args.training_data_path = TWITTER_COMPLAINTS_JSON_FORMAT
            data_args.response_template = "\n\n### Label:"
            data_args.dataset_text_field = "output"

            # initialize a config
            moe_config = FastMoeConfig(fast_moe=FastMoe(ep_degree=3))

            # 1. mock a plugin class
            # 2. register the mocked plugins
            # 3. call sft_trainer.train
            with build_framework_and_maybe_instantiate(
                [
                    (["training.moe.scattermoe"], ScatterMoEAccelerationPlugin),
                ],
                instantiate=False,
            ):
                with instantiate_model_patcher():
                    sft_trainer.train(
                        model_args,
                        data_args,
                        train_args,
                        fast_moe_config=moe_config,
                    )


@pytest.mark.skipif(
    not is_fms_accelerate_available(plugins="moe"),
    reason="Only runs if fms-accelerate is installed along with accelerated-moe plugin",
)
def test_error_raised_fast_moe_with_non_moe_model():
    """
    Ensure error is thrown when `--fast_moe` is passed and model is not MoE
    """
    with pytest.raises(
        AttributeError,
        match="'LlamaConfig' object has no attribute 'num_local_experts'",
    ):
        with tempfile.TemporaryDirectory() as tempdir:

            model_args = copy.deepcopy(MODEL_ARGS)
            model_args.model_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v0.3"
            model_args.torch_dtype = torch.bfloat16
            train_args = copy.deepcopy(TRAIN_ARGS)
            train_args.output_dir = tempdir
            train_args.save_strategy = "no"
            train_args.bf16 = True
            data_args = copy.deepcopy(DATA_ARGS)
            data_args.training_data_path = TWITTER_COMPLAINTS_JSON_FORMAT
            data_args.response_template = "\n\n### Label:"
            data_args.dataset_text_field = "output"

            # initialize a config
            moe_config = FastMoeConfig(fast_moe=FastMoe(ep_degree=1))

            # 1. mock a plugin class
            # 2. register the mocked plugins
            # 3. call sft_trainer.train
            with build_framework_and_maybe_instantiate(
                [
                    (["training.moe.scattermoe"], ScatterMoEAccelerationPlugin),
                ],
                instantiate=False,
            ):
                with instantiate_model_patcher():
                    sft_trainer.train(
                        model_args,
                        data_args,
                        train_args,
                        fast_moe_config=moe_config,
                    )


@pytest.mark.skipif(
    not is_fms_accelerate_available(plugins="foak"),
    reason="Only runs if fms-accelerate is installed along with \
        fused_ops_and_kernels plugin",
)
def test_fastkernels_with_full_finetuning_runs_successfully():
    """
    Ensure that a properly configured fastkernels dataclass will train with full FT.
    """
    with tempfile.TemporaryDirectory() as tempdir:

        model_args = copy.deepcopy(MODEL_ARGS)
        model_args.model_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v0.3"
        model_args.torch_dtype = torch.bfloat16
        train_args = copy.deepcopy(TRAIN_ARGS)
        train_args.output_dir = tempdir
        train_args.save_strategy = "no"
        train_args.bf16 = True
        data_args = copy.deepcopy(DATA_ARGS)
        data_args.training_data_path = TWITTER_COMPLAINTS_JSON_FORMAT
        data_args.response_template = "\n\n### Label:"
        data_args.dataset_text_field = "output"

        # initialize a FOAK config
        fusedops_kernels_config = FusedOpsAndKernelsConfig(
            fast_kernels=FastKernelsConfig(
                fast_loss=True, fast_rms_layernorm=True, fast_rope_embeddings=True
            ),
        )

        # create mocked plugin class for spying
        MockedPlugin1, spy = create_mock_plugin_class_and_spy(
            "FastKernelsMock", FastKernelsAccelerationPlugin
        )

        # 1. mock a plugin class
        # 2. register the mocked plugins
        # 3. call sft_trainer.train
        with build_framework_and_maybe_instantiate(
            [
                (["training.fused_ops_and_kernels"], MockedPlugin1),
            ],
            instantiate=False,
        ):
            with instantiate_model_patcher():
                sft_trainer.train(
                    model_args,
                    data_args,
                    train_args,
                    fusedops_kernels_config=fusedops_kernels_config,
                )

        # spy inside train to ensure that the aadp plugin is called
        assert spy["augmentation_calls"] == 1
        assert spy["get_ready_for_train_calls"] == 1
