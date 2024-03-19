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

# Third Party
import torch

# Local
import tuning.config.configs as tuning_config

NEW_MODEL_NAME = "Maykeye/TinyLLama-v0"


def test_model_argument_configs_default():
    data_arguments = tuning_config.DataArguments
    model_arguments = tuning_config.ModelArguments
    training_arguments = tuning_config.TrainingArguments
    # test model arguments default
    assert (
        model_arguments.model_name_or_path == tuning_config.DEFAULT_MODEL_NAME_OR_PATH
    )
    assert model_arguments.use_flash_attn == True
    assert isinstance(model_arguments.torch_dtype, torch.dtype)

    # test data arguments default
    assert data_arguments.data_path == None
    assert data_arguments.response_template == None
    assert data_arguments.dataset_text_field == None
    assert data_arguments.validation_data_path == None

    # test training arguments default
    assert training_arguments.cache_dir == None
    assert training_arguments.model_max_length == tuning_config.DEFAULT_CONTEXT_LENGTH
    assert training_arguments.packing == False


def test_model_argument_configs_init():
    # new data arguments
    data_arguments = tuning_config.DataArguments(
        data_path="/foo/bar",
        response_template="\n### Label:",
        dataset_text_field="output",
        validation_data_path="/foo/bar",
    )
    assert data_arguments.data_path == "/foo/bar"
    assert data_arguments.response_template == "\n### Label:"
    assert data_arguments.validation_data_path == "/foo/bar"

    # new model arguments
    model_arguments = tuning_config.ModelArguments(
        model_name_or_path=NEW_MODEL_NAME, use_flash_attn=False, torch_dtype=torch.int32
    )
    assert model_arguments.model_name_or_path == NEW_MODEL_NAME
    assert model_arguments.use_flash_attn == False
    assert model_arguments.torch_dtype == torch.int32

    # new training arguments
    training_arguments = tuning_config.TrainingArguments(
        cache_dir="/tmp/cache",
        model_max_length=1024,
        packing=True,
        output_dir="/tmp/output",
    )
    assert training_arguments.cache_dir == "/tmp/cache"
    assert training_arguments.model_max_length == 1024
    assert training_arguments.packing == True
    assert training_arguments.output_dir == "/tmp/output"
