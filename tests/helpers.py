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
import transformers

# Local
from tuning.config import configs, peft_config


def causal_lm_train_kwargs(train_kwargs):
    """Parse the kwargs for a valid train call to a Causal LM."""
    parser = transformers.HfArgumentParser(
        dataclass_types=(
            configs.ModelArguments,
            configs.DataArguments,
            configs.TrainingArguments,
            peft_config.LoraConfig,
            peft_config.PromptTuningConfig,
        )
    )
    (
        model_args,
        data_args,
        training_args,
        lora_config,
        prompt_tuning_config,
    ) = parser.parse_dict(train_kwargs, allow_extra_keys=True)

    # TODO: target_modules doesn't get set probably due to the way dataclass handles
    # mutable defaults, needs investigation on better way to handle this
    setattr(
        lora_config,
        "target_modules",
        lora_config.__dataclass_fields__.get("target_modules").default_factory(),
    )

    return (
        model_args,
        data_args,
        training_args,
        lora_config
        if train_kwargs.get("peft_method") == "lora"
        else prompt_tuning_config,
    )
