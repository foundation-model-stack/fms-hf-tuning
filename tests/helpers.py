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
    tuning_config = None
    if train_kwargs.get("peft_method") == "lora":
        tuning_config = lora_config
    elif train_kwargs.get("peft_method") == "pt":
        tuning_config = prompt_tuning_config
    return (model_args, data_args, training_args, tuning_config)
