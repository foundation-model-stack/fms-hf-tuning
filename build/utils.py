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
import logging

# Third Party
import transformers

# Local
from tuning.config import configs, peft_config


def process_launch_training_args(job_config_dict):
    """Return parsed config for tuning to pass to SFT Trainer
    Args:
        job_config_dict: dict
    Return:
        model_args: configs.ModelArguments
        data_args: configs.DataArguments
        training_args: configs.TrainingArguments
        tune_config: peft_config.LoraConfig | peft_config.PromptTuningConfig
        merge_model: bool
    """
    parser = transformers.HfArgumentParser(
        dataclass_types=(
            configs.ModelArguments,
            configs.DataArguments,
            configs.TrainingArguments,
            peft_config.LoraConfig,
            peft_config.PromptTuningConfig,
        )
    )
    peft_method_parsed = "pt"

    (
        model_args,
        data_args,
        training_args,
        lora_config,
        prompt_tuning_config,
    ) = parser.parse_dict(job_config_dict, allow_extra_keys=True)

    peft_method_parsed = job_config_dict.get("peft_method")

    tune_config = None
    merge_model = False
    if peft_method_parsed == "lora":
        tune_config = lora_config
        merge_model = True
    elif peft_method_parsed == "pt":
        tune_config = prompt_tuning_config

    logging.debug(
        "Parameters used to launch training: \
    model_args %s, data_args %s, training_args %s, tune_config %s",
        model_args,
        data_args,
        training_args,
        tune_config,
    )

    return model_args, data_args, training_args, tune_config, merge_model
