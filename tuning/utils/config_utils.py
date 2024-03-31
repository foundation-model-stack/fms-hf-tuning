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
from dataclasses import asdict
import logging

# Third Party
from peft import LoraConfig, PromptTuningConfig
import transformers

# Local
from tuning.config import configs, peft_config


def update_config(config, **kwargs):
    if isinstance(config, (tuple, list)):
        for c in config:
            update_config(c, **kwargs)
    else:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif "." in k:
                # allow --some_config.some_param=True
                config_name, param_name = k.split(".")
                if type(config).__name__ == config_name:
                    if hasattr(config, param_name):
                        setattr(config, param_name, v)
                    else:
                        # In case of specialized config we can warm user
                        print(f"Warning: {config_name} does not accept parameter: {k}")


def create_tuning_config(peft_method, **kwargs):
    """Create peft_config Tuning config
    Args:
        peft_method: str
           lora, pt or None
        kawrgs: parameters to initialize library configs with
     Return:
        peft_config.LoraConfig | peft_config.PromptTuningConfig | None
    """
    assert peft_method in [
        None,
        "lora",
        "pt",
        "None",
    ], f"peft config {peft_method} not defined in peft.py"
    if peft_method == "lora":
        tune_config = peft_config.LoraConfig()
        update_config(tune_config, **kwargs)
    elif peft_method == "pt":
        tune_config = peft_config.PromptTuningConfig()
        update_config(tune_config, **kwargs)
    else:
        tune_config = None  # full parameter tuning
    return tune_config


def get_hf_peft_config(task_type, tuning_config):
    """Return HF PEFT config for tuning based on type of tuning config passed
    Args:
        task_type: str
        tuning_config: peft_config.LoraConfig | peft_config.PromptTuningConfig | None
    Return: HF PEFT config or None
    """
    if isinstance(tuning_config, peft_config.LoraConfig):
        lora_config = asdict(tuning_config)
        if lora_config["target_modules"] == ["all-linear"]:
            lora_config["target_modules"] = "all-linear"
        hf_peft_config = LoraConfig(task_type=task_type, **lora_config)
    elif isinstance(tuning_config, peft_config.PromptTuningConfig):
        hf_peft_config = PromptTuningConfig(
            task_type=task_type, **asdict(tuning_config)
        )
    else:
        hf_peft_config = None  # full parameter tuning

    return hf_peft_config


def post_process_job_config(job_config_dict):
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
