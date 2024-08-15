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
import base64
import json
import os
import pickle

# Third Party
from peft import LoraConfig, PromptTuningConfig

# Local
from tuning.config import peft_config


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


def get_hf_peft_config(task_type, tuning_config, tokenizer_name_or_path):
    """Return HF PEFT config for tuning based on type of tuning config passed
    Args:
        task_type: str
        tuning_config: peft_config.LoraConfig | peft_config.PromptTuningConfig | None
        tokenizer_name_or_path: str
    Return: HF PEFT config or None
    """
    if isinstance(tuning_config, peft_config.LoraConfig):
        lora_config = asdict(tuning_config)
        if lora_config["target_modules"] == ["all-linear"]:
            lora_config["target_modules"] = "all-linear"
        hf_peft_config = LoraConfig(task_type=task_type, **lora_config)
    elif isinstance(tuning_config, peft_config.PromptTuningConfig):
        hf_peft_config = PromptTuningConfig(
            task_type=task_type,
            tokenizer_name_or_path=tokenizer_name_or_path,
            **asdict(tuning_config),
        )
    else:
        hf_peft_config = None  # full parameter tuning

    return hf_peft_config


def get_json_config():
    """Parses JSON configuration if provided via environment variables
    SFT_TRAINER_CONFIG_JSON_ENV_VAR or SFT_TRAINER_CONFIG_JSON_PATH.

    SFT_TRAINER_CONFIG_JSON_ENV_VAR is the base64 encoded JSON.
    SFT_TRAINER_CONFIG_JSON_PATH is the path to the JSON config file.

    Returns: dict or {}
    """
    json_env_var = os.getenv("SFT_TRAINER_CONFIG_JSON_ENV_VAR")
    json_path = os.getenv("SFT_TRAINER_CONFIG_JSON_PATH")

    # accepts either path to JSON file or encoded string config
    # env var takes precedent
    job_config_dict = {}
    if json_env_var:
        job_config_dict = txt_to_obj(json_env_var)
    elif json_path:
        with open(json_path, "r", encoding="utf-8") as f:
            job_config_dict = json.load(f)

    return job_config_dict


def txt_to_obj(txt):
    """Given encoded byte string, converts to base64 decoded dict.

    Args:
        txt: str
    Returns: dict[str, Any]
    """
    base64_bytes = txt.encode("ascii")
    message_bytes = base64.b64decode(base64_bytes)
    try:
        # If the bytes represent JSON string
        return json.loads(message_bytes)
    except UnicodeDecodeError:
        # Otherwise the bytes are a pickled python dictionary
        return pickle.loads(message_bytes)
