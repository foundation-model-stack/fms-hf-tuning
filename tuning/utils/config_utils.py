from peft import LoraConfig, PromptTuningConfig
from tuning.config.peft import lora_config, prompt_tuning_config
from dataclasses import asdict


def get_peft_config(train_config):
    assert train_config.peft_method in [None, "lora", "pt"], \
        f"peft config {train_config.peft_method} not defined in peft.py"

    if train_config.peft_method == "lora":
        peft_config = LoraConfig(**asdict(lora_config()))
    elif train_config.peft_method == "pt":
        peft_config = PromptTuningConfig(**asdict(prompt_tuning_config()))
    else:
        peft_config = None  # full parameter tuning

    return peft_config
