from peft import LoraConfig, PromptTuningConfig
from dataclasses import asdict

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

def create_tuning_config(train_config, **kwargs):
    """Create peft_config Tuning config
       Args:
           train_config: tuning.config.configs.TrainingArguments
           kawrgs: parameters to initialize library configs with
        Return:
           peft_config.LoraConfig | peft_config.PromptTuningConfig
    """
    assert train_config.peft_method in [None, "lora", "pt"], \
        f"peft config {train_config.peft_method} not defined in peft.py"
    if train_config.peft_method == "lora":
        tune_config = peft_config.LoraConfig()
        update_config(tune_config, **kwargs)
    if train_config.peft_method == "pt":
        tune_config = peft_config.PromptTuningConfig()
        update_config(tune_config, **kwargs)
    return tune_config


def get_peft_config(train_config, tuning_config):
    """Accept the train config and parameters and return HF PEFT config for tuning
       Args:
           train_config: tuning.config.configs.TrainingArguments
           tuning_config: peft_config.LoraConfig | peft_config.PromptTuningConfig
       Return: HF PEFT config
    """
    assert train_config.peft_method in [None, "lora", "pt"], \
        f"peft config {train_config.peft_method} not defined in peft.py"

    if train_config.peft_method == "lora":
        peft_config = LoraConfig(**asdict(tuning_config))
    elif train_config.peft_method == "pt":
        peft_config = PromptTuningConfig(**asdict(tuning_config))
    else:
        peft_config = None  # full parameter tuning

    return peft_config
