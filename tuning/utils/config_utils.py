from peft import LoraConfig, PromptTuningConfig
from tuning.config.peft import lora_config, prompt_tuning_config
from dataclasses import asdict


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


def get_peft_config(train_config, kwargs):
    assert train_config.peft_method in [None, "lora", "pt"], \
        f"peft config {train_config.peft_method} not defined in peft.py"

    if train_config.peft_method == "lora":
        config = lora_config()
        update_config(config, **kwargs)
        peft_config = LoraConfig(**asdict(config))
    elif train_config.peft_method == "pt":
        config = prompt_tuning_config()
        update_config(config, **kwargs)
        peft_config = PromptTuningConfig(**asdict(config))
    else:
        peft_config = None  # full parameter tuning

    return peft_config
