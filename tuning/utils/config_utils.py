# Standard
from dataclasses import asdict

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
