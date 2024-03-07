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
    return (
        model_args,
        data_args,
        training_args,
        lora_config
        if train_kwargs.get("peft_method") == "lora"
        else prompt_tuning_config,
    )
