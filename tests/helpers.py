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
    model_args, data_args, training_args, lora_config, prompt_tuning_config = parser.parse_dict(train_kwargs, allow_extra_keys=True)

    # TODO: target_modules doesn't get set probably due to the way dataclass handles
    # mutable defaults, needs investigation on better way to handle this
    setattr(lora_config, "target_modules", lora_config.__dataclass_fields__.get("target_modules").default_factory())
    
    return model_args, data_args, training_args, lora_config if train_kwargs.get("peft_method")=="lora" else prompt_tuning_config
