import fire
import transformers
from tuning.tuned_model import TunedCausalLM
from tuning.config import configs, peft_config

def main(**kwargs):
    parser = transformers.HfArgumentParser(dataclass_types=(configs.ModelArguments, 
                                                            configs.DataArguments,
                                                            configs.TrainingArguments,
                                                            peft_config.LoraConfig,
                                                            peft_config.PromptTuningConfig))
    parser.add_argument('--peft_method', type=str.lower, choices=['pt', 'lora', None, 'none'], default="pt")
    model_args, data_args, training_args, lora_config, prompt_tuning_config, peft_method, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    if peft_method.peft_method =="lora":
        tune_config=lora_config
    elif peft_method.peft_method =="pt":
        tune_config=prompt_tuning_config
    else:
        tune_config=None
    TunedCausalLM.train(model_args, data_args, training_args, tune_config)

if __name__ == "__main__":
    fire.Fire(main)
