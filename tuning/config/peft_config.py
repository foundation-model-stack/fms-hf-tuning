from dataclasses import dataclass, field
from typing import List, Union, Optional

@dataclass
class LoraConfig:
    r: int = 8
    lora_alpha: int = 32
    target_modules: Optional[Union[List[str], str]] = field(default_factory=lambda: ["q_proj", "v_proj"], metadata={
        "help": "The names of the modules to apply LORA to. When using a string, LORA uses regex matching. When using "
            "a list of strings, LORA selects modules which either completely match or end with one of the strings. "
            "If the value is the string 'all-linear', then LORA selects all linear and Conv1D modules except for "
            "the output layer. If the value is none, then LORA chooses modules depending on the model architecture. "
            "In that case, in that case LORA raises an exception prompting the user to manually select the modules."
    })
    bias = "none"
    lora_dropout: float = 0.05


@dataclass
class PromptTuningConfig:
    prompt_tuning_init: str = "TEXT"
    num_virtual_tokens: int = 8
    prompt_tuning_init_text: str = "Classify if the tweet is a complaint or not:"
    tokenizer_name_or_path: str = "llama-7b-hf"