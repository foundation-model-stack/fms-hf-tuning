from dataclasses import dataclass, field
from typing import List

@dataclass
class LoraConfig:
    r: int = 8
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias = "none"
    task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.05


@dataclass
class PromptTuningConfig:
    task_type: str = "CAUSAL_LM"
    prompt_tuning_init: str = "TEXT"
    num_virtual_tokens: int = 8
    prompt_tuning_init_text: str = "Classify if the tweet is a complaint or not:"
    tokenizer_name_or_path: str = "llama-7b-hf"