from dataclasses import dataclass, field
from typing import List

@dataclass
class lora_config:
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias = "none"
    task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.05

@dataclass
class prompt_tuning_config:
    task_type: str = "CAUSAL_LM"
    prompt_tuning_init: str = "TEXT"
    num_virtual_tokens: int = 8,
    prompt_tuning_init_text = "Classify if the tweet is a complaint or not:",
    tokenizer_name_or_path = "Llama-2-7b-hf"
