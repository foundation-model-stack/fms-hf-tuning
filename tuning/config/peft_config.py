# Copyright The FMS HF Tuning Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from dataclasses import dataclass, field
from typing import List


@dataclass
class LoraConfig:
    r: int = 8
    lora_alpha: int = 32
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"],
        metadata={
            "help": "The names of the modules to apply LORA to. LORA selects modules which either \
            completely match or "
            'end with one of the strings. If the value is ["all-linear"], \
            then LORA selects all linear and Conv1D '
            "modules except for the output layer."
        },
    )
    bias = "none"
    lora_dropout: float = 0.05


@dataclass
class PromptTuningConfig:
    prompt_tuning_init: str = "TEXT"
    num_virtual_tokens: int = 8
    prompt_tuning_init_text: str = "Classify if the tweet is a complaint or not:"
    tokenizer_name_or_path: str = "llama-7b-hf"
