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
from dataclasses import dataclass
import os

# Third Party
from transformers import (
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import torch

# Local
from .utils import ensure_nested_dataclasses_initialized, parsable_dataclass

is_recover_safetensors_from_dcp_available = True
try:
    # Third Party
    from fms_acceleration_moe.utils import recover_safetensors_from_dcp
except ImportError:
    is_recover_safetensors_from_dcp_available = False


@parsable_dataclass
@dataclass
class FastMoe:

    ep_degree: int = 1


@dataclass
class FastMoeConfig:

    fast_moe: FastMoe = None

    def __post_init__(self):
        # ensure nested dataclasses initialized
        ensure_nested_dataclasses_initialized(self)


def get_callbacks(**kwargs):
    pretrained_model_name_or_path = kwargs.pop("pretrained_model_name_or_path")
    trainer = kwargs.pop("trainer")
    callbacks = []
    if is_recover_safetensors_from_dcp_available:

        class ConvertAndSaveHFCheckpointAtEverySave(TrainerCallback):
            def __init__(self, pretrained_model_name_or_path: str, trainer: Trainer):
                self.pretrained_model_name_or_path = pretrained_model_name_or_path
                self.trainer = trainer

            def on_save(
                self,
                args: TrainingArguments,
                state: TrainerState,
                control: TrainerControl,
                **kwargs,
            ):
                """
                Save all HF files and convert dcp checkpoint to safetensors at every save operation.
                """

                def checkpoint():
                    checkpoint_dir = os.path.join(
                        args.output_dir,
                        f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}",
                    )
                    hf_converted_output_dir = os.path.join(
                        checkpoint_dir, "hf_converted_checkpoint"
                    )
                    if os.path.exists(hf_converted_output_dir):
                        # if the folder already exists
                        # we return, since this is possible to happen
                        # saving the checkpointing at the end of the training
                        return
                    os.mkdir(hf_converted_output_dir)
                    try:
                        recover_safetensors_from_dcp(
                            checkpoint_dir,
                            self.pretrained_model_name_or_path,
                            hf_converted_output_dir,
                        )
                        # save tokenizer
                        if self.trainer.processing_class:
                            self.trainer.processing_class.save_pretrained(
                                hf_converted_output_dir
                            )
                        # save training args
                        torch.save(
                            args,
                            os.path.join(hf_converted_output_dir, TRAINING_ARGS_NAME),
                        )
                        # save model config files
                        self.trainer.model.config.save_pretrained(
                            hf_converted_output_dir
                        )

                    except Exception as e:
                        raise ValueError(
                            f"Failed to convert the checkpoint {checkpoint_dir}\
                                to a HF compatible checkpoint"
                        ) from e

                if state.is_world_process_zero:
                    checkpoint()

        callbacks.append(
            ConvertAndSaveHFCheckpointAtEverySave(
                pretrained_model_name_or_path, trainer
            )
        )
    return callbacks
