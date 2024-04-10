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
"""Script wraps SFT Trainer to run for Train Conductor.
Read SFTTrainer configuration via environment variable `SFT_TRAINER_CONFIG_JSON_PATH`
for the path to the JSON config file with parameters or `SFT_TRAINER_CONFIG_JSON_ENV_VAR`
for the encoded config string to parse.
"""

# Standard
import os
import tempfile
import shutil

# First Party
import logging

# Local
from tuning import sft_trainer
from tuning.utils.merge_model_utils import create_merged_model
from build.utils import process_launch_training_args, get_job_config


def get_highest_checkpoint(dir_path):
    checkpoint_dir = ""
    for curr_dir in os.listdir(dir_path):
        if curr_dir.startswith("checkpoint"):
            if checkpoint_dir:
                curr_dir_num = int(checkpoint_dir.rsplit("-", maxsplit=1)[-1])
                new_dir_num = int(curr_dir.split("-")[-1])
                if new_dir_num > curr_dir_num:
                    checkpoint_dir = curr_dir
            else:
                checkpoint_dir = curr_dir

    return checkpoint_dir


def main():
    LOGLEVEL = os.environ.get("LOG_LEVEL", "WARNING").upper()
    logging.basicConfig(level=LOGLEVEL)

    logging.info("Initializing launch training script")

    job_config = get_job_config()

    logging.debug("Input params parsed: %s", job_config)

    (
        model_args,
        data_args,
        training_args,
        tune_config,
        merge_model,
    ) = process_launch_training_args(job_config)

    original_output_dir = training_args.output_dir
    with tempfile.TemporaryDirectory() as tempdir:
        training_args.output_dir = tempdir
        sft_trainer.train(model_args, data_args, training_args, tune_config)

        if merge_model:
            export_path = os.getenv(
                "LORA_MERGE_MODELS_EXPORT_PATH", original_output_dir
            )

            # get the highest checkpoint dir (last checkpoint)
            lora_checkpoint_dir = get_highest_checkpoint(training_args.output_dir)
            full_checkpoint_dir = os.path.join(
                training_args.output_dir, lora_checkpoint_dir
            )

            logging.info(
                "Merging lora tuned checkpoint %s with base model into output path: %s",
                lora_checkpoint_dir,
                export_path,
            )

            create_merged_model(
                checkpoint_models=full_checkpoint_dir,
                export_path=export_path,
                base_model=model_args.model_name_or_path,
                save_tokenizer=True,
            )
        else:
            # copy last checkpoint into mounted output dir
            pt_checkpoint_dir = get_highest_checkpoint(training_args.output_dir)
            logging.info(
                "Copying last checkpoint %s into output dir %s",
                pt_checkpoint_dir,
                original_output_dir,
            )
            shutil.copytree(
                os.path.join(training_args.output_dir, pt_checkpoint_dir),
                original_output_dir,
                dirs_exist_ok=True,
            )

        # copy over any loss logs
        train_logs_filepath = os.path.join(
            training_args.output_dir, sft_trainer.TRAINING_LOGS_FILENAME
        )
        if os.path.exists(train_logs_filepath):
            shutil.copy(train_logs_filepath, original_output_dir)


if __name__ == "__main__":
    main()
