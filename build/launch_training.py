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
import base64
import os
import pickle
import json
import tempfile
import shutil
import glob

# First Party
import logging

# Local
from tuning import sft_trainer
from tuning.utils.merge_model_utils import create_merged_model
from tuning.utils.config_utils import post_process_job_config


def txt_to_obj(txt):
    base64_bytes = txt.encode("ascii")
    message_bytes = base64.b64decode(base64_bytes)
    try:
        # If the bytes represent JSON string
        return json.loads(message_bytes)
    except UnicodeDecodeError:
        # Otherwise the bytes are a pickled python dictionary
        return pickle.loads(message_bytes)


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

    logging.info("Attempting to launch training script")

    json_path = os.getenv("SFT_TRAINER_CONFIG_JSON_PATH")
    json_env_var = os.getenv("SFT_TRAINER_CONFIG_JSON_ENV_VAR")

    # accepts either path to JSON file or encoded string config
    if json_path:
        with open(json_path, "r", encoding="utf-8") as f:
            job_config_dict = json.load(f)
    elif json_env_var:
        job_config_dict = txt_to_obj(json_env_var)
    else:
        raise ValueError(
            "Must set environment variable 'SFT_TRAINER_CONFIG_JSON_PATH' \
        or 'SFT_TRAINER_CONFIG_JSON_ENV_VAR'."
        )

    logging.debug("Input params parsed: %s", job_config_dict)

    (
        model_args,
        data_args,
        training_args,
        tune_config,
        merge_model,
    ) = post_process_job_config(job_config_dict)

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
        for file in glob.glob(f"{training_args.output_dir}/*loss.jsonl"):
            shutil.copy(file, original_output_dir)


if __name__ == "__main__":
    main()
