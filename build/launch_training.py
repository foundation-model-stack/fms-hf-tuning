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
from tuning import sft_trainer
from tuning.config import configs, peft_config
from tuning.utils.merge_model_utils import create_merged_model

# Third Party
import transformers


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

    logging.info("Initializing launch training script")

    parser = transformers.HfArgumentParser(
        dataclass_types=(
            configs.ModelArguments,
            configs.DataArguments,
            configs.TrainingArguments,
            peft_config.LoraConfig,
            peft_config.PromptTuningConfig,
        )
    )
    peft_method_parsed = "pt"
    json_path = os.getenv("SFT_TRAINER_CONFIG_JSON_PATH")
    json_env_var = os.getenv("SFT_TRAINER_CONFIG_JSON_ENV_VAR")

    # accepts either path to JSON file or encoded string config
    if json_path:
        (
            model_args,
            data_args,
            training_args,
            lora_config,
            prompt_tuning_config,
        ) = parser.parse_json_file(json_path, allow_extra_keys=True)

        contents = ""
        with open(json_path, "r", encoding="utf-8") as f:
            contents = json.load(f)
        peft_method_parsed = contents.get("peft_method")
        logging.debug("Input params parsed: %s", contents)
    elif json_env_var:
        job_config_dict = txt_to_obj(json_env_var)
        logging.debug("Input params parsed: %s", job_config_dict)

        (
            model_args,
            data_args,
            training_args,
            lora_config,
            prompt_tuning_config,
        ) = parser.parse_dict(job_config_dict, allow_extra_keys=True)

        peft_method_parsed = job_config_dict.get("peft_method")
    else:
        raise ValueError(
            "Must set environment variable 'SFT_TRAINER_CONFIG_JSON_PATH' \
        or 'SFT_TRAINER_CONFIG_JSON_ENV_VAR'."
        )

    tune_config = None
    merge_model = False
    if peft_method_parsed == "lora":
        tune_config = lora_config
        merge_model = True
    elif peft_method_parsed == "pt":
        tune_config = prompt_tuning_config

    logging.info(
        "Parameters used to launch training: \
    model_args %s, data_args %s, training_args %s, tune_config %s",
        model_args,
        data_args,
        training_args,
        tune_config,
    )

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
