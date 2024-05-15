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
import os
import json
import logging
import base64
import pickle

# Third Party
import torch
import transformers
from accelerate.commands.launch import launch_command_parser

# Local
from tuning.config import configs, peft_config, tracker_configs

# The USER_ERROR_EXIT_CODE will be thrown when the process must exit
# as result of a user input error. User-related errors should be
# >= 1 and <=127 due to how some kubernetes operators interpret them.
USER_ERROR_EXIT_CODE = 1
# The INTERNAL_ERROR_EXIT_CODE will be thrown when training
# abnormally terminates, and it is not clearly fault of the user.
# System-level errors should be >= 128 and <= 254
INTERNAL_ERROR_EXIT_CODE = 203


def write_termination_log(text):
    log_file = os.environ.get("TERMINATION_LOG_FILE", "/dev/termination-log")
    try:
        with open(log_file, "a", encoding="utf-8") as handle:
            handle.write(text)
    except Exception as e:  # pylint: disable=broad-except
        logging.warning("Unable to write termination log due to error {}".format(e))


def txt_to_obj(txt):
    base64_bytes = txt.encode("ascii")
    message_bytes = base64.b64decode(base64_bytes)
    try:
        # If the bytes represent JSON string
        return json.loads(message_bytes)
    except UnicodeDecodeError:
        # Otherwise the bytes are a pickled python dictionary
        return pickle.loads(message_bytes)


def get_job_config():
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
    return job_config_dict


def process_launch_training_args(job_config_dict):
    """Return parsed config for tuning to pass to SFT Trainer
    Args:
        job_config_dict: dict
    Return:
        model_args: configs.ModelArguments
        data_args: configs.DataArguments
        training_args: configs.TrainingArguments
        tune_config: peft_config.LoraConfig | peft_config.PromptTuningConfig
        merge_model: bool
        file_logger_config: tracker_configs.FileLoggingTrackerConfig
        aim_config: tracker_configs.AimConfig
    """
    parser = transformers.HfArgumentParser(
        dataclass_types=(
            configs.ModelArguments,
            configs.DataArguments,
            configs.TrainingArguments,
            peft_config.LoraConfig,
            peft_config.PromptTuningConfig,
            tracker_configs.FileLoggingTrackerConfig,
            tracker_configs.AimConfig,
        )
    )

    (
        model_args,
        data_args,
        training_args,
        lora_config,
        prompt_tuning_config,
        file_logger_config,
        aim_config,
    ) = parser.parse_dict(job_config_dict, allow_extra_keys=True)

    peft_method_parsed = job_config_dict.get("peft_method")

    tune_config = None
    merge_model = False
    if peft_method_parsed == "lora":
        tune_config = lora_config
        merge_model = True
    elif peft_method_parsed == "pt":
        tune_config = prompt_tuning_config

    logging.info(
        "Parameters used to launch training: \
    model_args %s, data_args %s, training_args %s, tune_config %s \
        file_logger_config %s aim_config %s",
        model_args,
        data_args,
        training_args,
        tune_config,
        file_logger_config,
        aim_config,
    )

    return (
        model_args,
        data_args,
        training_args,
        tune_config,
        merge_model,
        file_logger_config,
        aim_config,
    )


def process_accelerate_launch_args(job_config_dict):
    """Return parsed config for tuning to pass to SFT Trainer
    Args:
        job_config_dict: dict
    Return:
        args to pass to `accelerate launch`
    """
    parser = launch_command_parser()
    # Map to determine which flags don't require a value to be set
    actions_type_map = {
        action.dest: type(action).__name__ for action in parser._actions
    }

    # Parse accelerate_launch_args
    accelerate_launch_args = []
    accelerate_config = job_config_dict.get("accelerate_launch_args", {})
    if accelerate_config:
        logging.info("Using accelerate_launch_args configs: %s", accelerate_config)
        for key, val in accelerate_config.items():
            # skip num_processes to assign below based on SET_NUM_PROCESSES_TO_NUM_GPUS
            if key == "num_processes":
                continue

            if actions_type_map.get(key) == "_AppendAction":
                for param_val in val:
                    accelerate_launch_args.extend([f"--{key}", str(param_val)])
            elif (actions_type_map.get(key) == "_StoreTrueAction" and val) or (
                actions_type_map.get(key) == "_StoreFalseAction" and not val
            ):
                accelerate_launch_args.append(f"--{key}")
            else:
                accelerate_launch_args.append(f"--{key}")
                # Only need to add key for params that aren't flags ie. --quiet
                if actions_type_map.get(key) == "_StoreAction":
                    accelerate_launch_args.append(str(val))

    # accept setting SET_NUM_PROCESSES_TO_NUM_GPUS=True in Shell interpreted as string
    set_num_processes_to_num_gpus = os.getenv(
        "SET_NUM_PROCESSES_TO_NUM_GPUS", "True"
    ).lower()
    user_arg_num_processes = accelerate_config.get("num_processes")
    num_processes = 0
    if set_num_processes_to_num_gpus == "true":
        num_processes = torch.cuda.device_count()

        if user_arg_num_processes:
            logging.warning(
                "SET_NUM_PROCESSES_TO_NUM_GPUS=True, overwriting user set num_processes %s\
                to all GPUs available, %s.",
                user_arg_num_processes,
                num_processes,
            )
    elif user_arg_num_processes:
        num_processes = int(user_arg_num_processes)

    if num_processes:
        accelerate_launch_args.extend(["--num_processes", str(num_processes)])
        # if multi GPU setting and accelerate config_file not passed by user,
        # use the default config for default set of parameters
        if num_processes > 1 and not accelerate_config.get("config_file"):
            # Add default FSDP config
            fsdp_filepath = os.getenv(
                "FSDP_DEFAULTS_FILE_PATH", "/app/accelerate_fsdp_defaults.yaml"
            )
            if os.path.exists(fsdp_filepath):
                logging.info("Using accelerate config file: %s", fsdp_filepath)
                accelerate_launch_args.extend(["--config_file", fsdp_filepath])

        elif num_processes == 1:
            logging.info("num_processes=1 so setting env var CUDA_VISIBLE_DEVICES=0")
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        logging.warning(
            "num_processes param was not passed in. Value from config file (if available) will \
                be used or accelerate launch will determine number of processes automatically"
        )

    # Add training_script
    script = os.environ.get("LAUNCH_TRAINING_SCRIPT", "/app/launch_training.py")
    accelerate_launch_args.append(script)

    logging.debug("accelerate_launch_args: %s", accelerate_launch_args)
    args = parser.parse_args(args=accelerate_launch_args)
    return args
