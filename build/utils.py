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
import logging
import pickle
import base64

# Third Party
import torch
from accelerate.commands.launch import launch_command_parser
import shutil


def copy_checkpoint(source, destination, exclude_files: list[str] = None):
    if not os.path.exists(destination):
        os.makedirs(destination)
        shutil.copystat(source, destination)
    # Have a list of directory objects, now iterate over them.
    if exclude_files is None:
        exclude_files = []
    for item in os.listdir(source):
        if item in exclude_files:
            continue
        source_file = os.path.join(source, item)
        destination_file = os.path.join(destination, item)
        if os.path.isdir(source_file):
            # recursive call for subdirectories
            copy_checkpoint(source_file, destination_file)
        else:
            # straight copy.
            shutil.copy2(source_file, destination_file)


def get_highest_checkpoint(dir_path):
    """Given path to directory, returns name of highest checkpoint directory.
    Expects checkpoint directories to be formatted 'checkpoint-<number>'

    Args:
        dir_path: str
    Returns:
        str
    """
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


def serialize_args(args_json):
    """Given dict, converts to base64 byte representation.

    Args:
        args_json: dict
    Returns: str
    """
    message_bytes = pickle.dumps(args_json)
    base64_bytes = base64.b64encode(message_bytes)
    return base64_bytes.decode("ascii")


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
    script = os.environ.get("TRAINING_SCRIPT")
    if script:
        accelerate_launch_args.append(script)
    else:
        accelerate_launch_args.extend(["--module", "tuning.sft_trainer"])

    logging.debug("accelerate_launch_args: %s", accelerate_launch_args)
    args = parser.parse_args(args=accelerate_launch_args)
    return args
