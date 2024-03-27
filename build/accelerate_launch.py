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
"""Script wraps launch_training to run with accelerate for multi and single GPU cases.
Read accelerate_launch_args configuration via environment variable `SFT_TRAINER_CONFIG_JSON_PATH`
for the path to the JSON config file with parameters or `SFT_TRAINER_CONFIG_JSON_ENV_VAR`
for the encoded config string to parse.
"""

# Standard
import json
import os
import base64
import pickle
import logging

# Third Party
from accelerate.commands.launch import launch_command_parser, launch_command
import torch


def txt_to_obj(txt):
    base64_bytes = txt.encode("ascii")
    message_bytes = base64.b64decode(base64_bytes)
    try:
        # If the bytes represent JSON string
        return json.loads(message_bytes)
    except UnicodeDecodeError:
        # Otherwise the bytes are a pickled python dictionary
        return pickle.loads(message_bytes)


def main():
    LOGLEVEL = os.environ.get("LOG_LEVEL", "WARNING").upper()
    logging.basicConfig(level=LOGLEVEL)

    json_configs = {}
    json_path = os.getenv("SFT_TRAINER_CONFIG_JSON_PATH")
    json_env_var = os.getenv("SFT_TRAINER_CONFIG_JSON_ENV_VAR")

    if json_path:
        with open(json_path, "r", encoding="utf-8") as f:
            json_configs = json.load(f)

    elif json_env_var:
        json_configs = txt_to_obj(json_env_var)

    parser = launch_command_parser()
    # determine which flags are store true actions (don't require a value to be set)
    store_action_params = {}
    for action in parser._actions:
        if type(action).__name__ == "_StoreTrueAction":
            store_action_params[action.dest] = action

    # parse accelerate_launch_args
    accelerate_launch_args = []
    accelerate_config = json_configs.get("accelerate_launch_args", {})
    if accelerate_config:
        logging.info("Using accelerate_launch_args configs: %s", accelerate_config)
        for key, val in accelerate_config.items():
            # For flags that don't have value, ie. --quiet, only add if value is true
            if store_action_params.get(key) and val:
                accelerate_launch_args.append(f"--{key}")
            else:
                accelerate_launch_args.append(f"--{key}")
                accelerate_launch_args.append(str(val))

        if json_configs.get("multi_gpu"):
            # add FSDP config
            if not accelerate_config.get("config_file"):
                fsdp_filepath = os.getenv(
                    "FSDP_DEFAULTS_FILE_PATH", "/app/accelerate_fsdp_defaults.yaml"
                )
                if os.path.exists(fsdp_filepath):
                    logging.info("Setting accelerate config file to: %s", fsdp_filepath)
                    accelerate_launch_args.append("--config_file")
                    accelerate_launch_args.append(fsdp_filepath)

                # add num_processes to overwrite config file set one
                if not accelerate_config.get("num_processes"):
                    num_gpus = torch.cuda.device_count()
                    if num_gpus > 1:
                        logging.info("Setting accelerate num processes to %s", num_gpus)
                        accelerate_launch_args.append("--num_processes")
                        accelerate_launch_args.append(str(num_gpus))
        else:
            accelerate_launch_args.append("--num_processes")
            accelerate_launch_args.append("1")

    # add training_script
    accelerate_launch_args.append("/app/launch_training.py")

    logging.debug("accelerate_launch_args: %s", accelerate_launch_args)
    args = parser.parse_args(args=accelerate_launch_args)
    launch_command(args)


if __name__ == "__main__":
    main()
