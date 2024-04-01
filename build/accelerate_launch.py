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
    # Map to determine which flags don't require a value to be set
    actions_type_map = {
        action.dest: type(action).__name__ for action in parser._actions
    }

    # Parse accelerate_launch_args
    accelerate_launch_args = []
    accelerate_config = json_configs.get("accelerate_launch_args", {})
    if accelerate_config:
        logging.info("Using accelerate_launch_args configs: %s", accelerate_config)
        for key, val in accelerate_config.items():
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

    num_processes = accelerate_config.get("num_processes")
    if num_processes:
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
    accelerate_launch_args.append("/app/launch_training.py")

    logging.debug("accelerate_launch_args: %s", accelerate_launch_args)
    args = parser.parse_args(args=accelerate_launch_args)
    logging.debug("accelerate launch parsed args: %s", args)
    launch_command(args)


if __name__ == "__main__":
    main()
