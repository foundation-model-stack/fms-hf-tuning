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
Read multiGPU configuration via environment variable `SFT_TRAINER_CONFIG_JSON_PATH`
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

    # parse multiGPU args
    multi_gpu_args = []
    if json_configs.get("multiGPU"):
        logging.info("Using multi-GPU configs: %s", json_configs.get("multiGPU"))
        for key, val in json_configs["multiGPU"].items():
            multi_gpu_args.append(f"--{key}")
            multi_gpu_args.append(str(val))
    
        # add FSDP config
        fsdp_filepath = os.getenv("FSDP_DEFAULTS_FILE_PATH", "/app/accelerate_fsdp_defaults.yaml")
        multi_gpu_args.append("--config_file")
        multi_gpu_args.append(fsdp_filepath)

    # add training_script
    multi_gpu_args.append("/app/launch_training.py")
    
    logging.debug("multi_gpu_args: %s", multi_gpu_args)
    parser = launch_command_parser()
    args = parser.parse_args(args=multi_gpu_args)
    launch_command(args)

if __name__ == "__main__":
    main()