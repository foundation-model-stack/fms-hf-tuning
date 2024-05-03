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
import os
import logging
import traceback

# Third Party
from accelerate.commands.launch import launch_command

# Local
from build.utils import (
    process_accelerate_launch_args,
    get_job_config,
    write_termination_log,
)


def main():
    LOGLEVEL = os.environ.get("LOG_LEVEL", "WARNING").upper()
    logging.basicConfig(level=LOGLEVEL)

    try:
        job_config = get_job_config()

        args = process_accelerate_launch_args(job_config)
        logging.debug("accelerate launch parsed args: %s", args)
    except FileNotFoundError as e:
        logging.error(traceback.format_exc())
        write_termination_log("Unable to load file: {}".format(e))
        exit(1)
    except (TypeError, ValueError, EnvironmentError) as e:
        logging.error(traceback.format_exc())
        write_termination_log(
            "Exception raised during training. This may be a problem with your input: {}".format(
                e
            )
        )
        exit(1)
    except Exception as e:
        logging.error(traceback.format_exc())
        write_termination_log("Unhandled exception during training")
        exit(200)

    try:
        launch_command(args)
    except Exception as e:
        logging.error(traceback.format_exc)
        write_termination_log("Unhandled exception during training")
        exit(200)


if __name__ == "__main__":
    main()
