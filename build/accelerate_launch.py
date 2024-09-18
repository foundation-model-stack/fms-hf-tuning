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
"""Script wraps sft_trainer to run with accelerate for multi and single GPU cases.
Read accelerate_launch_args configuration via environment variable `SFT_TRAINER_CONFIG_JSON_PATH`
for the path to the JSON config file with parameters or `SFT_TRAINER_CONFIG_JSON_ENV_VAR`
for the encoded config string to parse.
"""

# Standard
import os
import logging
import subprocess
import sys
import traceback
from pathlib import Path

# Third Party
from accelerate.commands.launch import launch_command

# Local
from build.utils import (
    process_accelerate_launch_args,
)
from tuning.utils.config_utils import get_json_config
from tuning.utils.error_logging import (
    write_termination_log,
    USER_ERROR_EXIT_CODE,
    INTERNAL_ERROR_EXIT_CODE,
)

ERROR_LOG = "/dev/termination-log"


def main():
    if not os.getenv("TERMINATION_LOG_FILE"):
        os.environ["TERMINATION_LOG_FILE"] = ERROR_LOG

    ##########
    #
    # Parse arguments
    #
    ##########
    try:
        job_config = get_json_config()
        if not job_config:
            raise ValueError(
                "Must set environment variable 'SFT_TRAINER_CONFIG_JSON_PATH' \
            or 'SFT_TRAINER_CONFIG_JSON_ENV_VAR'."
            )

        # Configure log_level of python native logger.
        # CLI arg takes precedence over env var. And if neither is set, we use default "WARNING"
        log_level = job_config.get(
            "log_level"
        )  # this will be set to either the value found or None
        if (
            not log_level
        ):  # if log level not set by job_config aka by JSON, set it via env var or set default
            log_level = os.environ.get("LOG_LEVEL", "WARNING")
        log_level = log_level.upper()
        logging.basicConfig(level=log_level)

        args = process_accelerate_launch_args(job_config)
        logging.debug("accelerate launch parsed args: %s", args)
    except FileNotFoundError as e:
        logging.error(traceback.format_exc())
        write_termination_log("Unable to load file: {}".format(e))
        sys.exit(USER_ERROR_EXIT_CODE)
    except (TypeError, ValueError, EnvironmentError) as e:
        logging.error(traceback.format_exc())
        write_termination_log(
            f"Exception raised during training. This may be a problem with your input: {e}"
        )
        sys.exit(USER_ERROR_EXIT_CODE)
    except Exception as e:  # pylint: disable=broad-except
        logging.error(traceback.format_exc())
        write_termination_log(f"Unhandled exception during training. {e}")
        sys.exit(INTERNAL_ERROR_EXIT_CODE)

    ##########
    #
    # Launch training
    #
    ##########
    output_dir = job_config.get("output_dir")
    try:
        # checkpoints outputted to tempdir, only final checkpoint copied to output dir
        launch_command(args)
    except subprocess.CalledProcessError as e:
        # If the subprocess throws an exception, the base exception is hidden in the
        # subprocess call and is difficult to access at this level. However, that is not
        # an issue because sft_trainer.py would have already written the exception
        # message to termination log.
        logging.error(traceback.format_exc())
        # The exit code that sft_trainer.py threw is captured in e.returncode

        return_code = e.returncode
        if return_code not in [INTERNAL_ERROR_EXIT_CODE, USER_ERROR_EXIT_CODE]:
            return_code = INTERNAL_ERROR_EXIT_CODE
            write_termination_log(f"Unhandled exception during training. {e}")
        sys.exit(return_code)
    except Exception as e:  # pylint: disable=broad-except
        logging.error(traceback.format_exc())
        write_termination_log(f"Unhandled exception during training. {e}")
        sys.exit(INTERNAL_ERROR_EXIT_CODE)

    # The .complete file will signal to users that we are finished copying
    # files over
    if os.path.exists(output_dir):
        Path(os.path.join(output_dir, ".complete")).touch()

    return 0


if __name__ == "__main__":
    main()
