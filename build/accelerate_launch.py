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
import tempfile
import shutil
from pathlib import Path

# Third Party
from accelerate.commands.launch import launch_command

# Local
from build.utils import (
    process_accelerate_launch_args,
    serialize_args,
    get_highest_checkpoint,
)
from tuning.utils.config_utils import get_json_config
from tuning.utils.merge_model_utils import create_merged_model
from tuning.config.tracker_configs import FileLoggingTrackerConfig
from tuning.utils.error_logging import (
    write_termination_log,
    USER_ERROR_EXIT_CODE,
    INTERNAL_ERROR_EXIT_CODE,
)

ERROR_LOG = "/dev/termination-log"


def main():
    LOGLEVEL = os.environ.get("LOG_LEVEL", "WARNING").upper()
    logging.basicConfig(level=LOGLEVEL)

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
    original_output_dir = job_config.get("output_dir")
    with tempfile.TemporaryDirectory() as tempdir:
        try:
            # checkpoints outputted to tempdir, only final checkpoint copied to output dir
            job_config["output_dir"] = tempdir
            updated_args = serialize_args(job_config)
            os.environ["SFT_TRAINER_CONFIG_JSON_ENV_VAR"] = updated_args

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

        merge_model = False
        if job_config.get("peft_method") == "lora":
            merge_model = True

        if merge_model:
            try:
                export_path = os.getenv(
                    "LORA_MERGE_MODELS_EXPORT_PATH", original_output_dir
                )

                # get the highest checkpoint dir (last checkpoint)
                lora_checkpoint_dir = get_highest_checkpoint(tempdir)
                full_checkpoint_dir = os.path.join(tempdir, lora_checkpoint_dir)

                logging.info(
                    "Merging lora tuned checkpoint %s with base model into output path: %s",
                    lora_checkpoint_dir,
                    export_path,
                )

                # ensure checkpoint dir has correct files, important with multi-gpu tuning
                if os.path.exists(
                    os.path.join(full_checkpoint_dir, "adapter_config.json")
                ):
                    create_merged_model(
                        checkpoint_models=full_checkpoint_dir,
                        export_path=export_path,
                        save_tokenizer=True,
                    )
            except Exception as e:  # pylint: disable=broad-except
                logging.error(traceback.format_exc())
                write_termination_log(
                    f"Exception encountered merging base model with checkpoint. {e}"
                )
                sys.exit(INTERNAL_ERROR_EXIT_CODE)
        else:
            try:
                # copy last checkpoint into mounted output dir
                pt_checkpoint_dir = get_highest_checkpoint(tempdir)
                logging.info(
                    "Copying last checkpoint %s into output dir %s",
                    pt_checkpoint_dir,
                    original_output_dir,
                )
                shutil.copytree(
                    os.path.join(tempdir, pt_checkpoint_dir),
                    original_output_dir,
                    dirs_exist_ok=True,
                )
            except Exception as e:  # pylint: disable=broad-except
                logging.error(traceback.format_exc())
                write_termination_log(
                    f"Exception encountered writing output model to storage: {e}"
                )
                sys.exit(INTERNAL_ERROR_EXIT_CODE)

        # copy over any loss logs
        try:
            train_logs_filepath = os.path.join(
                tempdir,
                FileLoggingTrackerConfig.training_logs_filename,
            )
            if os.path.exists(train_logs_filepath):
                shutil.copy(train_logs_filepath, original_output_dir)

            # The .complete file will signal to users that we are finished copying
            # files over
            if os.path.exists(original_output_dir):
                Path(os.path.join(original_output_dir, ".complete")).touch()
        except Exception as e:  # pylint: disable=broad-except
            logging.error(traceback.format_exc())
            write_termination_log(
                f"Exception encountered in capturing training logs: {e}"
            )
            sys.exit(INTERNAL_ERROR_EXIT_CODE)

    return 0


if __name__ == "__main__":
    main()
