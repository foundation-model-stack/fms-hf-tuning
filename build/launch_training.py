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
<<<<<<< HEAD
=======
from pathlib import Path
import os
import tempfile
import shutil
>>>>>>> 62bb076 (add .complete file to output dir when done)
import sys
import traceback

# Third Party
from huggingface_hub.utils._validators import HFValidationError
from torch.cuda import OutOfMemoryError

# First Party
import logging

# Local
from tuning import sft_trainer
from tuning.config.tracker_configs import TrackerConfigFactory
from build.utils import (
    write_termination_log,
    USER_ERROR_EXIT_CODE,
    INTERNAL_ERROR_EXIT_CODE,
)


def main():

    try:
        sft_trainer.main()
    except (MemoryError, OutOfMemoryError) as e:
        logging.error(traceback.format_exc())
        write_termination_log(f"OOM error during training. {e}")
        sys.exit(INTERNAL_ERROR_EXIT_CODE)
    except FileNotFoundError as e:
        logging.error(traceback.format_exc())
        write_termination_log("Unable to load file: {}".format(e))
        sys.exit(USER_ERROR_EXIT_CODE)
    except HFValidationError as e:
        logging.error(traceback.format_exc())
        write_termination_log(
            f"There may be a problem with loading the model. Exception: {e}"
        )
        sys.exit(USER_ERROR_EXIT_CODE)
    except (TypeError, ValueError, EnvironmentError) as e:
        logging.error(traceback.format_exc())
        write_termination_log(
            f"Exception raised during training. This may be a problem with your input: {e}"
        )
        sys.exit(USER_ERROR_EXIT_CODE)
<<<<<<< HEAD
    except Exception as e:  # pylint: disable=broad-except
        logging.error(traceback.format_exc())
        write_termination_log(f"Unhandled exception during training: {e}")
        sys.exit(INTERNAL_ERROR_EXIT_CODE)
=======

    original_output_dir = training_args.output_dir
    with tempfile.TemporaryDirectory() as tempdir:
        training_args.output_dir = tempdir
        try:
            tracker_config_args = TrackerConfigFactory(
                file_logger_config=file_logger_config, aim_config=aim_config
            )
            sft_trainer.train(
                model_args=model_args,
                data_args=data_args,
                train_args=training_args,
                peft_config=tune_config,
                tracker_configs=tracker_config_args,
            )
        except (MemoryError, OutOfMemoryError) as e:
            logging.error(traceback.format_exc())
            write_termination_log(f"OOM error during training. {e}")
            sys.exit(INTERNAL_ERROR_EXIT_CODE)
        except FileNotFoundError as e:
            logging.error(traceback.format_exc())
            write_termination_log("Unable to load file: {}".format(e))
            sys.exit(USER_ERROR_EXIT_CODE)
        except HFValidationError as e:
            logging.error(traceback.format_exc())
            write_termination_log(
                f"There may be a problem with loading the model. Exception: {e}"
            )
            sys.exit(USER_ERROR_EXIT_CODE)
        except (TypeError, ValueError, EnvironmentError) as e:
            logging.error(traceback.format_exc())
            write_termination_log(
                f"Exception raised during training. This may be a problem with your input: {e}"
            )
            sys.exit(USER_ERROR_EXIT_CODE)
        except Exception as e:  # pylint: disable=broad-except
            logging.error(traceback.format_exc())
            write_termination_log(f"Unhandled exception during training: {e}")
            sys.exit(INTERNAL_ERROR_EXIT_CODE)

        if merge_model:
            try:
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
            except Exception as e:  # pylint: disable=broad-except
                logging.error(traceback.format_exc())
                write_termination_log(
                    f"Exception encountered merging base model with checkpoint. {e}"
                )
                sys.exit(INTERNAL_ERROR_EXIT_CODE)
        else:
            try:
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
            except Exception as e:  # pylint: disable=broad-except
                logging.error(traceback.format_exc())
                write_termination_log(
                    f"Exception encountered writing output model to storage: {e}"
                )
                sys.exit(INTERNAL_ERROR_EXIT_CODE)

        # copy over any loss logs
        try:
            train_logs_filepath = os.path.join(
                training_args.output_dir,
                tracker_config_args.file_logger_config.training_logs_filename,
            )
            if os.path.exists(train_logs_filepath):
                shutil.copy(train_logs_filepath, original_output_dir)
            # The .complete file will signal to users that we are finished copying
            # files over
            Path(os.path.join(original_output_dir, ".complete")).touch()
        except Exception as e:  # pylint: disable=broad-except
            logging.error(traceback.format_exc())
            write_termination_log(
                f"Exception encountered in capturing training logs: {e}"
            )
            sys.exit(INTERNAL_ERROR_EXIT_CODE)
>>>>>>> 62bb076 (add .complete file to output dir when done)

    return 0


if __name__ == "__main__":
    main()