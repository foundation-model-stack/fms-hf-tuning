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
import json

# Third Party
from accelerate.commands.launch import launch_command
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from torch import bfloat16

# Local
from build.utils import (
    process_accelerate_launch_args,
    get_highest_checkpoint,
)
from tuning.utils.config_utils import get_json_config
from tuning.utils.error_logging import (
    write_termination_log,
    USER_ERROR_EXIT_CODE,
    INTERNAL_ERROR_EXIT_CODE,
)
from tuning.data import tokenizer_data_utils

ERROR_LOG = "/dev/termination-log"


def get_base_model_from_adapter_config(adapter_config):
    """Given path to adapter_config.json file, returns the base model name"""
    with open(adapter_config, "r", encoding="utf-8") as config_file:
        adapter_config = json.load(config_file)
        return adapter_config.get("base_model_name_or_path")


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

    # remove lm_head from granite with llama arch models
    try:
        checkpoint_dir = job_config.get("save_model_dir")
        if not checkpoint_dir:
            checkpoint_dir = os.path.join(
                output_dir, get_highest_checkpoint(output_dir)
            )

        use_flash_attn = job_config.get("use_flash_attn", True)
        adapter_config_path = os.path.join(checkpoint_dir, "adapter_config.json")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

        if os.path.exists(adapter_config_path):
            base_model_path = get_base_model_from_adapter_config(adapter_config_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                attn_implementation="flash_attention_2" if use_flash_attn else None,
                torch_dtype=bfloat16 if use_flash_attn else None,
            )

            # since the peft library (PEFTModelForCausalLM) does not handle cases
            # where the model's layers are modified, in our case the embedding layer
            # is modified, so we resize the backbone model's embedding layer with our own
            # utility before passing it along to load the PEFT model.
            tokenizer_data_utils.tokenizer_and_embedding_resize(
                {}, tokenizer=tokenizer, model=base_model
            )
            model = PeftModel.from_pretrained(
                base_model,
                checkpoint_dir,
                attn_implementation="flash_attention_2" if use_flash_attn else None,
                torch_dtype=bfloat16 if use_flash_attn else None,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_dir,
                attn_implementation="flash_attention_2" if use_flash_attn else None,
                torch_dtype=bfloat16 if use_flash_attn else None,
            )

        model_arch = model.config.model_type
        # check that it is a granite model with llama architecture with tied weights
        # ie. lm_head is duplicate of embeddings

        # a fine tuned model will have params_dict.get("model.embed_tokens.weight")
        # a prompt adapter has params_dict.get("base_model.model.embed_tokens.weight")
        # a lora adapter has params_dict.get("base_model.model.model.embed_tokens.weight")
        if model_arch == "llama" and hasattr(model, "lm_head"):
            if (
                # lora tuned model has an addt model layer
                (
                    hasattr(model.model, "model")
                    and model.lm_head.weight.untyped_storage().data_ptr()
                    == model.model.model.embed_tokens.weight.untyped_storage().data_ptr()
                )
                # prompt tuned model or fine tuned model
                or (
                    hasattr(model.model, "embed_tokens")
                    and model.lm_head.weight.untyped_storage().data_ptr()
                    == model.model.embed_tokens.weight.untyped_storage().data_ptr()
                )
            ):

                logging.info("Removing lm_head from checkpoint")
                del model.lm_head.weight

                if hasattr(model, "lm_head.weight"):
                    logging.warning("Failed to delete lm_head.weight from model")

                logging.info("Saving checkpoint to %s", output_dir)
                model.save_pretrained(checkpoint_dir)
                # save tokenizer with model
                tokenizer.save_pretrained(checkpoint_dir)

    except Exception as e:  # pylint: disable=broad-except
        logging.error(traceback.format_exc())
        write_termination_log(f"Exception encountered removing lm_head from model: {e}")
        sys.exit(INTERNAL_ERROR_EXIT_CODE)

    # The .complete file will signal to users that we are finished copying
    # files over
    if os.path.exists(output_dir):
        Path(os.path.join(output_dir, ".complete")).touch()

    return 0


if __name__ == "__main__":
    main()
