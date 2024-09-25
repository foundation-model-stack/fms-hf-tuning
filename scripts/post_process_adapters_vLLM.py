""" Script to post-process tuned LoRA adapters for inference on vLLM. 
vLLM requires that any token embeddings added while tuning be moved to a new file \
called new_embeddings.safetensors. \
See the description in utility function \
/tuning/utils/merge_model_utils/post_process_vLLM_adapters_new_tokens for more details.

This script takes a path to tuned model artifacts containing adapters \
(or checkpoints with adapters) and the file 'added_tokens_info.json' produced while tuning. \
It will perform the post-processing as needed for inferencing on vLLM.
"""
# Standard
import argparse
import json
import logging
import os
import sys

# Local
from tuning.utils.merge_model_utils import (
    copy_files_to_directory,
    post_process_vLLM_adapters_new_tokens,
)


### Main & arg parsing
def main():
    parser = argparse.ArgumentParser(
        description="Post processes LoRA adapters due to addition of new tokens, as needed by vLLM"
    )
    parser.add_argument(
        "--model_path",
        help="Path to tuned model containing either one or multiple checkpoints. \
                              Path should have file added_tokens_info.json produced by tuning. \
                              Hint: This will be either output_dir or save_model_dir arguments while tuning. \
                              If multiple checkpoints are present, each checkpoint folder name \
                              should begin with 'checkpoint-'",
        required=True,
    )
    parser.add_argument(
        "--output_model_path",
        help="Output directory where post-processed artifacts will be stored. \
                                    If not provided, artifacts will be modified in place",
        default=None,
    )
    args = parser.parse_args()

    if args.output_model_path is None:
        output_model_path = args.model_path
    else:
        output_model_path = args.output_model_path
    if os.path.exists(os.path.join(args.model_path, "added_tokens_info.json")):
        with open(
            os.path.join(args.model_path, "added_tokens_info.json"), encoding="utf-8"
        ) as json_data:
            added_tokens_info = json.load(json_data)
            num_added_tokens = added_tokens_info["num_new_tokens"]
    else:
        raise ValueError(
            "file added_tokens_info.json not in model_path. \
                        Cannot post-processes"
        )
    if num_added_tokens == 0:
        logging.info("No new tokens added, hence post-processing not needed")
        sys.exit(0)

    found_adapters = 0
    if os.path.exists(os.path.join(args.model_path, "adapter_model.safetensors")):
        found_adapters = 1
        post_process_vLLM_adapters_new_tokens(
            args.model_path, output_model_path, num_added_tokens
        )
    # if multiple checkpoints in directory, process each checkpoint
    found_checkpoints = 0
    for _, dirs, _ in os.walk(args.model_path, topdown=False):
        for name in dirs:
            if "checkpoint-" in name.lower():
                post_process_vLLM_adapters_new_tokens(
                    os.path.join(args.model_path, name),
                    os.path.join(output_model_path, name),
                    num_added_tokens,
                )
                found_checkpoints = 1
    if found_checkpoints and output_model_path != args.model_path:
        copy_files_to_directory(
            args.model_path,
            output_model_path,
            exclude_files=["adapter_model.safetensors"],
        )
    if not found_adapters and not found_checkpoints:
        logging.warning("No adapters were found to process in model path provided")


if __name__ == "__main__":
    main()
