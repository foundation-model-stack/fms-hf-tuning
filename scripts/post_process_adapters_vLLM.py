# Standard
import argparse
import json
import os

# Local
from tuning.utils.merge_model_utils import post_process_vLLM_adapters_new_tokens


### Main & arg parsing
def main():
    parser = argparse.ArgumentParser(
        description="Post processes adapters due to addition of new tokens, as needed by vLLM"
    )
    parser.add_argument(
        "--model_path",
        help="Path to tuned model containing either one or multiple checkpoints \
                              Path should have file added_tokens_info.json produced by tuning \
                              Hint: This will be either output_dir or save_model_dir arguments while tuning \
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

    if os.path.exists(os.path.join(args.model_path, "added_tokens_info.json")):
        with open(
            os.path.join(args.model_path, "added_tokens_info.json"), encoding="utf-8"
        ) as json_data:
            added_tokens_info = json.loads(json_data)
            num_added_tokens = added_tokens_info["num_added_tokens"]
    else:
        print("file added_tokens_info.json not in model_path. Cannot post-processes")

    if os.path.exists(os.path.join(args.model_path, "adapter_model.safetensors")):
        post_process_vLLM_adapters_new_tokens(
            args.model_path, args.output_model_path, num_added_tokens
        )
    # if multiple checkpoints in directory, process each checkpoint
    for _, dirs, _ in os.walk(args.model_path, topdown=False):
        for name in dirs:
            if "checkpoint-" in name.lower():
                post_process_vLLM_adapters_new_tokens(
                    os.path.join(args.model_path, name),
                    os.path.join(args.output_model_path, name),
                    num_added_tokens,
                )