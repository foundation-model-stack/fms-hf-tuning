"""CLI for running loading a tuned model and running one or more inference calls on it.
"""
import argparse
import json
import os
from tqdm import tqdm
from tuning.tuned_model import TunedCausalLM

def main():
    parser = argparse.ArgumentParser(
        description="Loads a tuned model and runs an inference call(s) through it"
    )
    parser.add_argument("--model", help="Path to tuned model to be loaded", required=True)
    parser.add_argument(
        "--out_file",
        help="JSON file to write results to",
        default="inference_result.json",
    )
    parser.add_argument(
        "--base_model_name_or_path",
        help="Override for base model to be used [default: value in model adapter_config.json]",
        default=None
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Text to run inference on")
    group.add_argument("--text_file", help="File to be processed where each line is a text to run inference on")
    args = parser.parse_args()
    # If we passed a file, check if it exists before doing anything else
    if args.text_file and not os.path.isfile(args.text_file):
        raise FileNotFoundError(f"Text file: {args.text_file} does not exist!")

    # Load the model
    loaded_model = TunedCausalLM.load(
        checkpoint_path=args.model,
        base_model_name_or_path=args.base_model_name_or_path,
    )

    # Run inference on the text; if multiple were provided, process them all
    if args.text:
        texts = [args.text]
    else:
        with open(args.text_file, "r") as text_file:
            texts = [line.strip() for line in text_file.readlines()]

    # TODO: we should add batch inference support
    results = [
        {"input": text, "output": loaded_model.run(text)}
        for text in tqdm(texts)
    ]

    # Export the results to a file
    with open(args.out_file, "w") as out_file:
        json.dump(results, out_file, sort_keys=True, indent=4)

    print(f"Exported results to: {args.out_file}")

if __name__ == "__main__":
    main()
