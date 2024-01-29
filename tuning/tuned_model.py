"""Interface for loading and running trained causal LMs. In the future,
these capabilities will be unified with the sft_trainer's tuning capabilities.
"""
import argparse
import os
import json
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm

from tuning.utils import AdapterConfigPatcher

class TunedCausalLM:
    def __init__(self, model, tokenizer):
        self.peft_model = model
        self.tokenizer = tokenizer

    @classmethod
    def load(cls, checkpoint_path: str, base_model_name_or_path: str=None) -> "TunedCausalLM":
        """Loads an instance of this model.
        
        By default, the paths for the base model and tokenizer are contained within the adapter
        config of the tuned model. Note that in this context, a path may refer to a model to be
        downloaded from HF hub, or a local path on disk, the latter of which we must be careful
        with when using a model that was written on a different device.
        """
        overrides = {"base_model_name_or_path": base_model_name_or_path} if base_model_name_or_path is not None else {}
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        # Apply the configs to the adapter config of this model; if no overrides
        # are provided, then the context manager doesn't have any effect.
        with AdapterConfigPatcher(checkpoint_path, overrides):
            try:
                peft_model = AutoPeftModelForCausalLM.from_pretrained(checkpoint_path)
            except OSError as e:
                print("Failed to initialize checkpoint model!")
                raise e
        return cls(peft_model, tokenizer)

    def run(self, text: str) -> str:
        """Runs inference on an instance of this model."""
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        peft_outputs = self.peft_model.generate(input_ids=input_ids)
        decoded_result = self.tokenizer.batch_decode(peft_outputs, skip_special_tokens=False)[0]
        return decoded_result

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

if __name__ == "__main__":
    main()
