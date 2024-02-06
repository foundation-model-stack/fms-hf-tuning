"""CLI for running loading a tuned model and running one or more inference calls on it.
"""
import argparse
import json
import os
from peft import AutoPeftModelForCausalLM
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from tuning.utils import AdapterConfigPatcher


class TunedCausalLM:
    def __init__(self, model, tokenizer, device):
        self.peft_model = model
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def load(cls, checkpoint_path: str, base_model_name_or_path: str=None) -> "TunedCausalLM":
        """Loads an instance of this model.

        Args:
            checkpoint_path: str
                Checkpoint model to be loaded, which is a directory containing an
                adapter_config.json.
            base_model_name_or_path: str [Default: None]
                Override for the base model to be used.

        By default, the paths for the base model and tokenizer are contained within the adapter
        config of the tuned model. Note that in this context, a path may refer to a model to be
        downloaded from HF hub, or a local path on disk, the latter of which we must be careful
        with when using a model that was written on a different device.

        Returns:
            TunedCausalLM
                An instance of this class on which we can run inference.
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
        device = "cuda" if torch.cuda.is_available() else None
        print(f"Inferred device: {device}")
        peft_model.to(device)
        return cls(peft_model, tokenizer, device)


    def run(self, text: str, *, max_new_tokens: int) -> str:
        """Runs inference on an instance of this model.

        Args:
            text: str
                Text on which we want to run inference.
            max_new_tokens: int
                Max new tokens to use for inference.

        Returns:
            str
                Text generation result.          
        """
        tok_res = self.tokenizer(text, return_tensors="pt")
        input_ids = tok_res.input_ids.to(self.device)

        peft_outputs = self.peft_model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)
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
    parser.add_argument(
        "--max_new_tokens",
        help="max new tokens to use for inference",
        type=int,
        default=20,
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
        {"input": text, "output": loaded_model.run(text, max_new_tokens=args.max_new_tokens)}
        for text in tqdm(texts)
    ]

    # Export the results to a file
    with open(args.out_file, "w") as out_file:
        json.dump(results, out_file, sort_keys=True, indent=4)

    print(f"Exported results to: {args.out_file}")

if __name__ == "__main__":
    main()
