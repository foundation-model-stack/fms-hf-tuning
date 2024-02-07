"""CLI for running loading a tuned model and running one or more inference calls on it.

NOTE: For the moment, this script is intentionally written to contain all dependencies for two
reasons:
- to keep it portable and not deal with managing multiple local packages.
- because we don't currently plan on supporting inference as a library; i.e., this is only for
testing.

If these things change in the future, we should consider breaking it up.
"""
import argparse
import json
import os
from peft import AutoPeftModelForCausalLM
import torch
from tqdm import tqdm
from transformers import AutoTokenizer


### Utilities
class AdapterConfigPatcher:
    """Adapter config patcher is a context manager for patching overrides into a config;
    it will locate the adapter_config.json in a file and patch a dict of provided overrides
    when inside the dict block, and restore them when it leaves. This DOES actually write to
    the file, so it's NOT safe to use in parallel inference runs.

    Example:
        overrides = {"base_model_name_or_path": "foo"}
        with AdapterConfigPatcher(checkpoint_path, overrides):
            # When loaded in this block, the config's base_model_name_or_path is "foo"
            peft_model = AutoPeftModelForCausalLM.from_pretrained(checkpoint_path)
    """
    def __init__(self, checkpoint_path: str, overrides: dict):
        self.checkpoint_path = checkpoint_path
        self.overrides = overrides
        self.config_path = AdapterConfigPatcher._locate_adapter_config(self.checkpoint_path)
        # Values that we will patch later on
        self.patched_values = {}

    @staticmethod
    def _locate_adapter_config(checkpoint_path: str) -> str:
        """Given a path to a tuned checkpoint, tries to find the adapter_config
        that will be loaded through the Peft auto model API.

        Args:
            checkpoint_path: str
                Checkpoint model, which presumably holds an adapter config.

        Returns:
            str
                Path to the located adapter_config file.
        """
        config_path = os.path.join(checkpoint_path, "adapter_config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Could not locate adapter config: {config_path}")
        return config_path

    def _apply_config_changes(self, overrides: dict) -> dict:
        """Applies a patch to a config with some override dict, returning the values
        that we patched over so that they may be restored later.
        
        Args:
            overrides: dict
                Overrides to write into the adapter_config.json. Currently, we
                require all override keys to be defined in the config being patched.

        Returns:
            dict
                Dict containing the values that we patched over.
        """
        # If we have no overrides, this context manager is a noop; no need to do anything
        if not overrides:
            return {}
        with open(self.config_path, "r") as config_file:
            adapter_config = json.load(config_file)
        overridden_values = self._get_old_config_values(adapter_config, overrides)
        adapter_config = {**adapter_config, **overrides}
        with open(self.config_path, "w") as config_file:
            json.dump(adapter_config, config_file, indent=4)
        return overridden_values

    @staticmethod
    def _get_old_config_values(adapter_config: dict, overrides: dict) -> dict:
        """Grabs the existing config subdict that we are going to clobber from the
        loaded adapter_config.

        Args:
            adapter_config: dict
                Adapter config whose values we are interested in patching.
            overrides: dict
                Dict of overrides, containing keys definined in the adapter_config with
                new values.

        Returns:
            dict
                The subdictionary of adapter_config, containing the keys in overrides,
                with the values that we are going to replace.
        """
        # For now, we only expect to patch the base model; this may change in the future,
        # but ensure that anything we are patching is defined in the original config
        if not set(overrides.keys()).issubset(set(adapter_config.keys())):
            raise KeyError("Adapter config overrides must be set in the config being patched")
        return {key: adapter_config[key] for key in overrides}

    def __enter__(self):
        """Apply the config overrides and saved the patched values."""
        self.patched_values = self._apply_config_changes(self.overrides)

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Apply the patched values over our exported overrides."""
        self._apply_config_changes(self.patched_values)


### Funcs for loading and running models
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


### Main & arg parsing
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
