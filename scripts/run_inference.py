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
"""CLI for running loading a tuned model and running one or more inference calls on it.

NOTE: For the moment, this script is intentionally written to contain all dependencies for two
reasons:
- to keep it portable and not deal with managing multiple local packages.
- because we don't currently plan on supporting inference as a library; i.e., this is only for
testing.

If these things change in the future, we should consider breaking it up.
"""

# Standard
import argparse
import json
import os

# Third Party
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import torch

# Local
from tuning.utils.tokenizer_data_utils import tokenizer_and_embedding_resize


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
        self.config_path = AdapterConfigPatcher._locate_adapter_config(
            self.checkpoint_path
        )
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
        with open(self.config_path, "r", encoding="utf-8") as config_file:
            adapter_config = json.load(config_file)
        overridden_values = self._get_old_config_values(adapter_config, overrides)
        adapter_config = {**adapter_config, **overrides}
        with open(self.config_path, "w", encoding="utf-8") as config_file:
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
            raise KeyError(
                "Adapter config overrides must be set in the config being patched"
            )
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
    def load(
        cls,
        checkpoint_path: str,
        base_model_name_or_path: str = None,
        use_flash_attn: bool = False,
    ) -> "TunedCausalLM":
        """Loads an instance of this model.

        Args:
            checkpoint_path: str
                Checkpoint model to be loaded, which is a directory containing an
                adapter_config.json.
            base_model_name_or_path: str [Default: None]
                Override for the base model to be used.
            use_flash_attn: bool [Default: False]
                Whether to load the model using flash attention.

        By default, the paths for the base model and tokenizer are contained within the adapter
        config of the tuned model. Note that in this context, a path may refer to a model to be
        downloaded from HF hub, or a local path on disk, the latter of which we must be careful
        with when using a model that was written on a different device.

        Returns:
            TunedCausalLM
                An instance of this class on which we can run inference.
        """
        overrides = (
            {"base_model_name_or_path": base_model_name_or_path}
            if base_model_name_or_path is not None
            else {}
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        device = "cuda" if torch.cuda.is_available() else None
        print(f"Inferred device: {device}")
        # Apply the configs to the adapter config of this model; if no overrides
        # are provided, then the context manager doesn't have any effect.
        try:
            with AdapterConfigPatcher(checkpoint_path, overrides):
                try:
                    if base_model_name_or_path is None:
                        raise ValueError("base_model_name_or_path has to be passed")

                    if (
                        has_quantized_config(base_model_name_or_path)
                        and device == "cuda"
                    ):
                        # Using GPTQConfig from HF, avail params are here
                        # https://huggingface.co/docs/transformers/en/main_classes/quantization#transformers.GPTQConfig
                        # We only support 4-bit AutoGPTQ, so setting bits to 4
                        # setting exllama kernel to version 2 as it's a faster kernel
                        gptq_config = GPTQConfig(bits=4, exllama_config={"version": 2})

                        # Since we are using exllama kernel, we need torch.float16 as torch_dtype
                        base_model = AutoModelForCausalLM.from_pretrained(
                            base_model_name_or_path,
                            attn_implementation="flash_attention_2"
                            if use_flash_attn
                            else None,
                            device_map=device,
                            torch_dtype=torch.float16,
                            quantization_config=gptq_config,
                        )
                    else:
                        base_model = AutoModelForCausalLM.from_pretrained(
                            base_model_name_or_path,
                            attn_implementation="flash_attention_2"
                            if use_flash_attn
                            else None,
                            torch_dtype=torch.bfloat16 if use_flash_attn else None,
                        )

                    # since the peft library (PEFTModelForCausalLM) does not handle cases
                    # where the model's layers are modified, in our case the embedding layer
                    # is modified, so we resize the backbone model's embedding layer with our own
                    # utility before passing it along to load the PEFT model.
                    tokenizer_and_embedding_resize(
                        {}, tokenizer=tokenizer, model=base_model
                    )
                    model = PeftModel.from_pretrained(
                        base_model,
                        checkpoint_path,
                        attn_implementation="flash_attention_2"
                        if use_flash_attn
                        else None,
                        torch_dtype=torch.bfloat16 if use_flash_attn else None,
                    )
                except (OSError, ValueError) as e:
                    print("Failed to initialize checkpoint model!")
                    raise e
        except FileNotFoundError:
            print("No adapter config found! Loading as a merged model...")
            # Unable to find the adapter config; fall back to loading as a merged model
            if has_quantized_config(checkpoint_path) and device == "cuda":
                # Using GPTQConfig from HF, avail params are here
                # https://huggingface.co/docs/transformers/en/main_classes/quantization#transformers.GPTQConfig
                # We only support 4-bit AutoGPTQ, so setting bits to 4
                # setting exllama kernel to version 2 as it's a faster kernel
                gptq_config = GPTQConfig(bits=4, exllama_config={"version": 2})

                # Since we are using exllama kernel, we need torch.float16 as torch_dtype
                model = AutoModelForCausalLM.from_pretrained(
                    checkpoint_path,
                    attn_implementation="flash_attention_2" if use_flash_attn else None,
                    device_map=device,
                    torch_dtype=torch.float16,
                    quantization_config=gptq_config,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    checkpoint_path,
                    attn_implementation="flash_attention_2" if use_flash_attn else None,
                    torch_dtype=torch.bfloat16 if use_flash_attn else None,
                )

        model.to(device)
        return cls(model, tokenizer, device)

    def run(
        self, text: str, *, max_new_tokens: int, ret_gen_text_only: bool = False
    ) -> str:
        """Runs inference on an instance of this model.

        Args:
            text: str
                Text on which we want to run inference.
            max_new_tokens: int
                Max new tokens to use for inference.
            ret_gen_text_only: bool
                Indicates whether or not we should return the full text (i.e., input + new tokens)
                or just the newly generated tokens.

        Returns:
            str
                Text generation result.
        """
        tok_res = self.tokenizer(text, return_tensors="pt")
        input_ids = tok_res.input_ids.to(self.device)

        peft_outputs = self.peft_model.generate(
            input_ids=input_ids, max_new_tokens=max_new_tokens
        )
        if ret_gen_text_only:
            tok_to_decode = peft_outputs[:, input_ids.shape[1] :]
        else:
            tok_to_decode = peft_outputs
        decoded_result = self.tokenizer.batch_decode(
            tok_to_decode, skip_special_tokens=False
        )[0]
        return decoded_result


### Main & arg parsing
def main():
    parser = argparse.ArgumentParser(
        description="Loads a tuned model and runs an inference call(s) through it"
    )
    parser.add_argument(
        "--model", help="Path to tuned model / merged model to be loaded", required=True
    )
    parser.add_argument(
        "--out_file",
        help="JSON file to write results to",
        default="inference_result.json",
    )
    parser.add_argument(
        "--base_model_name_or_path",
        help="Override for base model to be used for non-merged models \
            [default: value in model adapter_config.json]",
        default=None,
    )
    parser.add_argument(
        "--max_new_tokens",
        help="max new tokens to use for inference",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--use_flash_attn",
        help="Whether to load the model using Flash Attention 2",
        action="store_true",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Text to run inference on")
    group.add_argument(
        "--text_file",
        help="File to be processed where each line is a text to run inference on",
    )
    args = parser.parse_args()
    # If we passed a file, check if it exists before doing anything else
    if args.text_file and not os.path.isfile(args.text_file):
        raise FileNotFoundError(f"Text file: {args.text_file} does not exist!")

    # Load the model
    loaded_model = TunedCausalLM.load(
        checkpoint_path=args.model,
        base_model_name_or_path=args.base_model_name_or_path,
        use_flash_attn=args.use_flash_attn,
    )

    # Run inference on the text; if multiple were provided, process them all
    if args.text:
        texts = [args.text]
    else:
        with open(args.text_file, "r", encoding="utf-8") as text_file:
            texts = [line.strip() for line in text_file.readlines()]

    # TODO: we should add batch inference support
    results = [
        {
            "input": text,
            "output": loaded_model.run(text, max_new_tokens=args.max_new_tokens),
        }
        for text in tqdm(texts)
    ]

    # Export the results to a file
    with open(args.out_file, "w", encoding="utf-8") as out_file:
        json.dump(results, out_file, sort_keys=True, indent=4)

    print(f"Exported results to: {args.out_file}")


def has_quantized_config(model_path: str):
    return os.path.exists(os.path.join(model_path, "quantize_config.json"))


if __name__ == "__main__":
    main()
