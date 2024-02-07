import argparse
import json
import os
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def create_merged_model(
    checkpoint_model: str,
    export_path: str=None,
    base_model: str=None,
    save_tokenizer: bool=True
):
    """Given a base model & a checkpoint model containing adapters, which were tuned with lora,
    load both into memory & create a merged model. If an export path is specified, write it
    to disk.

    Args:
        checkpoint_model: str
            Lora checkpoint containing adapters.
        export_path: str
            Path to export the merged model to.
        base_model: str
            Base model to be leveraged. If no base model is specified, the base model is pulled
            from the checkpoint model's adapter config.
        save_tokenizer: bool
            Indicates whether or not we should save the tokenizer from the base model. Only
            used if the export_path is set.

    Returns:
        transformers model
            Merged model created from the checkpoint / base model.
    """
    if base_model is None:
        base_model = fetch_base_model_from_checkpoint(checkpoint_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)
    model_combined = PeftModel.from_pretrained(model, checkpoint_model)
    model_combined = model_combined.merge_and_unload()
    if export_path is not None:
        model_combined.save_pretrained(export_path)
        # Export the tokenizer into the merged model dir
        if save_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            tokenizer.save_pretrained(export_path)
    return model_combined


def fetch_base_model_from_checkpoint(checkpoint_model: str) -> str:
    """Inspects the checkpoint model, locates the adapter config, and grabs the
    base_model_name_or_path.

    Args:
        checkpoint_model: str
            Checkpoint model containing the adapter config, which specifies the base model.
    
    Returns:
        str
            base_model_name_or_path specified in the adapter config of the tuned peft model.
    """
    adapter_config = os.path.join(checkpoint_model, "adapter_config.json")
    if not os.path.isfile(adapter_config):
        raise FileNotFoundError("Unable to locate adapter config to infer base model!")

    with open(adapter_config, "r") as cfg:
        adapter_dict = json.load(cfg)
    if "base_model_name_or_path" not in adapter_dict:
        raise KeyError("Base model adapter config exists, but has no base_model_name_or_path!")
    return adapter_dict["base_model_name_or_path"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates a merged lora model"
    )
    parser.add_argument("--checkpoint_model", help="Path to the checkpoint [tuned lora model]", required=True)
    parser.add_argument("--export_path", help="Path to write the merged model to", required=True)
    parser.add_argument("--base_model", help="Base model to be used [default=None; infers from adapter config]", default=None)
    parser.add_argument("--save_tokenizer", default=False, action="store_true", help="Whether or not we should export the tokenizer to the export_path")
    args = parser.parse_args()

    create_merged_model(
        checkpoint_model=args.checkpoint_model,
        export_path=args.export_path,
        base_model=args.base_model,
        save_tokenizer=args.save_tokenizer,
    )
