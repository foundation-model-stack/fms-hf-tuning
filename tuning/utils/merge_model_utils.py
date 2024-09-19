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

# Standard
from typing import Union
import json
import os
import shutil

# Third Party
from peft import PeftModel
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def create_merged_model(
    checkpoint_models: Union[str, list[str]],
    export_path: str = None,
    base_model: str = None,
    save_tokenizer: bool = True,
):
    """Given a base model & checkpoint model(s) which were tuned with lora, load into memory
    & create a merged model. If an export path is specified, write it to disk. If multiple
    checkpoint models are provided, we merge_and_unload() them one after the other, which
    combines them with equal weights.

    TODO: In the future, it's probably a good idea to explore different combination schemes,
    which can likely be done using a combination of add_weighted_adapter() and merge_and_unload().

    References:
    - https://github.com/huggingface/peft/issues/1040
    - https://github.com/huggingface/peft/issues/280#issuecomment-1500805831
    - https://huggingface.co/docs/peft/main/en/package_reference/lora#peft.LoraModel.add_weighted_adapter # pylint: disable=line-too-long

    Args:
        checkpoint_model: Union[str, list[str]]
            One or more lora checkpoints containing adapters.
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
    if isinstance(checkpoint_models, str):
        checkpoint_models = [checkpoint_models]

    if base_model is None:
        base_model = fetch_base_model_from_checkpoint(checkpoint_models[0])
    model = AutoModelForCausalLM.from_pretrained(base_model)

    # Merge each of the lora adapter models into the base model with equal weights
    for checkpoint_model in tqdm(checkpoint_models):
        model = PeftModel.from_pretrained(model, checkpoint_model)
        model = model.merge_and_unload()

    if export_path is not None:
        model.save_pretrained(export_path)
        # Export the tokenizer into the merged model dir
        if save_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            tokenizer.save_pretrained(export_path)
    return model


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

    with open(adapter_config, "r", encoding="utf-8") as cfg:
        adapter_dict = json.load(cfg)
    if "base_model_name_or_path" not in adapter_dict:
        raise KeyError(
            "Base model adapter config exists, but has no base_model_name_or_path!"
        )
    return adapter_dict["base_model_name_or_path"]


def _copy_files_to_directory(src: str, dest: str, exclude_files: list[str] = None):
    src_files = os.listdir(src)
    if exclude_files is None:
        exclude_files = []
    for file_name in src_files:
        if file_name in exclude_files:
            continue
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dest)


def post_process_vLLM_adapters_new_tokens(
    path_to_checkpoint: str,
    modified_checkpoint_path: str = None,
    num_added_tokens: int = 0,
):
    # if not set, original checkpoint will be modified
    if not modified_checkpoint_path:
        modified_checkpoint_path = path_to_checkpoint

    with safe_open(
        os.path.join(path_to_checkpoint, "adapter_model.safetensors"), framework="pt"
    ) as f:
        new_embeddings = {}
        adapters = {}
        embeddings_weights_in_adapters = False
        # Quickly check if post-processing is needed by checking adapters file for weights
        for k in f.keys():
            if "lm_head.weight" in k or "embed_tokens.weight" in k:
                embeddings_weights_in_adapters = True
                if num_added_tokens == 0:
                    raise NotImplementedError(
                        "Seems like embeddings are resized without adding new tokens. \
                        Cannot be post-processed to load on vLLM. Try setting \
                        parameter `embedding_size_multiple_of` to 1"
                    )

        # Post-processing is needed to copy out new vectors
        if embeddings_weights_in_adapters:
            for k in f.keys():
                if "lm_head.weight" in k:
                    lm_head = f.get_tensor(k)
                    # pull out tensor values of new tokens

                    new_output_embeddings = lm_head[-num_added_tokens:]
                    # vLLM requires renaming to output_embeddings
                    new_embeddings["output_embeddings"] = new_output_embeddings

                elif "embed_tokens.weight" in k:
                    embed_tokens = f.get_tensor(k)
                    # pull out tensor values of new tokens
                    new_input_embeddings = embed_tokens[-num_added_tokens:]
                    # vLLM requires renaming to input_embeddings
                    new_embeddings["input_embeddings"] = new_input_embeddings
                else:
                    # Retain all other weights in adapters.safetensors
                    adapters[k] = f.get_tensor(k)

            save_file(
                new_embeddings,
                os.path.join(modified_checkpoint_path, "new_embeddings.safetensors"),
            )
            save_file(
                adapters,
                os.path.join(modified_checkpoint_path, "adapter_model.safetensors"),
            )

            # copy out remaining files to desired path
            if modified_checkpoint_path != path_to_checkpoint:
                _copy_files_to_directory(
                    path_to_checkpoint,
                    modified_checkpoint_path,
                    exclude_files=["adapter_model.safetensors"],
                )
