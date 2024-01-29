"""Interface for loading and running trained causal LMs. In the future,
these capabilities will be unified with the sft_trainer's tuning capabilities.
"""

import argparse
import datasets
import os
import json
from typing import Optional, Union

from peft import AutoPeftModelForCausalLM
from peft.utils.other import fsdp_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaTokenizerFast, GPTNeoXTokenizerFast
from transformers.utils import logging
from transformers import TrainerCallback
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from tqdm import tqdm

from tuning.aim_loader import get_aimstack_callback
from tuning.config import configs, peft_config
from tuning.data import tokenizer_data_utils
from tuning.utils import AdapterConfigPatcher
from tuning.utils.config_utils import get_hf_peft_config
from tuning.utils.data_type_utils import get_torch_dtype

class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))


class TunedCausalLM:
    def __init__(self, model, tokenizer):
        self.peft_model = model
        self.tokenizer = tokenizer

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
        return cls(peft_model, tokenizer)


    def run(self, text: str) -> str:
        """Runs inference on an instance of this model.

        Args:
            text: str
                Text on which we want to run inference.

        Returns:
            str
                Text generation result.          
        """
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        peft_outputs = self.peft_model.generate(input_ids=input_ids)
        decoded_result = self.tokenizer.batch_decode(peft_outputs, skip_special_tokens=False)[0]
        return decoded_result


    @classmethod
    def train(
        cls,
        model_args: configs.ModelArguments,
        data_args: configs.DataArguments,
        train_args: configs.TrainingArguments,
        peft_config: Optional[Union[peft_config.LoraConfig, peft_config.PromptTuningConfig]] = None,
    ):
        """Call the SFTTrainer

        Args:
            model_args: tuning.config.configs.ModelArguments
            data_args: tuning.config.configs.DataArguments
            train_args: tuning.config.configs.TrainingArguments
            peft_config: peft_config.LoraConfig for Lora tuning | \
            peft_config.PromptTuningConfig for prompt tuning | \
            None for fine tuning
                The peft configuration to pass to trainer
        """
        run_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1

        logger = logging.get_logger("sft_trainer")

        # Validate parameters
        if (not isinstance(train_args.num_train_epochs, float)) or (train_args.num_train_epochs <= 0):
            raise ValueError("num_train_epochs has to be an integer/float >= 1")
        if (not isinstance(train_args.gradient_accumulation_steps , int)) or (train_args.gradient_accumulation_steps <= 0):
            raise ValueError("gradient_accumulation_steps has to be an integer >= 1")

        # make sure to unset FSDP args when running on single gpu
        if not run_distributed:
            train_args.fsdp = ""
            train_args.fsdp_config = {'xla':False}

        task_type = "CAUSAL_LM"
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=train_args.cache_dir,
            torch_dtype=get_torch_dtype(model_args.torch_dtype),
            use_flash_attention_2=model_args.use_flash_attn,
        )
        
        peft_config = get_hf_peft_config(task_type, peft_config)

        model.gradient_checkpointing_enable()

        # TODO: Move these to a config as well
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=train_args.cache_dir,
            model_max_length=train_args.model_max_length,
            padding_side="right",
            use_fast = True
        )
        if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
            tokenizer.add_special_tokens({
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            })
        elif isinstance(tokenizer, GPTNeoXTokenizerFast):
            tokenizer.add_special_tokens({
                "pad_token": "<pad>",
            })
        
        model_max_length = min(train_args.model_max_length, tokenizer.model_max_length)
        logger.info(f"Model max length {model_max_length}")
        if train_args.model_max_length > tokenizer.model_max_length:
            logger.warning(f"model_max_length {model_max_length} exceeds tokenizer.model_max_length {tokenizer.model_max_length}, using tokenizer.model_max_length {tokenizer.model_max_length}")
        
        # TODO: we need to change this, perhaps follow what open instruct does?
        special_tokens_dict = dict()
        if tokenizer.pad_token is None:
            logger.warning("PAD token set to default, missing in tokenizer")
            special_tokens_dict["pad_token"] = configs.DEFAULT_PAD_TOKEN
        if tokenizer.eos_token is None:
            logger.warning("EOS token set to default, missing in tokenizer")
            special_tokens_dict["eos_token"] = configs.DEFAULT_EOS_TOKEN
        if tokenizer.bos_token is None:
            logger.warning("BOS token set to default, missing in tokenizer")
            special_tokens_dict["bos_token"] = configs.DEFAULT_BOS_TOKEN
        if tokenizer.unk_token is None:
            logger.warning("UNK token set to default, missing in tokenizer")
            special_tokens_dict["unk_token"] = configs.DEFAULT_UNK_TOKEN

        tokenizer_data_utils.tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )
        
        # load the data by parsing JSON
        json_dataset = datasets.load_dataset('json', data_files=data_args.data_path)
        logger.info(f"Dataset length is {len(json_dataset['train'])}")

        aim_callback = get_aimstack_callback()
        callbacks=[aim_callback,PeftSavingCallback()]

        if train_args.packing:
            logger.info("Packing is set to True")
            data_collator = None
            packing = True
        else:
            logger.info("Packing is set to False")
            if data_args.response_template is None:
                logger.error("Error, response template is None, needs to be set for training")
                exit(-1)
            
            if data_args.dataset_text_field is None:
                logger.error("Error, dataset_text_field is None, needs to be set for training")
                exit(-1)

            response_template_ids = tokenizer.encode(data_args.response_template, add_special_tokens=False)[2:]
            data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer, ignore_index=configs.IGNORE_INDEX)
            packing = False

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=json_dataset['train'],
            packing=packing,
            data_collator=data_collator,
            dataset_text_field=data_args.dataset_text_field,
            args=train_args,
            max_seq_length=model_max_length,
            callbacks=callbacks,
            peft_config=peft_config,
        )

        if run_distributed and peft_config is not None:
            trainer.accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)
        trainer.train()


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
