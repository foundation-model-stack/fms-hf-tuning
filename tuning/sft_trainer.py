import os
import time
from typing import Optional, Union

import datasets
import fire
from peft.utils.other import fsdp_auto_wrap_policy
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaTokenizerFast, GPTNeoXTokenizerFast, GPT2Tokenizer
from transformers.utils import logging
from transformers import TrainerCallback
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from tuning.config import configs, peft_config, tracker_configs
from tuning.data import tokenizer_data_utils
from tuning.utils.config_utils import get_hf_peft_config
from tuning.utils.data_type_utils import get_torch_dtype
from tuning.tracker.tracker import Tracker
from tuning.tracker.aimstack_tracker import AimStackTracker

class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))

def train(
    model_args: configs.ModelArguments,
    data_args: configs.DataArguments,
    train_args: configs.TrainingArguments,
    peft_config: Optional[Union[peft_config.LoraConfig, peft_config.PromptTuningConfig]] = None,
    tracker_name: Optional[str] = None,
    tracker_config: Optional[Union[tracker_configs.AimConfig]] = None
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

    # Initialize the tracker early so we can calculate custom metrics like model_load_time.
 
    if tracker_name == 'aim':
        if tracker_config is not None:
            tracker = AimStackTracker(tracker_config)
        else:
            logger.error("Tracker name is set to "+tracker_name+" but config is None.")
    else:
        logger.info('No tracker set so just set a dummy API which does nothing')
        tracker = Tracker()

    task_type = "CAUSAL_LM"

    model_load_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=train_args.cache_dir,
        torch_dtype=get_torch_dtype(model_args.torch_dtype),
        use_flash_attention_2=model_args.use_flash_attn,
    )
    model_load_time = time.time() - model_load_time
    tracker.track(metric=model_load_time, name='model_load_time')

    peft_config = get_hf_peft_config(task_type, peft_config)

    model.gradient_checkpointing_enable()

    # TODO: Move these to a config as well
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=train_args.cache_dir,
        use_fast = True
    )

    # TODO: understand if we need to hardcode these here or just use defaults in model
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
        tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        })
    elif isinstance(tokenizer, GPTNeoXTokenizerFast) or isinstance(tokenizer, GPT2Tokenizer):
        tokenizer.add_special_tokens({
            "pad_token": "<pad>",
        })

    """TODO: near term - how response template ids are parsed out needs to be cleaned.
       The [2:] here applies if response template has \n prefix, it is needed to strip \n, otherwise template is not found.
       We will create issue to clean this out after we discuss data formats and collators we will support
    """
    response_template_ids = tokenizer.encode(data_args.response_template, add_special_tokens=False)[2:]
    # TODO: This is actually max_seq_length and not model_max_length. we should not override model_max_length 
    # as in current main. We need to change name of this parameter we expose to users.
    model_max_length = min(train_args.model_max_length, tokenizer.model_max_length)
    logger.info(f"Model max length {model_max_length}")
    if train_args.model_max_length > tokenizer.model_max_length:
        logger.warning(f"model_max_length {train_args.model_max_length} exceeds tokenizer.model_max_length {tokenizer.model_max_length}, using tokenizer.model_max_length {tokenizer.model_max_length}")
    
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

    # TODO: lower priority but understand if resizing impacts inference quality and why its needed.
    # It makes sense if we manipulate tokenizer that we also save it and provide it to inference.
    tokenizer_data_utils.tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    
    # load the data by parsing JSON
    json_dataset = datasets.load_dataset('json', data_files=data_args.data_path)
    formatted_dataset = json_dataset['train'].map(lambda example : {f"{data_args.dataset_text_field}" : example[f"{data_args.dataset_text_field}"] + tokenizer.eos_token})
    logger.info(f"Dataset length is {len(formatted_dataset)}")

    # club and register callbacks
    callbacks = [PeftSavingCallback()]

    tracker_callback = tracker.get_hf_callback()
    if tracker_callback is not None:
        callbacks.append(tracker_callback)

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
        
        data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer, ignore_index=configs.IGNORE_INDEX)
        packing = False

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset,
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


def main(**kwargs):
    parser = transformers.HfArgumentParser(dataclass_types=(configs.ModelArguments, 
                                                            configs.DataArguments,
                                                            configs.TrainingArguments,
                                                            peft_config.LoraConfig,
                                                            peft_config.PromptTuningConfig,
                                                            tracker_configs.AimConfig))
    parser.add_argument('--peft_method', type=str.lower, choices=['pt', 'lora', None, 'none'], default="pt")
    parser.add_argument('--tracker', type=str.lower, choices=['aim', None, 'none'], default="aim")
    (model_args, data_args, training_args,
    lora_config, prompt_tuning_config, aim_config,
        additional, _) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    peft_method = additional.peft_method
    tracker_name = additional.tracker

    if peft_method =="lora":
        tune_config=lora_config
    elif peft_method =="pt":
        tune_config=prompt_tuning_config
    else:
        tune_config=None

    if tracker_name == "aim":
        tracker_config=aim_config
    else:
        tracker_config=None

    train(model_args, data_args, training_args, tune_config, tracker_name, tracker_config)

if __name__ == "__main__":
    fire.Fire(main)
