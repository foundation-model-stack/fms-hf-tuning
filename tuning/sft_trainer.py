from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import transformers
import torch
import datasets

from tuning.data import tokenizer_data_utils
from tuning.config import configs

from aim_loader import get_aimstack_callback
from transformers.utils import logging

from peft import LoraConfig
import os
from transformers import TrainerCallback
from peft.utils.other import fsdp_auto_wrap_policy

class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))


def train():
    logger = logging.get_logger("sft_trainer")
    parser = transformers.HfArgumentParser((configs.ModelArguments, configs.DataArguments, configs.TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=model_args.use_flash_attn,
    )
    
    ## TODO: hard coding Lora config right now, we will deal with it later
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules = ["q_proj", "v_proj"],
    )

    model.gradient_checkpointing_enable()

    # TODO: Move these to a config as well
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast = True
    )
    
    model_max_length = tokenizer.model_max_length
    logger.info(f"Model max length {model_max_length}")
    
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

    if training_args.packing:
        logger.info("Packing is set to True")
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=json_dataset['train'],
            dataset_text_field=data_args.dataset_text_field,
            packing=True,
            args=training_args,
            max_seq_length=model_max_length,
            callbacks=callbacks,
            peft_config=peft_config,
        )
    else:
        logger.info("Packing is set to False")
        if data_args.response_template is None:
            logger.error("Error, response template is None, needs to be set for training")
            exit(-1)
        
        if data_args.dataset_text_field is None:
            logger.error("Error, dataset_text_field is None, needs to be set for training")
            exit(-1)
        
        #the below is specific to Llama since it uses sentencepiece tokenizer
        response_template_ids = tokenizer.encode(data_args.response_template, add_special_tokens=False)[2:]
        data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=json_dataset['train'],
            data_collator=data_collator,
            dataset_text_field=data_args.dataset_text_field,
            args=training_args,
            max_seq_length=model_max_length,
            callbacks=callbacks,
            peft_config=peft_config,
        )
    
    trainer.accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)
    trainer.train()
    
if __name__ == "__main__":
    train()
