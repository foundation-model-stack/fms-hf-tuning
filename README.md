# FMS HF Tuning

This repo aims to create basic tuning scripts with support for specific models, assumes the use of Hugging Face `SFTTrainer` and PyTorch FSDP only. Our approach to tuning is:
1. Models are loaded from Hugging Face `transformers` or the `foundation-model-stack` -- models are either optimized to use `Flash Attention v2` directly or through `SDPA`
2. Hugging Face `SFTTrainer` for the training loop
3. `FSDP` as the backend for training

## Installation

```
pip install -r requirements.txt
pip install -U datasets
pip install -e .
```

## Data format
The data format expectation is a single column text. The trainer is configured to expect a response template as a string. For example, if one wants to prepare the `alpaca` format data to feed into this trainer, it is quite easy and can be done with the following code.

```python
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def format_alpaca_fn(example):
    prompt_input, prompt_no_input = PROMPT_DICT['prompt_input'], PROMPT_DICT['prompt_no_input']
    output = prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
    output = f"{output} {example['output']}"
    return {"output": output}

ds = datasets.load_dataset('json', data_files='./stanford_alpaca/alpaca_data.json')

alpaca_ds = ds['train'].map(format_alpaca_fn, remove_columns=['instruction', 'input'])
alpaca_ds.to_json("sft_alpaca_data.json")
```

The `response template` corresponding to the above dataset and the `Llama` tokenizer is: `\n### Response:"`.

The same way can be applied to any dataset, with more info can be found [here](https://huggingface.co/docs/trl/main/en/sft_trainer#format-your-input-prompts).


## Supported Models

Current supported and tested models are `Llama2` (7 and 13B configurations have been tested) and `GPTBigCode`.

## Training

### Single GPU
```python
# if you want to use one GPU on multi-gpu machine
export CUDA_VISIBLE_DEVICES=0

python tuning/sft_trainer.py  \
--model_name_or_path $MODEL_PATH  \
--data_path $DATA_PATH  \
--output_dir $OUTPUT_PATH  \
--num_train_epochs 5  \
--per_device_train_batch_size 4  \
--per_device_eval_batch_size 4  \
--gradient_accumulation_steps 4  \
--evaluation_strategy "no"  \
--save_strategy "epoch"  \
--learning_rate 1e-5  \
--weight_decay 0.  \
--warmup_ratio 0.03  \
--lr_scheduler_type "cosine"  \
--logging_steps 1  \
--include_tokens_per_second  \
--packing False  \
--response_template "\n### Response:"  \
--dataset_text_field "output" 

```

### Multiple GPUs with FSDP
```
torchrun \
--nnodes=1 \
--nproc_per_node=8 \ 
--master_port=1234 \
tuning/sft_trainer.py \
--model_name_or_path $MODEL_PATH \
--data_path $DATA_PATH \
--bf16 True \
--output_dir $OUTPUT_PATH \
--num_train_epochs 5 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 4 \
--evaluation_strategy "no" \
--save_strategy "epoch" \
--learning_rate 1e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--fsdp "full_shard auto_wrap" \ 
--fsdp_config tuning/config/fsdp_config.json \
--include_tokens_per_second \
--packing False \
--response_template "\n### Response:" \
--dataset_text_field "output"
```

The above is an example. We would need to tune parameters depending on the model size, data size. The above example has been validated on 8 x A100 80GB.

For `GPTBigCode` models, Hugging Face has enabled Flash v2 and one can simply replace the `'LlamaDecoderLayer'` with `'GPTBigCodeBlock'` for proper sharding of the model.

## Validation

We can use [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) from EleutherAI for evaluating the generated model. For example, for the Llama-13B model, using the above command and the model at the end of Epoch 5, we evaluated MMLU score to be `53.9` compared to base model to be `52.8`.

How to run the validation:
```
pip install -U transformers
pip install -U datasets
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
python main.py 
--model hf-causal 
--model_args pretrained=$MODEL_PATH 
--output_path $OUTPUT_PATH/results.json 
--tasks boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,hendrycksTest-*
```

The above runs several tasks with `hendrycksTest-*` being MMLU.

## More Examples

[Prompt Tuning on Twitter Complaints](examples/prompt_tuning_twitter_complaints/README.md)

