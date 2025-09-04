# Training with fms-hf-tuning

- [Training with fms-hf-tuning](#training-with-fms-hf-tuning)
  - [Single GPU](#single-gpu)
    - [Using pre-processed dataset](#single-gpu)
    - [Using formatter with JSON/JSONL files](#single-gpu)
  - [Multiple GPUs with FSDP](#multiple-gpus-with-fsdp)
  - [Tips on Parameters to Set](#tips-on-parameters-to-set)
    - [Saving checkpoints while training](#saving-checkpoints-while-training-does-not-apply-to-activated-lora)
    - [Saving model after training](#saving-model-after-training)
      - [Ways you can use `save_model_dir` and more tips](#ways-you-can-use-save_model_dir-and-more-tips)
    - [Optimizing writing checkpoints](#optimizing-writing-checkpoints)
    - [Resuming tuning from checkpoints](#resuming-tuning-from-checkpoints)
    - [Setting Gradient Checkpointing](#setting-gradient-checkpointing)
  - [Training MXFP4 quantized with fms-hf-tuning](#training-mxfp4-quantized-models)


## Single GPU

Below example runs fine tuning with the given datasets and model:
1. Using pre-processed dataset for training. 

```bash
# if you want to use one GPU on multi-gpu machine
export CUDA_VISIBLE_DEVICES=0

# MODEL_PATH=meta-llama/Llama-2-7b-hf # Huggingface model id or path to a checkpoint
# TRAIN_DATA_PATH=twitter_complaints.json # Path to the dataset
                  # contains data in single sequence {"output": "### Input: text \n\n### Response: text"}
# OUTPUT_PATH=out # Path to the output folder where the checkpoints are saved

python tuning/sft_trainer.py  \
--model_name_or_path $MODEL_PATH  \
--tokenizer_name_or_path $MODEL_PATH \ # This field is optional and if not specified, tokenizer from model_name_or_path will be used
--training_data_path $TRAIN_DATA_PATH  \
--output_dir $OUTPUT_PATH  \
--num_train_epochs 5  \
--per_device_train_batch_size 4  \
--gradient_accumulation_steps 4  \
--learning_rate 1e-5  \
--response_template "\n### Response:"  \
--dataset_text_field "output"
```

2. Using formatter with JSON/JSONL files

```bash
# if you want to use one GPU on multi-gpu machine
export CUDA_VISIBLE_DEVICES=0

# MODEL_PATH=meta-llama/Llama-2-7b-hf # Huggingface model id or path to a checkpoint
# TRAIN_DATA_PATH=twitter_complaints.json # Path to the dataset
                  # contains data in form of [{"input": text , "output": text}]
# OUTPUT_PATH=out # Path to the output folder where the checkpoints are saved

python tuning/sft_trainer.py  \
--model_name_or_path $MODEL_PATH  \
--tokenizer_name_or_path $MODEL_PATH \ # This field is optional and if not specified, tokenizer from model_name_or_path will be used
--training_data_path $TRAIN_DATA_PATH  \
--output_dir $OUTPUT_PATH  \
--num_train_epochs 5  \
--per_device_train_batch_size 4  \
--gradient_accumulation_steps 4  \
--learning_rate 1e-5  \
--response_template "\n## Label:"  \
--data_formatter_template: "### Input: {{input}} \n\n## Label: {{output}}"

```

## Multiple GPUs with FSDP

The recommendation is to use [huggingface accelerate](https://huggingface.co/docs/accelerate/en/index) to launch multi-gpu jobs, in particular when using FSDP:
- `accelerate` is written on top of [`torch.distributed.run`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/run.py).
- `accelerate launch` CLI highly similar to `torchrun`, spawns multiple jobs (one for each gpu).
- tightly integrated with [huggingface Trainer](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py).

`accelerate launch` CLI to be run with specific command line arguments, see example below. Default arguments handled by passing in a 
`--config_file` argument; see [reference docs](https://huggingface.co/docs/accelerate/en/package_reference/cli#accelerate-launch) and [fixtures/accelerate_fsdp_defaults.yaml](./fixtures/accelerate_fsdp_defaults.yaml) for sample defaults.

Below example runs multi-GPU fine tuning on 8 GPUs with FSDP:
```bash
# Please set the environment variables:
# MASTER_PORT=1234 # The port at which the process with rank 0 listens to and should be set to an unused port
# MODEL_PATH=meta-llama/Llama-2-7b-hf # Huggingface model id or path to a checkpoint
# TRAIN_DATA_PATH=twitter_complaints.json # Path to the training dataset
# OUTPUT_PATH=out # Path to the output folder where the checkpoints are saved

accelerate launch \
--config_file fixtures/accelerate_fsdp_defaults.yaml \
--num_processes=8 \ 
--main_process_port=$MASTER_PORT \
tuning/sft_trainer.py \
--model_name_or_path $MODEL_PATH \
--training_data_path $TRAIN_DATA_PATH \
--torch_dtype bfloat16 \
--output_dir $OUTPUT_PATH \
--num_train_epochs 5 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--learning_rate 1e-5 \
--response_template "\n### Response:" \
--dataset_text_field "output" \
--tokenizer_name_or_path $MODEL_PATH  # This field is optional and if not specified, tokenizer from model_name_or_path will be used
```

To summarize you can pick either python for single-GPU jobs or use accelerate launch for multi-GPU jobs. The following tuning techniques can be applied:

# Tips on Parameters to Set

## Saving checkpoints while training (does not apply to Activated LoRA)

By default, [`save_strategy`](tuning/config/configs.py) is set to `"epoch"` in the TrainingArguments. This means that checkpoints will be saved on each epoch. This can also be set to `"steps"` to save on every `"save_steps"` or `"no"` to not save any checkpoints.

Checkpoints are saved to the given `output_dir`, which is a required field. If `save_strategy="no"`, the `output_dir` will only contain the training logs with loss details.

A useful flag to set to limit the number of checkpoints saved is [`save_total_limit`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.save_total_limit). Older checkpoints are deleted from the `output_dir` to limit the number of checkpoints, for example, if `save_total_limit=1`, this will only save the last checkpoint. However, while tuning, two checkpoints will exist in `output_dir` for a short time as the new checkpoint is created and then the older one will be deleted. If the user sets a validation dataset and [`load_best_model_at_end`](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments.load_best_model_at_end), then the best checkpoint will be saved.

## Saving model after training

`save_model_dir` can optionally be set to save the tuned model using `SFTTrainer.save_model()`. This can be used in tandem with `save_strategy="no"` to only save the designated checkpoint and not any intermediate checkpoints, which can help to save space.

`save_model_dir` can be set to a different directory than `output_dir`. If set to the same directory, the designated checkpoint, training logs, and any intermediate checkpoints will all be saved to the same directory as seen below.

<details>
<summary>Ways you can use `save_model_dir` and more tips:</summary>

For example, if `save_model_dir` is set to a sub-directory of `output_dir`and `save_total_limit=1` with LoRA tuning, the directory would look like:

```sh
$ ls /tmp/output_dir/
checkpoint-35  save_model_dir  training_logs.jsonl

$ ls /tmp/output_dir/save_model_dir/
README.md	     adapter_model.safetensors	special_tokens_map.json  tokenizer.model	training_args.bin
adapter_config.json  added_tokens.json		tokenizer.json		 tokenizer_config.json
```

Here is an fine tuning example of how the directory would look if `output_dir` is set to the same value as `save_model_dir` and `save_total_limit=2`. Note the checkpoint directories as well as the `training_logs.jsonl`:

```sh
$ ls /tmp/same_dir

added_tokens.json	model-00001-of-00006.safetensors  model-00006-of-00006.safetensors  tokenizer_config.json
checkpoint-16		model-00002-of-00006.safetensors  model.safetensors.index.json	    training_args.bin
checkpoint-20		model-00003-of-00006.safetensors  special_tokens_map.json	    training_logs.jsonl
config.json		model-00004-of-00006.safetensors  tokenizer.json
generation_config.json	model-00005-of-00006.safetensors  tokenizer.model
```

</details>

## Optimizing writing checkpoints
Writing models to Cloud Object Storage (COS) is an expensive operation. Saving model checkpoints to a local directory causes much faster training times than writing to COS. You can use `output_dir` and `save_model_dir` to control which type of storage you write your checkpoints and final model to.

You can set `output_dir` to a local directory and set `save_model_dir` to COS to save time on write operations while ensuring checkpoints are saved.

In order to achieve the fastest train time, set `save_strategy="no"`, as saving no checkpoints except for the final model will remove intermediate write operations all together.

## Resuming tuning from checkpoints
If the output directory already contains checkpoints, tuning will automatically resume from the latest checkpoint in the directory specified by the `output_dir` flag. To start tuning from scratch and ignore existing checkpoints, set the `resume_from_checkpoint` flag to False.

You can also use the resume_from_checkpoint flag to resume tuning from a specific checkpoint by providing the full path to the desired checkpoint as a string. This flag is passed as an argument to the [trainer.train()](https://github.com/huggingface/transformers/blob/db70426854fe7850f2c5834d633aff637f14772e/src/transformers/trainer.py#L1901) function of the SFTTrainer.

## Setting Gradient Checkpointing

Training large models requires the usage of a lot of GPU memory. To reduce memory usage while training, consider setting the [`gradient_checkpointing`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.gradient_checkpointing) flag. 

Gradient Checkpointing is a method that stores only certain intermediate activations during the backward pass for recomputation. This avoids storing all of the intermediate activations from the forward pass, thus saving memory. The resulting reduced memory costs allow fitting larger models on the same GPU, with the tradeoff of a ~20% increase in the time required to fully train the model. More information about Gradient Checkpointing can be found in [this paper](https://arxiv.org/abs/1604.06174), as well as [here](https://github.com/cybertronai/gradient-checkpointing?tab=readme-ov-file#how-it-works).

To enable this feature, add the `--gradient_checkpointing` flag as an argument when calling `sft_trainer`.

## Training MXFP4 Quantized Models

MXFP4 Quantized models like [gpt-oss](https://huggingface.co/openai/gpt-oss-120b) series models can be tuned by passing two extra parameters.

```
--quantization_method mxfp4 --dequantize True \
--flash_attn_implementation="kernels-community/vllm-flash-attn3"
```

1. Quantization method `mxfp4` and `dequantize=True` tells the code to dequantize the model and load it in `bf16` mode as training is not supported for `mxfp4`.
Even if support for training in `mxfp4` mode goes live it will be supported only on Hopper and above series of GPUs so users will need to specify `dequantize=False` when training on older GPUs e.g. `A100`s.

2. Flash attention 3 is supported by custom kernels support so users need to specify the correct argument else our code will fallback to `flash attention 2`

Full command for training GPT-OSS models can be like this - 

```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; \
rm -rf ~/gpt-oss-120b-multilingual-reasoner; \
accelerate launch \
--config_file ~/fsdp_config.yaml \
-m tuning.sft_trainer \
--model_name_or_path "openai/gpt-oss-120b" \
--output_dir ~/gpt-oss-120b-multilingual-reasoner \
--gradient_accumulation_steps 1 \
--per_device_train_batch_size 2 \
--num_train_epochs 1 \
--torch_dtype bfloat16 \
--learning_rate 2e-4 \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine_with_min_lr" \
--lr_scheduler_kwargs '{"min_lr_rate": 0.1}' \
--max_seq_length 4096 \
--logging_steps 1 \
--data_config ~/data_config.yaml \
--gradient_checkpointing True \
--peft_method lora \
--lora_r 8\
--lora_alpha 16 \
--lora_dropout 0.0 \
--target_modules "all-linear" \
--quantization_method mxfp4 --dequantize True \
--use_flash_attn True \
--tracker aim --aim_repo ~/aimrepo --experiment "gpt-oss-120b-lora-tuning-fa3-attn-torch-2.8" \
--flash_attn_implementation="kernels-community/vllm-flash-attn3"
```

With fsdp config passed to accelerate as - 
```
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD 
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_cpu_ram_efficient_loading: true
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
mixed_precision: bf16
machine_rank: 0
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
```

And [data config](./advanced-data-preprocessing.md#data-config) for [HuggingFace Reasoner dataset](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking) looks like 

```
dataprocessor:
    type: default
datasets:
  - name: dataset_1
    data_paths:
      - "HuggingFaceH4/Multilingual-Thinking"
    data_handlers:
      - name: tokenize_and_apply_chat_template_with_masking
        arguments:
          remove_columns: all
          fn_kwargs:
            conversation_column_name: "messages"
```