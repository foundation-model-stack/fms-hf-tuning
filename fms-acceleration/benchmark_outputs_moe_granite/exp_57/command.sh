#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export ACCELERATION_FRAMEWORK_CONFIG_FILE=/workspace/fms-acceleration/scripts/benchmarks/../../sample-configurations/moe-scattermoe-granite-ep4-padding-free-foak-sample-configuration.yaml
python -m tuning.sft_trainer --model_name_or_path ibm-research/moe-7b-1b-active-shared-experts --packing False --max_seq_len 4096 --training_data_path benchmark_outputs/data/cache_all.json --use_flash_attn True --response_template '
### Response:' --dataset_text_field output --include_tokens_per_second True --num_train_epochs 1 --gradient_checkpointing True --evaluation_strategy no --save_strategy no --weight_decay 0.01 --warmup_steps 10 --lr_scheduler_type linear --logging_strategy steps --max_steps 100 --learning_rate 5e-5 --torch_dtype bfloat16 --per_device_train_batch_size 8 --logging_steps 1 --adam_epsilon 1e-8 --gradient_accumulation_steps 16 --output_dir benchmark_outputs/exp_57/hf --skip_memory_metrics True