# Offline Data Preprocessing

Our library provides a [script](../scripts/offline_data_processing.py) that allows users to perform standalone data preprocessing, independent of tuning/training. This script enables users to process raw datasets, apply basic/advanced data preprocessing, and save the train and validation datasets in Parquet format inside the specified `output_dir`. When the `--num_dataset_shards` argument is specified, the datasets are divided and saved into multiple shards.

Users can pass any data config to this script. The goal of the script is to take the provided data config and generate a dataset that can be used directly for training, without requiring any online processing. As an example see this data config below:

```yaml
dataprocessor:
    type: default
    sampling_stopping_strategy: first_exhausted
    seed: 66
datasets:
  - name: dataset_1
    data_paths:
      - tests/artifacts/testdata/jsonl/twitter_complaints_input_output.jsonl
    data_handlers:
      - name: tokenize_and_apply_input_masking
        arguments:
          remove_columns: all
          batched: false
          fn_kwargs:
            input_field_name: input
            output_field_name: output
```

After preparing the data configuration YAML file, run the script with the following example command to perform offline data preprocessing:   

```
python scripts/offline_data_processing.py \
--data_config_path  /path/to/data_config.yaml \
--model_name_or_path "model_name"  \
--max_seq_length 4096 \
--output_dir /path/to/output/directory  \
--log_level info \
--num_dataset_shards 3
```

Additionally, once the offline data processing is complete, users can leverage the shards stored in `output_dir` for tuning by passing it through the `--training_data_path` flag or passing it via `data_paths` argument in data config yaml, provided they find the sharded datasets beneficial for training.

## Example Usage
### Applying Chat Template

This is a sample use case of the offline processing script being applied to a dataset with a chat template, after which the offline processed dataset is used to train a model.

In this use case, the chat template is applied to a dataset using the `apply_tokenizer_chat_template` handler, followed by additional data transformation handlers. 

**NOTE**: Streaming of the dataset is not supported when running the offline data preprocessing script. Therefore, in the data config, the `streaming` argument should either be set to `False` or left unassigned. 

```yaml
dataprocessor:
  type: default
  sampling_stopping_strategy: first_exhausted
  seed: 66
  streaming: False
  chat_template: |
   {%- for message in messages['messages'] %}
    {%- if message['role'] == 'system' %}
      {{ '<|start_of_role|>system<|end_of_role|>' + message['content'] + '<|end_of_text|>\n' }}
    {%- elif message['role'] == 'user' %}
      {{ '<|start_of_role|>user<|end_of_role|>' + message['content'] + '<|end_of_text|>\n' }}
    {%- elif message['role'] == 'assistant' %}
      {{ '<|start_of_role|>assistant<|end_of_role|>' + message['content'] + '<|end_of_text|>\n' }}
    {%- elif message['role'] == 'tools' %}
      {{ '<|start_of_role|>tools<|end_of_role|>' + message['content'] + '<|end_of_text|>\n' }}
    {%- elif message['role'] == 'tool' %}
      {{ '<|start_of_role|>tool<|end_of_role|>' + message['content'] + '<|end_of_text|>\n' }}
    {%- elif message['role'] == 'documents' %}
      {{ '<|start_of_role|>documents<|end_of_role|>' + message['content'] + '<|end_of_text|>\n' }}
    {%- else %}
      {{ '<|start_of_role|>unknown<|end_of_role|>' + message['content'] + '<|end_of_text|>\n' }} 
    {%- endif %}
   {%- endfor %}
datasets:
 - name: dataset_1
    retain_columns:
     - "formatted_chat"
   data_paths:
    - "/app/arb30_100.jsonl"
   data_handlers:
    - name: apply_tokenizer_chat_template
      arguments:
        fn_kwargs:
          dataset_text_field: "formatted_chat"
    - name: tokenize
      arguments:
        batched: false
        fn_kwargs:
          dataset_text_field: "formatted_chat"
          truncation: False
          max_length: 4096
    - name: skip_large_text
      arguments:
        fn_kwargs:
          column_name: "input_ids"
          max_length: 4096
    - name: retain_columns
      arguments:
        columns:
        - "formatted_chat"
```

Command to run the offline data processing script:

```yaml
python scripts/offline_data_processing.py \
--data_config_path "data_config.yaml" \
--instruction_template "<|start_of_role|>user<|end_of_role|>" \
--max_seq_length "8192" \
--model_name_or_path "/test/models/granite-3.1-8b-instruct" \
--output_dir "/test/data/offline_processing_shards" \
--packing "False" \
--response_template "<|start_of_role|>assistant<|end_of_role|>" \
--split_batches "true" \
--use_flash_attn "true" \
--num_dataset_shards "10"
```

The resulting shards are saved in the directory `/test/data/offline_processing_shards`, as specified by the `--output_dir` argument. These shards can then be used for tuning the model by pointing the `training_data_path` argument to the directory where the shards are stored—in this example, 
`/test/data/offline_processing_shards`.

Command to run the tuning:

```yaml
accelerate launch \
  --num_processes=8 \
  --dynamo_backend="no" \
  --fsdp_auto_wrap_policy="TRANSFORMER_BASED_WRAP" \
  --fsdp_cpu_ram_efficient_loading="true" \
  --fsdp_forward_prefetch="false" \
  --fsdp_offload_params="false" \
  --fsdp_sharding_strategy="HYBRID_SHARD" \
  --fsdp_state_dict_type="FULL_STATE_DICT" \
  --fsdp_sync_module_states="true" \
  --machine_rank="${RANK}" \
  --main_process_ip="${MASTER_ADDR}" \
  --main_process_port="${MASTER_PORT}" \
  --mixed_precision="no" \
  --num_machines="${WORLD_SIZE}" \
  --rdzv_backend="static" \
  --same_network \
  --use_fsdp \
  -m tuning.sft_trainer \
  --training_data_path "/test/data/offline_processing_shards" \
  --adam_beta1="0.9" \
  --adam_beta2="0.98" \
  --adam_epsilon="1e-10" \
  --aim_repo="${AIMSTACK_DB}" \
  --dataloader_drop_last="true" \
  --dataset_text_field="random" \
  --evaluation_strategy="no" \
  --experiment="train-nb-g8b-r26-e0e88b40-dbd8-41ae-a744-c853959495f2" \
  --gradient_accumulation_steps="1" \
  --gradient_checkpointing="true" \
  --include_tokens_per_second="false" \
  --instruction_template="<|start_of_role|>user<|end_of_role|>" \
  --learning_rate="1e-06" \
  --logging_steps="1" \
  --logging_strategy="steps" \
  --lr_scheduler_type="cosine" \
  --max_seq_length="8192" \
  --max_steps="12400" \
  --model_name_or_path="/test/models/granite-3.1-8b-instruct" \
  --num_train_epochs="3" \
  --optim="adamw_torch" \
  --output_dir="/hfcache/data_mixing/data_mixing/wca_summ/run26_rb_mix" \
  --packing="False" \
  --per_device_train_batch_size="32" \
  --response_template="<|start_of_role|>assistant<|end_of_role|>" \
  --save_steps="100" \
  --save_strategy="steps" \
  --split_batches="true" \
  --torch_dtype="bfloat16" \
  --use_flash_attn="true" \
  --use_reentrant="true" \
  --warmup_ratio="0.1" \
  --warmup_steps="200" \
  --weight_decay="0.1"
```