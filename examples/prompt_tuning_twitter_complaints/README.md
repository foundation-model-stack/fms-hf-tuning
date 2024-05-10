## Prompt Tuning on Twitter Complaints

This example follows [HF's Prompt Tuning example](https://huggingface.co/docs/peft/main/en/task_guides/clm-prompt-tuning)
which demonstrates how to apply our tuning to any HF example.

### Dataset

The first step is to make a `SFTTrainer`-compatible dataset. 

Let's firstly preprocess the data (copied from [here](https://huggingface.co/docs/peft/main/en/task_guides/clm-prompt-tuning#load-dataset)):
```python
from datasets import load_dataset
dataset_name = "twitter_complaints"
dataset = load_dataset("ought/raft", dataset_name)
classes = [k.replace("_", " ") for k in dataset["train"].features["Label"].names]
dataset = dataset.map(
    lambda x: {"text_label": [classes[label] for label in x["Label"]]},
    batched=True,
    num_proc=1,
)
```
Then let's make it `SFTTrainer` style (following instruction [here](https://huggingface.co/docs/trl/main/en/sft_trainer#format-your-input-prompts)):
```python
dataset = dataset["train"].map(
    lambda x: {"output": f"### Text: {x['Tweet text']}\n\n### Label: {x['text_label']}"},
)
dataset.to_json("twitter_complaints.json")
```

### Prompt Tuning
We will switch our PEFT method from LORA to Prompt Tuning (pt)
```bash
# Please set the environment variables:
# MASTER_PORT=1234 # The port at which the process with rank 0 listens to and should be set to an unused port
# MODEL_PATH=meta-llama/Llama-2-7b-hf # Huggingface model id or path to a checkpoint
# TRAIN_DATA_PATH=twitter_complaints.json # Path to the training dataset
# OUTPUT_PATH=out # Path to the output folder where the checkpoints are saved


accelerate launch \
--main_process_port $MASTER_PORT \
--config_file fixtures/accelerate_fsdp_defaults.yaml \
tuning/sft_trainer.py  \
--model_name_or_path $MODEL_PATH  \
--training_data_path $TRAIN_DATA_PATH  \
--output_dir $OUTPUT_PATH  \
--peft_method pt \
--torch_dtype bfloat16 \
--tokenizer_name_or_path $MODEL_PATH  \
--num_train_epochs 5  \
--per_device_train_batch_size 1  \
--per_device_eval_batch_size 1  \
--gradient_accumulation_steps 1  \
--evaluation_strategy "no"  \
--save_strategy "epoch"  \
--learning_rate 1e-5  \
--weight_decay 0.  \
--warmup_ratio 0.03  \
--lr_scheduler_type "cosine"  \
--logging_steps 1  \
--include_tokens_per_second  \
--packing False  \
--response_template "\n### Label:"  \
--dataset_text_field "output" 
```