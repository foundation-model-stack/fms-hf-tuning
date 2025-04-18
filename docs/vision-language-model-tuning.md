# Tuning Vision Language Models
Our library also supports full fine tuning and LoRA tuning for vision language models.

## Supported Dataset Format
We support tuning an `image+text` dataset that includes:
- A single text field, formatted using the model’s chat template.
- A single image field, which can contain either a list of images or a single image.

The text must follow the OpenAI conversational data format, which is defined as a list of message objects. Each message object must have two required fields: `role` and `content`:
- `role`: The speaker (e.g., "user" or "assistant").
- `content`: A list of dictionaries, each specifying:
   - `type`: `text` or `image`.
   - `text`: The text content (if applicable).

Example Format:
```json
[
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Who is this?"},
            {"type": "image"}
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Barack Obama"}
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is he famous for?"}
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "He is the 44th President of the United States."}
        ]
    }
]
```

## Processing of dataset

First, each dataset sample is processed by applying the [chat template](https://huggingface.co/docs/transformers/main/en/chat_templating) to the raw text, which formats the conversation as required. Then, the model’s [`processor`](https://huggingface.co/docs/transformers/main/en/processors) takes the formatted text and the corresponding image(s) and converts them into the final input representation (e.g., input_ids, attention masks, etc.) that the model uses for training.

**Note**: `Granite 3.2` and `Llava-1.5` Vision models expect a single image for each dataset sample. If a list of images is provided, only the first image will be used.

## Tuning configurations

Two parameters must be passed to specify which dataset columns to use:
- `dataset_text_field`: The column name that contains the conversational text.
- `dataset_image_field`: The column name that contains the images.

Below is a sample configuration file:
```json
{
  "model_name_or_path": "ibm-granite/granite-vision-3.2-2b", 
  "training_data_path": "HuggingFaceH4/llava-instruct-mix-vsft",
  "dataset_text_field": "messages",
  "dataset_image_field": "images",
  "output_dir": "/app/test",
  "num_train_epochs": 1.0,
  "per_device_train_batch_size": 8,
  "gradient_accumulation_steps": 2,
  "learning_rate": 1e-4,
  "bf16": true,
  "torch_dtype": "bfloat16",
  "use_flash_attn": true,
  "remove_unused_columns": false,
  "dataset_kwargs": {"skip_prepare_dataset": true},
  "gradient_checkpointing": true,
  "gradient_checkpointing_kwargs": {"use_reentrant": false},
  "accelerate_launch_args": { "fsdp_transformer_layer_cls_to_wrap": "GraniteDecoderLayer"}
}
```

## Running the Trainer

You can also run training by calling our trainer module directly using the command line. You can use `python` for single GPU or `accelerate launch` command for multi GPU.
For example:

Command for single GPU:

```sh
python tuning/sft_trainer.py  \
--model_name_or_path $MODEL_PATH  \
--training_data_path $TRAIN_DATA_PATH  \
--output_dir $OUTPUT_PATH  \
--num_train_epochs 5  \
--per_device_train_batch_size 4  \
--gradient_accumulation_steps 1  \
--learning_rate 1e-5  \
--dataset_text_field "messages" \
--dataset_image_field "images"
```

Command for multi GPU:

```sh
accelerate launch \
--num_processes=$NUM_PROCESSORS
--config_file fixtures/accelerate_fsdp_defaults.yaml \
tuning/sft_trainer.py  \
--model_name_or_path $MODEL_PATH  \
--training_data_path $TRAIN_DATA_PATH  \
--output_dir $OUTPUT_PATH  \
--num_train_epochs 5  \
--per_device_train_batch_size 4  \
--gradient_accumulation_steps 1  \
--learning_rate 1e-5  \
--dataset_text_field "messages" \
--dataset_image_field "images"
```

## Tuning Considerations for vision models

Flash Attention 2.0 is not supported by `MllamaForConditionalGeneration` models, thus when running tuning with the `Llama 3.2 Vision Models` set:

```json
"use_flash_attn": false
```
### Multi-GPU Tuning with FSDP:

When running `multi-GPU` tuning with `FSDP`, you need to wrap specific transformer layers. Use the following setting in FSDP config based on your model:

Granite 3.2 Vision Models:
```json
"accelerate_launch_args": { "fsdp_transformer_layer_cls_to_wrap": "GraniteDecoderLayer" }
```

Llava-Next and Llava-1.5 Models:
```json
"accelerate_launch_args": { "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer" }
```

Llava-1.6-Mistral Model:
```json
"accelerate_launch_args": { "fsdp_transformer_layer_cls_to_wrap": "MistralDecoderLayer" }
```

Llama 3.2 Vision Models: No additional configuration is required.

### Gradient Checkpointing:

We recommend running with argument `gradient_checkpointing=True` as enabling this will greatly reduce the memory needed to load and run the model.

When running with gradient checkpointing for the `Llava` and `Granite` vision models, you will need to also set `gradient_checkpointing_kwargs` to not use the activation checkpoint variant that requires reentrant autograd.  

```json
"gradient_checkpointing_kwargs": {"use_reentrant": false}
```

Without setting this, tuning will lead to error:

```sh
RuntimeError: mat2 must be a matrix, got 1-D tensor
RuntimeError: Expected weight to be of same shape as normalized_shape, but got weight of shape [0] and normalized_shape = [1152]
```

### Other arguments:

To prevent default text-only processing and ensure proper handling of multimodal data, we recommend setting:

```json
"remove_unused_columns": false
"dataset_kwargs": {"skip_prepare_dataset": true}
```

When performing LoRA tuning on vision models, you must specify the `target_modules` explicitly, as no defaults are provided.

