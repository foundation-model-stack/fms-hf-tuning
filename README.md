# FMS HF Tuning

- [Installation](#installation)
- [Data format support](#data-support)
- [Supported Models](#supported-models)
- [Training](#training)
  - [Single GPU](#single-gpu)
  - [Multiple GPUs with FSDP](#multiple-gpus-with-fsdp)
  - [Tips on Parameters to Set](#tips-on-parameters-to-set)
- [Tuning Techniques](#tuning-techniques)
  - [LoRA Tuning Example](#lora-tuning-example)
  - [GPTQ-LoRA with AutoGPTQ Tuning Example](#gptq-lora-with-autogptq-tuning-example)
  - [Prompt Tuning](#prompt-tuning)
  - [Fine Tuning](#fine-tuning)
  - [FMS Acceleration](#fms-acceleration)
- [Inference](#inference)
  - [Running a single example](#running-a-single-example)
  - [Running multiple examples](#running-multiple-examples)
  - [Inference Results Format](#inference-results-format)
  - [Changing the Base Model for Inference](#changing-the-base-model-for-inference)
- [Validation](#validation)
- [Trainer Controller Framework](#trainer-controller-framework)
- [Experiment Tracking](#experiment-tracking)
- [More Examples](#more-examples)

This repo provides basic tuning scripts with support for specific models. The repo relies on Hugging Face `SFTTrainer` and PyTorch FSDP. Our approach to tuning is:
1. Models are loaded from Hugging Face `transformers` or the [foundation-model-stack](https://github.com/foundation-model-stack/foundation-model-stack) -- models are either optimized to use `Flash Attention v2` directly or through `SDPA`
2. Hugging Face `SFTTrainer` for the training loop
3. `FSDP` as the backend for training

## Installation

### Basic Installation

```
pip install fms-hf-tuning
```

### Using FlashAttention

> Note: After installing, if you wish to use [FlashAttention](https://github.com/Dao-AILab/flash-attention), then you need to install these requirements:
```
pip install fms-hf-tuning[dev]
pip install fms-hf-tuning[flash-attn]
```
[FlashAttention](https://github.com/Dao-AILab/flash-attention) requires the [CUDA Toolit](https://developer.nvidia.com/cuda-toolkit) to be pre-installed.

### Using FMS-Acceleration

If you wish to use [fms-acceleration](https://github.com/foundation-model-stack/fms-acceleration), you need to install it. 
```
pip install fms-hf-tuning[fms-accel]
```
`fms-acceleration` is a collection of plugins that packages that accelerate fine-tuning / training of large models, as part of the `fms-hf-tuning` suite. For more details see [this section below](#fms-acceleration).

### Using Experiment Trackers

To use experiment tracking with popular tools like [Aim](https://github.com/aimhubio/aim), note that some trackers are considered optional dependencies and can be installed with the following command:
```
pip install fms-hf-tuning[aim]
```
For more details on how to enable and use the trackers, Please see, [the experiment tracking section below](#experiment-tracking).

## Data Support
Users can pass training data in a single file using the `--training_data_path` argument along with other arguments required for various [use cases](#use-cases-supported-with-training_data_path-argument) (see details below) and the file can be in any of the [supported formats](#supported-data-formats). Alternatively, you can use our powerful [data preprocessing backend](./docs/advanced-data-preprocessing.md) to preprocess datasets on the fly.


Below, we mention the list of supported data usecases via `--training_data_path` argument. For details of our advanced data preprocessing see more details in [Advanced Data Preprocessing](./docs/advanced-data-preprocessing.md).

## Supported Data Formats
We support the following data formats via `--training_data_path` argument

Data Format | Tested Support
------------|---------------
JSON        |   ✅
JSONL       |   ✅
PARQUET     |   ✅
ARROW       |   ✅

## Use cases supported with `training_data_path` argument

### 1. Data formats with a single sequence and a specified response_template to use for masking on completion.

#### 1.1 Pre-process the dataset
 Pre-process the dataset to contain a single sequence of each data instance containing input + response. The trainer is configured to expect a `response template` as a string. For example, if one wants to prepare the `alpaca` format data to feed into this trainer, it is quite easy and can be done with the following code.

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

Once the data is converted using the formatting function, pass the `dataset_text_field` containing the single sequence to the trainer. 

#### 1.2 Format the dataset on the fly
   Pass a dataset and a `data_formatter_template` to use the formatting function on the fly while tuning. The template should specify fields of the dataset with `{{field}}`. While tuning, the data will be converted to a single sequence using the template. Data fields can contain alpha-numeric characters, spaces and the following special symbols - "." , "_", "-".  

Example: Train.json
`[{ "input" : <text>,
    "output" : <text>,
  },
 ...
]`  
data_formatter_template: `### Input: {{input}} \n\n##Label: {{output}}`  

Formatting will happen on the fly while tuning. The keys in template should match fields in the dataset file. The `response template` corresponding to the above template will need to be supplied. in this case, `response template` = `\n## Label:`.

##### In conclusion, if using the reponse_template and single sequence, either the `data_formatter_template` argument or `dataset_text_field` needs to be supplied to the trainer.

### 2. Dataset with input and output fields (no response template)

  Pass a [supported dataset](#supported-data-formats) containing fields `"input"` with source text and `"output"` with class labels. Pre-format the input as you see fit. The output field will simply be concatenated to the end of input to create single sequence, and input will be masked.

  The `"input"` and `"output"` field names are mandatory and cannot be changed. 

Example: For a JSON dataset like, `Train.jsonl`

```
{"input": "### Input: Colorado is a state in USA ### Output:", "output": "USA : Location"} 
{"input": "### Input: Arizona is also a state in USA ### Output:", "output": "USA : Location"}
```

### 3. Chat Style Single/Multi turn datasets

  Pass a dataset containing single/multi turn chat dataset. Your dataset could follow this format:

```
$ head -n 1 train.jsonl
{"messages": [{"content": "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.", "role": "system"}, {"content": "Look up a word that rhymes with exist", "role": "user"}, {"content": "I found a word that rhymes with \"exist\":\n1\\. Mist", "role": "assistant"}], "group": "lab_extension", "dataset": "base/full-extension", "metadata": "{\"num_turns\": 1}"}
```

This format supports both single and multi-turn chat scenarios.

The chat template used to render the dataset will default to `tokenizer.chat_template` from the model's tokenizer configuration. This can be overridden using the `--chat_template <chat-template-string>` argument. For example, models like [ibm-granite/granite-3.0-8b-instruct](https://huggingface.co/ibm-granite/granite-3.0-8b-instruct), which include a [chat template](https://huggingface.co/ibm-granite/granite-3.0-8b-instruct/blob/e0a466fb25b9e07e9c2dc93380a360189700d1f8/tokenizer_config.json#L188) in their `tokenizer_config.json`, do not require users to provide a chat template to process the data.

Users do need to pass `--response_template` and `--instruction_template` which are pieces of text representing start of
`assistant` and `human` response inside the formatted chat template.
For the [granite model above](https://huggingface.co/ibm-granite/granite-3.0-8b-instruct/blob/main/tokenizer_config.json#L188) for example, the values shall be.
```
--instruction_template "<|start_of_role|>user<|end_of_role|>"
--response_template "<|start_of_role|>assistant<|end_of_role|>"
```

The code internally uses [`DataCollatorForCompletionOnlyLM`](https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py#L93) to perform masking of text ensuring model learns only on the `assistant` responses for both single and multi turn chat.

### 3. Pre tokenized datasets.

Users can also pass a pretokenized dataset (containing `input_ids` and `labels` columns) as `--training_data_path` argument e.g.

```
python tuning/sft_trainer.py ... --training_data_path twitter_complaints_tokenized_with_maykeye_tinyllama_v0.arrow
```

### 4. Advanced data preprocessing.

For advanced data preprocessing support including mixing and custom preprocessing of datasets please see [this document](./docs/advanced-data-preprocessing.md).

## Supported Models

- For each tuning technique, we run testing on a single large model of each architecture type and claim support for the smaller models. For example, with QLoRA technique, we tested on granite-34b GPTBigCode and claim support for granite-20b-multilingual.

- LoRA Layers supported : All the linear layers of a model + output `lm_head` layer. Users can specify layers as a list or use `all-linear` as a shortcut. Layers are specific to a model architecture and can be specified as noted [here](https://github.com/foundation-model-stack/fms-hf-tuning?tab=readme-ov-file#lora-tuning-example)

- Legend:

  ✅ Ready and available 

  ✔️ Ready and available - compatible architecture (*see first bullet point above)

  🚫 Not supported

  ? May be supported, but not tested

Model Name & Size  | Model Architecture | Full Finetuning | Low Rank Adaptation (i.e. LoRA) | qLoRA(quantized LoRA) | 
-------------------- | ---------------- | --------------- | ------------------------------- | --------------------- |
Granite PowerLM 3B   | GraniteForCausalLM | ✅* | ✅* | ✅* |
Granite 3.0 2B       | GraniteForCausalLM | ✔️* | ✔️* | ✔️* |
Granite 3.0 8B       | GraniteForCausalLM | ✅* | ✅* | ✔️ |
GraniteMoE 1B        | GraniteMoeForCausalLM  | ✅ | ✅** | ? |
GraniteMoE 3B        | GraniteMoeForCausalLM  | ✅ | ✅** | ? |
Granite 3B           | LlamawithCausalLM      | ✅ | ✔️  | ✔️ | 
Granite 8B           | LlamawithCausalLM      | ✅ | ✅ | ✅ |
Granite 13B          | GPTBigCodeForCausalLM  | ✅ | ✅ | ✔️  | 
Granite 20B          | GPTBigCodeForCausalLM  | ✅ | ✔️  | ✔️  | 
Granite 34B          | GPTBigCodeForCausalLM  | 🚫 | ✅ | ✅ | 
Llama3.1-8B          | LLaMA 3.1              | ✅*** | ✔️ | ✔️ |  
Llama3.1-70B(same architecture as llama3) | LLaMA 3.1 | 🚫 - same as Llama3-70B | ✔️  | ✔️ | 
Llama3.1-405B                             | LLaMA 3.1 | 🚫 | 🚫 | ✅ | 
Llama3-8B                                 | LLaMA 3   | ✅ | ✅ | ✔️ |  
Llama3-70B                                | LLaMA 3   | 🚫 | ✅ | ✅ |
aLLaM-13b                                 | LlamaForCausalLM |  ✅ | ✅ | ✅ |
Mixtral 8x7B                              | Mixtral   | ✅ | ✅ | ✅ |
Mistral-7b                                | Mistral   | ✅ | ✅ | ✅ |  
Mistral large                             | Mistral   | 🚫 | 🚫 | 🚫 | 

(*) - Supported with `fms-hf-tuning` v2.0.1 or later

(**) - Supported for q,k,v,o layers . `all-linear` target modules does not infer on vLLM yet.

(***) - Supported from platform up to 8k context length - same architecture as llama3-8b

## Training

### Single GPU

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
--data_formatter_template: "### Input: {{input}} \n\n##Label: {{output}}"

```

### Multiple GPUs with FSDP

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
--main_process_port $MASTER_PORT \
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

### Tips on Parameters to Set

#### Saving checkpoints while training

By default, [`save_strategy`](tuning/config/configs.py) is set to `"epoch"` in the TrainingArguments. This means that checkpoints will be saved on each epoch. This can also be set to `"steps"` to save on every `"save_steps"` or `"no"` to not save any checkpoints.

Checkpoints are saved to the given `output_dir`, which is a required field. If `save_strategy="no"`, the `output_dir` will only contain the training logs with loss details.

A useful flag to set to limit the number of checkpoints saved is [`save_total_limit`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.save_total_limit). Older checkpoints are deleted from the `output_dir` to limit the number of checkpoints, for example, if `save_total_limit=1`, this will only save the last checkpoint. However, while tuning, two checkpoints will exist in `output_dir` for a short time as the new checkpoint is created and then the older one will be deleted. If the user sets a validation dataset and [`load_best_model_at_end`](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments.load_best_model_at_end), then the best checkpoint will be saved.

#### Saving model after training

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

#### Optimizing writing checkpoints
Writing models to Cloud Object Storage (COS) is an expensive operation. Saving model checkpoints to a local directory causes much faster training times than writing to COS. You can use `output_dir` and `save_model_dir` to control which type of storage you write your checkpoints and final model to.

You can set `output_dir` to a local directory and set `save_model_dir` to COS to save time on write operations while ensuring checkpoints are saved.

In order to achieve the fastest train time, set `save_strategy="no"`, as saving no checkpoints except for the final model will remove intermediate write operations all together.

#### Resuming tuning from checkpoints
If the output directory already contains checkpoints, tuning will automatically resume from the latest checkpoint in the directory specified by the `output_dir` flag. To start tuning from scratch and ignore existing checkpoints, set the `resume_from_checkpoint` flag to False.

You can also use the resume_from_checkpoint flag to resume tuning from a specific checkpoint by providing the full path to the desired checkpoint as a string. This flag is passed as an argument to the [trainer.train()](https://github.com/huggingface/transformers/blob/db70426854fe7850f2c5834d633aff637f14772e/src/transformers/trainer.py#L1901) function of the SFTTrainer.

## Tuning Techniques:

### LoRA Tuning Example

Set `peft_method` to `"lora"`. You can additionally pass any arguments from [LoraConfig](https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/tuning/config/peft_config.py#L21).
```py
# Args you can pass
r: int =8 
lora_alpha: int = 32
target_modules: List[str] = field(
  default=None,
  metadata={
        "help": "The names of the modules to apply LORA to. LORA selects modules which either \
        completely match or "
        'end with one of the strings. If the value is ["all-linear"], \
        then LORA selects all linear and Conv1D '
        "modules except for the output layer."
  },
)
bias = "none"
lora_dropout: float = 0.05
```
Example command to run:

```bash
python tuning/sft_trainer.py \
--model_name_or_path $MODEL_PATH \
--tokenizer_name_or_path $MODEL_PATH \ # This field is optional and if not specified, tokenizer from model_name_or_path will be used
--training_data_path $TRAIN_DATA_PATH \
--output_dir $OUTPUT_PATH \
--num_train_epochs 40 \
--per_device_train_batch_size 4 \
---learning_rate 1e-4 \
--response_template "\n### Label:" \
--dataset_text_field "output" \
--peft_method "lora" \
--r 8 \
--lora_dropout 0.05 \
--lora_alpha 16 \
--target_modules c_attn c_proj
```

Equally you can pass in a JSON configuration for running tuning. See [build doc](./build/README.md) for more details. The above can also be passed in as JSON:
```json
{
    "model_name_or_path": $MODEL_PATH,
    "training_data_path": $TRAIN_DATA_PATH,
    "output_dir": $OUTPUT_PATH,
    "num_train_epochs": 40.0,
    "per_device_train_batch_size": 4,
    "learning_rate": 1e-4,
    "response_template": "\n### Label:",
    "dataset_text_field": "output",
    "peft_method": "lora",
    "r": 8,
    "lora_dropout": 0.05,
    "lora_alpha": 16,
    "target_modules": ["c_attn", "c_proj"]
}
```

Notice the `target_modules` are the names of the modules to apply the adapter to.
- If this is specified, only the modules with the specified names will be replaced. When passing a list of strings, either an exact match will be performed or it is checked if the name of the module ends with any of the passed strings. If this is specified as `all-linear`, then all linear/Conv1D modules are chosen, excluding the output layer. If this is specified as `lm_head` which is an output layer, the `lm_head` layer will be chosen. See the Note of this [section](#recommended-target-modules-per-model-architecture) on recommended target modules by model architecture.
- If this is not specified, modules will be chosen according to the model architecture. If the architecture is not known, an error will be raised — in this case, you should specify the target modules manually. See [HuggingFace docs](https://huggingface.co/docs/peft/en/package_reference/lora#peft.LoraConfig) for more details.

#### How to get list of LoRA target_modules of a model
For each model, the `target_modules` will depend on the type of model architecture. You can specify linear or attention layers to `target_modules`. To obtain list of `target_modules` for a model:

```py
from transformers import AutoModelForCausalLM
# load the model
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
# see the module list
model.modules

# to get just linear layers
import re
model_modules = str(model.modules)
pattern = r'\((\w+)\): Linear'
linear_layer_names = re.findall(pattern, model_modules)

names = []
for name in linear_layer_names:
    names.append(name)
target_modules = list(set(names))
```

For example for LLaMA model the modules look like:
```
<bound method Module.modules of LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 4096, padding_idx=0)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)>
```

You can specify attention or linear layers. With the CLI, you can specify layers with `--target_modules "q_proj" "v_proj" "k_proj" "o_proj"` or `--target_modules "all-linear"`.

#### Recommended target modules per model architecture 
As per [LoRA paper](https://arxiv.org/pdf/2106.09685), section 4.2 , by using the query and value projection matrices, we can achieve reasonable quality with efficient GPU utilization. Hence, while thinking about what LoRA adapters to specify, we recommend starting with query and value matrices. You could also refer to the defaults specified by PEFT library for popular model architectures in section [TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING](https://github.com/huggingface/peft/blob/7b1c08d2b5e13d3c99b7d6ee83eab90e1216d4ba/src/peft/utils/constants.py#L70) as a good starting point.

<details>

<summary>How to specify lm_head as a target module</summary>

Since `lm_head` is an output layer, it will **not** be included as a target module if you specify `all-linear`. You can, however, specify to apply the LoRA adapter to the `lm_head` layer by explicitly naming it in the `target_modules` arg.

**NOTE**: Specifying `["lm_head", "all-linear"]` will not tune the `lm_head` layer, but will run the equivalent of `["all-linear"]`. To include `lm_head`, you must explicitly specify all of the layers to tune on. Using the example of the Llama model above, you would need to list `"q_proj" "v_proj" "k_proj" "o_proj" "lm_head"` to tune the all linear layers including `lm_head`. These 5 layers will be produced in the LoRA adapter.

Example 1: 
```json
{
    "target_modules": ["lm_head"] // this produces lm_head layer only
}
```

Example 2:
```json
{
    "target_modules": ["lm_head", "c_proj", "c_attn", "c_fc"] // this produces lm_head, c_proj, c_attn and c_fc layers 
}
```

Example 3:
```json
{
    "target_modules": ["lm_head", "all-linear"] // this produces the equivalent of all-linear only, no lm_head
}
```

</details>

#### Post-processing needed for inference on VLLM

In order to run inference of LoRA adapters on vLLM, any new token embeddings added while tuning needs to be moved out of 'adapters.safetensors' to a new file 'new_embeddings.safetensors'. The 'adapters.safetensors' should only have LoRA weights and should not have modified embedding vectors. This is a requirement to support vLLM's paradigm that one base model can serve multiple adapters. New token embedding vectors are appended to the embedding matrix read from the base model by vLLM.

To do this postprocessing, the tuning script sft_trainer.py will generate a file 'added_tokens_info.json' with model artifacts. After tuning, you can run script 'post_process_adapters_vLLM.py' :

```bash
# model_path: Path to saved model artifacts which has file 'added_tokens_info.json'
# output_model_path: Optional. If you want to store modified \
#    artifacts in a different directory rather than modify in-place.
python scripts/post_process_adapters_vLLM.py \
--model_path "/testing/tuning/output/post-process-LoRA-saved" \
--output_model_path "/testing/tuning/output/post-process-LoRA-modified"
```

<details>
<summary> Alternatively, if using SDK :</summary>

```bash
# function in tuning/utils/merge_model_utils.py
post_process_vLLM_adapters_new_tokens(
    path_to_checkpoint="/testing/tuning/output/post-process-LoRA-saved",
    modified_checkpoint_path=None,
    num_added_tokens=1,
)
# where num_added_tokens is returned by sft_trainer.train()
```
</details>

_________________________


### GPTQ-LoRA with AutoGPTQ Tuning Example

This method is similar to LoRA Tuning, but the base model is a quantized model. We currently only support GPTQ-LoRA model that has been quantized with 4-bit AutoGPTQ technique. Bits-and-Bytes (BNB) quantized LoRA is not yet enabled.
The qLoRA tuning technique is enabled via the [fms-acceleration](https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/README.md#fms-acceleration) package.
You can see details on a sample configuration of Accelerated GPTQ-LoRA [here](https://github.com/foundation-model-stack/fms-acceleration/blob/main/sample-configurations/accelerated-peft-autogptq-sample-configuration.yaml).


To use GPTQ-LoRA technique, you can set the `quantized_lora_config` defined [here](https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/tuning/config/acceleration_configs/quantized_lora_config.py). See the Notes section of FMS Acceleration doc [below](https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/README.md#fms-acceleration) for usage. The only kernel we are supporting currently is `triton_v2`.

In addition, LoRA tuning technique is required to be used, set `peft_method` to `"lora"` and pass any arguments from [LoraConfig](https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/tuning/config/peft_config.py#L21).

Example command to run:

```bash
python tuning/sft_trainer.py \
--model_name_or_path $MODEL_PATH \
--tokenizer_name_or_path $MODEL_PATH \ # This field is optional and if not specified, tokenizer from model_name_or_path will be used
--training_data_path $TRAIN_DATA_PATH \
--output_dir $OUTPUT_PATH \
--num_train_epochs 40 \
--per_device_train_batch_size 4 \
--learning_rate 1e-4 \
--response_template "\n### Label:" \
--dataset_text_field "output" \
--peft_method "lora" \
--r 8 \
--lora_dropout 0.05 \
--lora_alpha 16 \
--target_modules c_attn c_proj \
--auto_gptq triton_v2 \ # setting quantized_lora_config 
--torch_dtype float16 \ # need this for triton_v2
--fp16 \ # need this for triton_v2
```

Equally you can pass in a JSON configuration for running tuning. See [build doc](./build/README.md) for more details. The above can also be passed in as JSON:

```json
{
    "model_name_or_path": $MODEL_PATH,
    "training_data_path": $TRAIN_DATA_PATH,
    "output_dir": $OUTPUT_PATH,
    "num_train_epochs": 40.0,
    "per_device_train_batch_size": 4,
    "learning_rate": 1e-4,
    "response_template": "\n### Label:",
    "dataset_text_field": "output",
    "peft_method": "lora",
    "r": 8,
    "lora_dropout": 0.05,
    "lora_alpha": 16,
    "target_modules": ["c_attn", "c_proj"],
    "auto_gptq": ["triton_v2"], // setting quantized_lora_config
    "torch_dtype": "float16", // need this for triton_v2
    "fp16": true // need this for triton_v2
}
```

Similarly to LoRA, the `target_modules` are the names of the modules to apply the adapter to. See the LoRA [section](#lora-tuning-example) on `target_modules` for more info.

Note that with LoRA tuning technique, setting `all-linear` on `target_modules` returns linear modules. And with qLoRA tuning technique, `all-linear` returns all quant linear modules, excluding `lm_head`.

_________________________

### Prompt Tuning:

Specify `peft_method` to `'pt'` . You can additionally pass any arguments from [PromptTuningConfig](https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/tuning/config/peft_config.py#L63).
```py
# prompt_tuning_init can be either "TEXT" or "RANDOM"
prompt_tuning_init: str = "TEXT"
num_virtual_tokens: int = 8
# prompt_tuning_init_text only applicable if prompt_tuning_init= "TEXT"
prompt_tuning_init_text: str = "Classify if the tweet is a complaint or not:"
tokenizer_name_or_path: str = "llama-7b-hf"
```

Example command you can run:  

```bash
python tuning/sft_trainer.py  \
--model_name_or_path $MODEL_PATH  \
--training_data_path $TRAIN_DATA_PATH  \
--output_dir $OUTPUT_PATH  \
--num_train_epochs 5  \
--per_device_train_batch_size 1  \
--learning_rate 0.03  \
--response_template "\n### Label:"  \
--dataset_text_field "output" \
--peft_method pt \
--tokenizer_name_or_path $MODEL_PATH \ # This field is optional and if not specified, tokenizer from model_name_or_path will be used
--prompt_tuning_init "RANDOM" \
--prompt_tuning_init_text "From the following input, identify target sentiment of following types: neutral, negative, positive"
```

Equally you can pass in a JSON configuration for running tuning. See [build doc](./build/README.md) for more details. The above can also be passed in as JSON:
```json
{
    "model_name_or_path": $MODEL_PATH,
    "training_data_path": $TRAIN_DATA_PATH,
    "output_dir": $OUTPUT_PATH,
    "num_train_epochs": 5.0,
    "per_device_train_batch_size": 1,
    "learning_rate": 0.03,
    "response_template": "\n### Label:",
    "dataset_text_field": "output",
    "peft_method": "pt",
    "tokenizer_name_or_path": $MODEL_PATH,
    "prompt_tuning_init": "RANDOM",
    "prompt_tuning_init_text": "From the following input, identify target sentiment of following types: neutral, negative, positive"
}
```

### Fine Tuning:

Set `peft_method` to `'None'` or do not provide `peft_method` flag.

Full fine tuning needs more compute resources, so it is advised to use the MultiGPU method. Example command:

```bash
accelerate launch \
--num_processes=4
--config_file fixtures/accelerate_fsdp_defaults.yaml \
tuning/sft_trainer.py  \
--model_name_or_path $MODEL_PATH  \
--tokenizer_name_or_path $MODEL_PATH \ # This field is optional and if not specified, tokenizer from model_name_or_path will be used
--training_data_path $TRAIN_DATA_PATH  \
--output_dir $OUTPUT_PATH  \
--num_train_epochs 5  \
--per_device_train_batch_size 4  \
--learning_rate 1e-5  \
--response_template "\n### Label:"  \
--dataset_text_field "output" \
--peft_method "None"
```

Equally you can pass in a JSON configuration for running tuning. See [build doc](./build/README.md) for more details. The above can also be passed in as JSON:
```json
{
    "model_name_or_path": $MODEL_PATH,
    "training_data_path": $TRAIN_DATA_PATH,
    "output_dir": $OUTPUT_PATH,
    "num_train_epochs": 5.0,
    "per_device_train_batch_size": 4,
    "learning_rate": 1e-5,
    "response_template": "\n### Label:",
    "dataset_text_field": "output",
    "peft_method": "None"
}
```

### FMS Acceleration

`fms-acceleration` is fuss-free approach to access a curated collection of acceleration plugins that acclerate your `tuning/sft-trainer.py` experience. Accelerations that apply to a variety of use-cases, e.g., PeFT / full-finetuning, are being planned for. As such, the accelerations are grouped into *plugins*; only install the plugins needed for the acceleration of interest. The plugins are housed in the [seperate repository found here](https://github.com/foundation-model-stack/fms-acceleration).

To access `fms-acceleration` features the `[fms-accel]` dependency must first be installed:
  ```
  $ pip install fms-hf-tuning[fms-accel]
  ```

Furthermore, the required `fms-acceleration` plugin must be installed. This is done via the command line utility `fms_acceleration.cli`. To show available plugins:
  ```
  $ python -m fms_acceleration.cli plugins
  ```
as well as to install the `fms_acceleration_peft`:

  ```
  $ python -m fms_acceleration.cli install fms_acceleration_peft
  ```

If you do not know what plugin to install (or forget), the framework will remind 

```
An acceleration feature is requested by specifying the '--auto_gptq' argument, but the this requires acceleration packages to be installed. Please do:
- python -m fms_acceleration.cli install fms_acceleration_peft
```

The list of configurations for various `fms_acceleration` plugins:
- [quantized_lora_config](./tuning/config/acceleration_configs/quantized_lora_config.py): For quantized 4bit LoRA training
  - `--auto_gptq`: 4bit GPTQ-LoRA with AutoGPTQ
  - `--bnb_qlora`: 4bit QLoRA with bitsandbytes
- [fused_ops_and_kernels](./tuning/config/acceleration_configs/fused_ops_and_kernels.py):
  - `--fused_lora`: fused lora for more efficient LoRA training.
  - `--fast_kernels`: fast cross-entropy, rope, rms loss kernels.
- [attention_and_distributed_packing](./tuning/config/acceleration_configs/attention_and_distributed_packing.py):
  - `--padding_free`: technique to process multiple examples in single batch without adding padding tokens that waste compute.
  - `--multipack`: technique for *multi-gpu training* to balance out number of tokens processed in each device, to minimize waiting time.

Notes: 
 * `quantized_lora_config` requires that it be used along with LoRA tuning technique. See [LoRA tuning section](https://github.com/foundation-model-stack/fms-hf-tuning/tree/main?tab=readme-ov-file#lora-tuning-example) on the LoRA parameters to pass.
 * When setting `--auto_gptq triton_v2` plus note to also pass `--torch_dtype float16` and `--fp16`, or an exception will be raised. This is because these kernels only support this dtype.
 * When using `fused_ops_and_kernels` together with `quantized_lora_config`,
 make sure to appropriately set `--fused_lora auto_gptq True` or `bitsandbytes True`; the `True` sets `fast_lora==True`.
 * `fused_ops_and_kernels` works for full-finetuning, LoRA, QLoRA and GPTQ-LORA, 
    - pass `--fast_kernels True True True` for full finetuning/LoRA
    - pass `--fast_kernels True True True --auto_gptq triton_v2 --fused_lora auto_gptq True` for GPTQ-LoRA
    - pass `--fast_kernels True True True --bitsandbytes nf4 --fused_lora bitsandbytes True` for QLoRA
    - Note the list of supported models [here](https://github.com/foundation-model-stack/fms-acceleration/blob/main/plugins/fused-ops-and-kernels/README.md#supported-models).
 * Notes on Padding Free
    - works for both *single* and *multi-gpu*. 
    - works on both *pretokenized* and *untokenized* datasets
    - verified against the version found in HF main, merged in via PR https://github.com/huggingface/transformers/pull/31629.
 * Notes on Multipack
    - works only for *multi-gpu*.
    - currently only includes the version of *multipack* optimized for linear attention implementations like *flash-attn*.

Note: To pass the above flags via a JSON config, each of the flags expects the value to be a mixed type list, so the values must be a list. For example:
```json
{
  "fast_kernels": [true, true, true],
  "padding_free": ["huggingface"],
  "multipack": [16],
  "auto_gptq": ["triton_v2"]
}
```

Activate `TRANSFORMERS_VERBOSITY=info` to see the huggingface trainer printouts and verify that `AccelerationFramework` is activated!

```
# this printout will be seen in huggingface trainer logs if acceleration is activated
***** FMS AccelerationFramework *****
Active Plugin: AutoGPTQAccelerationPlugin. Python package: fms_acceleration_peft. Version: 0.0.1.
***** Running training *****
Num examples = 1,549
Num Epochs = 1
Instantaneous batch size per device = 4
Total train batch size (w. parallel, distributed & accumulation) = 4
Gradient Accumulation steps = 1
Total optimization steps = 200
Number of trainable parameters = 13,631,488
```

The `fms_acceleration.cli` can do more to search for all available configs, plugins and arguments, [see the advanced flow](https://github.com/foundation-model-stack/fms-acceleration#advanced-flow).



## Inference
Currently, we do *not* offer inference support as part of the library, but we provide a standalone script for running inference on tuned models for testing purposes. For a full list of options run `python scripts/run_inference.py --help`. Note that no data formatting / templating is applied at inference time.

### Running a single example
If you want to run a single example through a model, you can pass it with the `--text` flag.

```bash
python scripts/run_inference.py \
--model my_checkpoint \
--text "This is a text the model will run inference on" \
--max_new_tokens 50 \
--out_file result.json
```

### Running multiple examples
To run multiple examples, pass a path to a file containing each source text as its own line. Example:

Contents of `source_texts.txt`
```
This is the first text to be processed.
And this is the second text to be processed.
```

```bash
python scripts/run_inference.py \
--model my_checkpoint \
--text_file source_texts.txt \
--max_new_tokens 50 \
--out_file result.json
```

### Inference Results Format
After running the inference script, the specified `--out_file` will be a JSON file, where each text has the original input string and the predicted output string, as follows. Note that due to the implementation of `.generate()` in Transformers, in general, the input string will be contained in the output string as well.
```
[
    {
        "input": "{{Your input string goes here}}",
        "output": "{{Generate result of processing your input string goes here}}"
    },
    ...
]
```

### Changing the Base Model for Inference
If you tuned a model using a *local* base model, then a machine-specific path will be saved into your checkpoint by Peft, specifically the `adapter_config.json`. This can be problematic if you are running inference on a different machine than you used for tuning.

As a workaround, the CLI for inference provides an arg for `--base_model_name_or_path`, where a new base model may be passed to run inference with. This will patch the `base_model_name_or_path` in your checkpoint's `adapter_config.json` while loading the model, and restore it to its original value after completion. Alternatively, if you like, you can change the config's value yourself.

NOTE: This can also be an issue for tokenizers (with the `tokenizer_name_or_path` config entry). We currently do not allow tokenizer patching since the tokenizer can also be explicitly configured within the base model and checkpoint model, but may choose to expose an override for the `tokenizer_name_or_path` in the future.

## Validation

We can use [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) from EleutherAI for evaluating the generated model. For example, for the Llama-13B model, using the above command and the model at the end of Epoch 5, we evaluated MMLU score to be `53.9` compared to base model to be `52.8`.

How to run the validation:
```bash
pip install -U transformers
pip install -U datasets
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
python main.py \ 
--model hf-causal \
--model_args pretrained=$MODEL_PATH \ 
--output_path $OUTPUT_PATH/results.json \ 
--tasks boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,hendrycksTest-*
```

The above runs several tasks with `hendrycksTest-*` being MMLU.

## Trainer Controller Framework

Trainer controller is a framework for controlling the trainer loop using user-defined rules and metrics.

This framework helps users define rules to capture scenarios like criteria for stopping an ongoing training (E.g validation loss reaching a certain target, validation loss increasing with epoch, training loss values for last 100 steps increasing etc).

For details about how you can use set a custom stopping criteria and perform custom operations, see [examples/trainercontroller_configs/Readme.md](examples/trainercontroller_configs/Readme.md)


## Experiment Tracking

Experiment tracking in fms-hf-tuning allows users to track their experiments with known trackers like [Aimstack](https://aimstack.io/), [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html) or custom trackers built into the code like
[FileLoggingTracker](./tuning/trackers/filelogging_tracker.py)

The code supports currently two trackers out of the box, 
* `FileLoggingTracker` : A built in tracker which supports logging training loss to a file.
* `Aimstack` : A popular opensource tracker which can be used to track any metrics or metadata from the experiments.
* `MLflow Tracking` : Another popular opensource tracker which stores metrics, metadata or even artifacts from experiments.

Further details on enabling and using the trackers mentioned above can be found [here](docs/experiment-tracking.md).  


## More Examples

[Prompt Tuning on Twitter Complaints](examples/prompt_tuning_twitter_complaints/README.md)

A good simple example can be found [here](examples/kfto-kueue-sft-trainer.yaml) which launches a Kubernetes-native `PyTorchJob` using the [Kubeflow Training Operator](https://github.com/kubeflow/training-operator/) with [Kueue](https://github.com/kubernetes-sigs/kueue) for the queue management of tuning jobs.