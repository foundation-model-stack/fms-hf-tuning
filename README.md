# FMS HF Tuning

This repo provides basic tuning scripts with support for specific models. The repo relies on Hugging Face `SFTTrainer` and PyTorch FSDP. Our approach to tuning is:
1. Models are loaded from Hugging Face `transformers` or the [foundation-model-stack](https://github.com/foundation-model-stack/foundation-model-stack) -- models are either optimized to use `Flash Attention v2` directly or through `SDPA`
2. Hugging Face `SFTTrainer` for the training loop
3. `FSDP` as the backend for training

## Installation

```
pip install fms-hf-tuning
```

> Note: After installing, if you wish to use [FlashAttention](https://github.com/Dao-AILab/flash-attention), then you need to install these requirements:
```
pip install fms-hf-tuning[dev]
pip install fms-hf-tuning[flash-attn]
```
[FlashAttention](https://github.com/Dao-AILab/flash-attention) requires the [CUDA Toolit](https://developer.nvidia.com/cuda-toolkit) to be pre-installed.

If you wish to use [aim](https://github.com/aimhubio/aim), then you need to install it:
```
pip install fms-hf-tuning[aim]
```

If you wish to use [fms-acceleration](https://github.com/foundation-model-stack/fms-acceleration), you need to install it. 
```
pip install git+https://github.com/foundation-model-stack/fms-acceleration.git#subdirectory=plugins/framework
```
`fms-acceleration` is a collection of plugins that packages that accelerate fine-tuning / training of large models, as part of the `fms-hf-tuning` suite. For more details on see [this section below](#fms-acceleration).

## Data format
We support two data formats:

1. #### Pre-process the JSON/JSONL dataset
 Pre-process the JSON/JSONL dataset to contain a single sequence of each data instance containing input + Response. The trainer is configured to expect a response template as a string. For example, if one wants to prepare the `alpaca` format data to feed into this trainer, it is quite easy and can be done with the following code.

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

Once the JSON is converted using the formatting function, pass the `dataset_text_field` containing the single sequence to the trainer. 

2.  #### Format JSON/JSONL on the fly
   Pass a JSON/JSONL and a `data_formatter_template` to use the formatting function on the fly while tuning. The template should specify fields of JSON with `{{field}}`. While tuning, the data will be converted to a single sequence using the template.  
   JSON fields can contain alpha-numeric characters, spaces and the following special symbols - "." , "_", "-".  

Example: Train.json
`[{ "input" : <text>,
    "output" : <text>,
  },
 ...
]`  
data_formatter_template: `### Input: {{input}} \n\n##Label: {{output}}`  

Formatting will happen on the fly while tuning. The keys in template should match fields in JSON file. The `response template` corresponding to the above template will need to be supplied. in this case, `response template` = `\n## Label:`.


##### In conclusion, either the `data_formatter_template` argument or `dataset_text_field` needs to be supplied to the trainer. 

## Supported Models

Current supported and tested models are `Llama2` (7 and 13B configurations have been tested) and `GPTBigCode`.

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
--dataset_text_field "output"
```

To summarize you can pick either python for single-GPU jobs or use accelerate launch for multi-GPU jobs. The following tuning techniques can be applied:

## Tuning Techniques:

### LoRA Tuning Example

Set `peft_method` to `"lora"`. You can additionally pass any arguments from [LoraConfig](https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/tuning/config/peft_config.py#L21).
```py
# Args you can pass
r: int =8 
lora_alpha: int = 32
target_modules: List[str] = field(
  default_factory=lambda: ["q_proj", "v_proj"],
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
--target_modules ["c_attn", "c_proj"]
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

Notice the `target_modules` that are set are the default values. `target_modules` are the names of the modules to apply the adapter to. If this is specified, only the modules with the specified names will be replaced. When passing a list of strings, either an exact match will be performed or it is checked if the name of the module ends with any of the passed strings. If this is specified as `all-linear`, then all linear/Conv1D modules are chosen, excluding the output layer. If this is not specified, modules will be chosen according to the model architecture. If the architecture is not known, an error will be raised — in this case, you should specify the target modules manually. See [HuggingFace docs](https://huggingface.co/docs/peft/en/package_reference/lora#peft.LoraConfig) for more details.

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
--tokenizer_name_or_path $MODEL_PATH
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
  $ pip install https://github.com/foundation-model-stack/fms-acceleration.git#subdirectory=plugins/framework
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
- [fused_ops_and_kernels](./tuning/config/acceleration_configs/fused_ops_and_kernels.py) (experimental):
  - `--fused_lora`: fused lora for more efficient LoRA training.
  - `--fast_kernels`: fast cross-entropy, rope, rms loss kernels.

Notes: 
 * `quantized_lora_config` requires that it be used along with LoRA tuning technique. See [LoRA tuning section](https://github.com/foundation-model-stack/fms-hf-tuning/tree/main?tab=readme-ov-file#lora-tuning-example) on the LoRA parameters to pass.
 * When setting `--auto_gptq triton_v2` plus note to also pass `--torch_dtype float16` and `--fp16`, or an exception will be raised. This is because these kernels only support this dtype.
 * Currently, the `fused_ops_and_kernels` is to be used used together QLoRA or GPTQ-LORA via the `quantized_lora_config`. In the future it may be made more flexible such that `fast_kernels` can even be used with full-finetuning.
 * When using `fused_ops_and_kernels` together with `quantized_lora_config`,
 make sure to appropriately set `--fused_lora auto_gptq True` or `bitsandbytes True`; the `True` sets `fast_lora==True`.
 * Currently `fused_ops_and_kernels` only supports activating `fast_loss,fast_rsm_layernorm,fast_rope_embeddings` all to `True`, so pass `--fast_kernels True True True`.


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

## More Examples

[Prompt Tuning on Twitter Complaints](examples/prompt_tuning_twitter_complaints/README.md)

A good simple example can be found [here](examples/kfto-kueue-sft-trainer.yaml) which launches a Kubernetes-native `PyTorchJob` using the [Kubeflow Training Operator](https://github.com/kubeflow/training-operator/) with [Kueue](https://github.com/kubernetes-sigs/kueue) for the queue management of tuning jobs.
