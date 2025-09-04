# Table of Contents

- [LoRA Tuning Example](#lora-tuning-example)
  - [How to get list of LoRA target_modules of a model](#how-to-get-list-of-lora-target_modules-of-a-model)
  - [Recommended target modules per model architecture](#recommended-target-modules-per-model-architecture)
    - [How to specify lm_head as a target module](#how-to-specify-lm_head-as-a-target-module)
  - [Post-processing needed for inference on VLLM](#post-processing-needed-for-inference-on-vllm)
    - [Alternatively, if using SDK](#alternatively-if-using-sdk)

- [Activated LoRA Tuning Example](#activated-lora-tuning-example)
  - [How to get list of aLoRA target_modules of a model](#how-to-get-list-of-alora-target_modules-of-a-model)
  - [Recommended target modules per model architecture](#recommended-target-modules-per-model-architecture-1)
  - [Intermediate checkpoint saving](#intermediate-checkpoint-saving)
  - [Inference with aLoRA models](#inference-with-alora-models)
  - [Running aLoRA models on VLLM](#running-alora-models-on-vllm)

- [GPTQ-LoRA with AutoGPTQ Tuning Example](#gptq-lora-with-autogptq-tuning-example)

- [Fine Tuning](#fine-tuning)

- [FMS Acceleration](#fms-acceleration)

- [Extended Pre-Training](#extended-pre-training)

## LoRA Tuning Example

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
--response_template "\n## Label:" \
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
    "response_template": "\n## Label:",
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

### How to get list of LoRA target_modules of a model
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

### Recommended target modules per model architecture 
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

### Post-processing needed for inference on VLLM

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

## Activated LoRA Tuning Example

Activated LoRA (aLoRA) is a new low rank adapter architecture that allows for reusing existing base model KV cache for more efficient inference. This approach is best suited for inference pipelines which rely on the base model for most tasks/generations, but use aLoRA adapter(s) to perform specialized task(s) within the chain. For example, checking or rewriting generated outputs of the base model.

[Paper](https://arxiv.org/abs/2504.12397)

[IBM Research Blogpost](https://research.ibm.com/blog/inference-friendly-aloras)

[Github](https://github.com/IBM/activated-lora)

**Usage** Usage is very similar to standard LoRA, with the key difference that an invocation_string must be specified so that the model knows when to turn on i.e "activate" the adapter weights. The model will scan any input strings (during training or at test time) for this invocation_string, and activate the adapter weights 1 token after the start of the sequence. If there are multiple instances of the invocation_string in the same input, it will activate at the last such instance.

**Note** Often (not always) aLoRA requires higher rank (r) than LoRA. r=32 can be a good starting point for challenging tasks.

**Installation** The Activated LoRA requirements are an optional install in pyproject.toml (activated-lora)

Set `peft_method` to `"alora"`. 

You *must* pass in an invocation_string argument. This invocation_string *must be present* in both training data inputs and the input at test time. A good solution is to set invocation_string = response_template, this will ensure that every training input will have the invocation_string present. We keep these separate arguments for flexibility. It is most robust if the invocation_string begins and ends with special tokens.

You can additionally pass any arguments from [aLoraConfig](https://github.com/IBM/activated-lora/blob/fms-hf-tuning/alora/config.py#L35), see the LoRA section for examples.

Example command to run, here using the ([Granite Instruct response template](https://huggingface.co/ibm-granite/granite-3.0-8b-instruct/blob/main/tokenizer_config.json#L188)) as the invocation sequence:

```bash
python tuning/sft_trainer.py \
--model_name_or_path $MODEL_PATH \
--tokenizer_name_or_path $MODEL_PATH \ # This field is optional and if not specified, tokenizer from model_name_or_path will be used
--training_data_path $TRAIN_DATA_PATH \
--output_dir $OUTPUT_PATH \
--num_train_epochs 40 \
--per_device_train_batch_size 4 \
---learning_rate 1e-4 \
--response_template "<|start_of_role|>assistant<|end_of_role|>" \ #this example uses special tokens in the Granite tokenizer, adjust for other models
--invocation_string "<|start_of_role|>assistant<|end_of_role|>" \
--dataset_text_field "output" \
--peft_method "alora" \
--r 32 \
--lora_dropout 0.05 \
--lora_alpha 16 \
--target_modules q_proj k_proj v_proj
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
    "response_template": "<|start_of_role|>assistant<|end_of_role|>",
    "invocation_string": "<|start_of_role|>assistant<|end_of_role|>",
    "dataset_text_field": "output",
    "peft_method": "alora",
    "r": 32,
    "lora_dropout": 0.05,
    "lora_alpha": 16,
    "target_modules": ["q_proj", "k_proj", "v_proj"]
}
```

Notice the `target_modules` are the names of the modules to apply the adapter to.
- If this is specified, only the modules with the specified names will be replaced. When passing a list of strings, either an exact match will be performed or it is checked if the name of the module ends with any of the passed strings. If this is specified as `all-linear`, then all linear/Conv1D modules are chosen, excluding the output layer. 
- If this is not specified, modules will be chosen according to the model architecture. If the architecture is not known, an error will be raised — in this case, you should specify the target modules manually. See [HuggingFace docs](https://huggingface.co/docs/peft/en/package_reference/lora#peft.LoraConfig) for more details.


### How to get list of aLoRA target_modules of a model
See [How to get list of LoRA target_modules of a model](#how-to-get-list-of-lora-target_modules-of-a-model). 

### Recommended target modules per model architecture 
As per [aLoRA paper](https://arxiv.org/abs/2504.12397), by using the key, query and value projection matrices, we can achieve good quality with efficient GPU utilization. Hence, while thinking about what aLoRA adapters to specify, we recommend starting with key, query and value matrices. 

### Intermediate checkpoint saving
Note that `sft_trainer.py` will always save the final trained model for you. If you want to save intermediate checkpoints from within the training process, the below applies.

For now, `save_strategy` is not supported (it is always reset to `none`). You can either save the model once training is complete, or pass in a custom callback in `additional_callbacks` directly to `tuning.sft_trainer.train` to perform saving. For example the following (from [alora github](https://github.com/IBM/activated-lora/blob/fms-hf-tuning/train_scripts/finetune_example_callback.py)) saves and updates the best performing model so far, checking whenever eval is called according to `eval_strategy`:
```py
class SaveBestModelCallback(TrainerCallback):
    def __init__(self):
        self.best_eval_loss = float("inf")  # Track best loss

    def on_evaluate(self, args, state, control, **kwargs):
        """Save the best model manually during evaluation."""

        model = kwargs["model"]
        metrics = kwargs["metrics"]
        
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None and eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss  # Update best loss

            # Manually save best model
            model.save_pretrained(args.output_dir)
```
### Inference with aLoRA models
*Important* Inference with aLoRA models requires nsuring that the invocation string is present in the input (usually the end).

Example inference:
```py
# Load the model
loaded_model = TunedCausalLM.load(ALORA_MODEL, BASE_MODEL_NAME, use_alora=True)

# Retrieve the invocation string from the model config
invocation_string = loaded_model.peft_model.peft_config[
    loaded_model.peft_model.active_adapter
].invocation_string

# In this case, we have the invocation string at the end of the input 
input_string = "Simply put, the theory of relativity states that \n" + invocation_string

# Run inference on the text
output_inference = loaded_model.run(
    input_string, 
    max_new_tokens=50,
)
```

### Running aLoRA models on VLLM

Coming soon! For now, there is inference support in this package, or see [aLoRA github](https://github.com/IBM/activated-lora/experiments/inference_example.py) for example code demonstrating KV cache reuse from prior base model calls.

__________



## GPTQ-LoRA with AutoGPTQ Tuning Example

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
--response_template "\n## Label:" \
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
    "response_template": "\n## Label:",
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

## Fine Tuning:

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
--response_template "\n## Label:"  \
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
    "response_template": "\n## Label:",
    "dataset_text_field": "output",
    "peft_method": "None"
}
```

## FMS Acceleration

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
- [fast_moe_config](./tuning/config/acceleration_configs/fast_moe.py) (experimental):
  - `--fast_moe`: trains MoE models in parallel with [Scatter MoE kernels](https://github.com/foundation-model-stack/fms-acceleration/tree/main/plugins/accelerated-moe#fms-acceleration-for-mixture-of-experts), increasing throughput and decreasing memory usage.

Notes: 
 * `quantized_lora_config` requires that it be used along with LoRA tuning technique. See [LoRA tuning section](https://github.com/foundation-model-stack/fms-hf-tuning/tree/main?tab=readme-ov-file#lora-tuning-example) on the LoRA parameters to pass.
 * When setting `--auto_gptq triton_v2` plus note to also pass `--torch_dtype float16` and `--fp16`, or an exception will be raised. This is because these kernels only support this dtype.
 * When using `fused_ops_and_kernels` together with `quantized_lora_config`,
 make sure to appropriately set `--fused_lora auto_gptq True` or `bitsandbytes True`; the `True` sets `fast_lora==True`.
 * `fused_ops_and_kernels` works for full-finetuning, LoRA, QLoRA and GPTQ-LORA, 
    - Pass `--fast_kernels True True True` for full finetuning/LoRA
    - Pass `--fast_kernels True True True --auto_gptq triton_v2 --fused_lora auto_gptq True` for GPTQ-LoRA
    - Pass `--fast_kernels True True True --bitsandbytes nf4 --fused_lora bitsandbytes True` for QLoRA
    - Note the list of supported models [here](https://github.com/foundation-model-stack/fms-acceleration/blob/main/plugins/fused-ops-and-kernels/README.md#supported-models).
 * Notes on Padding Free
    - Works for both *single* and *multi-gpu*. 
    - Works on both *pretokenized* and *untokenized* datasets
    - Verified against the version found in HF main, merged in via PR https://github.com/huggingface/transformers/pull/31629.
 * Notes on Multipack
    - Works only for *multi-gpu*.
    - Currently only includes the version of *multipack* optimized for linear attention implementations like *flash-attn*.
    - Streaming datasets or use of `IterableDatasets` is not compatible with the fms-acceleration multipack plugin because multipack sampler has to run thorugh the full dataset every epoch. Using multipack and streaming together will raise an error.
 * Notes on Fast MoE
    - `--fast_moe` takes either an integer or boolean value.
      - When an integer `n` is passed, it enables expert parallel sharding with the expert parallel degree as `n` along with Scatter MoE kernels enabled.
      - When a boolean is passed, the expert parallel degree defaults to 1 and further the behaviour would be as follows:
          - if True, it is Scatter MoE Kernels with experts sharded based on the top level sharding protocol (e.g. FSDP).
          - if False, Scatter MoE Kernels with complete replication of experts across ranks.
    - FSDP must be used when lora tuning with `--fast_moe`
    - lora tuning with ScatterMoE is supported, but because of inference restrictions on vLLM/vanilla PEFT, the expert layers and router linear layer should not be trained as `target_modules` for models being tuned with ScatterMoE. Users have control over which `target_modules` they wish to train:
        - At this time, only attention layers are trainable when using LoRA with scatterMoE. Until support for the router linear layer is added in, target modules must be specified explicitly (i.e `target_modules: ["q_proj", "v_proj", "o_proj", "k_proj"]`) instead of passing `target_modules: ["all-linear"]`.
    - `world_size` must be divisible by the `ep_degree`
    - `number of experts` in the MoE module must be divisible by the `ep_degree`
    - Running fast moe modifies the state dict of the model, and must be post-processed which happens automatically and the converted checkpoint can be found at `hf_converted_checkpoint` folder within every saved checkpoint directory. Alternatively, we can perform similar option manually through [checkpoint utils](https://github.com/foundation-model-stack/fms-acceleration/blob/main/plugins/accelerated-moe/src/fms_acceleration_moe/utils/checkpoint_utils.py) script.
      - The typical usecase for this script is to run:
        ```
        python -m fms_acceleration_moe.utils.checkpoint_utils \
        <checkpoint file> \
        <output file> \
        <original model>
        ```

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


## Extended Pre-Training

We also have support for extended pre training where users might wanna pretrain a model with large number of samples. Please refer our separate doc on [EPT Use Cases](./ept.md)