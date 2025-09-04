# FMS HF Tuning

- [Installation](#installation)
- [Tuning Techniques](#tuning-techniques)
- [Training and Training Parameter Selection](#training-and-training-parameters)
- [Supported Models](#supported-models)
- [Data format support](#data-support)
  - [Advanced Data Processing](./docs/advanced-data-preprocessing.md#data-config)
  - [Guidelines on supported data formats](./docs/advanced-data-preprocessing.md#use-cases-supported-via-command-line-argument-training_data_path)
  - [Offline data processing](#offline-data-preprocessing)
- [Additional Frameworks](#additional-frameworks)
  - [Inference](#inference)
  - [Validation](#validation)
  - [Training controller](#trainer-controller-framework)
  - [More Examples](#more-examples)

This repo provides basic tuning scripts with support for specific models. The repo relies on Hugging Face `SFTTrainer` and PyTorch FSDP. Our approach to tuning is:
1. Models are loaded from Hugging Face `transformers` or the [foundation-model-stack](https://github.com/foundation-model-stack/foundation-model-stack) -- models are either optimized to use `Flash Attention v2` directly or through `SDPA`
2. Hugging Face `SFTTrainer` for the training loop
3. `FSDP` as the backend for multi gpu training

## Installation

Refer our [Installation](./docs/installations.md) guide for details on how to install the library.

## Tuning Techniques:

Please refer to our [tuning techniques document](./docs/tuning-techniques.md) for details on how to perform - 
* [LoRA](./docs/tuning-techniques.md#lora-tuning-example)
* [Activated LoRA](./docs/tuning-techniques.md#activated-lora-tuning-example)
* [GPTQ-LoRA](./docs/tuning-techniques.md#gptq-lora-with-autogptq-tuning-example) 
* [Full Fine Tuning](./docs/tuning-techniques.md#fine-tuning)
* [Use FMS Acceleration](./docs/tuning-techniques.md#fms-acceleration)
* [Extended Pre-Training](./docs/tuning-techniques.md#extended-pre-training) 

## Training and Training Parameters:

Please refer our [document](./docs/training.md) to see how to start [Single GPU](./docs/training.md#single-gpu) or [Multi-GPU](./docs/training.md#multiple-gpus-with-fsdp) runs with fms-hf-tuning.

You can also refer the same [document](./docs/training.md#tips-on-parameters-to-set) on how to use various training arguments.

### *Debug recommendation:*
While training, if you encounter flash-attn errors such as `undefined symbol`, you can follow the below steps for clean installation of flash binaries. This may occur when having multiple environments sharing the pip cache directory or torch version is updated.

```sh
pip uninstall flash-attn
pip cache purge
pip install fms-hf-tuning[flash-attn]
```

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
[Granite 4.0 Tiny Preview](https://huggingface.co/ibm-granite/granite-4.0-tiny-preview) | GraniteMoeHybridForCausalLM | ✅ | ✅ | ? |
[Granite PowerLM 3B](https://huggingface.co/ibm-research/PowerLM-3b) | GraniteForCausalLM | ✅* | ✅* | ✅* |
[Granite 3.1 1B](https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-base)       | GraniteForCausalLM | ✔️* | ✔️* | ✔️* |
[Granite 3.1 2B](https://huggingface.co/ibm-granite/granite-3.1-2b-base)             | GraniteForCausalLM | ✔️* | ✔️* | ✔️* |
[Granite 3.1 8B](https://huggingface.co/ibm-granite/granite-3.1-8b-base)       | GraniteForCausalLM | ✔️* | ✔️* | ✔️* |
[Granite 3.0 2B](https://huggingface.co/ibm-granite/granite-3.0-2b-base)       | GraniteForCausalLM | ✔️* | ✔️* | ✔️* |
[Granite 3.0 8B](https://huggingface.co/ibm-granite/granite-3.0-8b-base)       | GraniteForCausalLM | ✅* | ✅* | ✔️ |
[GraniteMoE 1B](https://huggingface.co/ibm-granite/granite-3.0-1b-a400m-base)        | GraniteMoeForCausalLM  | ✅ | ✅** | ? |
[GraniteMoE 3B](https://huggingface.co/ibm-granite/granite-3.0-3b-a800m-base)        | GraniteMoeForCausalLM  | ✅ | ✅** | ? |
[Granite 3B Code](https://huggingface.co/ibm-granite/granite-3b-code-base-2k)           | LlamaForCausalLM      | ✅ | ✔️  | ✔️ | 
[Granite 8B Code](https://huggingface.co/ibm-granite/granite-8b-code-base-4k)           | LlamaForCausalLM      | ✅ | ✅ | ✅ |
Granite 13B          | GPTBigCodeForCausalLM  | ✅ | ✅ | ✔️  | 
Granite 20B          | GPTBigCodeForCausalLM  | ✅ | ✔️  | ✔️  | 
[Granite 34B Code](https://huggingface.co/ibm-granite/granite-34b-code-instruct-8k)            | GPTBigCodeForCausalLM  | 🚫 | ✅ | ✅ | 
[Llama3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)          | LlamaForCausalLM               | ✅*** | ✔️ | ✔️ |  
[Llama3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B)(same architecture as llama3) | LlamaForCausalLM   | 🚫 - same as Llama3-70B | ✔️  | ✔️ | 
[Llama3.1-405B](https://huggingface.co/meta-llama/Llama-3.1-405B)                            | LlamaForCausalLM   | 🚫 | 🚫 | ✅ | 
[Llama3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)                               | LlamaForCausalLM   | ✅ | ✅ | ✔️ |  
[Llama3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B)                             | LlamaForCausalLM   | 🚫 | ✅ | ✅ |
aLLaM-13b                                 | LlamaForCausalLM |  ✅ | ✅ | ✅ |
[Mixtral 8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)                              | MixtralForCausalLM   | ✅ | ✅ | ✅ |
[Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)                                  | MistralForCausalLM   | ✅ | ✅ | ✅ |  
Mistral large                             | MistralForCausalLM   | 🚫 | 🚫 | 🚫 | 
[GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b)                                  | GptOssForCausalLM   | ✅ | ✅ | ? |  
[GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b)                                  | GptOssForCausalLM   | ✅ | ✅ | ? |  

(*) - Supported with `fms-hf-tuning` v2.4.0 or later.

(**) - Supported for q,k,v,o layers . `all-linear` target modules does not infer on vLLM yet.

(***) - Supported from platform up to 8k context length - same architecture as llama3-8b.

### Supported vision model

We also support full fine-tuning and LoRA tuning for vision language models - `Granite 3.2 Vision`, `Llama 3.2 Vision`, and `LLaVa-Next` from `v2.8.1` onwards.
For information on supported dataset formats and how to tune a vision-language model, please see [this document](./vision-language-model-tuning.md).

Model Name & Size  | Model Architecture | LoRA Tuning | Full Finetuning |
-------------------- | ---------------- | --------------- | --------------- |
Llama 3.2-11B Vision  | MllamaForConditionalGeneration | ✅ | ✅ |
Llama 3.2-90B Vision  | MllamaForConditionalGeneration | ✔️ | ✔️ |
Granite 3.2-2B Vision  | LlavaNextForConditionalGeneration | ✅ | ✅ |
Llava Mistral 1.6-7B  | LlavaNextForConditionalGeneration | ✅ | ✅ |
Llava 1.6-34B  | LlavaNextForConditionalGeneration | ✔️ | ✔️ |
Llava 1.5-7B  | LlavaForConditionalGeneration | ✅ | ✅ |
Llava 1.5-13B  | LlavaForConditionalGeneration | ✔️ | ✔️ |

**Note**:
* vLLM currently does not support inference with LoRA-tuned vision models. To use a tuned LoRA adapter of vision model, please merge it with the base model before running vLLM inference.

## Data Support
Users can pass training data as either a single file or a Hugging Face dataset ID using the `--training_data_path` argument along with other arguments required for various [use cases](./docs/advanced-data-preprocessing.md#use-cases-supported-via-command-line-argument-training_data_path). If user choose to pass a file, it can be in any of the [supported formats](#supported-data-formats). Alternatively, you can use our powerful [data preprocessing backend](./docs/advanced-data-preprocessing.md) to preprocess datasets on the fly.

Below, we mention the list of supported data usecases via `--training_data_path` argument. For details of our advanced data preprocessing see more details in [Advanced Data Preprocessing](./docs/advanced-data-preprocessing.md).

EOS tokens are added to all data formats listed below (EOS token is appended to the end of each data point, like a sentence or paragraph within the dataset), except for pretokenized data format at this time. For more info, see [pretokenized](./docs/advanced-data-preprocessing.md#4-pre-tokenized-datasets).

## Offline Data Preprocessing

We also provide an interface for the user to perform standalone data preprocessing. This is especially useful if:

1. The user is working with a large dataset and wants to perform the processing in one shot and then train the model directly on the processed dataset.

2. The user wants to test out the data preprocessing outcome before training.

Please refer to [this document](docs/offline-data-preprocessing.md) for details on how to perform offline data processing.

## Additional Frameworks

### Inference
Currently, we do *not* offer inference support as part of the library, but we provide a standalone script for running inference on tuned models for testing purposes. For a full list of options run `python scripts/run_inference.py --help`. Note that no data formatting / templating is applied at inference time.

#### Running a single example
If you want to run a single example through a model, you can pass it with the `--text` flag.

```bash
python scripts/run_inference.py \
--model my_checkpoint \
--text "This is a text the model will run inference on" \
--max_new_tokens 50 \
--out_file result.json
```

#### Running multiple examples
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

#### Inference Results Format
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

#### Changing the Base Model for Inference
If you tuned a model using a *local* base model, then a machine-specific path will be saved into your checkpoint by Peft, specifically the `adapter_config.json`. This can be problematic if you are running inference on a different machine than you used for tuning.

As a workaround, the CLI for inference provides an arg for `--base_model_name_or_path`, where a new base model may be passed to run inference with. This will patch the `base_model_name_or_path` in your checkpoint's `adapter_config.json` while loading the model, and restore it to its original value after completion. Alternatively, if you like, you can change the config's value yourself.

NOTE: This can also be an issue for tokenizers (with the `tokenizer_name_or_path` config entry). We currently do not allow tokenizer patching since the tokenizer can also be explicitly configured within the base model and checkpoint model, but may choose to expose an override for the `tokenizer_name_or_path` in the future.

### Validation

For examples on how to run inference on models trained via fms-hf-tuning see [Inference](./docs/tuning.md#inference) document.

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

### Trainer Controller Framework

Trainer controller is a framework for controlling the trainer loop using user-defined rules and metrics. For details about how you can use set a custom stopping criteria and perform custom operations, see [examples/trainercontroller_configs/Readme.md](examples/trainercontroller_configs/Readme.md)

### More Examples
A good simple example can be found [here](examples/kfto-kueue-sft-trainer.yaml) which launches a Kubernetes-native `PyTorchJob` using the [Kubeflow Training Operator](https://github.com/kubeflow/training-operator/) with [Kueue](https://github.com/kubernetes-sigs/kueue) for the queue management of tuning jobs.
