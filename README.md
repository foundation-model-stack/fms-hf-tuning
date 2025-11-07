# FMS HF Tuning

- [Installation](#installation)
- [Tuning Techniques](#tuning-techniques)
- [Training and Training Parameter Selection](#training-and-training-parameters)
- [Supported Models](#supported-models)
- [Data format support](#data-support)
  - [Advanced Data Processing](./docs/advanced-data-preprocessing.md#data-config)
  - [Guidelines on supported data formats](./docs/advanced-data-preprocessing.md#use-cases-supported-via-command-line-argument-training_data_path)
  - [Offline data processing](#offline-data-preprocessing)
  - [Online data mixing](./docs/online-data-mixing.md)
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

Refer our [Installation](./docs/installation.md) guide for details on how to install the library.

## Tuning Techniques:

Please refer to our [tuning techniques document](./docs/tuning-techniques.md) for details on how to perform - 
* [LoRA](./docs/tuning-techniques.md#lora-tuning-example)
* [Activated LoRA](./docs/tuning-techniques.md#activated-lora-tuning-example)
* [GPTQ-LoRA](./docs/tuning-techniques.md#gptq-lora-with-autogptq-tuning-example) 
* [Full Fine Tuning](./docs/tuning-techniques.md#fine-tuning)
* [Use FMS Acceleration](./docs/tuning-techniques.md#fms-acceleration)
* [Extended Pre-Training](./docs/tuning-techniques.md#extended-pre-training)

## Training and Training Parameters:

* Please refer our document on [training](./docs/training.md) to see how to start [Single GPU](./docs/training.md#single-gpu) or [Multi-GPU](./docs/training.md#multiple-gpus-with-fsdp) runs with fms-hf-tuning.
* You can also refer the same a different [section](./docs/training.md#tips-on-parameters-to-set) of the same document on tips to set various training arguments.

### *Debug recommendation:*
While training, if you encounter flash-attn errors such as `undefined symbol`, you can follow the below steps for clean installation of flash binaries. This may occur when having multiple environments sharing the pip cache directory or torch version is updated.

```sh
pip uninstall flash-attn
pip cache purge
pip install fms-hf-tuning[flash-attn]
```

## Supported Models

- While we expect most Hugging Face decoder models to work, we have primarily tested fine-tuning for below family of models.
  * [IBM Granite](https://huggingface.co/ibm-granite)
  * [Meta Llama](https://huggingface.co/meta-llama)
  * [Mistral Ai](https://huggingface.co/mistralai) and
  * [OpenAI GPT-OSS](https://huggingface.co/collections/openai/gpt-oss-68911959590a1634ba11c7a4)

- LoRA Layers supported : All the linear layers of a model + output `lm_head` layer. Users can specify layers as a list or use `all-linear` as a shortcut. Layers are specific to a model architecture and can be specified as noted [here](https://github.com/foundation-model-stack/fms-hf-tuning?tab=readme-ov-file#lora-tuning-example)

An extended list for tested models is maintaned in the [supported models](./docs/supported-models.md) document but might have outdated information.

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
