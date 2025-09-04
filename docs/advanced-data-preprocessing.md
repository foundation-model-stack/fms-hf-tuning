# Advanced Data Processing
Our library also supports a powerful data processing backend which can be used by the users to perform custom data preprocessing including
1. Support for multiple datasets  
1. Creating custom data processing pipeline for the datasets.  
1. Combining multiple datasets into one, even if they have different formats.  
1. Mixing datasets as required and sampling each dataset with different weights.

These things are supported via what we call a [`data_config`](#data-config) which can be passed as an argument to sft trainer.

## Supported Data File Formats
We support the following file formats via `--training_data_path` argument

Data Format | Tested Support
------------|---------------
JSON        |   ✅
JSONL       |   ✅
PARQUET     |   ✅
ARROW       |   ✅

As iterated above, we also support passing a HF dataset ID directly via `--training_data_path` argument.

**NOTE**: Due to the variety of supported data formats and file types, `--training_data_path` is handled as follows:
- If `--training_data_path` ends in a valid file extension (e.g., .json, .csv), it is treated as a file.
- If `--training_data_path` points to a valid folder, it is treated as a folder.
- If neither of these are true, the data preprocessor tries to load `--training_data_path` as a Hugging Face (HF) dataset ID.

## Data Config

Data config is a configuration file which `sft_trainer.py` supports as an argument via `--data_config_path` flag. In this
configuration users can describe multiple datasets, configurations on how to load the datasets and configuration on how to 
process the datasets. Users can currently pass both YAML or JSON based configuration files as data_configs.

### What is data config schema 
The data config schema is designed to define datasets and their processing strategies in a structured way.

It consists of the following top-level keys:
 - `datapreprocessor`: Defines global data processing parameters, such as the type (`default`), sampling stopping strategy (`all_exhausted` or `first_exhausted`), and sampling seed for reproducibility.
 - `datasets`: A list of dataset configurations, each describing the dataset name, paths, optional builders, sampling ratios, and data handlers.

At the top level, the data config schema looks like this:
```yaml
definitions:
  data_config:
    type: object
    additionalProperties: false
    properties:
      dataprocessor:
        $ref: '#/definitions/Dataprocessor'
      datasets:
        type: array
        items:
          $ref: '#/definitions/Dataset'
    required:
      - dataprocessor
      - datasets
    title: data_config
  Dataprocessor:
    type: object
    additionalProperties: false
    properties:
      type:
        type: string
      sampling_stopping_strategy:
        type: string
      seed:
        type: integer
      chat_template:
        type: string
    required:
      - type
    title: Dataprocessor
  Dataset:
    type: object
    additionalProperties: false
    properties:
      name:
        type: string
      sampling:
        type: float
      builder:
        type: string
      data_paths:
        type: array
        items:
          type: string
      data_handlers:
        type: array
        items:
          $ref: '#/definitions/DataHandler'
    required:
      - data_paths
      - name
    title: Dataset
  DataHandler:
    type: object
    additionalProperties: false
    properties:
      name:
        type: string
      arguments:
        $ref: '#/definitions/DataHandlerArguments'
    required:
      - arguments
      - name
    title: DataHandler
  DataHandlerArguments:
    type: object
    additionalProperties: false
    properties:
      remove_columns:
        type: string
      batched:
        type: boolean
      fn_kwargs:
        $ref: '#/definitions/DataHandlerFnKwargs'
    required:
      - fn_kwargs
      - remove_columns
    title: DataHandlerArguments
  DataHandlerFnKwargs:
    type: object
    properties:
      str:
        type: str
    title: DataHandlerFnKwargs
```

### How the user can write data configs
Users can create a data config file in any of YAML or JSON format they choose (we provide examples of YAML for ease of use). The file should follow the schema outlined above with the following parameters:

`datapreprocessor`:
 - `type` (optional, str): Type of data preprocessor, `default` is currently the only supported type.
 - `streaming` (optional, bool): Stream datasets using [IterableDatasets](https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/main_classes#datasets.IterableDataset).
 - `sampling_stopping_strategy` (optional, str): Dataset interleave stopping strategy in case of choosing to mix multiple datasets by weight, supported values are [`all_exhausted` or `first_exhausted`](https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/main_classes#datasets.interleave_datasets.stopping_strategy), defaults to `all_exhausted`.
 - `seed` (optional, int): [seed](https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/main_classes#datasets.interleave_datasets.seed) to use for interleaving datasets, for reproducibility choose same value, defaults to 42.
 - `chat_template` (optional, str): pass `chat_template` via data_config for multi-turn data, replaces existing default chat template.

`datasets` (list):
  - `name` (optional, str): A unique identifier for the dataset.
    - `data_paths` (optional, list): A `list` of file paths or directories containing the dataset.
    - `builder` (optional, str): Specifies a [Hugging Face dataset builder](https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/loading_methods#datasets.load_dataset.path), if applicable.
    - `sampling` (optional, float): The sampling ratio (0.0 to 1.0) with which to sample a dataset in case of interleaving.
    - `split` (optional, dict[str: float]): Defines how to split the dataset into training and validation sets. Requires both `train` and `validation` keys.
    - `data_handlers` (optional, list): A list of data handler configurations which preprocess the dataset.

Data handlers are customizable components within the data config that allow users to preprocess or manipulate individual datasets. We use [Hugging Face Map API](https://huggingface.co/docs/datasets/en/process#map) to apply these routines.
These functions can process the dataset in any way users require and the `list` of data handlers specified for each dataset are applied in order.
Each data handler has:
- `name`: The handler's unique identifier.
- `arguments`: A dictionary of parameters specific to the handler.

We do provide some sample `data_configs` here, [predefined_data_configs](../tests/artifacts/predefined_data_configs/).

### How users can pass the datasets 
Users can provide single or multiple file paths, folder paths, or [Hugging Face dataset IDs](https://huggingface.co/datasets) through the `data_paths` argument. These datasets can be in various supported formats such as JSON, JSONL, Parquet, and Arrow. For a more up-to-date supported format list see [README.md](../README.md#supported-data-formats). Additionally, users can pass globbing patterns to specify files or folder paths matching specific regex patterns.

When passing multiple datasets with differing column structures, users should ensure appropriate handlers are specified to process the datasets correctly.

The `builder` argument can also be optionally included to provide additional information for dataset loading, this argument is directly passed to [HF `load_dataset` API](https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/loading_methods#datasets.load_dataset) as the first argument.


User can pass `builder` in `DataSetConfig` to mention the specific loader for the passed `file/folder/pattern`.
We support the following,
- Passing file paths that include a file extension in filename, or specifying a `builder` if file extension is not provided in filename.
- Passing of folder paths, with or without `builder`. 

Not Supported:
- Passing file paths that do not include a file extension in filename and do not specify a `builder`.
- Passing a folder as a wildcard globbing pattern.

Currently there's no support for sampling under multiple data paths which are defined inside a dataset definition.
All dataset paths that will be specified inside one dataset will be [concatenated](https://huggingface.co/docs/datasets/v3.2.0/en/process#concatenate) after loading them, while across datasets users can specify [mixing via sampling datasets](#data-mixing)


### Data Handlers

Data handlers, as explained above, are routines which process the dataset using [HF process frameworks](https://huggingface.co/docs/datasets/en/process) including map, filter, remove, select, and rename. 

For a thorough explanation of data handlers, how to use them, see the [data handlers document](./advanced-data-handlers.md)

### Data Mixing
Dataset mixing allows users to mix multiple datasets often with different `sampling ratios` to ensure the model is trained on a mix of some datasets in specific proportion. 

If users want to train a model on just a straight forward [`concatenation`](https://huggingface.co/docs/datasets/v3.2.0/en/process#concatenate) of the datasets then they need not enable data mixing. Users can specify just different datasets via `data_paths` as shown [above](#how-the-user-can-pass-the-datasets) and all the datasets will be concatenated via [`concatenate_datasets`](https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/main_classes#datasets.concatenate_datasets).

If users want to enable data mixing they need to enable `sampling` of the datasets by specifying `sampling` ratio for each dataset as described [above](#how-the-user-can-write-data-configs). The library will then collect sampling ratios from each dataset definition in the `data_config` and create a new [interleaved](https://huggingface.co/docs/datasets/v3.2.0/en/process#interleave) dataset which is a combination of all the sampled datasets via [`interleave_datasets()`](https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/main_classes#datasets.interleave_datasets) api.

Needless to say the sampling ratio of a datasets is a float and all the sampling ratios must sum to 1.

We also allow users to pass a [`seed`](https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/main_classes#datasets.interleave_datasets.seed) to randomize the interleaving of datasets and a [`stopping_strategy`](https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/main_classes#datasets.interleave_datasets.stopping_strategy) to describe when to stop sampling. Both values should remain the same for experiment reproducibility. Both these values are common for all datasets and should be supplied at top level in the `datapreprocessor` as shown [above](#how-the-user-can-write-data-configs). For a list of the supported values of these arguments see the corresponding HF API.


Note: If a user specifies data sampling they can expect the datasets to be mixed and individual samples in the dataset to not be broken unless the max_seq_len argument is smaller than the length of individual samples in the dataset

### Dataset Splitting

In addition to [sampling and mixing](#data-mixing), our library supports **dataset splitting**, which allows users to split a dataset into training and validation splits using the `split` field in the dataset config.

This is especially useful when users want to split a single dataset (or multiple datasets) internally instead of supplying separate files for training and validation.

#### How to Use

The `split` field in each dataset config allows users to internally divide a dataset into `train` and `validation` sets using fractional ratios.

To use it, specify both `train` and `validation` ratios values under the `split` key for each dataset. Example:

```yaml
datasets:
  - name: my_dataset
    split:
      train: 0.8
      validation: 0.2
    data_paths:
      - "path/to/data.jsonl"
```

### Split Support for Streaming vs Non-Streaming Datasets

**Non-Streaming Datasets (`Dataset`, `DatasetDict`)**:
- Supports arbitrary train/validation splits.
- Both `train` and `validation` keys must be present under `split`.
- The sum of `train + validation` must be in `(0, 1]`; less than 1.0 implies subset usage.
- If no `split` is defined, the dataset is returned unchanged.

**Streaming Datasets (`IterableDataset`, `IterableDatasetDict`)**:
- Only supports full splits:
  - Either `train: 1.0, validation: 0.0`
  - Or `train: 0.0, validation: 1.0`
- Partial splits like `train: 0.8, validation: 0.2` are not supported and will raise a `NotImplementedError`.
- If no `split` is defined, the dataset is returned unchanged.
- Streaming behavior must be explicitly enabled via `dataprocessor.streaming: true`.

### Using Separate Files for Train and Validation Splits

If you want to use **separate files for training and validation**, you can define them as **separate dataset entries** in the `datasets` section of your config.  
In each entry:

- Specify the corresponding file in the `data_paths` field.
- Set the `split` value to either `train: 1.0` or `validation: 1.0` as appropriate.

This allows you to fully control which file is used for which purpose, without relying on automatic or in-place splitting.

#### Example

```yaml
datasets:
  - name: my_train_set
    split:
      train: 1.0
    data_paths:
      - "path/to/train.jsonl"
  - name: my_val_set
    split:
      validation: 1.0
    data_paths:
      - "path/to/val.jsonl"
```

### **Note:**
> - While passing a validation dataset via the command line is possible using the `validation_data_path` argument, **this argument is not compatible with `data_config`**. If you're using a `data_config`, define the validation set within it using a `split: validation: 1.0` entry instead as shown [here](#using-separate-files-for-train-and-validation-splits).
> - Dataset splitting is performed based on the `split` configuration, supporting only `"train"` and `"validation"` splits. Support for a `"test"` split is not yet available.
> - **Only the `"train"` split is sampled**, and **sampling is done after splitting**. This ensures that validation remains consistent and unbiased, while allowing training to be performed on a controlled subset if desired.
> - **⚠️ Users must explicitly set the [`eval_strategy`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.eval_strategy) in the Trainer's arguments to a valid value (e.g., `"steps"` or `"epoch"`) for evaluation to run. Splitting the dataset alone does not trigger evaluation and will likely result in an error if `evaluation_strategy` is left unset.**

### Data Streaming
Dataset streaming allows users to utilize the functionality of iterable datasets to pass in data piece by piece, avoiding memory constraints with large datasets for use-cases like extended pre-training.

Users can use streaming by setting `streaming` to `true` in the `datapreprocessor` config. This top-level variable must be set for all datasets in the config, and cannot differ from dataset to dataset. When `streaming` is `true`, the dataset is loaded as an `IterableDataset` ([docs](https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/main_classes#datasets.IterableDataset)) instead of a regular `Dataset`, this means the dataset is loaded chunk-by-chunk rather than all at once and is processed lazily. For more details on the differences, see the [HF Blog](https://huggingface.co/docs/datasets/en/about_mapstyle_vs_iterable).

In a data config this looks like (see [ept document](./ept.md#large-non-tokenized-dataset) for a more in-depth example):
```
dataprocessor:
    type: default
    streaming: true
```

When using streaming, `split_batches` in the `TrainingArguments` will automatically be set to `True`, by doing so, the main process will fetch a full batch and slice it into `num_processes` batches for each process. This means that `num_processes` must be divisible by `batch_size`. This will replace the global batch size.

Note: Streaming datasets or use of `IterableDatasets` is not compatible with the fms-acceleration multipack plugin because multipack sampler has to run thorugh the full dataset every epoch. Using multipack and streaming together will raise an error.

**When using streaming, the user must set `max_steps` in the `TrainingArguments` instead of `num_train_epochs`.** Since iterable datasets are loaded chunk-by-chunk, data cannot run through epochs in a typical fashion as the **Trainer** can not know length of the dataset as it is being passed through. If both `max_steps` and `num_train_epochs` are given in a training config, `max_steps` will overwrite `num_train_epochs` since `max_steps` directly specifies the total number of optimization steps, which is needed when dataset length cannot be known. 

If the dataset size is known to the user, `max_steps` can be calculated as the total number of samples divided by the batch size.

### How users can specify the chat template

In the `data_config.yaml` file:

**✅ USE:**

```yaml
dataprocessor:
  chat_template: "my single line chat template"
```

The recommended way is to copy paste the chat template from the official checkpoint https://huggingface.co/ibm-granite/granite-3.1-8b-instruct/blob/main/tokenizer_config.json#L188


**✅ (Optional) USE:**

```yaml
dataprocessor:
  chat_template: |
    my multi-line chat template
```

Specifying a multi-line chat template will requires some manual effort on the user's part to ensure new lines are specified correctly.
This approach is mainly useful for readability, especially if you are customizing the chat template.

Example:

```yaml
dataprocessor:
  chat_template: |
    {%- if messages[0]['role'] == 'system' %}
        {%- set system_message = messages[0]['content'] %}
        {%- set loop_messages = messages[1:] %}
    {%- else %}
        {%- set system_message = "Knowledge Cutoff Date: April 2024.
    Today's Date: " + strftime_now('%B %d, %Y') + ".
    You are Granite, developed by IBM." %}
        {%- if tools and documents %}
    ................
```

**❌ DO NOT USE:**

```yaml
dataprocessor:
  chat_template: |
    my single line chat template
```

This can add extra backslashes to your chat template causing it to become invalid.

### Example data configs.

We provide some example data configs [here](../tests/artifacts/predefined_data_configs/)

# Use cases supported via command line argument `training_data_path`

For basic users who want to pass command line argument directly to our stack you can refer to the following supported data formats.

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
data_formatter_template: `### Input: {{input}} \n\n## Label: {{output}}`  

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

#### Aligning dataset formats
In some cases the chat template might not be aligned with the data format of the dataset. For example, consider the following data sample and suppose we want to use the list of contents associated with the `messages` key from the data sample for our multi-turn training job!

```
{
  "messages": [
    {"content": "You are an AI...", "role": "system"},
    {"content": "Look up a word...", "role": "user"},
    {"content": "A word that rhymes is 'mist'", "role": "assistant"}
  ],
  "group": "lab_extension",
  "dataset": "base/full-extension",
  "metadata": "{\"num_turns\": 2}"
}
```
Different Chat templates support different data formats and the chat template might not always align with the data format of the dataset!

Here is a example of chat template that iterates over the nested data sample by addressing the "messages" key in `for message in messages['messages']` :
```
{% for message in messages['messages'] %}\
  {% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + eos_token }}\
  {% elif message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + eos_token }}\
  {% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n'  + message['content'] + eos_token }}\
  {% endif %}\
  {% if loop.last and add_generation_prompt %}{{ '<|assistant|>' }}\
  {% endif %}\
{% endfor %}
```
While the above template might be suitable for certain data formats, not all chat templates access the nested contents in a data sample.

In the following example notice the `for message in messages` line which does not access any nested contents in the data and expects the nested content to be passed directly to the chat template!

```
{%- for message in messages %}\
  {%- if message['role'] == 'system' %}\
  {{- '<|system|>\n' + message['content'] + '\n' }}\
  {%- elif message['role'] == 'user' %}\
  {{- '<|user|>\n' + message['content'] + '\n' }}\
  {%- elif message['role'] == 'assistant' %}\
  {%- if not loop.last %}\
  {{- '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}\
  {%- else %}\
  {{- '<|assistant|>\n'  + message['content'] + eos_token }}\
  {%- endif %}\
  {%- endif %}\
  {%- if loop.last and add_generation_prompt %}\
  {{- '<|assistant|>\n' }}\
  {%- endif %}\
{%- endfor %}
```

When working with multi-turn datasets, it's often necessary to extract specific fields from the data depending on the format. For example, in many multi-turn datasets, conversations may be stored under a dedicated key (e.g., `conversations`, `messages`, etc), and you may only need the content of that key for processing.

```
{
  "conversations": [
    {"content": "You are an AI...", "role": "system"},
    {"content": "Look up a word...", "role": "user"},
    {"content": "A word that rhymes is 'mist'", "role": "assistant"}
  ],
  "group": "lab_extension",
  "dataset": "base/full-extension",
  "metadata": "{\"num_turns\": 2}"
}

```
To extract and use the conversations field, pass the following flag when running:
```
--dataset_conversation_field "conversations"
``` 

*Note:* For most cases, users using `Granite3.1+ Instruct` series models which already contain chat template should look to pass `--dataset_conversation_field "messages"` while using multi-turn data on the commandline or use `conversations_column` argument in the [data handler](https://github.com/foundation-model-stack/fms-hf-tuning/blob/30ceecc63f3e2bf3aadba2dfc3336b62187c240f/tests/artifacts/predefined_data_configs/mt_data_granite_3_1B_tokenize_and_mask_handler.yaml#L63) which processes chat template 

We recommend inspecting the data and chat template to decide if you need to pass this flag.

### Guidelines

Depending on various scenarios users might need to decide on how to use chat template with their data or which chat template to use for their use case.  

Following are the Guidelines from us in a flow chart :  
![guidelines for chat template](docs/images/chat_template_guide.jpg)  

Here are some scenarios addressed in the flow chart:  
1. Depending on the model the tokenizer for the model may or may not have a chat template  
2. If the template is available then the `json object schema` of the dataset might not match the chat template's `string format`
3. There might be special tokens used in chat template which the tokenizer might be unaware of, for example `<|start_of_role|>` which can cause issues during tokenization as it might not be treated as a single token  


#### Add Special Tokens
Working with multi-turn chat data might require the tokenizer to use a few new control tokens ( ex: `<|assistant|>`, `[SYS]` ) as described above in the guidelines. These special tokens might not be present in the tokenizer's vocabulary if the user is using base model.

Users can pass `--add_special_tokens` argument which would add the required tokens to the tokenizer's vocabulary.  
For example required special tokens used in `--instruction_template`/`--response_template` can be passed as follows:

```
python -m tuning.sft_trainer \
...
--add_special_tokens "<|start_of_role|>" "<|end_of_role|>" \
--instruction_template "<|start_of_role|>user<|end_of_role|>" \
--response_template "<|start_of_role|>assistant<|end_of_role|>"
```

### 4. Pre tokenized datasets.

Users can also pass a pretokenized dataset (containing `input_ids` and `labels` columns) as `--training_data_path` argument e.g.

At this time, the data preprocessor does not add EOS tokens to pretokenized datasets, users must ensure EOS tokens are included in their pretokenized data if needed.

```
python tuning/sft_trainer.py ... --training_data_path twitter_complaints_tokenized_with_maykeye_tinyllama_v0.arrow
```
