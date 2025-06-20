# Advanced Data Processing
Our library also supports a powerful data processing backend which can be used by the users to perform custom data preprocessing including
1. Support for multiple datasets  
1. Creating custom data processing pipeline for the datasets.  
1. Combining multiple datasets into one, even if they have different formats.  
1. Mixing datasets as required and sampling each dataset with different weights.

These things are supported via what we call a [`data_config`](#data-config) which can be passed as an argument to sft trainer.

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
      rename_columns:
        type: object
      retain_columns:
        type: object
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
    - `rename_columns` (optional, dict[str:str]): Specifies a dictionary of columns to rename like `{"old_name": "new_name"}` at dataset load time. *Applied before `retain_columns` if both are specified*.
    - `retain_columns` (optional, list[str]): Specifies a list of columns to retain `["input_ids", "labels"]` every other column will be dropped at dataset load time. *Applied strictly after `rename_columns` if both are specified*.
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

Probably something like this:

Additionally while loading the dataset, users can specify which columns to rename via `rename_columns` and which to retain via `retain_columns` arguments above.
The order of application of these operations is *strictly rename followed by retain* so users should note that an old column name which is renamed will not be available in retain and hence should be careful while applying these operations. The code will throw a `ValueError` in case user specified a column requested to be renamed via rename argument in retain argument as well. 

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


