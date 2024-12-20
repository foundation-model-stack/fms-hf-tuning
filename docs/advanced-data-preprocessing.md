# Advanced Data Processing
Our library also supports a powerful data processing backed which can be used by the users to perform custom data preprocessing including
1. Support for multiple datasets
1. Creating custom data processing pipeline for the datasets.
1. Combining multiple datasets into one, even if they have different formats.
1. Mixing datasets as requried and sampling if needed each with different weights.

These things are supported via what we call a [`data_config`](#data-config) which can be passed an an argument to sft trainer.

## Data Config

Data config is a YAML configuration file which users can provide to `sft_trainer.py`, in this file they can include a variety of datasets and configurations. This data config is passed to `sft_trainer` through the `--data_config` flag.

### What is data config schema 
The data config schema is designed to define datasets and their processing strategies in a structured way. It consists of the following top-level keys:
 - `datapreprocessor`: Defines global data processing parameters, such as the type (`default`), sampling stopping strategy (`all_exhausted` or `first_exhausted`), and sampling seed for reproducibility.
 - `datasets`: A list of dataset configurations, each describing the dataset name, paths, optional builders, sampling ratios, and data handlers.

At the top level, the data config looks like this:
```yaml
datapreprocessor:
    ...
datasets:
    ...
```

### How the user can write data configs
Users can create a data config file in YAML format. The file should follow the schema outlined above with the following parameters:

`datapreprocessor`:
 - `type` (optional): Type of data preprocessor, `default` is currently the only supported type.
 - `sampling_stopping_strategy` (optional): Stopping strategy, either `all_exhausted` or `first_exhausted`, defaults to `all_exhausted`.
 - `sampling_seed` (optional): An int for reproducibility, defaults to 42.

`datasets`:
 - `name`: A unique identifier for the dataset.
 - `data_paths`: A list of file paths or directories containing the dataset.
 - `builder` (optional): Specifies a Hugging Face dataset builder, if applicable.
 - `sampling` (optional): A float representing the sampling ratio (0.0 to 1.0).
 - `data_handlers` (optional): A list of data handler configurations.

For examples, see [predefined_data_configs](../tests/artifacts/predefined_data_configs/).

### What are data handlers
Data handlers are customizable components within the data config that allow users to preprocess or manipulate individual datasets. Each data handler has:
- `name`: The handler's unique identifier.
- `arguments`: A dictionary of parameters specific to the handler.

#### Preexisting data handlers
This library currently supports four preexisting data handlers:
 - `tokenize_and_apply_input_masking`: Tokenizes input text and applies masking to the labels for causal language modeling tasks, good for input/output datasets.
 - `apply_dataset_formatting`: Formats a dataset by appending an EOS token to a specified field.
 - `apply_custom_data_formatting_template`: Applies a custom template (e.g., Alpaca style) to format dataset elements.
 - `apply_tokenizer_chat_template`: Uses a tokenizer's chat template to preprocess dataset elements, good for single/multi turn chat templates.

#### Extra data handlers
Users can define custom data handlers by implementing their logic and specifying their names and arguments in the data config.

### How the user can pass the datasets 
`data_paths` can be either file and/or folder paths and can combine datasets with different formats as long as they have the same columns. Users can also use globbing patterns to pass in datasets following a specific regex pattern.

### What kind of datasets can be passed
The library supports datasets of type JSON, JSONL, Parquet, and Arrow. To see an up to date version of supported dataset types, see the `get_loader_for_filepath` function in [utils.py](../tuning/utils/utils.py).
Of these datatypes, the library supports pretokenized datasets and other datasets supported by data handlers, either preexisting or custom.

### How the user can perform sampling
 - What does sampling means?
    - Sampling allows users to specify a subset of the data to be processed. For example, a sampling ratio of 0.5 means 50% of the dataset will be used during training.
 - How will it affect the datasets
    - Sampling reduces the size of the dataset processed during training, which can speed up processing or focus on specific portions of the data.

### How the user can create a data config for the existing use cases.

### Corner cases which needs attention.