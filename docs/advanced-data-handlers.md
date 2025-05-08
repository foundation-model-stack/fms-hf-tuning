# Data Handlers
Data handlers, are routines which process a dataset using [HF process frameworks](https://huggingface.co/docs/datasets/en/process) including map, filter, remove, select, and rename. 
All data handler routines are registered with our data preprocessor as a `k:func` object where
`k` is the name (`str`) of the data handler and `func` (`callable`) is the function which is called.

In the data config, users can request which data handler to apply by requesting the corresponding `name`
with which the data handler was registered and specifying the appropriate `arguments`. Each data handler accepts two types of arguments via `DataHandlerArguments` (as defined in the data preprocessor [schema](./advanced-data-preprocessing.md#what-is-data-config-schema)), as shown below.

```yaml
datapreprocessor:
    ...
datasets:
  - name: ...
    data_paths:
      - ...
    data_handlers:
      - name: str
        arguments:
          argument: object
          ...
          argument: object
          fn_kwargs:
            fn_kwarg: object
            ...
            fn_kwarg: object
  ...
```

Arguments to the data handlers are of two types,

Each data handler is a routine passed to an underlying HF API so the `kwargs` supported by the underlying API can be passed via the `arguments` section of the data handler config. In our pre-existing handlers the supported underlying API is either:
 - [Map](https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/main_classes#datasets.Dataset.map)
 - [Filter](https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/main_classes#datasets.Dataset.filter)
 - [Rename](https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/main_classes#datasets.Dataset.rename_columns)
 - [Select](https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/main_classes#datasets.Dataset.select)
 - [Remove](https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/main_classes#datasets.Dataset.remove_columns)

The operations performed for pre-existing handlers can be found in the [Preexisting data handlers](#preexisting-data-handlers) section

For example, users can pass `batched` through `arguments` to ensure [batched processing](https://huggingface.co/docs/datasets/en/about_map_batch) of the data handler.

Users can also pass any number of `kwargs` arguments required for each data handling `routine` function as [`fn_kwargs`](https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/main_classes#datasets.Dataset.map.fn_kwargs) inside the arguments.


## Preexisting data handlers
This library currently supports the following preexisting data handlers. These handlers could be requested by their same name and users can lookup the function args from [data handlers source code](https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/tuning/data/data_handlers.py):


### `tokenize_and_apply_input_masking`:
Tokenizes input text and applies masking to the labels for causal language modeling tasks, good for input/output datasets.
By default this handler adds `EOS_TOKEN` which can be disabled by a handler argument, [see](https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/tests/artifacts/predefined_data_configs/tokenize_and_apply_input_masking.yaml) 

Type: MAP

Args:
 - `element`: the HF Dataset element.
 - `tokenizer`: Tokenizer to be used for tokenization.
 - `column_names`: Name of all the columns in the dataset.
 - `input_field_name`: Name of the input (instruction) field in dataset
 - `output_field_name`: Name of the output field in dataset
 - `add_eos_token`: should add tokenizer.eos_token to text or not, defaults to True

Returns formatted Dataset element with input_ids, labels and attention_mask columns

### `add_tokenizer_eos_token`:
Appends the tokenizer's EOS token to a specified dataset field.

Type: MAP

Args:
 - `element`: the HF Dataset element.
 - `tokenizer`: Tokenizer to be used for the EOS token, which will be appended when formatting the data into a single sequence. Defaults to empty.
 - `dataset_text_field`: Text column name of the dataset where EOS is to be added.

Returns formatted Dataset element with EOS added to dataset_text_field of the element.

### `apply_custom_data_formatting_template`:
Applies a custom template (e.g., Alpaca style) to format dataset elements.
By default this handler adds `EOS_TOKEN` which can be disabled by a handler argument, [see](https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/tests/artifacts/predefined_data_configs/apply_custom_template.yaml)

Type: MAP

Args:
 - `element`: the HF Dataset element.
 - `tokenizer`: Tokenizer to be used for the EOS token, which will be appended when formatting the data into a single sequence. Defaults to empty.
 - `dataset_text_field`: Text column name of the dataset where formatted text is saved.
 - `template`: Template to format data with. Features of Dataset should be referred to by their key.
 - `add_eos_token`: should add tokenizer.eos_token to text or not, defaults to True.

Returns formatted Dataset element by formatting dataset with template+tokenizer.EOS_TOKEN, saving the result to dataset_text_field argument.

### `apply_custom_jinja_template`:
Applies a custom jinja template (e.g., Alpaca style) to format dataset elements.
By default this handler adds `EOS_TOKEN` which can be disabled by a handler argument, [see](https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/tests/artifacts/predefined_data_configs/apply_custom_jinja_template.yaml)

Type: MAP

Args:
 - `element`: the HF Dataset element
 - `tokenizer`: Tokenizer to be used for the EOS token, which will be appended
    when formatting the data into a single sequence. Defaults to empty.
 - `dataset_text_field`: formatted_dataset_field.
 - `template`: Template to format data with. Features of Dataset should be referred to by their key.
 - `add_eos_token`: should add tokenizer.eos_token to text or not, defaults to True.

Returns formatted HF Dataset element by formatting dataset with provided jinja template, saving the result to dataset_text_field argument.

### `apply_tokenizer_chat_template`:
Uses a tokenizer's chat template to preprocess dataset elements, good for single/multi turn chat templates.

Type: MAP

Args:
 - `element`: the HF Dataset element.
 - `tokenizer`: Tokenizer to be used.
 - `dataset_text_field`: the field in which to store the rendered text.
 - `conversation_column`: column name where the chat template expects the conversation

Returns formatted HF Dataset element by formatting dataset with tokenizer's chat template, saving the result to dataset_text_field argument.

### `tokenize`:
Tokenizes one column of the dataset passed as input `dataset_text_field`.

Type: MAP

Args:
 - `element`: the HF Dataset element.
 - `tokenizer`: Tokenizer to be used.
 - `dataset_text_field`: The dataset field to tokenize.
 - `truncation`: Truncation strategy to use, refer the link (https://huggingface.co/docs/transformers/en/pad_truncation).
 - `max_length`: Max length to truncate the samples to.

Returns tokenized dataset element field `dataset_text_field`

### `duplicate_columns`:
Duplicate one columne of a dataset to another new column.

Type: MAP

Args:
 - `element`: the HF Dataset element
 - `old_column`: Name of the column to be duplicated
 - `new_column`: Name of the new column where dyplicated column is saved

Returns formatted HF dataset element with `new_column` where `old_column` content is deep copied.

### `skip_large_columns`:
Skips elements which contains certain columns larger than the passed max length in the dataset.

Type: FILTER

Args:
 - `element`: HF dataset element.
 - `column_name`: Name of column to be filtered.
 - `max_length`: Max allowed lenght of column in either characters or tokens.

 Returns a filtered dataset which contains elements with length shorter than max length

### `remove_columns`:
Directly calls [remove_columns](https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/main_classes#datasets.Dataset.remove_columns) in HF API

Type: REMOVE

Args:
 - `column_names`: Names of columns to be removed from dataset

Removes specified columns of dataset

### `select_columns`:
Directly calls [select](https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/main_classes#datasets.Dataset.select) in HF API

Type: SELECT

Args:
 - `column_names`: Names of columns to be retained in the new dataset

Create a new dataset with rows selected following the list/array of indices.

### `rename_columns`:
Directly calls [rename_columns](https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/main_classes#datasets.Dataset.rename_columns) in HF API

Type: RENAME

Args:
 - `column_mapping`: Column names passed as `str:str` from `old_name:new_name`

Returns renamed columns in dataset using provided column mapping.


## Extra data handlers
Users are also allowed to pass custom data handlers using [`sft_trainer.py::train()`](https://github.com/foundation-model-stack/fms-hf-tuning/blob/d7f06f5fc898eb700a9e89f08793b2735d97889c/tuning/sft_trainer.py#L71) API call via the [`additional_data_handlers`](https://github.com/foundation-model-stack/fms-hf-tuning/blob/d7f06f5fc898eb700a9e89f08793b2735d97889c/tuning/sft_trainer.py#L89) argument.

The argument expects users to pass a map similar to the existing data handlers `k(str):func(callable)` which will be registered with the data preprocessor via its [`register_data_handlers`](https://github.com/foundation-model-stack/fms-hf-tuning/blob/d7f06f5fc898eb700a9e89f08793b2735d97889c/tuning/data/data_processors.py#L65) api

## Examples
To see typical use-cases and how handlers are linked together, see [data preprocessing recipes](./data-preprocessing-recipes.md).