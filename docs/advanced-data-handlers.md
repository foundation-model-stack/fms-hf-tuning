# Data Handlers
Please note, this document is intended for advanced users who want to customize data handler arguments and use data handler functions to perform
complex operations on the data configs.

Data handlers, are routines which process a dataset using [HF process frameworks](https://huggingface.co/docs/datasets/en/process) including map, filter, remove, select, and rename. 
All data handler routines are registered with our data preprocessor as a `k:func` object where
`k` is the name (`str`) of the data handler and `func` (`callable`) is the function which is called.

In the data config, users can request which data handler to apply by requesting the corresponding `name`
with which the data handler was registered and specifying the appropriate `arguments`. Each data handler accepts two types of arguments via `DataHandlerArguments` (as defined in the data preprocessor [schema](./advanced-data-preprocessing.md#what-is-data-config-schema)), as shown below.

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

A typical YAML snippet where you'd specify arguments to the handlers
```
datapreprocessor:
  ...
  datasets:
    - name: my_dataset
      data_paths:
        - /path/to/my_dataset
      data_handlers:
        - name: tokenize
          arguments:
            # Additional kwargs passed directly to the underlying HF API call
            batched: false
            num_proc: 10
            
            fn_kwargs:
              # Any arguments specific to the tokenize handler itself
              truncation: true
              max_length: 1280
```

For example, `num_proc` and `batched` in the snippet above are passed straight to 
`datasets.Dataset.map(...) ` while, the `truncation` and `max_length` arguments 
in the snippet above directly control how the handler performs tokenization.

For native handlers like `REMOVE` `RENAME` `SELECT` (see below) you don't need to pass `fn_kwargs` and args need to be provided in `arguments`.

### Default Arguments
Each data handler supports many arguments and some of them are automatically provided to the data handler via the data processor framework. 
The data processor framework makes these arguments available to the data handlers via `kwargs`.

1. `tokenizer`: The `AutoTokenizer` representation of the `tokenizer_name_or_path` or from `model_name_or_path` arg passed to the library.
2. `column_names`: The names of the columns of the current dataset being processed.

**Also one special handling data preprocessor provides is to pass in `remove_columns` as `all` which will internally be translated to all column names to the `Map` of `Filter` data handler routines.**

## Preexisting data handlers
This library currently supports the following preexisting data handlers. These handlers could be requested by their same name and users can lookup the function args from [data handlers source code](https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/tuning/data/data_handlers.py):

### `tokenize_and_apply_input_masking`:
Tokenizes input text and applies masking to the labels for causal language modeling tasks, good for input/output datasets.
By default this handler adds `EOS_TOKEN` which can be disabled by a handler argument, see [this](https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/tests/artifacts/predefined_data_configs/tokenize_and_apply_input_masking.yaml) or the `add_eos_token` argument below.

Users don't need to pass any extra `response` or `instruction` templates here.

**Type: MAP**

**arguments**
 - Any argument supported by the [HF MAP API](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.map)

**fn_args:**
 - `element`: the HF Dataset element.
 - `input_column_name`: Name of the input (instruction) field in dataset
 - `output_column_name`: Name of the output field in dataset
 - `add_eos_token`: should add tokenizer.eos_token to text or not, defaults to True

**Returns:**
- tokenized Dataset element with input_ids, labels and attention_mask columns where labels contain masking of the `input` section of the dataset.

### `apply_custom_jinja_template`:
Applies a custom jinja template (e.g., Alpaca style) to format dataset elements.
Returns dataset which contains column `formatted_text_column_name` containing the string formatted using provided template.

Users need to pass in appropriate `response_template` if they specify this handler as the final handler to ensure the 
`DataCollatorForCompletionOnlyLM` used underneath to apply proper masking ensure the model learns only on responses.

**Type: MAP**

**arguments**
 - Any argument supported by the [HF MAP API](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.map)

**fn_args:**
 - `element`: the HF Dataset element
 - `formatted_text_column_name`: formatted_dataset_field.
 - `template`: Jinja template to format data with. Features of Dataset should be referred to by their key.

**Returns:**
- Formatted HF Dataset element by formatting dataset with provided jinja template, saving the result to `formatted_text_column_name` argument.

### `apply_tokenizer_chat_template`:
Uses tokenizer's chat template to preprocess dataset elements, good for single/multi turn chat templates.
Returns dataset which contains column `formatted_text_column_name` containing the chat template formatted string. 

Since this handler does not tokenize the dataset users need to provide appropriate `resonse_template` and `instruction_template` for the
`DataCollatorForCompletionOnlyLM` used underneath to apply proper masking ensure the model learns only on assistant responses.

**Type: MAP**

**arguments**
 - Any argument supported by the [HF MAP API](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.map)

**fn_args:**
 - `element`: the HF Dataset element.
 - `formatted_text_column_name`: the field in which to store the rendered text.
 - `conversation_column`: column name where the chat template expects the conversation

**Returns:**
- Formatted HF Dataset element by formatting dataset with tokenizer's chat template, saving the result to `formatted_text_column_name` argument.

### `tokenize_and_apply_chat_template_with_masking`:
Uses tokenizer's chat template to preprocess dataset elements, good for single/multi turn chat templates.
Then tokenizes the dataset while masking all user and system conversations ensuring model learns only on assistant responses.
Tokenizes the dataset so you don't need to pass any extra arguments for data collator. 

**Type: MAP**

**arguments**
 - Any argument supported by the [HF MAP API](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.map)
 *Note: - Always recommended to be used with `remove_columns:all` as argument as you don't want to retain text columns and tokenized columns alongside while training which can cause a potential crash.

**fn_args:**
 - `element`: the HF Dataset element.
 - `formatted_text_column_name`: the field in which to store the rendered text.
 - `conversation_column`: column name where the chat template expects the conversation

**Returns:**
- Tokenized Dataset element containing `input_ids` `labels` and `attention_mask`.

### `tokenize`:
Tokenizes one column of the dataset passed as input `text_column_name`.

**Type: MAP**

**arguments**
 - Any argument supported by the [HF MAP API](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.map)

**fn_kwargs:**
 - `element`: the HF Dataset element.
 - `text_column_name`: The dataset field to tokenize.
 - `truncation`: Truncation strategy to use, refer the link (https://huggingface.co/docs/transformers/en/pad_truncation).
 - `max_length`: Max length to truncate the samples to.

**Return:**
- Tokenized dataset element field `text_column_name` containing `input_ids` and `labels`

### `duplicate_columns`:
Duplicate one columne of a dataset to another new column.

**Type: MAP**

**arguments**
 - Any argument supported by the [HF MAP API](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.map)

**fn_args:**
 - `element`: the HF Dataset element
 - `existing_column_name`: Name of the column to be duplicated
 - `new_column_name`: Name of the new column where dyplicated column is saved

**Return:**
- Formatted HF dataset element with `new_column_name` where `existing_column_name` content is copied.

### `skip_samples_with_large_columns`:
Skips elements which contains certain columns larger than the passed max length in the dataset.

**Type: FILTER**

**arguments**
 - Any arguments supported by [HF Filter API](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.filter)

**fn_args**:
 - `element`: HF dataset element.
 - `column_name`: Name of column to be filtered.
 - `max_allowed_length`: Max allowed lenght of column in either characters or tokens.

**Return:**
- A filtered dataset which contains elements with length of column `column_name` shorter than max allowed length

### `remove_columns`:
Directly calls [remove_columns](https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/main_classes#datasets.Dataset.remove_columns) in HF API over the dataset.

**Type: REMOVE**

**arguments**:
 - `column_names`: Names of columns to be removed from dataset

**fn_args**:
 - Nil. As this is a Native API.

**Returns:**
- Dataset with specified `column_names` removed

### `select_columns`:
Directly calls [select](https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/main_classes#datasets.Dataset.select) in HF API

**Type: SELECT**

**arguments**:
 - `column_names`: Names of columns to be retained in the new dataset

**fn_args**:
- Nil. As this is a Native API.

**Returns:**
- Dataset where only columns specified in `column_names` are retained.

### `rename_columns`:
Directly calls [rename_columns](https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/main_classes#datasets.Dataset.rename_columns) in HF API

**Type: RENAME**

**arguments**:
 - `column_mapping`: Column names passed as `str:str` from `old_name:new_name`

**fn_args**:
 - Nil. As this is a Native API.

**Returns:**
- Dataset where columns are renamed using provided column mapping.

## Additional arguments
Please note that the choice of extra arguments needed with handler depends on how the dataset looks post processing which is a combination post
application of the full DAG of the data handlers and should be used be referring to our other documentation [here](https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/README.md) and [here](https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/docs/advanced-data-preprocessing.md) and reference templates provided [here](https://github.com/foundation-model-stack/fms-hf-tuning/tree/main/tests/artifacts/predefined_data_configs)


## Extra data handlers
Users are also allowed to pass custom data handlers using [`sft_trainer.py::train()`](https://github.com/foundation-model-stack/fms-hf-tuning/blob/d7f06f5fc898eb700a9e89f08793b2735d97889c/tuning/sft_trainer.py#L71) API call via the [`additional_data_handlers`](https://github.com/foundation-model-stack/fms-hf-tuning/blob/d7f06f5fc898eb700a9e89f08793b2735d97889c/tuning/sft_trainer.py#L89) argument.

The argument expects users to pass a map similar to the existing data handlers `k(str):func(callable)` which will be registered with the data preprocessor via its [`register_data_handlers`](https://github.com/foundation-model-stack/fms-hf-tuning/blob/d7f06f5fc898eb700a9e89f08793b2735d97889c/tuning/data/data_processors.py#L65) api