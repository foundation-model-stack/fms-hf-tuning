# Data Handlers
Data handlers are routines which process a dataset using [Hugging Face process frameworks](https://huggingface.co/docs/datasets/en/process) including `map`, `filter`, `remove`, `select`, and `rename_columns`. Each data handler routine is registered with our data preprocessor as a `k:func` object, where:

- k (str): The name of the data handler.
- func (callable): The function called to perform the operation.

In the data config, you can request a particular data handler by referring to its registered `name` and specifying the appropriate arguments. Each data handler accepts two types of arguments via `DataHandlerArguments` (defined in the data preprocessor [schema](https://github.com/foundation-model-stack/fms-hf-tuning/pull/494/advanced-data-preprocessing.md#what-is-data-config-schema)):
- Top-level arguments `arguments` – Control the behavior of the handler itself.
- Key word arguments `fn_kwargs` – Passed directly to the underlying Hugging Face Datasets API call (e.g., `map`, `filter`, etc.).

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

A typical YAML snippet might look like:

```yaml
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


## Preexisting data handlers
This library currently supports the following preexisting data handlers. These handlers can be requested by their same name and users can look up the function args from [data handlers source code](https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/tuning/data/data_handlers.py):


### `tokenize_and_apply_input_masking`:
Tokenizes input text and applies masking to the labels for causal language modeling tasks, good for input/output datasets.
By default, this handler appends the tokenizer’s `EOS_TOKEN` to the combined input-output string. This behavior can be disabled using the handler argument `add_eos_token`, [see](https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/tests/artifacts/predefined_data_configs/tokenize_and_apply_input_masking.yaml).

**Type**: MAP

**Args**:
 - `element`: The HF Dataset element.
 - `tokenizer`: Tokenizer to be used for tokenization.
 - `column_names`: List of all the columns in the dataset.
 - `input_column_name`: Name of the input (instruction) column in dataset.
 - `output_column_name`: Name of the output (response) column in dataset.
 - `add_eos_token`: Whether to append the tokenizer's EOS token to the text. Defaults to True. 

**Returns**: Formatted Dataset element with `input_ids`, `labels` and `attention_mask` columns.

### `add_tokenizer_eos_token`:
Appends the tokenizer's EOS token to a specified text column in a dataset. This handler is designed to ensure that the specified column always ends with the EOS token.

**Type**: MAP

**Args**:
 - `element`: The HF Dataset element.
 - `tokenizer`: Tokenizer to be used for the EOS token, which will be appended when formatting the data into a single sequence. Defaults to empty.
 - `text_column_name`: Name of the text column where the EOS token should be appended.

**Returns**: Formatted Dataset element with EOS added to `text_column_name` of the element.

### `apply_custom_data_formatting_template`:
Applies a custom template (e.g., Alpaca style) to format dataset elements.
By default, this handler appends the tokenizer's `EOS_TOKEN` to the formatted text. This behavior can be disabled using the handler argument `add_eos_token`, [see](https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/tests/artifacts/predefined_data_configs/apply_custom_template.yaml).

**Type**: MAP

**Args**:
 - `element`: the HF Dataset element.
 - `tokenizer`: Tokenizer to be used for the EOS token, which will be appended when formatting the data into a single sequence. Defaults to empty.
 - `formatted_text_column_name`: Name of the dataset column where formatted text is to be saved. If doesn't exist a new column will be created.
 - `template`: Custom template used for formatting the data. Dataset features should be referenced using the placeholder syntax `{{key}}`.
 - `add_eos_token`: Whether to append the tokenizer's EOS token to the template. Defaults to True.

**Returns**: Formatted Dataset element, by formatting dataset with template+tokenizer.EOS_TOKEN, saving the result to the column specified by `formatted_text_column_name` argument.

### `apply_custom_jinja_template`:
Applies a custom jinja template (e.g., Alpaca style) to format dataset elements.
This handler renders a Jinja template by replacing placeholders with corresponding values from the dataset element. By default, the handler appends the tokenizer's EOS token to the template output, though this can be disabled via the add_eos_token argument, [see](https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/tests/artifacts/predefined_data_configs/apply_custom_jinja_template.yaml).

**Type**: MAP

**Args**:
 - `element`: The HF Dataset element.
 - `tokenizer`: Tokenizer to be used for the EOS token, which will be appended when formatting the data into a single sequence. Defaults to empty.
 - `formatted_text_column_name`: Name of the dataset column where formatted text is to be saved. If doesn't exist a new column will be created.
 - `template`: The Jinja template used for formatting the data. Dataset features should be referenced using the placeholder syntax (e.g., `{{key}}`).
 - `add_eos_token`: Whether to append the tokenizer's EOS token to the formatted text. Defaults to True.

**Returns**: Formatted HF Dataset element, by formatting dataset with provided jinja template, saving the result to the column specified by `formatted_text_column_name` argument.

### `apply_tokenizer_chat_template`:
Uses a tokenizer's chat template to preprocess dataset elements, good for single/multi turn chat templates. This handler formats the dataset element using the tokenizer’s chat template without performing tokenization.

**Type**: MAP

**Args**:
 - `element`: The HF Dataset element.
 - `tokenizer`: The tokenizer whose chat template is applied.
 - `formatted_text_column_name`: The name of the column where the rendered text will be stored.
 - `conversation_column`: The column name from which to extract the conversation. If not provided or not found in the element, the entire element is used as the conversation.

**Returns**: Formatted HF Dataset element, by formatting dataset with tokenizer's chat template, saving the result to the column specified by `formatted_text_column_name` argument.

### `tokenize`:
Tokenizes the text from a specified dataset column `text_column_name` using a provided tokenizer. 

**Type**: MAP

**Args**:
 - `element`: The HF Dataset element.
 - `tokenizer`: Tokenizer to be used.
 - `text_column_name`: The dataset column containing text to be tokenized.
 - `truncation`: The truncation strategy to use. Defaults to True. For more details, refer to the [Hugging Face documentation](https://huggingface.co/docs/transformers/en/pad_truncation).
 - `max_length`: The maximum length to which the samples should be truncated.

**Returns**: Tokenized dataset element with the tokenized output of text column `text_column_name`.

### `duplicate_columns`:
Duplicates the content of one dataset column into a new column. 

**Type**: MAP

**Args**:
 - `element`: The HF Dataset element.
 - `old_column`: The name of the column to duplicate.
 - `new_column`: The name of the new column where the duplicated column content will be stored.

**Returns**: Formatted dataset element with both the original (`old_column_name`) and the duplicated (`new_column_name`) columns present. The new column is a deep copy of the original.

### `skip_large_columns`:
Skips and filters out dataset elements that contains the specified column with a length exceeding passed `max_length`.

**Type**: FILTER

**Args**:
 - `element`: The HF dataset element.
 - `column_name`: The name of the column to be checked for filter.
 - `max_length`: The maximum allowed length for the column’s value. If the column contains token IDs (e.g., `input_ids`), the length is measured in tokens. Otherwise, the length is measured in characters.

**Returns**: Filtered dataset which contains elements with length shorter than or equal to `max_length`.

### `remove_columns`:
Directly calls the [remove_columns](https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/main_classes#datasets.Dataset.remove_columns) function from the Hugging Face Datasets API to remove specified columns from a dataset.

**Type**: REMOVE

**Args**:
 - `column_names`: List of column names to be removed from the dataset.

**Behavior**: Removes the specified columns from the dataset, returning a new dataset without those columns.

### `select_columns`:
Directly calls the [select function](https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/main_classes#datasets.Dataset.select) from the Hugging Face Datasets API to retain only the specified columns.

**Type**: SELECT

**Args**:
 - `column_names`: List of column names to be retained in the dataset.

**Behavior**: Creates a new dataset by selecting only the columns listed in column_names, effectively filtering out the rest.

### `rename_columns`:
Directly calls the [rename_columns function](https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/main_classes#datasets.Dataset.rename_columns) from the Hugging Face Datasets API to rename dataset columns.

**Type**: RENAME

**Args**:
 - `column_mapping`: A mapping of column names in the format {old_name (str): new_name (str)}.

**Returns**: A dataset with columns renamed as specified in the provided `column_mapping`.


## Extra data handlers
Users are also allowed to pass custom data handlers using [`sft_trainer.py::train()`](https://github.com/foundation-model-stack/fms-hf-tuning/blob/d7f06f5fc898eb700a9e89f08793b2735d97889c/tuning/sft_trainer.py#L71) API call via the [`additional_data_handlers`](https://github.com/foundation-model-stack/fms-hf-tuning/blob/d7f06f5fc898eb700a9e89f08793b2735d97889c/tuning/sft_trainer.py#L89) argument.

The argument expects users to pass a map similar to the existing data handlers `k(str):func(callable)` which will be registered with the data preprocessor via its [`register_data_handlers`](https://github.com/foundation-model-stack/fms-hf-tuning/blob/d7f06f5fc898eb700a9e89f08793b2735d97889c/tuning/data/data_processors.py#L65) API.

## Examples
To see typical use-cases and how handlers are linked together, see [data preprocessing recipes](./data-preprocessing-recipes.md).