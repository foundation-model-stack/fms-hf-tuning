# Extended Pre Training Support
Our library also supports Extended Pre-Training (EPT), which is generally useful when users want to train a pretrained model on a large number of samples. The training behaviour of EPT is similar to that of pretraining where users might wanna make sure the models runs through entire corpus of data available and be trained on whole set of tokens without any specific masking.

See [below](#additional-information) for information on when this document was last updated and the release which supports this feature.

## Packing support

We support training via `packing` dataset samples by specifing `--packing=True` in the command line parameters. Users can choose to specify `--max_seq_len=<value like 4k/8k>` to provide the maxium sequence length of each chunk post packing.

We provide below details on how to use different style of datasets with the library.

## Non-Tokenized Dataset

### Single Non-Tokenized Dataset
Users can pass a single dataset to the library by using a [data_config](./advanced-data-preprocessing.md#data-config). 
Lets say you have a `JSONL` data file which contains text to be trained on in each line that you want to perform EPT on, you can create a `data_config` for the dataset in this manner,

Example dataset,

```
{"Tweet":"@HMRCcustomers No this is my first job","ID":0,"Label":2,"text_label":"no complaint","output":"### Text: @HMRCcustomers No this is my first job\n\n### Label: no complaint"}
{"Tweet":"@KristaMariePark Thank you for your interest! If you decide to cancel, you can call Customer Care at 1-800-NYTIMES.","ID":1,"Label":2,"text_label":"no complaint","output":"### Text: @KristaMariePark Thank you for your interest! If you decide to cancel, you can call Customer Care at 1-800-NYTIMES.\n\n### Label: no complaint"}
...
```

Sample data config for the above use case.
```
dataprocessor:
    type: default
datasets:
  - name: non_tokenized_text_dataset
    data_paths:
      - "<path-to-the-jsonl-dataset>"
        data_handlers:
        - name: add_tokenizer_eos_token
            arguments:
            remove_columns: all
            batched: false
            fn_kwargs:
                dataset_text_field: "dataset_text_field"
```

And the commandline passed to the library should include following.

```
--data_config <path to the data config> --packing=True --max_seq_len 8192
```

Please note that for non tokenized dataset our code adds `EOS_TOKEN` to the lines, for e.g. `Tweet` column before passing that as a dataset.

### Multiple Non Tokenized Datasets

If a user wants to utilize multiple datasets and want to [`sample`](./advanced-data-preprocessing.md#how-the-user-can-write-data-configs) the datasets. This can be achieved by specifying multiple datasets in the data config with different sampling ratios.

Sample data config for sampling among multiple datasets
```
dataprocessor:
    type: default
    sampling_stopping_strategy: first_exhausted
    seed: 66
datasets:
  - name: non_tokenized_text_dataset_1
    sampling: 0.3
    data_paths:
      - "FILE_PATH"
    data_handlers:
      - name: apply_custom_data_formatting_template
        arguments:
          remove_columns: all
          batched: false
          fn_kwargs:
            dataset_text_field: "dataset_text_field"
            template: "dataset_template"
  - name: non_tokenized_text_dataset_2
    sampling: 0.4
    data_paths:
      - "FILE_PATH"
    data_handlers:
      - name: apply_custom_data_formatting_template
        arguments:
          remove_columns: all
          batched: false
          fn_kwargs:
            dataset_text_field: "dataset_text_field"
            template: "dataset_template"
  - name: non_tokenized_text_dataset_3
    sampling: 0.3
    data_paths:
      - "FILE_PATH"
    data_handlers:
      - name: apply_custom_data_formatting_template
        arguments:
          remove_columns: all
          batched: false
          fn_kwargs:
            dataset_text_field: "dataset_text_field"
            template: "dataset_template"
```

NOTE: More in-depth documentation of `sampling_stopping_strategy` and how to specify data mixing parameters in the `data_config` is covered in the [data mixing](./advanced-data-preprocessing.md#data-mixing) section of the advanced data preprocessing documentation

Here also the command line arguments would be 

```
--data_config <path to the data config> --packing=True --max_seq_len 8192
```

The code again would add `EOS_TOKEN` to the non tokenized data before using it and also note that the `dataset_text_field` is assumed to be same across all datasets for now.

### Large Non-Tokenized Dataset
Let's say you have a large JSONL data file that cannot all fit into memory at once and you want to perform EPT on it, you can use the streaming feature to efficiently load and process data in chunks. To enable streaming, you can define a data_config as follows:

Sample data config for the above use case.
```
dataprocessor:
    type: default
    streaming: true
datasets:
  - name: non_tokenized_text_dataset
    data_paths:
      - "<path-to-the-jsonl-dataset>"
        data_handlers:
        - name: add_tokenizer_eos_token
            arguments:
            remove_columns: all
            batched: false
            fn_kwargs:
                dataset_text_field: "dataset_text_field"
```

The command-line arguments passed to the library should include the following:

```
--data_config <path to the data config> --packing=True --max_seq_len 8192 --max_steps <num training steps>
```

Please note when using streaming, user must pass `max_steps` instead of `num_train_epochs`. See advanced data preprocessing [document](./advanced-data-preprocessing.md#data-streaming) for more info.

### Additional Information
This feature is supported post [v2.3.1](https://github.com/foundation-model-stack/fms-hf-tuning/releases/tag/v2.3.1) of this library.
Post Last Updated On: 12-02-2025