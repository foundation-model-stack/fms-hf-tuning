# Extended Pre Training Support
Our library also supports Extended Pre-Training or (EPT) which is generally useful when users might want to to train a pretrained model on large number of samples. The training behaviour of EPT is similar to that of pretraining where users might wanna make sure the models runs through entire corpus of data available and be trained on whole set of tokens without any specific masking.

## Packing support

We support training via `packing` dataset samples by specifing `--packing=True` in the command line parameters. Users can choose to specify `--max_seq_len=<value like 4k/8k>` to provide the maxium sequence length of each chunk post packing.

We provide below details on how to use different style of datasets with the library.

## Non-Tokenized Dataset

### Single Non-Tokenized Dataset
Users can pass a single dataset to the library by using a [data_config](./advanced-data-preprocessing.md#data-config). 
Lets say you have a `JSONL` data file which contains text to be trained on in each line and you want to perform EPT on the same, you can create a `data_config` for the same in this manner,

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
```

And the commandline passed to the library should include following.

```
--data_config <path to the data config> --dataset_text_field "Tweet" --packing=True --max_seq_len 8192
```

Please note that for non tokenized dataset our code adds `EOS_TOKEN` to the lines, for e.g. `Tweet` column before passing that as a dataset.

### Multiple Non Tokenized Datasets

If a user wants to utilise multiple datasets and want to [`sample`](./advanced-data-preprocessing.md#how-the-user-can-write-data-configs) the datasets. This can be acheived by specifying in the data config multiple datasets with differnt sampling ratios.

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
  - name: non_tokenized_text_dataset_2
    sampling: 0.4
    data_paths:
      - "FILE_PATH"
  - name: non_tokenized_text_dataset_3
    sampling: 0.3
    data_paths:
      - "FILE_PATH"
```

Please note we cover what different `sampling_strategies` mean and how to specify them in `data_config` as part of our document on [data mixing in advanced data preprocessing](./advanced-data-preprocessing.md#data-mixing)

Here also the command line arguments would be 

```
--data_config <path to the data config> --dataset_text_field "Tweet" --packing=True --max_seq_len 8192
```

The code again would add `EOS_TOKEN` to the non tokenized data before using it and also note that the `dataset_text_field` is assumed to be same across all datasets for now.