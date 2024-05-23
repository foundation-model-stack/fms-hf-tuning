# Data Formatting / Evaluation
This doc describes how to pull & format datasets into something that can be run against this repository for evaluation.

## Pulling and Converting Datasets Into Alpaca Format
In order to pull and format the datasets, you'll need to set the following environment variables:

```bash
export S3_ACCESS_KEY_ID={YOUR S3 KEY}
export S3_SECRET_ACCESS_KEY={YOUR S3 SECRET ACCESS KEY}
export S3_ENDPOINT={YOUR S3 ENDPOINT}
```

Next, pull and format the datasets. In order to run the data formatting & evaluation, you'll need a few extra dependencies, namely `boto3` and `scikit-learn`, which you can pull from the dev dependencies.

NOTE: If you are running this from inside of a container with the library already installed, the easiest way to run evaluation is to copy & install the project's dev dependencies e.g., with `pip3 install .[dev]`, inside of a virtual environment in the container.

To pull and format the datasets into Alpaca format, run the following command.
```bash
python3 pull_and_format_datasets.py
```

Make sure you see everything under `formatted`! It should look something like this. Note that the numeric prefix for train/test files indicates how many samples were randomly sampled into the Alpaca formatted file:
```bash
ls -R formatted_data/

formatted_data/:
Entities  cc_tone  tsa_mams

formatted_data/Entities:
1000_IBM_NET_2019_DELIVERABLE_VERSION_C_train.json  IBM_NET_2019_DELIVERABLE_VERSION_C_dev.json
500_IBM_NET_2019_DELIVERABLE_VERSION_C_test.json

formatted_data/cc_tone:
1000_train.json  500_test.json

formatted_data/tsa_mams:
1000_train.json  500_test.json validation.json
```

## Converting Alpaca Format Datasets Into SFT Format

In order to run a tuning on the datasets, you'll need to convert the Alpaca format datasets into the SFT format. You can do this by with `alpaca_to_sft_format.py`; pass each Alpaca formatted file that you want to convert as part of the `--files` argument, as shown below.

```bash
python3 alpaca_to_sft_format.py \
    --files \
    formatted_data/cc_tone/1000_train.json \
    formatted_data/Entities/1000_IBM_NET_2019_DELIVERABLE_VERSION_C_train.json \
    formatted_data/tsa_mams/1000_train.json
```

Now, you should have a `sft_format_{file_name}` in the same location as each of your aforementioned files. E.g.,
```bash
ls formatted_data/cc_tone/

1000_train.json # Alpaca format - don't worry about this one anymore
500_test.json # You will use this one for evaluation
sft_format_1000_train.json # You will use this one for tuning

# And anything for validation, you can ignore
```

### Adding New Datasets to Pull / Format Script
The above command pulls and formats different datasets in the Alpaca format. In general, this needs to be updated per dataset being consumed, as it is dependent on the raw format of the dataset being converted.

To add a new dataset, you'll need to add a new entry to the `DATASET_INFOS` in `pull_and_format_datasets.py`, which includes:

- The location in the COS instance to pull from; it's assumed that all datasets live in the same COS instance
- A formatting function which loads the raw dataset and exports it in the expected format for the task being considered; there are currently examples for entities/TSA and tone. Running the script and inspecting the Alpaca formatted examples for these datasets is the quickest way to understand the format for those tasks. Succinctly:
    - For tone, the output per example is the labels separated by `, `, e.g., `"polite, excited, satisfied, sympathetic"`
    - For TSA/entities, the output per example is the entities in the format `<entity>: <type>` join by `, `, e.g., `engineer: JobTitle, Thursday: Date`, or `"atmosphere: positive, drinks: neutral"`

### Running a Tuning
There are some resources for how to run a tuning against a cluster that you can find in the [wiki](https://github.com/foundation-model-stack/fms-hf-tuning/wiki/Installing-and-Testing-OpenShift-fms%E2%80%90hf%E2%80%90tuning-Stack#6-testing) which should be useful. If you are planning to run a tuning / evaluation from inside of a running container with GPUs available, the easiest way to configure your tuning job is to create a local `config.json` with your training specs that is analogous to the mounted configmap, and repoint the `SFT_TRAINER_CONFIG_JSON_PATH` env var to that local config. After doing this, you can trigger a tuning by running `python3 /app/accelerate_launch.py`.

There are a few good things to be careful of here:

- When tuning, make sure to use `"\n### Response:"` as your response template
- If you see a bunch of warnings like `Could not find response key `[19371, 27]` in the following instance:`, you're doing something wrong! Most likely offenders are pointing tuning at the Alpaca data, or using the response template. This should not happen.
- Make sure your `SFT_TRAINER_CONFIG_JSON_PATH` is pointing at the right place if you're inside of a pod with a mounted configmap, otherwise your tuning will be pointing at the wrong config!

### Running Evaluation
To run the evaluation, point the evaluation script at your exported model and test data. For example:

```bash
python3 run_evaluation.py \
    --model <YOUR_TUNED_MODEL>  \
    --delimiter , \
    --data_path formatted_data/cc_tone/500_test.json \
    --max_new_tokens 100
```

To understand the files created by the evaluation, check out the comments in this [PR](https://github.com/foundation-model-stack/fms-hf-tuning/pull/102).
