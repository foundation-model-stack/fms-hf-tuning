# Formatting / Tuning / Evaluating Notes

This assumes you are working off of this fork and is a quick and somewhat msesy guide to make a pod, format the data, run a tuning, and an evaluation.

If you're running it off of `main` in the upstream, you'll want to copy:
- `alpaca_to_sft_format.py`
- `pull_and_format_datasets.py`


### 1. Make a Pod and Copy Scripts Inside
Update the pod spec (which we will call `cluster_k8s.yaml`) to sit still for awhile with the following command:
```yaml
      command: ["sleep", "99999"]
```
And make the pod with 
```bash
$ oc apply -f cluster_k8s.yaml
```

Set your pod name and copy the scripts in (assumes you're running from inside of `scripts` in `fms-hf-tuning`):

```bash
$ POD_NAME=sft-trainer-test-alex-granite-13b

$ oc cp ./pull_and_format_datasets.py $POD_NAME:/app/pull_and_format_datasets.py
$ oc cp ./alpaca_to_sft_format.py $POD_NAME:/app/alpaca_to_sft_format.py
$ oc cp ./run_evaluation.py $POD_NAME:/app/run_evaluation.py
$ oc cp ./run_inference.py $POD_NAME:/app/run_inference.py
$ oc cp ../requirements.txt $POD_NAME:/app/requirements.txt
```

The role of each of these scripts is as follows:
- `pull_and_format_datasets.py` - downloads the data from s3 and converts it to alpaca format
- `alpaca_to_sft_format.py` - converts alpaca format data to the format our package expects (needed for tuning data only)
- `run_evaluation.py` - runs the evaluation on a tuned model
- `run_inference.py` - used by the evaluation script to actually call the model

### 2. Downloading and Formatting the Data
Now go into the pod, set the creds, and pull + format the data. We need some extra dependencies here and there are permission restrictions on the pod, so the easiest workaround is to make a virtual environment for formatting / envaluation and install the needed dependencies + the rest of the project there.

```bash
$ oc exec $POD_NAME -it -- bash

$ export S3_ACCESS_KEY_ID={YOUR S3 KEY}
$ export S3_SECRET_ACCESS_KEY={YOUR S3 SECRET ACCESS KEY}
$ export S3_ENDPOINT={YOUR S3 ENDPOINT}

$ python3 -m venv venv
$ source venv/bin/activate
$ pip install boto3 scikit-learn==1.4.2 datasets

$ python3 pull_and_format_datasets.py
```

Make sure you see everything under `formatted`! It should look something like this. Note that the numeric prefix for train/test files indicates how many samples were randomly sampled into the Alpaca formatted file:
```bash
$ ls -R formatted_data/

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

TWe need to run one more conversion step to get the data into the format we expect for tuning. You can do this by running `alpaca_to_sft_format.py` and pointing it at your train files as command line args.

```bash
python3 alpaca_to_sft_format.py \
    --files \
    formatted_data/cc_tone/1000_train.json \
    formatted_data/Entities/1000_IBM_NET_2019_DELIVERABLE_VERSION_C_train.json \
    formatted_data/tsa_mams/1000_train.json
```

Now, you should have a `sft_format_{file_name}` in the same location as each of your aforementioned files. E.g.,
```bash
$ ls formatted_data/cc_tone/

1000_train.json # Alpaca format - don't worry about this one anymore
500_test.json # Use this one for evaluation
sft_format_1000_train.json # Use this one for tuning

# And anything for validation, you can ignore
```

### 3. Running Tuning Inside the Pod
Now we can actually run our prompt tuning job. A couple of things to be careful about:

- When tuning, make sure to use `"\n### Response:"` as your response template
- If you see a bunch of warnings `Could not find response key `[19371, 27]` in the following instance:`, you're doing something wrong! Most likely offenders are pointing tuning at the Alpaca data, or using the response template. This should not happen.

Next, create a file named `config.json` with your training specs. An example is shown below:
```json
{
    "model_name_or_path": "/granite/granite-13b-base-v2/step_300000_ckpt",
    "training_data_path": "/app/formatted_data/cc_tone/sft_format_1000_train.json",
    "output_dir": "/data/brooks/tuning_output/granite-13b-pt-multi-gpu-epoch-1",
    "num_train_epochs": 1.0,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "evaluation_strategy": "no",
    "save_strategy": "epoch",
    "learning_rate": 1e-5,
    "response_template": "\n### Response:",
    "dataset_text_field": "output",
    "tokenizer_name_or_path": "/granite/granite-13b-base-v2/step_300000_ckpt",
    "peft_method": "pt"
}
```

Now repoint the env var to our config - make sure to do this, otherwise it'll point at the mounted configmap by default!
```bash
export SFT_TRAINER_CONFIG_JSON_PATH=/app/config.json
```

Next,launch the training:
```bash
python /app/accelerate_launch.py
```

### 4. Running Evaluation Inside the Pod 
To run the evaluation, point the evaluation script at your exported model and test data.

```bash
python run_evaluation.py \
    --model /data/brooks/tuning_output/granite-13b-pt-multi-gpu-epoch-1  \
    --delimiter , \
    --data_path formatted_data/cc_tone/500_test.json \
    --max_new_tokens 10
```

To understand the files created by the evaluation, check out the comments in this [PR](https://github.com/foundation-model-stack/fms-hf-tuning/pull/102).
