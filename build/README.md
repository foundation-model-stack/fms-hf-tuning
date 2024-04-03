# Building fms-hf-tuning as an Image

The Dockerfile provides a way of running fms-hf-tuning SFT Trainer. It installs the dependencies needed and adds two additional scripts that helps to parse arguments to pass to SFT Trainer. The `accelerate_launch.py` script is run by default when running the image to trigger SFT trainer for single or multi GPU by parsing arguments and running `accelerate launch launch_training.py`. 

## Configuration

The scripts accept a JSON formatted config which are set by environment variables. `SFT_TRAINER_CONFIG_JSON_PATH` can be set to the mounted path of the JSON config. Alternatively, `SFT_TRAINER_CONFIG_JSON_ENV_VAR` can be set to the encoded JSON config using the below function:

```py
import base64

def encode_json(my_json_string):
    base64_bytes = base64.b64encode(my_json_string.encode("ascii"))
    txt = base64_bytes.decode("ascii")
    return txt

with open("test_config.json") as f:
    contents = f.read()

encode_json(contents)
```

The keys for the JSON config are all of the flags available to use with [SFT Trainer](https://huggingface.co/docs/trl/sft_trainer#trl.SFTTrainer).

For configuring `accelerate launch`, use key `accelerate_launch_args` and pass the set of flags accepted by [accelerate launch](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-launch). Since these flags are passed via the JSON config, the key matches the long formed flag name. For example, to enable flag `--quiet`, use JSON key `"quiet"`, using the short formed `"q"` will fail.

For example, the below config is used for running with two GPUs and FSDP for fine tuning:

```json
{
    "accelerate_launch_args": {
        "num_machines": 1,
        "main_process_port": 1234,
        "num_processes": 2,
        "use_fsdp": true,
        "fsdp_backward_prefetch_policy": "TRANSFORMER_BASED_WRAP",
        "fsdp_sharding_strategy": 1,
        "fsdp_state_dict_type": "FULL_STATE_DICT",
        "fsdp_cpu_ram_efficient_loading": true,
        "fsdp_sync_module_states": true
    },
    "model_name_or_path": "/llama/13B",
    "training_data_path": "/data/twitter_complaints.json",
    "output_dir": "/output/llama-7b-pt-multigpu",
    "num_train_epochs": 5.0,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "save_strategy": "epoch",
    "learning_rate": 0.03,
    "weight_decay": 0.0,
    "lr_scheduler_type": "cosine",
    "logging_steps": 1.0,
    "packing": false,
    "include_tokens_per_second": true,
    "response_template": "\n### Label:",
    "dataset_text_field": "output",
    "use_flash_attn": true,
    "torch_dtype": "bfloat16",
    "tokenizer_name_or_path": "/llama/13B"
}
```

Users should always set `num_processes` to be explicit about the number of processes to run tuning on. When `num_processes` is greater than 1, the [FSDP config](https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/fixtures/accelerate_fsdp_defaults.yaml) is used by default. Thus in the above example, you don't need to pass in the FSDP flags since they match the ones used in the default FSDP config. You can also set your own default values by specifying your own config file using key `config_file`. Any of these values in configs can be overwritten by passing in flags via `accelerate_launch_args` in the JSON config.

Note that `num_processes` which is the total number of processes to be launched in parallel, should match the number of GPUs to run on. The number of GPUs used can also be set by setting environment variable `CUDA_VISIBLE_DEVICES`. If ``num_processes=1`, the script will assume single-GPU.


## Building the Image

With docker, build the image at the top level with:

```sh
docker build . -t sft-trainer:mytag -f build/Dockerfile
```

## Running the Image

Run sft-trainer-image with the JSON env var and mounts set up.

```sh
docker run -v config.json:/app/config.json -v $MODEL_PATH:/model -v $TRAINING_DATA_PATH:/data/twitter_complaints.json --env SFT_TRAINER_CONFIG_JSON_PATH=/app/config.json sft-trainer:mytag
```

This will run `accelerate_launch.py` with the JSON config passed.

An example Kubernetes Pod for deploying sft-trainer which requires creating PVCs with the model and input dataset and any mounts needed for the outputted tuned model:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
name: sft-trainer-config
data:
config.json: |
    {
        "accelerate_launch_args": {
            "num_machines": 1,
            "main_process_port": 1234,
            "num_processes": 2,
            "use_fsdp": true,
            "fsdp_backward_prefetch_policy": "TRANSFORMER_BASED_WRAP",
            "fsdp_sharding_strategy": 1,
            "fsdp_state_dict_type": "FULL_STATE_DICT",
            "fsdp_cpu_ram_efficient_loading": true,
            "fsdp_sync_module_states": true
        },
        "model_name_or_path": "/llama/13B",
        "training_data_path": "/data/twitter_complaints.json",
        "output_dir": "/output/llama-7b-pt-multigpu",
        "num_train_epochs": 5.0,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "save_strategy": "epoch",
        "learning_rate": 0.03,
        "weight_decay": 0.0,
        "lr_scheduler_type": "cosine",
        "logging_steps": 1.0,
        "packing": false,
        "include_tokens_per_second": true,
        "response_template": "\n### Label:",
        "dataset_text_field": "output",
        "use_flash_attn": true,
        "torch_dtype": "bfloat16",
        "tokenizer_name_or_path": "/llama/13B"
    }
---
apiVersion: v1
kind: Pod
metadata:
name: sft-trainer-test
spec:
containers:
    env:
        - name: SFT_TRAINER_CONFIG_JSON_PATH
        value: /config/config.json
    image: sft-trainer:mytag
    imagePullPolicy: IfNotPresent
    name: tuning-test
    resources:
        limits:
            nvidia.com/gpu: "2"
            memory: 200Gi
            cpu: "10"
            ephemeral-storage: 2Ti
        requests:
            memory: 80Gi
            cpu: "5"
    volumeMounts:
        - mountPath: /data/input
        name: input-data
        - mountPath: /data/output
        name: output-data
        - mountPath: /config
        name: sft-trainer-config
restartPolicy: Never
terminationGracePeriodSeconds: 30
volumes:
    - name: input-data
    persistentVolumeClaim:
        claimName: input-pvc
    - name: output-data
    persistentVolumeClaim:
        claimName: output-pvc
    - name: sft-trainer-config
    configMap:
        name: sft-trainer-config
```

The above kube resource values are not hard-defined. However, they are useful when running some models (such as LLaMa-13b model). If ephemeral storage is not defined, you will likely hit into error `The node was low on resource: ephemeral-storage. Container was using 1498072868Ki, which exceeds its request of 0.` where the pod runs low on storage while tuning the model.