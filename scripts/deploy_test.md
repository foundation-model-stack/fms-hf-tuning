# Test deploy script

## Test Cases

### Tuning Pod

1. Can deploy fully rendered ConfigMap and Pod combined
   - [Test cases](https://github.ibm.com/ai-foundation/fmaas-tuning-docs/tree/main/test) in fmaas-tuning-docs
   - technically can also deploy a PytorchJob that is already rendered if the resource doesn't already exist. if it does, it will fail
   - example command: `python scripts/run_deploy_test.py --tgis_yaml scripts/deploy-tests/tgis-ft.yaml --tuning_yaml scripts/deploy-tests/fine-tune.yaml --mount_fms_tuning_pvc --inference_text "### Input:\nHowever the casual atmosphere comes in handy if you want a good place to drop in and get food.\\n\\n### Response:" --overwrite`

2. Deploys default fine tune test case
   - Multi-GPU (2 GPUs), granite-13b-base-v2, 10 epochs
   - User must provide training data path and output directory.
   - Default ConfigMap:
    ```json
    {
        "model_name_or_path": "/granite/granite-13b-base-v2/step_300000_ckpt",
        "training_data_path": $TRAINING_DATA,
        "output_dir": $OUTPUT_DIR,
        "num_train_epochs": 5,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4,
        "num_virtual_tokens": 100,
        "dataset_text_field": "output",
        "response_template": "\n### Response:"
    }
    ```

3. Deploys default LoRA test case
   - Multi-GPU (2 GPUs), llama-13b-base, 10 epochs
   - User must provide training data path and output directory.
   - Default ConfigMap:
    ```json
    {
        "model_name_or_path": "/llama/LLaMa/models/hf/13B", 
        "training_data_path": $TRAINING_DATA, 
        "output_dir": $OUTPUT_DIR,
        "num_train_epochs": 5, 
        "per_device_train_batch_size": 4, 
        "gradient_accumulation_steps": 1, 
        "learning_rate": 1e-4, 
        "num_virtual_tokens": 100, 
        "dataset_text_field": "output", 
        "response_template": "\n### Response:",
         "peft_method": "lora", 
         "r": 8, 
         "lora_alpha": 16, 
         "lora_dropout": 0.05, 
         "target_modules": ["q_proj", "v_proj"]
    }
    ```

Example command:
```sh
python3.9 scripts/run_deploy_test.py --output_dir "/data/anhuong/tuning_output/test-script-ft-default-config-twitter-with-inference" --mount_fms_tuning_pvc --data_path /data/anhuong/twitter_complaints.json --sft_image_pull_secret artifactory-docker-anh --run_as_root --num_train_epochs 1 --tuning_technique ft --response_template "\n### Label:" --inference_text "### Text: This wallpaper is so cute\n\n### Label:"
```

Tuning params can adjust:
- User must provide training data path (--model_path) and output directory (--output_dir).
- response_template
- num_train_epochs
- learning_rate

Tuning pod configs can adjust:
- resource_name (same one used for pod and configmap)
- sft_image_name
- sft_image_pull_secret
- mount_fms_tuning_pvc
- run_as_root

Prereqs:
- assumes PVCs already exist with names...
- will use image pull secret... by default or 
- will use image... by default
- will delete ConfigMap and Pod of the same name if they already exist --> TODO
- assumes image pull secret created beforehand

Assumptions:
- expects differentiated label selectors between TGIS instances
    - currently the label selector is only based on the tuning technique
    - TODO: add model name to label selector

### TGIS Pod
1. Can deploy fully rendered yaml
   - Thus you may need to run `kustomize build` beforehand to generate a yaml file of the TGIS resources.
2. Deploys preset TGIS inference server
   - Image: quay.io/wxpe/text-gen-server:main.9b4aea8
   - 1 GPU with flash attention
   - NVIDIA-A100-SXM4-80GB


## Testing Running Example

Run coded tuning and TGIS resources with LoRA. Note this still does the LoRA merging of the adapter since it is deploying TGIS and not vLLM. TODO: update that if using LoRA will update TGIS resource to deploy vLLM.