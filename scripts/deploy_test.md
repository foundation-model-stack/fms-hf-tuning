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

Note that since my connection to the cluster is insecure, I get the below warning after every kubebernetes API call:
```sh
/usr/local/lib/python3.9/site-packages/urllib3/connectionpool.py:1045: InsecureRequestWarning: Unverified HTTPS request is being made to host 'api.pok.res.ibm.com'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/1.26.x/advanced-usage.html#ssl-warnings
```

I have removed these from the below example for ease in reading the output.

```sh
$ python3.9 scripts/run_deploy_test.py --output_dir "/data/anhuong/tuning_output/test-script-lora-default-config-twitter-with-inference" --mount_fms_tuning_pvc --data_path /data/anhuong/twitter_complaints.json --sft_image_pull_secret artifactory-docker-anh --run_as_root --num_train_epochs 5 --tuning_technique lora --response_template "\n### Label:" --inference_text "### Text: This wallpaper is so cute\n\n### Label:" --overwrite

Deploying Configmap sft-trainer-test-lora-tune into namespace fms-tuning
Deploying Pod sft-trainer-test-lora-tune into namespace fms-tuning
WARNING:root:Pod sft-trainer-test-lora-tune already exists
WARNING:root:Deleting Pod: sft-trainer-test-lora-tune
Deploying Pod sft-trainer-test-lora-tune into namespace fms-tuning
Waiting for tuning pod sft-trainer-test-lora-tune state to be complete...

[01:44:39] Trying to find a pod using label_selector None and field_selector                                    pods.py:248
           metadata.name=sft-trainer-test-lora-tune                                                                        
/usr/local/lib/python3.9/site-packages/urllib3/connectionpool.py:1045: InsecureRequestWarning: Unverified HTTPS request is being made to host 'api.pok.res.ibm.com'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/1.26.x/advanced-usage.html#ssl-warnings
  warnings.warn(
           Pod sft-trainer-test-lora-tune is in phase Pending                                                   pods.py:307
[01:44:42] Container status:  {'allocated_resources': None,                                                     pods.py:317
            'container_id': None,                                                                                          
            'image':                                                                                                       
           'docker-na-public.artifactory.swg-devops.com/wcp-ai-foundation-team-docker-virtual/sft-trainer:b6b59            
           2f_ubi9_py311',                                                                                                 
            'image_id': '',                                                                                                
            'last_state': {'running': None, 'terminated': None, 'waiting': None},                                          
            'name': 'sft-training',                                                                                        
            'ready': False,                                                                                                
            'resources': None,                                                                                             
            'restart_count': 0,                                                                                            
            'started': False,                                                                                              
            'state': {'running': None,                                                                                     
                      'terminated': None,                                                                                  
                      'waiting': {'message': None, 'reason': 'ContainerCreating'}}}                                        
           Container is Waiting with reason ContainerCreating                                                   pods.py:339
[01:44:48] Container status:  {'allocated_resources': None,                                                     pods.py:317
            'container_id': None,                                                                                          
            'image':                                                                                                       
           'docker-na-public.artifactory.swg-devops.com/wcp-ai-foundation-team-docker-virtual/sft-trainer:b6b59            
           2f_ubi9_py311',                                                                                                 
            'image_id': '',                                                                                                
            'last_state': {'running': None, 'terminated': None, 'waiting': None},                                          
            'name': 'sft-training',                                                                                        
            'ready': False,                                                                                                
            'resources': None,                                                                                             
            'restart_count': 0,                                                                                            
            'started': False,                                                                                              
            'state': {'running': None,                                                                                     
                      'terminated': None,                                                                                  
                      'waiting': {'message': None, 'reason': 'ContainerCreating'}}}                                        
           Container is Waiting with reason ContainerCreating                                                   pods.py:339
[01:44:52] Pod sft-trainer-test-lora-tune is in phase Running                                                   pods.py:307
           Container status:  {'allocated_resources': None,                                                     pods.py:317
            'container_id': 'cri-o://31ecdaf7d7f4617891921e39dc56934ed8268a68d6acfd89bb33d7fb2ad5900f',                    
            'image':                                                                                                       
           'docker-na-public.artifactory.swg-devops.com/wcp-ai-foundation-team-docker-virtual/sft-trainer:b6b59            
           2f_ubi9_py311',                                                                                                 
            'image_id':                                                                                                    
           'docker-na-public.artifactory.swg-devops.com/wcp-ai-foundation-team-docker-virtual/sft-trainer@sha25            
           6:a44ac7403100b7ec100052e58ca1f93e7fd043e5ad031ef6bc2ccb13301e2ee1',                                            
            'last_state': {'running': None, 'terminated': None, 'waiting': None},                                          
            'name': 'sft-training',                                                                                        
            'ready': True,                                                                                                 
            'resources': None,                                                                                             
            'restart_count': 0,                                                                                            
            'started': True,                                                                                               
            'state': {'running': {'started_at': datetime.datetime(2024, 6, 29, 7, 44, 51, tzinfo=tzutc())},                
                      'terminated': None,                                                                                  
                      'waiting': None}}                                                                                    
[01:51:29] Container status:  {'allocated_resources': None,                                                     pods.py:317
            'container_id': 'cri-o://31ecdaf7d7f4617891921e39dc56934ed8268a68d6acfd89bb33d7fb2ad5900f',                    
            'image':                                                                                                       
           'docker-na-public.artifactory.swg-devops.com/wcp-ai-foundation-team-docker-virtual/sft-trainer:b6b59            
           2f_ubi9_py311',                                                                                                 
            'image_id':                                                                                                    
           'docker-na-public.artifactory.swg-devops.com/wcp-ai-foundation-team-docker-virtual/sft-trainer@sha25            
           6:a44ac7403100b7ec100052e58ca1f93e7fd043e5ad031ef6bc2ccb13301e2ee1',                                            
            'last_state': {'running': None, 'terminated': None, 'waiting': None},                                          
            'name': 'sft-training',                                                                                        
            'ready': False,                                                                                                
            'resources': None,                                                                                             
            'restart_count': 0,                                                                                            
            'started': False,                                                                                              
            'state': {'running': None,                                                                                     
                      'terminated': {'container_id':                                                                       
           'cri-o://31ecdaf7d7f4617891921e39dc56934ed8268a68d6acfd89bb33d7fb2ad5900f',                                     
                                     'exit_code': 0,                                                                       
                                     'finished_at': datetime.datetime(2024, 6, 29, 7, 51, 28, tzinfo=tzutc()),             
                                     'message': None,                                                                      
                                     'reason': 'Completed',                                                                
                                     'signal': None,                                                                       
                                     'started_at': datetime.datetime(2024, 6, 29, 7, 44, 51, tzinfo=tzutc())},             
                      'waiting': None}}         
[01:51:33] Pod sft-trainer-test-lora-tune is in phase Succeeded                                                 pods.py:286
Deploying Service tuning-dev-inference-server-lora into namespace fms-tuning
WARNING:root:Service tuning-dev-inference-server-lora already exists
WARNING:root:Deleting Service: tuning-dev-inference-server-lora
Deploying Service tuning-dev-inference-server-lora into namespace fms-tuning
Deploying Deployment tuning-dev-inference-server-lora into namespace fms-tuning
Waiting for TGIS pod state to be Running...
[01:54:41] Trying to find a pod using label_selector app=text-gen-tuning-dev-lora and field_selector None       pods.py:248
[01:54:42] Pod tuning-dev-inference-server-lora-6df66bd64b-8dwct is in phase Running                            pods.py:286
Sending batch request...
input: ### Text: This wallpaper is so cute\n\n### Label:
response:  no complaint
stop_reason: 2
```