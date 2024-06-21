# Standard
import os
import json
import time
import argparse
import yaml
import logging
from typing import List

# Third Party
import grpc
import kubernetes
from kr8s.objects import Pod
# pip install caikit-test-harness --> ai-foundation, useful tools
from testharness.utils.k8s.pods import *
# requires running `python -m grpc_tools.protoc -I ./proto --python_out=. --grpc_python_out=. generation.proto` in TGIS repo
import tgis.generation_pb2
import tgis.generation_pb2_grpc

# TODO: turn print statements into logging warnings

# Default configurations to use for preset ConfigMap and tuning Pod
FINE_TUNE_RESOURCE_NAME = "sft-trainer-test-fine-tune"
LORA_TUNE_RESOURCE_NAME = "sft-trainer-test-lora-tune"
FINE_TUNE_MODEL = "/granite/granite-13b-base-v2/step_300000_ckpt"
LORA_TUNE_MODEL = "/llama/LLaMa/models/hf/13B"
RESPONSE_TEMPLATE = "\n### Response:"
SFT_TRAINER_IMAGE = "docker-na-public.artifactory.swg-devops.com/wcp-ai-foundation-team-docker-virtual/sft-trainer:b6b592f_ubi9_py311"
IMAGE_PULL_SECRET = "artifactory-docker"

def get_configmap(
    name: str,
    namespace: str,
    tuning_technique: str,
    output_dir: str,
    data_path: str,
    model: str,
    response_template: str,
    target_modules: str = ""
) -> tuple[kubernetes.client.V1ObjectMeta, str]:

    data = {
      "model_name_or_path": model,
      "training_data_path": data_path,
      "output_dir": output_dir,
      "num_train_epochs": 10,
      "per_device_train_batch_size": 4,
      "gradient_accumulation_steps": 1,
      "learning_rate": 1e-4,
      "num_virtual_tokens": 100,
      "dataset_text_field": "output",
      "response_template": response_template
    } 
    if tuning_technique == "lora":
        if not target_modules:
            target_modules = ["q_proj", "v_proj"]
            if "granite" in model:
                target_modules = ["c_attn", "c_proj"]

        data.update(
            {
                "peft_method": "lora",
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": target_modules
            }
        )
    
    metadata = kubernetes.client.V1ObjectMeta(
        name=name,
        namespace=namespace,
    )

    return kubernetes.client.V1ConfigMap(
        metadata=metadata,
        data={
            "config.json": json.dumps(data)
        },
    )


def get_pod_template(
    name: str,
    namespace: str,
    configmap_name: str,
    image: str,
    image_pull_secret: str,
):
    metadata = kubernetes.client.V1ObjectMeta(
        name=name,
        namespace=namespace,
    )

    security_context = kubernetes.client.V1PodSecurityContext(
        run_as_user=1000,
        run_as_group=0,
        fs_group=1000,
        fs_group_change_policy="OnRootMismatch"
    )

    container = kubernetes.client.V1Container(
        name="sft-training",
        # v0.3.0 of fms-hf-tuning image
        image=image,
        env=[
            kubernetes.client.V1EnvVar(
                name="SFT_TRAINER_CONFIG_JSON_PATH",
                value="/config/config.json"
            ),
            # kubernetes.client.V1EnvVar(
            #     name="LOG_LEVEL",
            #     value="debug"
            # ),
        ],
        # TODO: lower for devstage?
        # TODO: single vs multi-gpu?
        resources=kubernetes.client.V1ResourceRequirements(
            requests={"cpu": "10", "memory": "80Gi", "ephemeral-storage": "1600Gi", "nvidia.com/gpu": "2"},
            limits={"cpu": "5", "memory": "200Gi", "ephemeral-storage": "3Ti"},
        ),
        # TODO: make sure across devstage + research cluster the PVCs are the same name
        # TODO: make sure the mount_paths are teh same across both, how people use
        volume_mounts=[
            kubernetes.client.V1VolumeMount(
                mount_path="/config",
                name="sft-trainer-config"
            ),
            kubernetes.client.V1VolumeMount(
                mount_path="/granite",
                name="ibm-granite-pvc"
            ),
            kubernetes.client.V1VolumeMount(
                mount_path="/llama",
                name="llama-eval-pvc"
            ),
            kubernetes.client.V1VolumeMount(
                mount_path="/fmaas-integration-tests",
                name="fmaas-integration-tests"
            ),
        ],
    )

    # Create the specification of pod
    spec = kubernetes.client.V1PodSpec(
        containers=[container],
        security_context=security_context,
        # TODO: should create this secret beforehand if it doesn't exist?
        image_pull_secrets=[
            kubernetes.client.V1LocalObjectReference(
                name=image_pull_secret,
            )
        ],
        restart_policy="Never",
        termination_grace_period_seconds=30,
        volumes=[
            kubernetes.client.V1Volume(
                name="sft-trainer-config",
                config_map=kubernetes.client.V1ConfigMapVolumeSource(name=configmap_name)
            ),
            kubernetes.client.V1Volume(
                name="ibm-granite-pvc",
                config_map=kubernetes.client.V1PersistentVolumeClaimVolumeSource(claim_name="ibm-granite-pvc")
            ),
            kubernetes.client.V1Volume(
                name="llama-eval-pvc",
                config_map=kubernetes.client.V1PersistentVolumeClaimVolumeSource(claim_name="llama-eval-pvc")
            ),
            kubernetes.client.V1Volume(
                name="fmaas-integration-tests",
                config_map=kubernetes.client.V1PersistentVolumeClaimVolumeSource(claim_name="fmaas-integration-tests")
            ),
        ]
    )

    # Instantiate the pod object
    return kubernetes.client.V1Pod(
        metadata=metadata,
        spec=spec,
    )

def delete_kube_resource_and_wait(api_instance, kube_resource_name, namespace, kube_kind):
    deleting = True
    while deleting:
        try:
            logging.warning(f"Deleting {kube_kind}: {kube_resource_name}")
            if kube_kind == "Pod":
                api_instance.delete_namespaced_pod(kube_resource_name, namespace)
            elif kube_kind == "ConfigMap":
                api_instance.delete_namespaced_config_map(kube_resource_name, namespace)
            else:
                raise RuntimeError(f"Unknown kube resource provided {kube_kind}, failed to delete")
            # give small time to delete before looping
            time.sleep(3)
        except kubernetes.client.exceptions.ApiException as e:
            has_deleted = json.loads(e.body)['code'] == 404
            if has_deleted:
                deleting = False

def get_api_client() -> kubernetes.client.ApiClient:
    """Return kube API client with kube config loaded."""
    return kubernetes.client.api_client.ApiClient(configuration=load_kube_configuration())

def get_current_namespace() -> str:
    """Returns current namespace, assumes already logged into kube cluster."""
    contexts, current = kubernetes.config.list_kube_config_contexts()
    return current["context"]["namespace"]

def load_kube_configuration() -> kubernetes.client.configuration.Configuration:
    """Loads local kube config file to connect to cluster, uses current session."""
    try:
        kubernetes.config.load_kube_config()
    except kubernetes.config.ConfigException:
        kubernetes.config.load_incluster_config()
    return kubernetes.client.configuration.Configuration.get_default_copy()

def parse_args():
    """Parse the arguments and ensures paths are valid."""
    parser = argparse.ArgumentParser(
        description="Deploys configuration for tuning and TGIS config to test tuning and inference."
    )
    #########################   yamls to deploy   #########################
    parser.add_argument(
        "--tuning_yaml", help="Path to kube yaml to deploy tuning Pod and ConfigMap", required=False
    )
    parser.add_argument(
        "--tgis_yaml", help="Path to kube yaml to deploy TGIS server", required=True
    )
    #########################   preset tuning example configurations   #########################
    parser.add_argument(
        "--model_path", help="Path to model in cluster to add to ConfigMap", required=False
    )
    parser.add_argument(
        "--output_dir", help="Path to output_dir in cluster to add to ConfigMap", required=False
    )
    parser.add_argument(
        "--data_path", help="Path to dataset in cluster to add to ConfigMap", required=False
    )
    parser.add_argument(
        "--response_template", help="Tuning response template to add to ConfigMap", required=False
    )
    parser.add_argument(
        "--resource_name", help="Name to use for ConfigMap and Pod", required=False
    )
    parser.add_argument(
        "--tuning_technique", choices=['lora', 'fine-tune', 'ft'], help="Which preset tuning yaml to deploy", required=False, default="fine-tune"
    )
    parser.add_argument(
        "--sft_image_name", help="Full sft-trainer image name to use in pod. Ex. docker-na-public.artifactory.swg-devops.com/wcp-ai-foundation-team-docker-virtual/sft-trainer:b6b592f_ubi9_py311", required=False
    )
    parser.add_argument(
        "--sft_image_pull_secret", help="Kube Secret to use in pod to pull the sft-trainer image", required=False
    )
    parsed_args = parser.parse_args()

    if not os.path.exists(parsed_args.tuning_yaml):
        raise RuntimeError(f"Path to tuning yaml does not exist: {parsed_args.tuning_yaml}")
    if not os.path.exists(parsed_args.tgis_yaml):
        raise RuntimeError(f"Path to tuning yaml does not exist: {parsed_args.tgis_yaml}")

    return parsed_args


def main():
    #########################   setup   #########################
    LOGLEVEL = os.environ.get("LOG_LEVEL", "WARNING").upper()
    logging.basicConfig(level=LOGLEVEL)

    args = parse_args()

    # need to pass tuning_yaml or data_path and output_dir
    if not (args.tuning_yaml or (args.output_dir and args.data_path)):
        raise RuntimeError("Need to provide either tuning_yaml or output_dir and data_path.")

    # setup kube - requires logged in already
    client = get_api_client()
    current_namespace = get_current_namespace()
    print("Current kube namespace:", current_namespace)
    v1_api_instance = kubernetes.client.CoreV1Api(client)

    output_dir = ""  # TODO: could pull output_dir and use in TGIS yaml
    #########################   deploy user provided tuning yaml   #########################
    if args.tuning_yaml:
        pod_name = ""
        fine_tune_resources = {}
        if args.output_dir or args.data_path or args.model_path:
            logging.warning("Deploying provided tuning_yaml verbatim, ignoring values provided for data_path, model_name, and output_dir")

        with open(args.tuning_yaml) as f:
            yaml_resources = yaml.safe_load_all(f)
            for item in yaml_resources:
                name = item["metadata"]["name"]
                kind = item["kind"]
                fine_tune_resources[kind] = name
                if kind == "Pod":
                    pod_name = name
                if kind == "ConfigMap":
                    configmap_data = item.get("data")
                    if not configmap_data:
                        logging.warn("Could not get output_dir from ConfigMap, unable to find data")
                    
                    for values in configmap_data.values():
                        tuning_configs = json.loads(values)
                        output_dir = tuning_configs.get("output_dir")

        # deploy fine tuned model
        # deletes kube resources if they already exist
        creating = True
        while creating:
            try:
                fine_tune_resources = kubernetes.utils.create_from_yaml(client, args.tuning_yaml, namespace=current_namespace)
                creating = False
            except kubernetes.utils.FailToCreateError as e:
                if "already exists" in e.__str__():
                    for kind, name in fine_tune_resources.items():
                        delete_kube_resource_and_wait(v1_api_instance, name, current_namespace, kind)

        logging.info(f"Waiting for tuning pod {pod_name} state to be complete...")
        # timeout_seconds by default set to 1hr
        wait_for_pod_state(
            namespace=current_namespace,
            name=pod_name,
            state=PodState.Succeeded,
            label_selector=None,
            debug=True,
        )

    #########################   deploy preset configmap and tuning pod   #########################
    else:
        configmap_and_pod_name = args.resource_name
        model_name = args.model_path
        image_name = args.sft_image_name or SFT_TRAINER_IMAGE
        image_pull_secret = args.sft_image_pull_secret or IMAGE_PULL_SECRET
        output_dir = args.output_dir
        if args.tuning_technique == "ft" or args.tuning_technique == "fine-tune":
            if not configmap_and_pod_name:
                configmap_and_pod_name = FINE_TUNE_RESOURCE_NAME
            if not args.model_path:
                model_name = FINE_TUNE_MODEL
        elif args.tuning_technique == "lora":
            if not configmap_and_pod_name:
                configmap_and_pod_name = LORA_TUNE_RESOURCE_NAME
            if not args.model_path:
                model_name = LORA_TUNE_MODEL

        configmap = get_configmap(
            name=configmap_and_pod_name,
            namespace=current_namespace,
            tuning_technique=args.tuning_technique,
            output_dir=args.output_dir,
            data_path=args.data_path,
            model=model_name,
            response_template=args.response_template
        )

        creating_configmap = True
        while creating_configmap:
            try:
                fine_tune_resources = v1_api_instance.create_namespaced_config_map(body=configmap, namespace=current_namespace)
                creating_configmap = False
            except kubernetes.utils.FailToCreateError as e:
                if "already exists" in e.__str__():
                    delete_kube_resource_and_wait(v1_api_instance, configmap_and_pod_name, current_namespace, "ConfigMap")

        tuning_pod = get_pod_template(
            name=configmap_and_pod_name,
            namespace=current_namespace,
            configmap_name=configmap_and_pod_name,
            image=image_name,
            image_pull_secret=image_pull_secret
        )

        creating_pod = True
        while creating_pod:
            try:
                fine_tune_resources = v1_api_instance.create_namespaced_pod(body=tuning_pod, namespace=current_namespace)
                creating_pod = False
            except kubernetes.utils.FailToCreateError as e:
                if "already exists" in e.__str__():
                    delete_kube_resource_and_wait(v1_api_instance, configmap_and_pod_name, current_namespace, "Pod")

        logging.info(f"Waiting for tuning pod {configmap_and_pod_name} state to be complete...")
        # timeout_seconds by default set to 1hr
        wait_for_pod_state(
            namespace=current_namespace,
            name=configmap_and_pod_name,
            state=PodState.Succeeded,
            label_selector=None,
            debug=True,
        )


    #########################   deploy TGIS   #########################
    # TODO: need to resuse output dir as MODEL_NAME
    try:
        tgis_ft_resources = kubernetes.utils.create_from_yaml(client, args.tgis_yaml, namespace=current_namespace)
    except kubernetes.utils.FailToCreateError as e:
        if "already exists" not in e.__str__():
            raise RuntimeError(e)
        else:
            print("TGIS pod already exists, will reuse")


    print(f"Waiting for TGIS pod state to be complete...")
    # TODO: check that this works if pod already up
    tgis_pod_name = wait_for_pod_state(
        namespace=current_namespace,
        name=None,
        state=PodState.Running,
        label_selector="app=text-gen-tuning-dev-ft-granite",
        debug=True,
    )

    # run TGIS inference
    # # Build the request parameters
    stopping = tgis.generation_pb2.StoppingCriteria(
        max_new_tokens=100
    )
    params = tgis.generation_pb2.Parameters(
        decoding=tgis.generation_pb2.DecodingParameters(
            repetition_penalty=2.5,
        ),
        stopping=stopping,
    )

    sentiment_eval_text = "### Input: The piano guy isn't there all the time, but when he is it's a great addition to the meal.\\n\\n### Response:"
    expected_eval_text = "piano: negative, meal: positive"
    sentiment_train_text = "### Input:\nHowever the casual atmosphere comes in handy if you want a good place to drop in and get food.\\n\\n### Response:"
    expected_train_text = "atmosphere: positive, food: neutral"
    gen_reqs = [
        tgis.generation_pb2.GenerationRequest(text=sentiment_eval_text),
        tgis.generation_pb2.GenerationRequest(text=sentiment_train_text),
    ]
    batch_request = tgis.generation_pb2.BatchedGenerationRequest(
        requests=gen_reqs,
        params=params,
    )

    tgis_pod = Pod.get(tgis_pod_name)
    with tgis_pod.portforward(remote_port=8033, local_port=8033):
        with grpc.insecure_channel("localhost:8033") as channel:
            stub = tgis.generation_pb2_grpc.GenerationServiceStub(channel)
            print("Sending batch request...")
            responses = stub.Generate(batch_request)
            for resp in responses.responses:
                print(resp)
                print(resp.text)
    

if __name__ == "__main__":
    main()
