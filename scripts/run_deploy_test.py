# Standard
from typing import Dict, List
import argparse
import json
import logging
import os
import sys
import time
import yaml

# Third Party
import grpc
import kubernetes
from kr8s.objects import Pod

# pip install caikit-test-harness --> ai-foundation, useful kube methods
from testharness.utils.k8s.deployments import *
from testharness.utils.k8s.pods import *

# requires running `python -m grpc_tools.protoc -I ./proto --python_out=. --grpc_python_out=. generation.proto` in TGIS repo
import tgis.generation_pb2
import tgis.generation_pb2_grpc

# Default configurations to use for preset ConfigMap and tuning Pod
FINE_TUNE_RESOURCE_NAME = "sft-trainer-test-fine-tune"
LORA_TUNE_RESOURCE_NAME = "sft-trainer-test-lora-tune"
FINE_TUNE_MODEL = "/granite/granite-13b-base-v2/step_300000_ckpt"
LORA_TUNE_MODEL = "/llama/LLaMa/models/hf/13B"
RESPONSE_TEMPLATE = "\n### Label:"
SFT_TRAINER_IMAGE = "docker-na-public.artifactory.swg-devops.com/wcp-ai-foundation-team-docker-virtual/sft-trainer:b6b592f_ubi9_py311"
IMAGE_PULL_SECRET = "artifactory-docker"
TGIS_IMAGE = "quay.io/wxpe/text-gen-server:main.9b4aea8"


def get_tuning_configmap(
    name: str,
    namespace: str,
    tuning_technique: str,
    output_dir: str,
    data_path: str,
    model: str,
    num_train_epochs: int = 5,
    learning_rate: float = 1e-4,
    response_template: str = RESPONSE_TEMPLATE,
    target_modules: str = "",
) -> tuple[kubernetes.client.V1ObjectMeta, str]:

    data = {
        "model_name_or_path": model,
        "training_data_path": data_path,
        "output_dir": output_dir,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "learning_rate": learning_rate,
        "num_virtual_tokens": 100,
        "dataset_text_field": "output",
        "response_template": response_template,
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
                "target_modules": target_modules,
            }
        )

    metadata = kubernetes.client.V1ObjectMeta(
        name=name,
        namespace=namespace,
    )

    return kubernetes.client.V1ConfigMap(
        metadata=metadata,
        data={"config.json": json.dumps(data)},
    )


def get_tuning_pod_template(
    name: str,
    namespace: str,
    configmap_name: str,
    image: str,
    image_pull_secret: str,
    mount_fms_tuning_pvc: bool = False,
    run_as_root: bool = False,
):
    metadata = kubernetes.client.V1ObjectMeta(
        name=name,
        namespace=namespace,
    )

    # tuning user created in Dockerfile
    user = 1000
    if run_as_root:
        user = 0

    security_context = kubernetes.client.V1PodSecurityContext(
        run_as_user=user,
        run_as_group=0,
        fs_group=user,
        fs_group_change_policy="OnRootMismatch",
    )

    container = kubernetes.client.V1Container(
        name="sft-training",
        # v0.3.0 of fms-hf-tuning image
        image=image,
        env=[
            kubernetes.client.V1EnvVar(
                name="SFT_TRAINER_CONFIG_JSON_PATH", value="/config/config.json"
            ),
            # TODO: create debug flag
            kubernetes.client.V1EnvVar(name="LOG_LEVEL", value="debug"),
        ],
        # TODO: lower for devstage?
        # TODO: single vs multi-gpu?
        resources=kubernetes.client.V1ResourceRequirements(
            requests={
                "cpu": "5",
                "memory": "80Gi",
                "ephemeral-storage": "1600Gi",
                "nvidia.com/gpu": "2",
            },
            limits={
                "cpu": "10",
                "memory": "200Gi",
                "ephemeral-storage": "3Ti",
                "nvidia.com/gpu": "2",
            },
        ),
        # TODO: make sure across devstage + research cluster the PVCs are the same name
        # TODO: make sure the mount_paths are teh same across both, how people use
        volume_mounts=get_pod_volume_mounts(
            mount_sft_config=True, mount_fms_tuning_pvc=mount_fms_tuning_pvc
        ),
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
        volumes=get_pod_volumes(
            mount_sft_config=True,
            mount_fms_tuning_pvc=mount_fms_tuning_pvc,
            configmap_name=configmap_name,
        ),
    )

    # Instantiate the pod object
    return kubernetes.client.V1Pod(
        metadata=metadata,
        spec=spec,
    )


def get_pod_volume_mounts(
    mount_sft_config: bool, mount_fms_tuning_pvc: bool
) -> List[kubernetes.client.V1VolumeMount]:
    volume_mounts = [
        kubernetes.client.V1VolumeMount(mount_path="/granite", name="ibm-granite-pvc"),
        kubernetes.client.V1VolumeMount(mount_path="/llama", name="llama-eval-pvc"),
        kubernetes.client.V1VolumeMount(
            mount_path="/fmaas-integration-tests", name="fmaas-integration-tests"
        ),
    ]

    if mount_sft_config:
        volume_mounts.append(
            kubernetes.client.V1VolumeMount(
                mount_path="/config", name="sft-trainer-config"
            ),
        )
    if mount_fms_tuning_pvc:
        volume_mounts.append(
            kubernetes.client.V1VolumeMount(mount_path="/data", name="fms-tuning-pvc"),
        )

    return volume_mounts


def get_pod_volumes(
    mount_sft_config: bool, mount_fms_tuning_pvc: bool, configmap_name: str = ""
) -> List[kubernetes.client.V1Volume]:
    volumes = [
        kubernetes.client.V1Volume(
            name="ibm-granite-pvc",
            persistent_volume_claim=kubernetes.client.V1PersistentVolumeClaimVolumeSource(
                claim_name="ibm-granite-pvc"
            ),
        ),
        kubernetes.client.V1Volume(
            name="llama-eval-pvc",
            persistent_volume_claim=kubernetes.client.V1PersistentVolumeClaimVolumeSource(
                claim_name="llama-eval-pvc"
            ),
        ),
        kubernetes.client.V1Volume(
            name="fmaas-integration-tests",
            persistent_volume_claim=kubernetes.client.V1PersistentVolumeClaimVolumeSource(
                claim_name="fmaas-integration-tests"
            ),
        ),
    ]
    if mount_sft_config and configmap_name:
        volumes.append(
            kubernetes.client.V1Volume(
                name="sft-trainer-config",
                config_map=kubernetes.client.V1ConfigMapVolumeSource(
                    name=configmap_name
                ),
            )
        )
    if mount_fms_tuning_pvc:
        volumes.append(
            kubernetes.client.V1Volume(
                name="fms-tuning-pvc",
                persistent_volume_claim=kubernetes.client.V1PersistentVolumeClaimVolumeSource(
                    claim_name="fms-tuning-pvc"
                ),
            ),
        )

    return volumes


def get_tgis_service_template(
    name: str,
    namespace: str,
    label_selectors: Dict[str, str],
):

    metadata = kubernetes.client.V1ObjectMeta(
        name=name,
        namespace=namespace,
        labels=label_selectors,
    )

    spec = kubernetes.client.V1ServiceSpec(
        type="ClusterIP",
        cluster_ip="None",
        ports=[
            kubernetes.client.V1ServicePort(
                port=8033,
                target_port="grpc",
            ),
        ],
        selector=label_selectors,
    )

    return kubernetes.client.V1Service(metadata=metadata, spec=spec)


def get_tgis_deployment_template(
    model_path: str,
    name: str,
    namespace: str,
    label_selectors: Dict[str, str],
    tgis_image: str = TGIS_IMAGE,
    mount_fms_tuning_pvc: bool = False,
):

    metadata = kubernetes.client.V1ObjectMeta(
        name=name,
        namespace=namespace,
        labels=label_selectors,
    )

    env_vars = [
        kubernetes.client.V1EnvVar(
            name="SFT_TRAINER_CONFIG_JSON_PATH", value="/config/config.json"
        ),
        kubernetes.client.V1EnvVar(name="DEPLOYMENT_FRAMEWORK", value="tgis_native"),
        kubernetes.client.V1EnvVar(name="FLASH_ATTENTION", value="true"),
        kubernetes.client.V1EnvVar(name="NUM_GPUS", value="1"),
        kubernetes.client.V1EnvVar(name="CUDA_VISIBLE_DEVICES", value="0"),
        kubernetes.client.V1EnvVar(
            name="HF_HUB_CACHE", value="/home/tgis/transformers_cache"
        ),
        kubernetes.client.V1EnvVar(name="DTYPE_STR", value="float16"),
        kubernetes.client.V1EnvVar(name="MAX_BATCH_SIZE", value="256"),
        kubernetes.client.V1EnvVar(name="MAX_CONCURRENT_REQUESTS", value="320"),
        kubernetes.client.V1EnvVar(name="MODEL_NAME", value=model_path),
    ]

    # llama2 values
    max_seq_length = 4096
    max_new_tokens = 1536
    if "granite" in model_path:
        max_seq_length = 8192
        max_new_tokens = 4096

    env_vars.extend(
        [
            kubernetes.client.V1EnvVar(
                name="MAX_SEQUENCE_LENGTH", value=str(max_seq_length)
            ),
            kubernetes.client.V1EnvVar(
                name="MAX_NEW_TOKENS", value=str(max_new_tokens)
            ),
        ]
    )

    container = kubernetes.client.V1Container(
        name="server",
        image=tgis_image,
        env=env_vars,
        resources=kubernetes.client.V1ResourceRequirements(
            requests={"cpu": "8", "nvidia.com/gpu": "1"},
            limits={"cpu": "16", "memory": "96Gi", "nvidia.com/gpu": "1"},
        ),
        # TODO: make sure across devstage + research cluster the PVCs are the same name
        # TODO: make sure the mount_paths are teh same across both, how people use
        volume_mounts=get_pod_volume_mounts(
            mount_sft_config=False, mount_fms_tuning_pvc=mount_fms_tuning_pvc
        ),
        startup_probe=kubernetes.client.V1Probe(
            failure_threshold=24,
            period_seconds=30,
            http_get=kubernetes.client.V1HTTPGetAction(
                path="/health",
                port="http",
            ),
        ),
        liveness_probe=kubernetes.client.V1Probe(
            timeout_seconds=8,
            period_seconds=100,
            http_get=kubernetes.client.V1HTTPGetAction(
                path="/health",
                port="http",
            ),
        ),
        readiness_probe=kubernetes.client.V1Probe(
            timeout_seconds=4,
            period_seconds=30,
            http_get=kubernetes.client.V1HTTPGetAction(
                path="/health",
                port="http",
            ),
        ),
        ports=[
            kubernetes.client.V1ContainerPort(
                container_port=3000,
                name="http",
            ),
            kubernetes.client.V1ContainerPort(
                container_port=8033,
                name="grpc",
            ),
        ],
        security_context=kubernetes.client.V1SecurityContext(
            allow_privilege_escalation=False,
            capabilities=kubernetes.client.V1Capabilities(drop=["ALL"]),
            privileged=False,
            run_as_non_root=True,
            seccomp_profile=kubernetes.client.V1SeccompProfile(type="RuntimeDefault"),
        ),
        termination_message_policy="FallbackToLogsOnError",
    )

    affinity = kubernetes.client.V1Affinity(
        node_affinity=kubernetes.client.V1NodeAffinity(
            required_during_scheduling_ignored_during_execution=kubernetes.client.V1NodeSelector(
                node_selector_terms=[
                    kubernetes.client.V1NodeSelectorTerm(
                        match_expressions=[
                            kubernetes.client.V1NodeSelectorRequirement(
                                key="nvidia.com/gpu.product",
                                operator="In",
                                values=["NVIDIA-A100-SXM4-80GB"],
                            ),
                        ],
                    ),
                ],
            ),
        ),
        pod_affinity=kubernetes.client.V1PodAffinity(
            preferred_during_scheduling_ignored_during_execution=[
                kubernetes.client.V1WeightedPodAffinityTerm(
                    pod_affinity_term=kubernetes.client.V1PodAffinityTerm(
                        label_selector=kubernetes.client.V1LabelSelector(
                            match_expressions=[
                                kubernetes.client.V1LabelSelectorRequirement(
                                    key="bam-placement",
                                    operator="In",
                                    values=["nvidia-a100"],
                                ),
                            ],
                        ),
                        topology_key="kubernetes.io/hostname",
                    ),
                    weight=100,
                ),
            ],
        ),
    )

    pod_spec = kubernetes.client.V1PodSpec(
        affinity=affinity,
        containers=[container],
        # TODO: should create this secret beforehand if it doesn't exist?
        # image_pull_secrets=[
        #     kubernetes.client.V1LocalObjectReference(
        #         name=image_pull_secret,
        #     )
        # ],
        enable_service_links=False,
        priority_class_name="system-node-critical",
        termination_grace_period_seconds=120,
        volumes=get_pod_volumes(
            mount_fms_tuning_pvc=mount_fms_tuning_pvc, mount_sft_config=False
        ),
    )

    annotations = {
        "prometheus.io/port": "3000",
        "prometheus.io/scrape": "true",
    }

    # Create the specification of deployment
    spec = kubernetes.client.V1DeploymentSpec(
        replicas=1,
        template=kubernetes.client.V1PodTemplateSpec(
            spec=pod_spec,
            metadata=kubernetes.client.V1ObjectMeta(
                name=name,
                annotations=annotations,
                namespace=namespace,
                labels={**label_selectors, **{"bam-placement": "nvidia-a100"}},
            ),
        ),
        selector=kubernetes.client.V1LabelSelector(match_labels=label_selectors),
        strategy=kubernetes.client.V1DeploymentStrategy(
            rolling_update=kubernetes.client.V1RollingUpdateDeployment(max_surge=1)
        ),
    )

    # Instantiate the pod object
    return kubernetes.client.V1Deployment(
        metadata=metadata,
        spec=spec,
    )


def delete_kube_resource_and_wait(
    api_instance, kube_resource_name, namespace, kube_kind
):
    logging.warning(f"Deleting {kube_kind}: {kube_resource_name}")
    deleting = True
    while deleting:
        try:
            if kube_kind == "Pod":
                api_instance.delete_namespaced_pod(kube_resource_name, namespace)
            elif kube_kind == "ConfigMap":
                api_instance.delete_namespaced_config_map(kube_resource_name, namespace)
            elif kube_kind == "Service":
                api_instance.delete_namespaced_service(kube_resource_name, namespace)
            elif kube_kind == "Deployment":
                api_instance.delete_namespaced_deployment(kube_resource_name, namespace)
            else:
                raise RuntimeError(
                    f"Unknown kube resource provided {kube_kind}, failed to delete"
                )
            # give small time to delete before looping
            time.sleep(3)
        except kubernetes.client.exceptions.ApiException as e:
            has_deleted = json.loads(e.body)["code"] == 404
            if has_deleted:
                deleting = False


def get_api_client() -> kubernetes.client.ApiClient:
    """Return kube API client with kube config loaded."""
    return kubernetes.client.api_client.ApiClient(
        configuration=load_kube_configuration()
    )


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
        description="Deploys tuning Pod and ConfigMap and TGIS server to test tuning and inference."
    )
    parser.add_argument(
        "--timeout",
        help="Time in seconds to wait for pod to complete before failing",
        required=False,
        type=int,
        default=3600,
    )
    parser.add_argument(
        "--overwrite",
        help="Whether to delete kube resources if they already exist",
        action="store_true",
    )
    #########################   yamls to deploy   #########################
    parser.add_argument(
        "--tuning_yaml",
        help="Path to kube yaml to deploy tuning Pod and ConfigMap",
        required=False,
    )
    parser.add_argument(
        "--tgis_yaml", help="Path to kube yaml to deploy TGIS server", required=False
    )
    #########################    tuning configurations   #########################
    parser.add_argument(
        "--model_path",
        help="Path to model in cluster to add to ConfigMap",
        required=False,
    )
    parser.add_argument(
        "--output_dir",
        help="Path to output_dir in cluster to add to ConfigMap",
        required=False,
    )
    parser.add_argument(
        "--data_path",
        help="Path to dataset in cluster to add to ConfigMap",
        required=False,
    )
    parser.add_argument(
        "--response_template",
        help="Tuning response template to add to ConfigMap",
        required=False,
        default=RESPONSE_TEMPLATE,
    )
    parser.add_argument(
        "--num_train_epochs",
        help="Number of train epochs to tune",
        required=False,
        type=int,
        default=5,
    )
    parser.add_argument(
        "--learning_rate",
        help="Learning rate to set for tuning",
        required=False,
        type=float,
        default=1e-4,
    )
    #########################   tune pod configurations   #########################
    parser.add_argument(
        "--tuning_technique",
        choices=["lora", "fine-tune", "ft"],
        help="Which preset tuning yaml to deploy",
        required=False,
        default="fine-tune",
    )
    parser.add_argument(
        "--resource_name", help="Name to use for ConfigMap and Pod", required=False
    )
    parser.add_argument(
        "--sft_image_name",
        help="Full sft-trainer image name to use in pod. Ex. docker-na-public.artifactory.swg-devops.com/wcp-ai-foundation-team-docker-virtual/sft-trainer:b6b592f_ubi9_py311",
        required=False,
        default=SFT_TRAINER_IMAGE,
    )
    parser.add_argument(
        "--sft_image_pull_secret",
        help="Kube Secret to use in pod to pull the sft-trainer image",
        required=False,
        default=IMAGE_PULL_SECRET,
    )
    parser.add_argument(
        "--mount_fms_tuning_pvc",
        help="Whether to mount fms-tuning-pvc in research cluster",
        action="store_true",
    )
    parser.add_argument(
        "--run_as_root",
        help="Whether to run tuning pod as root user",
        action="store_true",
    )
    #########################   tgis configurations   #########################
    parser.add_argument(
        "--no_deploy_tgis",
        help="Whether to not deploy TGIS pod",
        action="store_true",
    )
    parser.add_argument(
        "--tgis_image_name",
        help="Full TGIS image name to use in pod. Ex. quay.io/wxpe/text-gen-server:main.9b4aea8",
        required=False,
        default=TGIS_IMAGE,
    )
    parser.add_argument(
        "--inference_text", help="Text to run inference against TGIS", required=False
    )
    parsed_args = parser.parse_args()

    if parsed_args.tuning_yaml and not os.path.exists(parsed_args.tuning_yaml):
        raise RuntimeError(
            f"Path to tuning yaml does not exist: {parsed_args.tuning_yaml}"
        )
    if parsed_args.tgis_yaml and not os.path.exists(parsed_args.tgis_yaml):
        raise RuntimeError(
            f"Path to tuning yaml does not exist: {parsed_args.tgis_yaml}"
        )

    return parsed_args


def main():
    #########################   setup   #########################
    LOGLEVEL = os.environ.get("LOG_LEVEL", "WARNING").upper()
    logging.basicConfig(level=LOGLEVEL)

    args = parse_args()

    # need to pass tuning_yaml or data_path and output_dir
    if not (args.tuning_yaml or (args.output_dir and args.data_path)):
        raise RuntimeError(
            "Need to provide either tuning_yaml OR output_dir and data_path."
        )

    if not args.no_deploy_tgis:
        if not (args.tgis_yaml or args.output_dir) or not args.inference_text:
            raise RuntimeError(
                "Need to provide either tgis_yaml OR output_dir to mount model in TGIS pod AND inference_text."
            )

    # setup kube - requires logged in already
    client = get_api_client()
    current_namespace = get_current_namespace()
    logging.info("Current kube namespace:", current_namespace)
    core_v1_api_instance = kubernetes.client.CoreV1Api(client)
    apps_v1_api_instance = kubernetes.client.AppsV1Api(client)

    output_dir = ""
    #########################   deploy user provided tuning yaml   #########################
    if args.tuning_yaml:
        tuning_resources = {}
        if args.output_dir or args.data_path or args.model_path:
            logging.warning(
                "Deploying provided tuning_yaml verbatim, ignoring configurations provided for tuning or kube."
            )

        with open(args.tuning_yaml) as f:
            yaml_resources = yaml.safe_load_all(f)
            for item in yaml_resources:
                name = item["metadata"]["name"]
                kind = item["kind"]
                tuning_resources[kind] = name

                if kind == "ConfigMap":
                    configmap_data = item.get("data")
                    if not configmap_data:
                        logging.warn(
                            "Could not get output_dir from ConfigMap, unable to find data"
                        )

                    for values in configmap_data.values():
                        tuning_configs = json.loads(values)
                        output_dir = tuning_configs.get("output_dir")

        # deploy fine tuned model
        # deletes kube resources if they already exist
        creating = True
        while creating:
            try:
                logging.info(
                    f"Deploying file {args.tuning_yaml} into namespace {current_namespace}"
                )
                created_resources = kubernetes.utils.create_from_yaml(
                    client, args.tuning_yaml, namespace=current_namespace
                )
                # returns List[List[Dict(of kube resource)]] so created_resources[0][0] = ConfigMap and created_resources[1][0] = Pod
                creating = False
            except kubernetes.utils.FailToCreateError as e:
                if args.overwrite and "already exists" in e.__str__():
                    for kind, name in tuning_resources.items():
                        delete_kube_resource_and_wait(
                            core_v1_api_instance, name, current_namespace, kind
                        )
                else:
                    raise e

        pod_name = tuning_resources.get("Pod")
        if pod_name:
            print(f"Waiting for tuning pod {pod_name} state to be complete...")
            # timeout_seconds by default set to 1hr
            wait_for_pod_state(
                namespace=current_namespace,
                name=pod_name,
                state=PodState.Succeeded,
                label_selector=None,
                debug=True,
                timeout_seconds=args.timeout,
            )

    #########################   deploy preset configmap and tuning pod   #########################
    else:
        configmap_and_pod_name = args.resource_name
        model_name = args.model_path
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

        configmap = get_tuning_configmap(
            name=configmap_and_pod_name,
            namespace=current_namespace,
            tuning_technique=args.tuning_technique,
            output_dir=args.output_dir,
            data_path=args.data_path,
            model=model_name,
            response_template=args.response_template,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
        )

        creating_configmap = True
        while creating_configmap:
            try:
                print(
                    f"Deploying Configmap {configmap_and_pod_name} into namespace {current_namespace}"
                )
                configmap_resource = core_v1_api_instance.create_namespaced_config_map(
                    body=configmap, namespace=current_namespace
                )
                creating_configmap = False
            except kubernetes.client.exceptions.ApiException as e:
                if args.overwrite and "already exists" in e.__str__():
                    logging.warning(
                        f"ConfigMap {configmap_and_pod_name} already exists"
                    )
                    delete_kube_resource_and_wait(
                        core_v1_api_instance,
                        configmap_and_pod_name,
                        current_namespace,
                        "ConfigMap",
                    )
                else:
                    raise e

        tuning_pod = get_tuning_pod_template(
            name=configmap_and_pod_name,
            namespace=current_namespace,
            configmap_name=configmap_and_pod_name,
            image=args.sft_image_name,
            image_pull_secret=args.sft_image_pull_secret,
            mount_fms_tuning_pvc=args.mount_fms_tuning_pvc,
            run_as_root=args.run_as_root,
        )

        creating_pod = True
        while creating_pod:
            try:
                print(
                    f"Deploying Pod {configmap_and_pod_name} into namespace {current_namespace}"
                )
                pod_resource = core_v1_api_instance.create_namespaced_pod(
                    body=tuning_pod, namespace=current_namespace
                )
                creating_pod = False
            except kubernetes.client.exceptions.ApiException as e:
                if args.overwrite and "already exists" in e.__str__():
                    logging.warning(f"Pod {configmap_and_pod_name} already exists")
                    delete_kube_resource_and_wait(
                        core_v1_api_instance,
                        configmap_and_pod_name,
                        current_namespace,
                        "Pod",
                    )
                else:
                    raise e

        print(
            f"Waiting for tuning pod {configmap_and_pod_name} state to be complete..."
        )
        # timeout_seconds by default set to 1hr
        wait_for_pod_state(
            namespace=current_namespace,
            name=configmap_and_pod_name,
            state=PodState.Succeeded,
            label_selector=None,
            debug=True,
            timeout_seconds=args.timeout,
        )

    #########################   deploy TGIS   #########################
    #########################   deploy user provided tgis yaml   #########################
    if args.no_deploy_tgis:
        logging.debug("Not deploying TGIS, exiting")
        sys.exit()

    if args.tgis_yaml:
        tgis_resources = {}
        label_selectors = {}
        tgis_pod_name = ""
        with open(args.tgis_yaml) as f:
            yaml_resources = yaml.safe_load_all(f)
            for item in yaml_resources:
                name = item["metadata"]["name"]
                kind = item["kind"]
                tgis_resources[kind] = name

                if kind == "Deployment":
                    label_selectors = item["spec"].get("selector").get("matchLabels")

        creating_tgis = True
        while creating_tgis:
            try:
                print(
                    f"Deploying TGIS yaml {args.tgis_yaml} into namespace {current_namespace}"
                )
                created_resources = kubernetes.utils.create_from_yaml(
                    client, args.tgis_yaml, namespace=current_namespace
                )
                creating_tgis = False
            except kubernetes.utils.FailToCreateError as e:
                if args.overwrite and "already exists" in e.__str__():
                    logging.warning(f"TGIS resources already exists")
                    for kind, name in tgis_resources.items():
                        if kind == "Deployment":
                            delete_kube_resource_and_wait(
                                apps_v1_api_instance, name, current_namespace, kind
                            )
                        else:
                            delete_kube_resource_and_wait(
                                core_v1_api_instance, name, current_namespace, kind
                            )
                else:
                    raise e

        if tgis_resources.get("Deployment"):
            print(f"Waiting for TGIS pod state to be Running...")

            label_selector_match = ""
            if label_selectors.get("app"):
                label_selector_match = f"app={label_selectors.get('app')}"
            else:
                key = label_selectors.keys()[0]
                label_selector_match = f"{key}={label_selectors[key]}"
            # timeout_seconds by default set to 1hr
            tgis_pod_name = wait_for_pod_state(
                namespace=current_namespace,
                name=None,
                state=PodState.Running,
                label_selector=label_selector_match,
                debug=True,
                timeout_seconds=args.timeout,
            )

    #########################   deploy preset tgis yaml   #########################
    else:
        tgis_svc_and_deploy_name = (
            f"tuning-dev-inference-server-{args.tuning_technique}"
        )
        tgis_label_selectors = {
            "app": f"text-gen-tuning-dev-{args.tuning_technique}",
            "component": "fmaas-inference-server",
        }

        service = get_tgis_service_template(
            name=tgis_svc_and_deploy_name,
            namespace=current_namespace,
            label_selectors=tgis_label_selectors,
        )

        creating_service = True
        while creating_service:
            try:
                print(
                    f"Deploying Service {tgis_svc_and_deploy_name} into namespace {current_namespace}"
                )
                svc_resource = core_v1_api_instance.create_namespaced_service(
                    body=service, namespace=current_namespace
                )
                creating_service = False
            except kubernetes.client.exceptions.ApiException as e:
                if args.overwrite and "already exists" in e.__str__():
                    logging.warning(
                        f"Service {tgis_svc_and_deploy_name} already exists"
                    )
                    delete_kube_resource_and_wait(
                        core_v1_api_instance,
                        tgis_svc_and_deploy_name,
                        current_namespace,
                        "Service",
                    )
                else:
                    raise e

        tgis_deploy = get_tgis_deployment_template(
            model_path=output_dir,
            name=tgis_svc_and_deploy_name,
            namespace=current_namespace,
            label_selectors=tgis_label_selectors,
            tgis_image=args.tgis_image_name,
            mount_fms_tuning_pvc=args.mount_fms_tuning_pvc,
        )

        creating_deploy = True
        while creating_deploy:
            try:
                print(
                    f"Deploying Deployment {tgis_svc_and_deploy_name} into namespace {current_namespace}"
                )
                deploy_resource = apps_v1_api_instance.create_namespaced_deployment(
                    body=tgis_deploy, namespace=current_namespace
                )
                creating_deploy = False
            except kubernetes.client.exceptions.ApiException as e:
                if args.overwrite and "already exists" in e.__str__():
                    logging.warning(
                        f"Deployment {tgis_svc_and_deploy_name} already exists"
                    )
                    delete_kube_resource_and_wait(
                        apps_v1_api_instance,
                        tgis_svc_and_deploy_name,
                        current_namespace,
                        "Deployment",
                    )
                else:
                    raise e

        print(f"Waiting for TGIS pod state to be Running...")
        # timeout_seconds by default set to 1hr
        tgis_deployment = wait_for_deployment_replicas_available(
            namespace=current_namespace,
            name=tgis_svc_and_deploy_name,
            label_selector=None,
            replicas_wanted=1,
            timeout_seconds=args.timeout,
        )

        tgis_pod_name = wait_for_pod_state(
            namespace=current_namespace,
            name=None,
            state=PodState.Running,
            label_selector=f"app={tgis_label_selectors.get('app')}",
            debug=True,
            timeout_seconds=args.timeout,
        )

    #########################   run TGIS inference   #########################
    # Build the request parameters
    stopping = tgis.generation_pb2.StoppingCriteria(max_new_tokens=100)
    params = tgis.generation_pb2.Parameters(
        decoding=tgis.generation_pb2.DecodingParameters(
            repetition_penalty=2.5,
        ),
        stopping=stopping,
    )

    gen_reqs = [
        tgis.generation_pb2.GenerationRequest(text=args.inference_text),
    ]
    batch_request = tgis.generation_pb2.BatchedGenerationRequest(
        requests=gen_reqs,
        params=params,
    )

    # tgis_pod = Pod.get(f"app={tgis_label_selectors.get('app')}")
    tgis_pod = Pod.get(tgis_pod_name)
    tgis_pod.wait("condition=Ready")
    with tgis_pod.portforward(remote_port=8033, local_port=8033):
        with grpc.insecure_channel("localhost:8033") as channel:
            stub = tgis.generation_pb2_grpc.GenerationServiceStub(channel)
            print("Sending batch request...")
            responses = stub.Generate(batch_request)
            for resp in responses.responses:
                print(
                    f"input: {args.inference_text}\nresponse: {resp.text}\nstop_reason: {resp.stop_reason}"
                )
                logging.debug(f"Full TGIS response: {resp}")


if __name__ == "__main__":
    main()
