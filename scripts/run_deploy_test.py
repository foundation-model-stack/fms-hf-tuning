# Standard
import os
import json
import time
import argparse
import yaml
import logging
# from typing import Optional

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
    parser.add_argument(
        "--tuning_yaml", help="Path to kube yaml to deploy tuning Pod/PytorchJob/Job", required=True
    )
    parser.add_argument(
        "--tuning_configmap", help="Path to kube yaml with ConfigMap for tuning", required=False
    )
    parser.add_argument(
        "--tgis_yaml", help="Path to kube yaml to deploy TGIS server", required=True
    )
    parsed_args = parser.parse_args()

    if not os.path.exists(parsed_args.tuning_yaml):
        raise RuntimeError(f"Path to tuning yaml does not exist: {parsed_args.tuning_yaml}")
    if not os.path.exists(parsed_args.tgis_yaml):
        raise RuntimeError(f"Path to tuning yaml does not exist: {parsed_args.tgis_yaml}")

    return parsed_args

def main():
    LOGLEVEL = os.environ.get("LOG_LEVEL", "WARNING").upper()
    logging.basicConfig(level=LOGLEVEL)

    args = parse_args()

    # setup kube - requires logged in already
    client = get_api_client()
    current_namespace = get_current_namespace()
    print("Current kube namespace:", current_namespace)
    v1_api_instance = kubernetes.client.CoreV1Api(client)

    # deploy fine tuned model
    # in future can turn whole kube yaml into python kube object
    # will be easier to track status i think
    creating = True
    fine_tune_resources = {}
    pod_name = ""
    with open(args.tuning_yaml) as f:
        yaml_resources = yaml.safe_load_all(f)
        for item in yaml_resources:
            name = item["metadata"]["name"]
            kind = item["kind"]
            fine_tune_resources[kind] = name
            if kind == "Pod":
                pod_name = name

    # deletes kube resources if they already exist
    while creating:
        try:
            fine_tune_resources = kubernetes.utils.create_from_yaml(client, args.tuning_yaml, namespace=current_namespace)
            creating = False
        except kubernetes.utils.FailToCreateError as e:
            if "already exists" in e.__str__():
                for kind, name in fine_tune_resources.items():
                    deleting = True
                    while deleting:
                        try:
                            print(f"Deleting kube resource {kind} {name}")
                            if kind == "ConfigMap":
                                v1_api_instance.delete_namespaced_config_map(name, current_namespace)
                            elif kind == "Pod":
                                v1_api_instance.delete_namespaced_pod(name, current_namespace)
                            else:
                                raise RuntimeError(f"Unable to delete kube resource {name} of kind {kind}")
                            
                            # give small time to delete before looping
                            time.sleep(3)
                        except kubernetes.client.exceptions.ApiException as e:
                            has_deleted = json.loads(e.body)['code'] == 404
                            if has_deleted:
                                deleting = False

    # deploy TGIS-es
    # TODO: need to resuse output dir as MODEL_NAME
    try:
        tgis_ft_resources = kubernetes.utils.create_from_yaml(client, args.tgis_yaml, namespace=current_namespace)
    except kubernetes.utils.FailToCreateError as e:
        if "already exists" not in e.__str__():
            raise RuntimeError(e)
        else:
            print("TGIS pod already exists, will reuse")

    print(f"Waiting for tuning pod {pod_name} state to be complete...")
    # timeout_seconds by default set to 1hr
    wait_for_pod_state(
        namespace=current_namespace,
        name=pod_name,
        state=PodState.Succeeded,
        label_selector=None,
        debug=True,
    )

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
