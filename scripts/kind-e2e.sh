#!/bin/bash

# Copyright 2022 IBM, Red Hat
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail
: "${INGRESS_NGINX_VERSION:=controller-v1.9.6}"
: "${KinD:=false}"

# Setup KinD
if ${KinD}
then
echo "Creating KinD cluster"
kind delete cluster -n training-operator-cluster
cat <<EOF | kind create cluster --name training-operator-cluster --config -
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
    image: kindest/node:v1.25.3@sha256:f52781bc0d7a19fb6c405c2af83abfeb311f130707a0e219175677e366cc45d1
    extraPortMappings:
    - containerPort: 80
      hostPort: 80
      protocol: TCP
    kubeadmConfigPatches:
      - |
        kind: InitConfiguration
        nodeRegistration:
          kubeletExtraArgs:
            node-labels: "ingress-ready=true"
EOF
fi
echo "Deploying Ingress controller into KinD cluster"
curl https://raw.githubusercontent.com/kubernetes/ingress-nginx/"${INGRESS_NGINX_VERSION}"/deploy/static/provider/kind/deploy.yaml | sed "s/--publish-status-address=localhost/--report-node-internal-ip-address\\n        - --status-update-interval=10/g" | kubectl apply -f -
kubectl annotate ingressclass nginx "ingressclass.kubernetes.io/is-default-class=true"
kubectl -n ingress-nginx wait --timeout=300s --for=condition=Available deployments --all

# Kueue
: "${KUEUE_VERSION:=v0.6.1}" # v0.6.1 is current release
: "${KFTO_IMG:=training-operator:dev}"

echo "Installing Kueue into the cluster"
kubectl apply --server-side -f https://github.com/kubernetes-sigs/kueue/releases/download/${KUEUE_VERSION}/manifests.yaml

echo "Wait for Kueue deployment"
kubectl -n kueue-system wait --timeout=300s --for=condition=Available deployments --all

echo "Creating Kueue Resources"
cat <<EOF | kubectl apply -f -
apiVersion: kueue.x-k8s.io/v1beta1
kind: ResourceFlavor
metadata:
  name: "cpu-flavor"
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: "cq-small"
spec:
  namespaceSelector: {} # match all.
  resourceGroups:
  - coveredResources: ["cpu", "memory"]
    flavors:
    - name: "cpu-flavor"
      resources:
      - name: "cpu"
        nominalQuota: 5
      - name: "memory"
        nominalQuota: 20Gi
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata:
  name: lq-trainer
  namespace: default
spec:
  clusterQueue: cq-small
EOF

echo "Install training-operator (stable release)"
# docker pull kubeflow/training-operator:v1-83ddd3b
# kind load docker-image kubeflow/training-operator:v1-855e096 -n training-operator-cluster
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.7.0"
kubectl -n kubeflow wait --timeout=300s --for=condition=Available deployments --all

echo "Build fms-hf-tuning image"
docker build --progress=plain -t fms-hf-tuning:dev . -f build/Dockerfile
kind load docker-image fms-hf-tuning:dev -n training-operator-cluster

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: my-config
data:
  config.json: |
    {
      "accelerate_launch_args": {
        "num_machines": 1,
        "num_processes": 2
      },
      "model_name_or_path": "bigscience/bloom-560m",
      "training_data_path": "/etc/config/twitter_complaints_small.json",
      "output_dir": "/tmp/out",
      "num_train_epochs": 1.0,
      "per_device_train_batch_size": 4,
      "per_device_eval_batch_size": 4,
      "gradient_accumulation_steps": 4,
      "evaluation_strategy": "no",
      "save_strategy": "epoch",
      "learning_rate": 1e-5,
      "weight_decay": 0.0,
      "lr_scheduler_type": "cosine",
      "logging_steps": 1.0,
      "packing": false,
      "include_tokens_per_second": true,
      "response_template": "\n### Label:",
      "dataset_text_field": "output",
      "use_flash_attn": false,
      "torch_dtype": "float32",
      "peft_method": "pt",
      "tokenizer_name_or_path": "bigscience/bloom"
    }
  twitter_complaints_small.json: |
    {"Tweet text":"@HMRCcustomers No this is my first job","ID":0,"Label":2,"text_label":"no complaint","output":"### Text: @HMRCcustomers No this is my first job\n\n### Label: no complaint"}
    {"Tweet text":"@KristaMariePark Thank you for your interest! If you decide to cancel, you can call Customer Care at 1-800-NYTIMES.","ID":1,"Label":2,"text_label":"no complaint","output":"### Text: @KristaMariePark Thank you for your interest! If you decide to cancel, you can call Customer Care at 1-800-NYTIMES.\n\n### Label: no complaint"}
    {"Tweet text":"If I can't get my 3rd pair of @beatsbydre powerbeats to work today I'm doneski man. This is a slap in my balls. Your next @Bose @BoseService","ID":2,"Label":1,"text_label":"complaint","output":"### Text: If I can't get my 3rd pair of @beatsbydre powerbeats to work today I'm doneski man. This is a slap in my balls. Your next @Bose @BoseService\n\n### Label: complaint"}
    {"Tweet text":"@EE On Rosneath Arial having good upload and download speeds but terrible latency 200ms. Why is this.","ID":3,"Label":1,"text_label":"complaint","output":"### Text: @EE On Rosneath Arial having good upload and download speeds but terrible latency 200ms. Why is this.\n\n### Label: complaint"}
    {"Tweet text":"Couples wallpaper, so cute. :) #BrothersAtHome","ID":4,"Label":2,"text_label":"no complaint","output":"### Text: Couples wallpaper, so cute. :) #BrothersAtHome\n\n### Label: no complaint"}
    {"Tweet text":"@mckelldogs This might just be me, but-- eyedrops? Artificial tears are so useful when you're sleep-deprived and sp\u2026 https:\/\/t.co\/WRtNsokblG","ID":5,"Label":2,"text_label":"no complaint","output":"### Text: @mckelldogs This might just be me, but-- eyedrops? Artificial tears are so useful when you're sleep-deprived and sp\u2026 https:\/\/t.co\/WRtNsokblG\n\n### Label: no complaint"}
    {"Tweet text":"@Yelp can we get the exact calculations for a business rating (for example if its 4 stars but actually 4.2) or do we use a 3rd party site?","ID":6,"Label":2,"text_label":"no complaint","output":"### Text: @Yelp can we get the exact calculations for a business rating (for example if its 4 stars but actually 4.2) or do we use a 3rd party site?\n\n### Label: no complaint"}
    {"Tweet text":"@nationalgridus I have no water and the bill is current and paid. Can you do something about this?","ID":7,"Label":1,"text_label":"complaint","output":"### Text: @nationalgridus I have no water and the bill is current and paid. Can you do something about this?\n\n### Label: complaint"}
    {"Tweet text":"Never shopping at @MACcosmetics again. Every time I go in there, their employees are super rude\/condescending. I'll take my $$ to @Sephora","ID":8,"Label":1,"text_label":"complaint","output":"### Text: Never shopping at @MACcosmetics again. Every time I go in there, their employees are super rude\/condescending. I'll take my $$ to @Sephora\n\n### Label: complaint"}
    {"Tweet text":"@JenniferTilly Merry Christmas to as well. You get more stunning every year \ufffd\ufffd","ID":9,"Label":2,"text_label":"no complaint","output":"### Text: @JenniferTilly Merry Christmas to as well. You get more stunning every year \ufffd\ufffd\n\n### Label: no complaint"}
---
apiVersion: "kubeflow.org/v1"
kind: PyTorchJob
metadata:
  name: kfto-sft
  labels:
    kueue.x-k8s.io/queue-name: lq-trainer
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: Never # Do not restart the pod on failure. If you do set it to OnFailure, be sure to also set backoffLimit
      template:
        spec:
          containers:
            - name: pytorch
              # This is the temp location util image is officially released
              image: fms-hf-tuning:dev
              imagePullPolicy: IfNotPresent
              command:
                - "python"
                - "/app/accelerate_launch.py"
              env:
                - name: SFT_TRAINER_CONFIG_JSON_PATH
                  value: /etc/config/config.json
              volumeMounts:
              - name: config-volume
                mountPath: /etc/config
          volumes:
          - name: config-volume
            configMap:
              name: my-config
              items:
              - key: config.json
                path: config.json
              - key: twitter_complaints_small.json
                path: twitter_complaints_small.json
EOF

kubectl -n default wait --timeout=300s --for=condition=Created pytorchjobs kfto-sft
echo "pytorchjobs created"
kubectl -n default wait --timeout=300s --for=condition=Running pytorchjobs kfto-sft
echo "pytorchjobs running"
kubectl logs -f kfto-sft-master-0 -n default
kubectl -n default wait --timeout=300s --for=condition=Succeeded pytorchjobs kfto-sft
echo "e2e test finished successfully"