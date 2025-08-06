# Copyright The FMS HF Tuning Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


## Global Args #################################################################
ARG BASE_UBI_IMAGE_TAG=latest
ARG USER=tuning
ARG USER_UID=1000
ARG PYTHON_VERSION=3.12
ARG WHEEL_VERSION=""

## Change these args if requested via ENABLE_XX set to "true" or "false"
ARG ENABLE_FMS_ACCELERATION=true
ARG ENABLE_AIM=false
ARG ENABLE_ALORA=false
ARG ENABLE_MLFLOW=false
ARG ENABLE_SCANNER=false
ARG ENABLE_CLEARML=false
ARG ENABLE_TRITON_KERNELS=true
ARG ENABLE_MAMBA_SUPPORT=true

## Base Layer ##################################################################
FROM registry.access.redhat.com/ubi9/ubi:${BASE_UBI_IMAGE_TAG} AS base

ARG PYTHON_VERSION
ARG USER
ARG USER_UID

# Note this is tested to be working for version 3.9, 3.11, 3.12
RUN dnf remove -y --disableplugin=subscription-manager \
        subscription-manager \
    && dnf install -y python${PYTHON_VERSION} procps g++ python${PYTHON_VERSION}-devel \
    && ln -s /usr/bin/python${PYTHON_VERSION} /bin/python \
    && python -m ensurepip --upgrade \
    && python -m pip install --upgrade pip \
    && python -m pip install --upgrade setuptools \
    && dnf update -y \
    && dnf clean all

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN useradd -u $USER_UID ${USER} -m -g 0 --system && \
    chmod g+rx /home/${USER}

## Used as base of the Release stage to removed unrelated the packages and CVEs
FROM base AS release-base

# Removes the python code to eliminate possible CVEs.  Also removes dnf
RUN rpm -e $(dnf repoquery python3-* -q --installed) dnf python3 yum crypto-policies-scripts


## CUDA Base ###################################################################
FROM base AS cuda-base

# Ref: https://docs.nvidia.com/cuda/archive/12.1.0/cuda-toolkit-release-notes/
ENV CUDA_VERSION_MAJOR=12 \
    CUDA_VERSION_MINOR=8 \
    CUDA_VERSION_SUB_MINOR=1 \
    CUDA_VERSION=${CUDA_VERSION_MAJOR}.{CUDA_VERSION_MINOR}.${CUDA_VERSION_SUB_MINOR} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN dnf install -y dnf-plugins-core && dnf clean all

RUN dnf config-manager \
       --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo \
    && dnf install -y \
        cuda-cudart-${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR} \
        cuda-compat-${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR} \
    && echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf \
    && dnf clean all

ENV CUDA_HOME="/usr/local/cuda" \
    PATH="/usr/local/nvidia/bin:${CUDA_HOME}/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"

## CUDA Development ############################################################
FROM cuda-base AS cuda-devel

# Ref: https://developer.nvidia.com/nccl/nccl-legacy-downloads
ENV CUDA_VERSION_MAJOR=12 \
    CUDA_VERSION_MINOR=8 \
    CUDA_VERSION_SUB_MINOR=1 \
    CUDA_VERSION=${CUDA_VERSION_MAJOR}.{CUDA_VERSION_MINOR}.${CUDA_VERSION_SUB_MINOR} \
    NV_LIBNCCL_DEV_PACKAGE_VERSION=2.26.2-1+cuda12.8

RUN dnf config-manager \
       --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo \
    && dnf install -y \
        cuda-command-line-tools-${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR} \
        cuda-libraries-devel-${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR} \
        cuda-minimal-build-${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR} \
        cuda-cudart-devel-${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR} \
        cuda-nvml-devel-${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR} \
        libcublas-devel-${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR} \
        libnpp-devel-${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR} \
        libnccl-devel-${NV_LIBNCCL_DEV_PACKAGE_VERSION} \
    && dnf clean all

# opening connection for too long in one go was resulting in timeouts
RUN dnf config-manager \
       --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo \
    && dnf clean packages \
    && dnf install -y \
        libcusparselt0 libcusparselt-devel \
        cudnn9-cuda-${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR} \
    && dnf clean all

ENV LIBRARY_PATH="$CUDA_HOME/lib64/stubs"

FROM cuda-devel AS base-python-installations

ARG WHEEL_VERSION
ARG USER
ARG USER_UID

RUN dnf install -y git && \
    # perl-Net-SSLeay.x86_64 and server_key.pem are installed with git as dependencies
    # Twistlock detects it as H severity: Private keys stored in image
    rm -f /usr/share/doc/perl-Net-SSLeay/examples/server_key.pem && \
    dnf clean all
USER ${USER}
WORKDIR /tmp
RUN --mount=type=cache,target=/home/${USER}/.cache/pip,uid=${USER_UID} \
    python -m pip install --user build
COPY --chown=${USER}:root tuning tuning
COPY .git .git
COPY pyproject.toml pyproject.toml

# Build a wheel if PyPi wheel_version is empty else download the wheel from PyPi
RUN if [[ -z "${WHEEL_VERSION}" ]]; \
    then python -m build --wheel --outdir /tmp; \
    else pip download fms-hf-tuning==${WHEEL_VERSION} --dest /tmp --only-binary=:all: --no-deps; \
    fi && \
    ls /tmp/*.whl >/tmp/bdist_name

# Ensures to always build mamba_ssm from source
ENV PIP_NO_BINARY=mamba-ssm,mamba_ssm

# Install from the wheel
RUN --mount=type=cache,target=/home/${USER}/.cache/pip,uid=${USER_UID} \
    python -m pip install --user wheel && \
    python -m pip install --user "$(head bdist_name)" && \
    python -m pip install --user "$(head bdist_name)[flash-attn]" && \
    python -m pip install --user --no-build-isolation "$(head bdist_name)[mamba]"

FROM base-python-installations AS python-installations

ARG ENABLE_FMS_ACCELERATION
ARG ENABLE_AIM
ARG ENABLE_MLFLOW
ARG ENABLE_ALORA
ARG ENABLE_SCANNER
ARG ENABLE_CLEARML
ARG ENABLE_TRITON_KERNELS
ARG ENABLE_MAMBA_SUPPORT

# fms_acceleration_peft = PEFT-training, e.g., 4bit QLoRA
# fms_acceleration_foak = Fused LoRA and triton kernels
# fms_acceleration_aadp = Padding-Free Flash Attention Computation
# fms_acceleration_moe = Parallelized Mixture of Experts
RUN if [[ "${ENABLE_FMS_ACCELERATION}" == "true" ]]; then \
        python -m pip install --user "$(head bdist_name)[fms-accel]"; \
        python -m fms_acceleration.cli install fms_acceleration_peft; \
        python -m fms_acceleration.cli install fms_acceleration_foak; \
        python -m fms_acceleration.cli install fms_acceleration_aadp; \
        python -m fms_acceleration.cli install fms_acceleration_moe; \
    fi

# Training support
RUN if [[ "${ENABLE_ALORA}" == "true" ]]; then \
        python -m pip install --user "$(head bdist_name)[activated-lora]"; \
    fi

# Trackers
RUN if [[ "${ENABLE_AIM}" == "true" ]]; then \
        python -m pip install --user "$(head bdist_name)[aim]"; \
    fi
RUN if [[ "${ENABLE_MLFLOW}" == "true" ]]; then \
    python -m pip install --user "$(head bdist_name)[mlflow]"; \
    fi
RUN if [[ "${ENABLE_SCANNER}" == "true" ]]; then \
        python -m pip install --user "$(head bdist_name)[scanner-dev]"; \
    fi
RUN if [[ "${ENABLE_CLEARML}" == "true" ]]; then \
        python -m pip install --user "$(head bdist_name)[clearml]"; \
    fi

# Model support
RUN if [[ "${ENABLE_MAMBA_SUPPORT}" == "true" ]]; then \
        python -m pip install --user "$(head bdist_name)[mamba]"; \
    fi
RUN if [[ "${ENABLE_TRITON_KERNELS}" == "true" ]]; then \
        python -m pip install --user "git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels"; \
    fi

    # Clean up the wheel module. It's only needed by flash-attn install
RUN python -m pip uninstall wheel build -y && \
    # Cleanup the bdist whl file
    rm $(head bdist_name) /tmp/bdist_name

## Final image ################################################
FROM python-installations AS release
ARG USER
ARG PYTHON_VERSION
ARG ENABLE_AIM

RUN mkdir -p /licenses
COPY LICENSE /licenses/

RUN mkdir /app && \
    chown -R $USER:0 /app /tmp && \
    chmod -R g+rwX /app /tmp

# Set Triton environment variables for qLoRA
ENV TRITON_HOME="/tmp/triton_home"
ENV TRITON_DUMP_DIR="/tmp/triton_dump_dir"
ENV TRITON_CACHE_DIR="/tmp/triton_cache_dir"
ENV TRITON_OVERRIDE_DIR="/tmp/triton_override_dir"

# Need a better way to address these hacks
RUN if [[ "${ENABLE_AIM}" == "true" ]] ; then \
        touch /.aim_profile && \
        chmod -R 777 /.aim_profile; \
    fi
RUN mkdir /.cache && \
    chmod -R 777 /.cache

# Copy scripts and default configs
COPY build/accelerate_launch.py fixtures/accelerate_fsdp_defaults.yaml /app/
COPY build/utils.py /app/build/
RUN chmod +x /app/accelerate_launch.py

ENV FSDP_DEFAULTS_FILE_PATH="/app/accelerate_fsdp_defaults.yaml"
ENV SET_NUM_PROCESSES_TO_NUM_GPUS="True"

WORKDIR /app
USER ${USER}
COPY --from=python-installations /home/${USER}/.local /home/${USER}/.local
ENV PYTHONPATH="/home/${USER}/.local/lib/python${PYTHON_VERSION}/site-packages"

CMD [ "python", "/app/accelerate_launch.py" ]
