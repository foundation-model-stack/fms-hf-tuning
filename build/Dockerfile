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
ARG PYTHON_VERSION=3.11
ARG WHEEL_VERSION=""

## Base Layer ##################################################################
FROM registry.access.redhat.com/ubi9/ubi:${BASE_UBI_IMAGE_TAG} as base

ARG PYTHON_VERSION
ARG USER
ARG USER_UID

RUN dnf remove -y --disableplugin=subscription-manager \
        subscription-manager \
    && dnf install -y python${PYTHON_VERSION} procps \
    && ln -s /usr/bin/python${PYTHON_VERSION} /bin/python \
    && python -m ensurepip --upgrade \
    && python -m pip install --upgrade pip \
    && dnf update -y \
    && dnf clean all

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN useradd -u $USER_UID ${USER} -m -g 0 --system && \
    chmod g+rx /home/${USER}

## Used as base of the Release stage to removed unrelated the packages and CVEs
FROM base as release-base

# Removes the python3.9 code to eliminate possible CVEs.  Also removes dnf
RUN rpm -e $(dnf repoquery python3-* -q --installed) dnf python3 yum crypto-policies-scripts


## CUDA Base ###################################################################
FROM base as cuda-base

# Ref: https://docs.nvidia.com/cuda/archive/12.1.0/cuda-toolkit-release-notes/
ENV CUDA_VERSION=12.1.0 \
    NV_CUDA_LIB_VERSION=12.1.0-1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NV_CUDA_CUDART_VERSION=12.1.55-1 \
    NV_CUDA_COMPAT_VERSION=530.30.02-1

RUN dnf config-manager \
       --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo \
    && dnf install -y \
        cuda-cudart-12-1-${NV_CUDA_CUDART_VERSION} \
        cuda-compat-12-1-${NV_CUDA_COMPAT_VERSION} \
    && echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf \
    && dnf clean all

ENV CUDA_HOME="/usr/local/cuda" \
    PATH="/usr/local/nvidia/bin:${CUDA_HOME}/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"

## CUDA Development ############################################################
FROM cuda-base as cuda-devel

# Ref: https://developer.nvidia.com/nccl/nccl-legacy-downloads
ENV NV_CUDA_CUDART_DEV_VERSION=12.1.55-1 \
    NV_NVML_DEV_VERSION=12.1.55-1 \
    NV_LIBCUBLAS_DEV_VERSION=12.1.0.26-1 \
    NV_LIBNPP_DEV_VERSION=12.0.2.50-1 \
    NV_LIBNCCL_DEV_PACKAGE_VERSION=2.18.3-1+cuda12.1

RUN dnf config-manager \
       --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo \
    && dnf install -y \
        cuda-command-line-tools-12-1-${NV_CUDA_LIB_VERSION} \
        cuda-libraries-devel-12-1-${NV_CUDA_LIB_VERSION} \
        cuda-minimal-build-12-1-${NV_CUDA_LIB_VERSION} \
        cuda-cudart-devel-12-1-${NV_CUDA_CUDART_DEV_VERSION} \
        cuda-nvml-devel-12-1-${NV_NVML_DEV_VERSION} \
        libcublas-devel-12-1-${NV_LIBCUBLAS_DEV_VERSION} \
        libnpp-devel-12-1-${NV_LIBNPP_DEV_VERSION} \
        libnccl-devel-${NV_LIBNCCL_DEV_PACKAGE_VERSION} \
    && dnf clean all

ENV LIBRARY_PATH="$CUDA_HOME/lib64/stubs"

FROM cuda-devel as python-installations

ARG WHEEL_VERSION
ARG USER
ARG USER_UID

RUN dnf install -y git && \
    # perl-Net-SSLeay.x86_64 and server_key.pem are installed with git as dependencies
    # Twistlock detects it as H severity: Private keys stored in image
    rm -f /usr/share/doc/perl-Net-SSLeay/examples/server_key.pem && \
    dnf clean all
USER ${USER}
# Ensure that git directory is owned by current user, otherwise git raises
# "fatal: detected dubious ownership" for `/tmp`
WORKDIR /tmp/fms-hf-tuning

RUN --mount=type=cache,target=/home/${USER}/.cache/pip,uid=${USER_UID} \
    python -m pip install --user wheel packaging poetry poetry-plugin-export

COPY --chown=${USER}:root tuning tuning
COPY --chown=${USER}:root .git .git
COPY --chown=${USER}:root pyproject.toml pyproject.toml
COPY --chown=${USER}:root poetry.lock poetry.lock
COPY --chown=${USER}:root launcher launcher
COPY README.md README.md

# Install using poetry if PyPi wheel_version is empty else download the wheel from PyPi
RUN --mount=type=cache,target=/home/${USER}/.cache/pip,uid=${USER_UID} \
    if [[ -z "${WHEEL_VERSION}" ]]; then \
      # Extract requirements from poetry and install them in ~/.local
      python -m pip install --user --requirement <(python -m poetry export --format requirements.txt) &&  \
      # Next install the package with flash-attn \
      python -m pip install --user ".[flash-attn]"; \
    else \
      # This will use whatever dependency versions satisfy the pyproject.toml constraints
      python -m pip install --user "fms-hf-tuning[flash-attn]==${WHEEL_VERSION}" ; \
    fi && \
      python -m pip uninstall wheel -y

## Final image ################################################
FROM release-base as release
ARG USER
ARG PYTHON_VERSION

RUN mkdir -p /licenses
COPY LICENSE /licenses/

RUN mkdir /app && \
    chown -R $USER:0 /app /tmp && \
    chmod -R g+rwX /app /tmp

# Copy scripts and default configs
COPY launcher/accelerate_launch.py fixtures/accelerate_fsdp_defaults.yaml /app/
COPY launcher/utils.py /app/build/
RUN chmod +x /app/accelerate_launch.py

ENV FSDP_DEFAULTS_FILE_PATH="/app/accelerate_fsdp_defaults.yaml"
ENV SET_NUM_PROCESSES_TO_NUM_GPUS="True"

# Need a better way to address this hack
RUN touch /.aim_profile && \
    chmod -R 777 /.aim_profile && \
    mkdir /.cache && \
    chmod -R 777 /.cache

WORKDIR /app
USER ${USER}
COPY --from=python-installations /home/${USER}/.local /home/${USER}/.local
ENV PYTHONPATH="/home/${USER}/.local/lib/python${PYTHON_VERSION}/site-packages"

CMD [ "python", "/app/accelerate_launch.py" ]
