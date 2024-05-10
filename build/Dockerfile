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

<<<<<<< HEAD
FROM registry.access.redhat.com/ubi9/python-311 as wheel

ARG WHEEL_VERSION=""
USER root
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    python -m pip install build
COPY tuning tuning
COPY .git .git
COPY pyproject.toml pyproject.toml
# build wheel if wheel version is empty else download the wheel from PyPi
RUN if [[ -z "${WHEEL_VERSION}" ]]; \
    then python -m build --wheel --outdir /tmp; \
    else pip download fms-hf-tuning==${WHEEL_VERSION} --dest /tmp --only-binary=:all: --no-deps; \
    fi && \
    ls /tmp/*.whl >/tmp/bdist_name


FROM registry.access.redhat.com/ubi9/ubi AS release

ARG CUDA_VERSION=11.8.0
ARG USER=tuning
ARG USER_UID=1000
ARG SET_NUM_PROCESSES_TO_NUM_GPUS=True
=======

## Global Args #################################################################
ARG BASE_UBI_IMAGE_TAG=9.3-1552
ARG PROTOC_VERSION=25.2
ARG PYTORCH_INDEX="https://download.pytorch.org/whl"
>>>>>>> d876974 (Multi-stage build)

# match PyTorch version that was used to compile flash-attention v2 pre-built wheels
# e.g. flash-attn v2.5.2 => torch ['1.12.1', '1.13.1', '2.0.1', '2.1.2', '2.2.0', '2.3.0.dev20240126']
# https://github.com/Dao-AILab/flash-attention/blob/v2.5.2/.github/workflows/publish.yml#L47
# use nightly build index for torch .dev pre-release versions
ARG PYTORCH_VERSION=2.2.1

ARG PYTHON_VERSION=3.11
ARG PYTHON_VERSION_SHORT=311


## Base Layer ##################################################################
FROM registry.access.redhat.com/ubi9/ubi:${BASE_UBI_IMAGE_TAG} as base
WORKDIR /app

ARG PYTHON_VERSION

RUN dnf remove -y --disableplugin=subscription-manager \
        subscription-manager \
        # we install newer version of requests via pip
    python${PYTHON_VERSION}-requests \
    && dnf install -y make \
        # to help with debugging
        procps \
    && dnf clean all

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8


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

######## CUDA Runtime, needs to be verified if needed
# ENV NV_NVTX_VERSION=11.8.86-1 \
#     NV_LIBNPP_VERSION=11.8.0.86-1 \
#     NV_LIBCUBLAS_VERSION=11.11.3.6-1 \
#     NV_LIBNCCL_PACKAGE_VERSION=2.15.5-1+cuda11.8

# RUN dnf config-manager \
#        --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo \
#     && dnf install -y \
#         cuda-libraries-11-8-${NV_CUDA_LIB_VERSION} \
#         cuda-nvtx-11-8-${NV_NVTX_VERSION} \
#         libnpp-11-8-${NV_LIBNPP_VERSION} \
#         libcublas-11-8-${NV_LIBCUBLAS_VERSION} \
#         libnccl-${NV_LIBNCCL_PACKAGE_VERSION} \
#     && dnf clean all

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

<<<<<<< HEAD
RUN dnf install -y python3.11 git \
    && ln -s /usr/bin/python3.11 /bin/python \
    && python -m ensurepip --upgrade \
    && dnf update -y \
    && dnf clean all

# Removes the example private key to avoid high severity vulnerability warning
RUN rm -f /usr/share/doc/perl-Net-SSLeay/examples/server_key.pem

# Removes the python3.9 code to eliminate possible CVEs.  Also removes dnf
RUN rpm -e $(dnf repoquery python3-* -q --installed) dnf python3 yum crypto-policies-scripts

WORKDIR /tmp
COPY --from=wheel /tmp/*.whl /tmp/bdist_name /tmp/
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    python -m pip install wheel && \
    python -m pip install "$(head bdist_name)" && \
    # Due to FIPS tolerance issues, removing aim at this time
    #python -m pip install "$(head bdist_name)[aim]" && \
    python -m pip install "$(head bdist_name)[flash-attn]" && \
    # Clean up the wheel module. It's only needed by flash-attn install
    python -m pip uninstall wheel -y && \
    # Cleanup the bdist whl file
    rm $(head bdist_name) /tmp/bdist_name
=======

## Python builder #############################################################
FROM cuda-devel as python-builder
ARG PYTORCH_INDEX
ARG PYTORCH_VERSION
ARG PYTHON_VERSION
ARG MINIFORGE_VERSION=23.11.0-0

# consistent arch support anywhere we compile CUDA code
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX;8.9"

RUN dnf install -y unzip git ninja-build && dnf clean all

RUN curl -fsSL -v -o ~/miniforge3.sh -O  "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-$(uname)-$(uname -m).sh" && \
    chmod +x ~/miniforge3.sh && \
    bash ~/miniforge3.sh -b -p /opt/conda && \
    source "/opt/conda/etc/profile.d/conda.sh" && \
    conda create -y -p /opt/tuning python=${PYTHON_VERSION} && \
    conda activate /opt/tuning && \
    rm ~/miniforge3.sh

ENV PATH=/opt/tuning/bin/:$PATH

# Install specific version of torch
RUN pip install ninja==1.11.1.1 --no-cache-dir
RUN pip install packaging --no-cache-dir
RUN pip install torch==$PYTORCH_VERSION+cu121 --index-url "${PYTORCH_INDEX}/cu121" --no-cache-dir


## Build flash attention v2 ####################################################
FROM python-builder as flash-att-v2-builder
ARG FLASH_ATT_VERSION=v2.5.6

WORKDIR /usr/src/flash-attention-v2

# Download the wheel or build it if a pre-compiled release doesn't exist
# MAX_JOBS: For CI, limit number of parallel compilation threads otherwise the github runner goes OOM
RUN MAX_JOBS=2  pip --verbose wheel --no-deps flash-attn==${FLASH_ATT_VERSION} \
    # "git+https://github.com/Dao-AILab/flash-attention.git@${FLASH_ATT_VERSION}#subdirectory=csrc/layer_norm" \
    # "git+https://github.com/Dao-AILab/flash-attention.git@${FLASH_ATT_VERSION}#subdirectory=csrc/rotary" \
    --no-build-isolation --no-cache-dir


## Flash attention v2 cached build image #######################################
FROM base as flash-att-v2-cache

# Copy just the wheels we built for flash-attention
COPY --from=flash-att-v2-builder /usr/src/flash-attention-v2 /usr/src/flash-attention-v2

<<<<<<< HEAD
=======

## Wheel Layer #################################################################
FROM registry.access.redhat.com/ubi9/python-${PYTHON_VERSION_SHORT} as wheel

ARG WHEEL_VERSION=""

USER root
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    python -m pip install build
COPY tuning tuning
COPY .git .git
COPY pyproject.toml pyproject.toml
# build wheel if wheel version is empty else download the wheel from PyPi
RUN if [[ -z "${WHEEL_VERSION}" ]]; \
    then python -m build --wheel --outdir /tmp; \
    else pip download fms-hf-tuning==${WHEEL_VERSION} --dest /tmp --only-binary=:all: --no-deps; \
    fi && \
    ls /tmp/*.whl >/tmp/bdist_name


>>>>>>> a3d039e (Tiny fix for app dir)
## Full set of python installations for server release #########################

FROM python-builder as python-installations

ARG PYTHON_VERSION
ARG SITE_PACKAGES=/opt/tuning/lib/python${PYTHON_VERSION}/site-packages

# `pip` is installed in the venv here
ENV PATH=/opt/tuning/bin:$PATH

# Install flash attention v2 from the cache build
RUN --mount=type=bind,from=flash-att-v2-cache,src=/usr/src/flash-attention-v2,target=/usr/src/flash-attention-v2 \
    pip install /usr/src/flash-attention-v2/*.whl --no-cache-dir

WORKDIR /tmp

<<<<<<< HEAD
# TODO Move to installing wheel once we have proper releases setup instead of cloning the repo
RUN git clone https://github.com/foundation-model-stack/fms-hf-tuning.git && \
    cd fms-hf-tuning && \
    python -m pip install ".[dev]" && \
    python -m pip install ".[aim]" && \
    python -m pip install -U datasets
>>>>>>> d876974 (Multi-stage build)
=======
RUN dnf install -y python${PYTHON_VERSION} git \
    && ln -s /usr/bin/python${PYTHON_VERSION} /bin/python \
    && python -m ensurepip --upgrade \
    && dnf clean all

# Removes the example private key to avoid high severity vulnerability warning
RUN rm -f /usr/share/doc/perl-Net-SSLeay/examples/server_key.pem

WORKDIR /tmp
COPY --from=wheel /tmp/*.whl /tmp/bdist_name /tmp/
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    python -m pip install wheel && \
    python -m pip install "$(head bdist_name)" && \
    python -m pip install "$(head bdist_name)[aim]" && \
    python -m pip install "$(head bdist_name)[flash-attn]" && \
    # Clean up the wheel module. It's only needed by flash-attn install
    python -m pip uninstall wheel -y && \
    # Cleanup the bdist whl file
    rm $(head bdist_name) /tmp/bdist_name
>>>>>>> a3d039e (Tiny fix for app dir)

## Final image ################################################
FROM base as release
ARG PYTHON_VERSION
ARG SITE_PACKAGES=/opt/tuning/lib/python${PYTHON_VERSION}/site-packages
ARG USER=tuning
ARG USER_UID=1000

# Copy in the full python environment
COPY --from=python-installations /opt/tuning /opt/tuning

ENV PATH=/opt/tuning/bin:$PATH

# Print a list of all installed packages and versions
RUN pip list -v --disable-pip-version-check --no-python-version-warning

ENV HOME=/home/tuning

RUN mkdir -p /licenses
COPY LICENSE /licenses/

<<<<<<< HEAD
<<<<<<< HEAD
RUN mkdir /app
=======
>>>>>>> a3d039e (Tiny fix for app dir)
# Copy scripts and default configs
COPY build/launch_training.py build/accelerate_launch.py fixtures/accelerate_fsdp_defaults.yaml /app/
COPY build/utils.py /app/build/
RUN chmod +x /app/launch_training.py /app/accelerate_launch.py

ENV FSDP_DEFAULTS_FILE_PATH="/app/accelerate_fsdp_defaults.yaml"
=======
COPY build/launch_training.py /app
RUN chmod +x /app/launch_training.py
>>>>>>> d876974 (Multi-stage build)

# Need a better way to address this hack
RUN touch /.aim_profile && \
    chmod -R 777 /.aim_profile && \
    mkdir /.cache && \
    chmod -R 777 /.cache

# Runs as arbitrary user in OpenShift
RUN useradd -u $USER_UID tuning -m -g 0 --system && \
<<<<<<< HEAD
    chown -R $USER:0 /app /tmp && \
    chmod -R g+rwX /app /tmp
=======
    chown -R $USER:0 ${HOME} && \
    chmod -R g+rwX ${HOME}
>>>>>>> d876974 (Multi-stage build)

# Run as non-root user by default
USER ${USER}

CMD [ "python", "/app/accelerate_launch.py" ]
