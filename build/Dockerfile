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

USER root

RUN dnf remove -y --disableplugin=subscription-manager \
        subscription-manager \
        # we install newer version of requests via pip
        python3.11-requests \
    && dnf install -y make \
        # to help with debugging
        procps \
    && dnf clean all

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

ENV CUDA_VERSION=$CUDA_VERSION \
    NV_CUDA_LIB_VERSION=11.8.0-1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NV_CUDA_CUDART_VERSION=11.8.89-1 \
    NV_CUDA_COMPAT_VERSION=520.61.05-1

RUN dnf config-manager \
       --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo \
    && dnf install -y \
        cuda-cudart-11-8-${NV_CUDA_CUDART_VERSION} \
        cuda-compat-11-8-${NV_CUDA_COMPAT_VERSION} \
    && echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf \
    && dnf clean all

ENV CUDA_HOME="/usr/local/cuda" \
    PATH="/usr/local/nvidia/bin:${CUDA_HOME}/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"


ENV NV_NVTX_VERSION=11.8.86-1 \
    NV_LIBNPP_VERSION=11.8.0.86-1 \
    NV_LIBCUBLAS_VERSION=11.11.3.6-1 \
    NV_LIBNCCL_PACKAGE_VERSION=2.15.5-1+cuda11.8

RUN dnf config-manager \
       --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo \
    && dnf install -y \
        cuda-libraries-11-8-${NV_CUDA_LIB_VERSION} \
        cuda-nvtx-11-8-${NV_NVTX_VERSION} \
        libnpp-11-8-${NV_LIBNPP_VERSION} \
        libcublas-11-8-${NV_LIBCUBLAS_VERSION} \
        libnccl-${NV_LIBNCCL_PACKAGE_VERSION} \
    && dnf clean all

ENV NV_CUDA_CUDART_DEV_VERSION=11.8.89-1 \
    NV_NVML_DEV_VERSION=11.8.86-1 \
    NV_LIBCUBLAS_DEV_VERSION=11.11.3.6-1 \
    NV_LIBNPP_DEV_VERSION=11.8.0.86-1 \
    NV_LIBNCCL_DEV_PACKAGE_VERSION=2.15.5-1+cuda11.8

RUN dnf config-manager \
       --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo \
    && dnf install -y \
        cuda-command-line-tools-11-8-${NV_CUDA_LIB_VERSION} \
        cuda-libraries-devel-11-8-${NV_CUDA_LIB_VERSION} \
        cuda-minimal-build-11-8-${NV_CUDA_LIB_VERSION} \
        cuda-cudart-devel-11-8-${NV_CUDA_CUDART_DEV_VERSION} \
        cuda-nvml-devel-11-8-${NV_NVML_DEV_VERSION} \
        libcublas-devel-11-8-${NV_LIBCUBLAS_DEV_VERSION} \
        libnpp-devel-11-8-${NV_LIBNPP_DEV_VERSION} \
        libnccl-devel-${NV_LIBNCCL_DEV_PACKAGE_VERSION} \
    && dnf clean all

ENV LIBRARY_PATH="$CUDA_HOME/lib64/stubs"

RUN dnf install -y python3.11 git && \
    ln -s /usr/bin/python3.11 /bin/python && \
    python -m ensurepip --upgrade && \
    dnf clean all

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

RUN mkdir -p /licenses
COPY LICENSE /licenses/

RUN mkdir /app
# Copy scripts and default configs
COPY build/launch_training.py build/accelerate_launch.py fixtures/accelerate_fsdp_defaults.yaml /app/
COPY build/utils.py /app/build/
RUN chmod +x /app/launch_training.py /app/accelerate_launch.py

ENV FSDP_DEFAULTS_FILE_PATH="/app/accelerate_fsdp_defaults.yaml"

# Need a better way to address this hack
RUN touch /.aim_profile && \
    chmod -R 777 /.aim_profile && \
    mkdir /.cache && \
    chmod -R 777 /.cache

# create tuning user and give ownership to dirs
RUN useradd -u $USER_UID tuning -m -g 0 --system && \
    chown -R $USER:0 /app /tmp && \
    chmod -R g+rwX /app /tmp

WORKDIR /app
USER ${USER}

CMD [ "python", "/app/accelerate_launch.py" ]
