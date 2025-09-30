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
## If the nvcr container is updated, ensure to check the torch and python
## installation version inside the dockerfile before pushing changes.
ARG NVCR_IMAGE_VERSION=25.02-py3

# This is based on what is inside the NVCR image already
ARG PYTHON_VERSION=3.12

## Base Layer ##################################################################
FROM nvcr.io/nvidia/pytorch:${NVCR_IMAGE_VERSION} AS dev

ARG USER=root
ARG USER_UID=0
ARG WORKDIR=/app
ARG SOURCE_DIR=${WORKDIR}/fms-hf-tuning

ARG ENABLE_FMS_ACCELERATION=true
ARG ENABLE_AIM=true
ARG ENABLE_ALORA=true
ARG ENABLE_MLFLOW=true
ARG ENABLE_SCANNER=true
ARG ENABLE_CLEARML=true
ARG ENABLE_TRITON_KERNELS=true
ARG ENABLE_MAMBA_SUPPORT=true

# Ensures to always build mamba_ssm from source
ENV PIP_NO_BINARY=mamba-ssm,mamba_ssm

RUN python -m pip install --upgrade pip

# upgrade torch as the base layer contains only torch 2.7
RUN pip install --upgrade --force-reinstall torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128

# Install main package + flash attention
COPY . ${SOURCE_DIR}
RUN cd ${SOURCE_DIR}
RUN pip install --no-cache-dir ${SOURCE_DIR} && \
    pip install --no-cache-dir ${SOURCE_DIR}[flash-attn]

# Optional extras
RUN if [[ "${ENABLE_FMS_ACCELERATION}" == "true" ]]; then \
        pip install --no-cache-dir ${SOURCE_DIR}[fms-accel] && \
        python -m fms_acceleration.cli install fms_acceleration_peft && \
        python -m fms_acceleration.cli install fms_acceleration_foak && \
        python -m fms_acceleration.cli install fms_acceleration_aadp && \
        python -m fms_acceleration.cli install fms_acceleration_moe; \
    fi

RUN if [[ "${ENABLE_ALORA}" == "true" ]]; then \
        pip install --no-cache-dir ${SOURCE_DIR}[activated-lora]; \
    fi
RUN if [[ "${ENABLE_AIM}" == "true" ]]; then \
        pip install --no-cache-dir ${SOURCE_DIR}[aim]; \
    fi
RUN if [[ "${ENABLE_MLFLOW}" == "true" ]]; then \
        pip install --no-cache-dir ${SOURCE_DIR}[mlflow]; \
    fi
RUN if [[ "${ENABLE_SCANNER}" == "true" ]]; then \
        pip install --no-cache-dir ${SOURCE_DIR}[scanner-dev]; \
    fi
RUN if [[ "${ENABLE_CLEARML}" == "true" ]]; then \
        pip install --no-cache-dir ${SOURCE_DIR}[clearml]; \
    fi
RUN if [[ "${ENABLE_MAMBA_SUPPORT}" == "true" ]]; then \
        pip install --no-cache-dir ${SOURCE_DIR}[mamba]; \
    fi
RUN if [[ "${ENABLE_TRITON_KERNELS}" == "true" ]]; then \
        pip install --no-cache-dir "git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels"; \
    fi

RUN chmod -R g+rwX $WORKDIR /tmp
RUN mkdir -p /.cache && chmod -R 777 /.cache

# Set Triton environment variables for qLoRA
ENV TRITON_HOME="/tmp/triton_home"
ENV TRITON_DUMP_DIR="/tmp/triton_dump_dir"
ENV TRITON_CACHE_DIR="/tmp/triton_cache_dir"
ENV TRITON_OVERRIDE_DIR="/tmp/triton_override_dir"

WORKDIR $WORKDIR

CMD ["${SOURCE_DIR}/build/accelerate_launch.py"]
