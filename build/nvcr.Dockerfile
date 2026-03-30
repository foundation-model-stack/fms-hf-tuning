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

######################## BUILDER ########################
FROM nvcr.io/nvidia/pytorch:${NVCR_IMAGE_VERSION} AS builder

ARG USER=root
ARG USER_UID=0
ARG WORKDIR=/app
ARG SOURCE_DIR=${WORKDIR}/fms-hf-tuning

ARG ENABLE_FMS_ACCELERATION=true
ARG ENABLE_AIM=false
ARG ENABLE_MLFLOW=false
ARG ENABLE_SCANNER=false
ARG ENABLE_CLEARML=true
ARG ENABLE_TRITON_KERNELS=true
ARG ENABLE_RECOMMENDER=true

# Ensures to always build mamba_ssm from source
ENV PIP_NO_BINARY=mamba-ssm,mamba_ssm

# upgrade torch as the base layer contains only torch 2.7
RUN python -m pip install --upgrade pip && \
    pip install --upgrade setuptools && \
    pip install --upgrade --force-reinstall torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128

# Install main package + flash attention
COPY . ${SOURCE_DIR}
RUN cd ${SOURCE_DIR}

RUN pip install --no-cache-dir ${SOURCE_DIR} && \
    pip install --no-cache-dir --no-build-isolation ${SOURCE_DIR}[flash-attn] && \
    pip install --no-cache-dir --no-build-isolation ${SOURCE_DIR}[mamba]

# Optional extras
RUN if [[ "${ENABLE_FMS_ACCELERATION}" == "true" ]]; then \
        pip install --no-cache-dir ${SOURCE_DIR}[fms-accel] && \
        python -m fms_acceleration.cli install fms_acceleration_peft && \
        python -m fms_acceleration.cli install fms_acceleration_foak && \
        python -m fms_acceleration.cli install fms_acceleration_aadp && \
        python -m fms_acceleration.cli install fms_acceleration_moe && \
        python -m fms_acceleration.cli install fms_acceleration_odm; \
    fi

RUN if [[ "${ENABLE_TRITON_KERNELS}" == "true" ]]; then \
        pip install --no-cache-dir "git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels"; \
    fi
RUN if [[ "${ENABLE_CLEARML}" == "true" ]]; then \
        pip install --no-cache-dir ${SOURCE_DIR}[clearml]; \
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
RUN if [[ "${ENABLE_RECOMMENDER}" == "true" ]]; then \
        pip install --no-cache-dir ${SOURCE_DIR}[tuning_config_recommender]; \
    fi

# cleanup build artifacts and caches
RUN rm -rf /root/.cache /tmp/pip-* \
    && find /usr/local/lib/python3.12/dist-packages \
        \( -type d -name "__pycache__" -o -type d -name "tests" -o -type d -name "test" \) \
        -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.12/dist-packages -name "*.pyc" -delete 2>/dev/null || true

######################## RUNTIME ########################
FROM nvcr.io/nvidia/pytorch:${NVCR_IMAGE_VERSION}

ARG WORKDIR=/app
ARG SOURCE_DIR=${WORKDIR}/fms-hf-tuning

# Remove bloat from the base image in a SINGLE layer so deletions reduce size.
# - /opt/pytorch: PyTorch source/examples bundled in NVCR
# - CUDA static libs (*.a): only needed for static linking at compile time
# - CUDA samples/docs: not needed at runtime
# - pip cache and tmp
RUN rm -rf \
        /opt/pytorch \
        /root/.cache \
        /tmp/* \
        /usr/local/cuda/targets/x86_64-linux/lib/*.a \
        /usr/local/cuda/doc \
        /usr/local/cuda/samples \
    && find /usr/local/lib/python3.12/dist-packages \
        \( -type d -name "__pycache__" -o -type d -name "tests" -o -type d -name "test" \) \
        -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.12/dist-packages -name "*.pyc" -delete 2>/dev/null || true \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app \
    && chown -R root:0 /app /tmp \
    && chmod -R g+rwX /app /tmp

WORKDIR /app

# Copy Python site-packages, binaries, and app from builder
COPY --from=builder /usr/local/lib/python3.12/dist-packages \
                    /usr/local/lib/python3.12/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder ${SOURCE_DIR} ${SOURCE_DIR}

RUN chmod -R g+rwX /app /tmp && \
    mkdir -p /.cache && chmod -R 777 /.cache

# Copy scripts and default configs
COPY build/accelerate_launch.py fixtures/accelerate_fsdp_defaults.yaml /app/
COPY build/utils.py /app/build/
RUN chmod +x /app/accelerate_launch.py

ENV FSDP_DEFAULTS_FILE_PATH="/app/accelerate_fsdp_defaults.yaml"
ENV SET_NUM_PROCESSES_TO_NUM_GPUS="True"
ENV HOME="/app"

# Set Triton environment variables for qLoRA
ENV TRITON_HOME="/tmp/triton_home"
ENV TRITON_DUMP_DIR="/tmp/triton_dump_dir"
ENV TRITON_CACHE_DIR="/tmp/triton_cache_dir"
ENV TRITON_OVERRIDE_DIR="/tmp/triton_override_dir"

RUN pip install -U accelerate

CMD ["python", "/app/accelerate_launch.py"]
