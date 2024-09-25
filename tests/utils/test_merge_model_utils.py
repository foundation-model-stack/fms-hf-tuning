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

"""Unit Tests for SFT Trainer's merge_model_utils functions
"""

# Standard
import os
import shutil
import tempfile

# Third Party
from safetensors import safe_open
import pytest

# Local
from tuning.utils.merge_model_utils import post_process_vLLM_adapters_new_tokens

dir_path = os.path.dirname(os.path.realpath(__file__))
DUMMY_TUNED_LLAMA_WITH_ADDED_TOKENS = os.path.join(
    dir_path, "../artifacts/tuned_llama_with_added_tokens"
)


def test_post_process_vllm_adapters_new_tokens():
    """Ensure that in post-process, we output the correct format supported by vLLM for added_tokens
    - we should output a new_embeddings.safetensors
    - we should not have lm_head.weight in adapter_model.safetensors
    """
    # first, double check dummy tuned llama has a lm_head.weight
    found_lm_head = False
    with safe_open(
        os.path.join(DUMMY_TUNED_LLAMA_WITH_ADDED_TOKENS, "adapter_model.safetensors"),
        framework="pt",
    ) as f:
        for k in f.keys():
            if "lm_head.weight" in k:
                found_lm_head = True
    assert found_lm_head

    # do the post processing
    with tempfile.TemporaryDirectory() as tempdir:
        post_process_vLLM_adapters_new_tokens(
            DUMMY_TUNED_LLAMA_WITH_ADDED_TOKENS, tempdir, num_added_tokens=1
        )

        # check that new_embeddings.safetensors exist
        new_embeddings = os.path.join(tempdir, "new_embeddings.safetensors")
        assert os.path.exists(new_embeddings)

        # check that lm_head.weight NOT in the new outputted adapter_model.safetensors
        adapter_model = os.path.join(tempdir, "adapter_model.safetensors")
        assert os.path.exists(adapter_model)

        found_lm_head = False
        with safe_open(adapter_model, framework="pt") as f:
            for k in f.keys():
                if "lm_head.weight" in k:
                    found_lm_head = True
        assert not found_lm_head


def test_post_process_vllm_adapters_no_new_tokens():
    """Ensure that an error is returned if no added tokens while tuning, \
          but embeddings resized.
    """
    # first, double check dummy tuned llama has a lm_head.weight
    found_lm_head = False
    with safe_open(
        os.path.join(DUMMY_TUNED_LLAMA_WITH_ADDED_TOKENS, "adapter_model.safetensors"),
        framework="pt",
    ) as f:
        for k in f.keys():
            if "lm_head.weight" in k:
                found_lm_head = True
    assert found_lm_head
    # do the post processing
    with pytest.raises(NotImplementedError):
        post_process_vLLM_adapters_new_tokens(
            DUMMY_TUNED_LLAMA_WITH_ADDED_TOKENS, None, num_added_tokens=0
        )


def test_post_process_in_place_vllm_adapters_new_tokens():
    """Ensure that in post-process, we output the correct format supported by vLLM for added_tokens
    - if output dir is not specified, it should modify files in place
    - we should output a new_embeddings.safetensors
    - we should not have lm_head.weight in adapter_model.safetensors
    """
    # first, double check dummy tuned llama has a lm_head.weight
    found_lm_head = False
    with safe_open(
        os.path.join(DUMMY_TUNED_LLAMA_WITH_ADDED_TOKENS, "adapter_model.safetensors"),
        framework="pt",
    ) as f:
        for k in f.keys():
            if "lm_head.weight" in k:
                found_lm_head = True
    assert found_lm_head

    # do the post processing
    with tempfile.TemporaryDirectory() as tempdir:
        shutil.copytree(
            DUMMY_TUNED_LLAMA_WITH_ADDED_TOKENS, tempdir, dirs_exist_ok=True
        )
        post_process_vLLM_adapters_new_tokens(tempdir, None, num_added_tokens=1)

        # check that new_embeddings.safetensors exist
        new_embeddings = os.path.join(tempdir, "new_embeddings.safetensors")
        assert os.path.exists(new_embeddings)

        # check that lm_head.weight NOT in the new outputted adapter_model.safetensors
        adapter_model = os.path.join(tempdir, "adapter_model.safetensors")
        assert os.path.exists(adapter_model)

        found_lm_head = False
        with safe_open(adapter_model, framework="pt") as f:
            for k in f.keys():
                if "lm_head.weight" in k:
                    found_lm_head = True
        assert not found_lm_head
