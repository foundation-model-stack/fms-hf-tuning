# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Glm4 model."""

import unittest

import pytest

from transformers import AutoModelForCausalLM, AutoTokenizer, Glm4Config, is_torch_available
from transformers.testing_utils import (
    require_flash_attn,
    require_torch,
    require_torch_large_gpu,
    require_torch_sdpa,
    slow,
    torch_device,
)

from ...models.gemma.test_modeling_gemma import GemmaModelTest, GemmaModelTester
from ...test_configuration_common import ConfigTester


if is_torch_available():
    import torch

    from transformers import (
        Glm4ForCausalLM,
        Glm4ForSequenceClassification,
        Glm4ForTokenClassification,
        Glm4Model,
    )


class Glm4ModelTester(GemmaModelTester):
    if is_torch_available():
        config_class = Glm4Config
        model_class = Glm4Model
        for_causal_lm_class = Glm4ForCausalLM
        for_sequence_class = Glm4ForSequenceClassification
        for_token_class = Glm4ForTokenClassification


@require_torch
class Glm4ModelTest(GemmaModelTest, unittest.TestCase):
    all_model_classes = (
        (Glm4Model, Glm4ForCausalLM, Glm4ForSequenceClassification, Glm4ForTokenClassification)
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": Glm4Model,
            "text-classification": Glm4ForSequenceClassification,
            "token-classification": Glm4ForTokenClassification,
            "text-generation": Glm4ForCausalLM,
            "zero-shot": Glm4ForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    _is_stateful = True
    model_split_percents = [0.5, 0.6]

    def setUp(self):
        self.model_tester = Glm4ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Glm4Config, hidden_size=37)


@slow
@require_torch_large_gpu
class Glm4IntegrationTest(unittest.TestCase):
    input_text = ["Hello I am doing", "Hi today"]
    model_id = "THUDM/glm-4-0414-9b-chat"
    revision = "refs/pr/15"
    # This variable is used to determine which CUDA device are we using for our runners (A10 or T4)
    # Depending on the hardware we get different logits / generations
    cuda_compute_capability_major_version = None

    @classmethod
    def setUpClass(cls):
        if is_torch_available() and torch.cuda.is_available():
            # 8 is for A100 / A10 and 7 for T4
            cls.cuda_compute_capability_major_version = torch.cuda.get_device_capability()[0]

    def test_model_9b_fp16(self):
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the history of the internetSolution:\n\nStep 1: Introduction\nThe history of the",
            "Hi today I am going to show you how to make a simple and easy to make a DIY paper flower.",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16, revision=self.revision
        ).to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_9b_bf16(self):
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the history of the internetSolution:\n\nStep 1: Introduction\nThe history of the",
            "Hi today I am going to show you how to make a simple and easy to make a DIY paper flower.",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, revision=self.revision
        ).to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_9b_eager(self):
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the history of the internetSolution:\n\nStep 1: Introduction\nThe history of the",
            "Hi today I am going to show you how to make a simple and easy to make a DIY paper flower.",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            revision=self.revision,
        )
        model.to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_torch_sdpa
    def test_model_9b_sdpa(self):
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the history of the internetSolution:\n\nStep 1: Introduction\nThe history of the",
            "Hi today I am going to show you how to make a simple and easy to make a DIY paper flower.",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            revision=self.revision,
        )
        model.to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_flash_attn
    @pytest.mark.flash_attn_test
    def test_model_9b_flash_attn(self):
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the history of the internetSolution:\n\nStep 1: Introduction\nThe history of the",
            "Hi today I am going to show you how to make a simple and easy to make a DIY paper flower.",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            revision=self.revision,
        )
        model.to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)
