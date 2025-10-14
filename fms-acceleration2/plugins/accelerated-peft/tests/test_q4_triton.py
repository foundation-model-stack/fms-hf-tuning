###############################################################################
# Adapted from https://github.com/ModelCloud/GPTQModel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################
# -- do not touch
# Standard
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

# Standard
import unittest  # noqa: E402

# Third Party
from transformers import AutoTokenizer  # noqa: E402
import torch  # noqa: E402

CUDA_AVAILABLE = False
if torch.cuda.is_available():
    # First Party
    from fms_acceleration_peft.gptqmodel import Backend, GPTQModel  # noqa: E402
    from fms_acceleration_peft.gptqmodel.nn_modules.qlinear.qlinear_tritonv2 import (  # noqa: E402
        QuantLinear as TritonV2QuantLinear,
    )

    CUDA_AVAILABLE = True


GENERATE_EVAL_SIZE = 100


class TestsQ4Triton(unittest.TestCase):
    @unittest.skipIf(
        CUDA_AVAILABLE is False,
        "Only runs if there is a cuda device available",
    )
    def test_generation_desc_act_false(self):
        prompt = "I am in Paris and"

        reference_output = "<s> I am in Paris and I am in love with you.\n\nScene 2:\n\n(The stage is now dark, but the audience can see the characters walking around the stage.)\n\n(The stage is now lit up, but the audience can only see the characters' silhouettes.)\n\n("
        new_tokens = 60

        model_id = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

        model_q = GPTQModel.from_quantized(
            model_id,
            device="cuda:0",
            backend=Backend.TRITON,
            torch_dtype=torch.float16,
        )
        for _, submodule in model_q.named_modules():
            if isinstance(submodule, TritonV2QuantLinear):
                break
        else:
            raise ValueError("Did not find a tritonv2 linear layer")

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        inp = tokenizer(prompt, return_tensors="pt").to("cuda:0")

        # This one uses Autocast.
        res = model_q.generate(
            **inp, num_beams=1, min_new_tokens=new_tokens, max_new_tokens=new_tokens
        )
        predicted_text = tokenizer.decode(res[0])

        self.assertEqual(
            predicted_text[:GENERATE_EVAL_SIZE], reference_output[:GENERATE_EVAL_SIZE]
        )

        # This one does not.
        res = model_q.model.generate(
            **inp, num_beams=1, min_new_tokens=new_tokens, max_new_tokens=new_tokens
        )
        predicted_text = tokenizer.decode(res[0])

        self.assertEqual(
            predicted_text[:GENERATE_EVAL_SIZE], reference_output[:GENERATE_EVAL_SIZE]
        )

    @unittest.skipIf(
        CUDA_AVAILABLE is False,
        "Only runs if there is a cuda device available",
    )
    def test_generation_desc_act_true(self):
        prompt = "I am in Paris and"
        device = torch.device("cuda:0")

        # Reference generated with the cuda-old kernel
        reference_output = "<s> I am in Paris and I am in love with you.\n\nScene 2:\n\nThe stage is now set in a Parisian café. The café is filled with people, including a group of friends, a couple, and a group of tourists. The friends are discussing their plans for the"

        model_id = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"
        revision = "desc_act_true"

        model_q = GPTQModel.from_quantized(
            model_id,
            device="cuda:0",
            backend=Backend.TRITON,
            revision=revision,
        )
        for _, submodule in model_q.named_modules():
            if isinstance(submodule, TritonV2QuantLinear):
                break
        else:
            raise ValueError("Did not find a tritonv2 linear layer")

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        inp = tokenizer(prompt, return_tensors="pt").to(device)

        res = model_q.generate(**inp, num_beams=1, min_new_tokens=60, max_new_tokens=60)

        predicted_text = tokenizer.decode(res[0])

        self.assertEqual(
            predicted_text[:GENERATE_EVAL_SIZE], reference_output[:GENERATE_EVAL_SIZE]
        )
