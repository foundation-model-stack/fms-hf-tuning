# Copyright 2025 HuggingFace Inc.
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

import json
import shutil
import tempfile
import unittest

from transformers import AutoProcessor, LlamaTokenizerFast, LlavaNextVideoProcessor
from transformers.testing_utils import require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import LlavaNextImageProcessor, LlavaNextVideoImageProcessor

if is_torch_available:
    pass


@require_vision
class LlavaNextVideoProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = LlavaNextVideoProcessor

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()
        image_processor = LlavaNextImageProcessor()
        video_processor = LlavaNextVideoImageProcessor()
        tokenizer = LlamaTokenizerFast.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")
        processor_kwargs = cls.prepare_processor_dict()

        processor = LlavaNextVideoProcessor(
            video_processor=video_processor, image_processor=image_processor, tokenizer=tokenizer, **processor_kwargs
        )
        processor.save_pretrained(cls.tmpdirname)
        cls.image_token = processor.image_token
        cls.video_token = processor.video_token

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def get_video_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).video_processor

    @classmethod
    def prepare_processor_dict(cls):
        return {
            "chat_template": "{% for message in messages %}{{'<|im_start|>' + message['role'] + ' '}}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>' }}{% endfor %}{# Render all video then #}{% for content in message['content'] | selectattr('type', 'equalto', 'video') %}{{ '<video>' }}{% endfor %}{# Render all text next #}{% if message['role'] != 'assistant' %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ '\n' + content['text'] }}{% endfor %}{% else %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{% generation %}{{ '\n' + content['text'] }}{% endgeneration %}{% endfor %}{% endif %}{{'<|im_end|>'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
            "num_additional_image_tokens": 0,
            "patch_size": 128,
            "vision_feature_select_strategy": "default",
        }

    # Copied from tests.models.llava.test_processor_llava.LlavaProcessorTest.test_chat_template_is_saved
    def test_chat_template_is_saved(self):
        processor_loaded = self.processor_class.from_pretrained(self.tmpdirname)
        processor_dict_loaded = json.loads(processor_loaded.to_json_string())
        # chat templates aren't serialized to json in processors
        self.assertFalse("chat_template" in processor_dict_loaded.keys())

        # they have to be saved as separate file and loaded back from that file
        # so we check if the same template is loaded
        processor_dict = self.prepare_processor_dict()
        self.assertTrue(processor_loaded.chat_template == processor_dict.get("chat_template", None))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)
