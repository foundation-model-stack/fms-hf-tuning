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

"""Helpful saved vison models for unit tests.
"""
# Standard
import os

### Constants used for model path
PREDEFINED_MODEL_PATH = os.path.join(os.path.dirname(__file__))
TINY_LLAMA_VISION_MODEL_NAME = os.path.join(
    PREDEFINED_MODEL_PATH, "tiny_llama_vision_model"
)
TINY_GRANITE_VISION_MODEL_NAME = os.path.join(
    PREDEFINED_MODEL_PATH, "tiny_granite_vision_model"
)
