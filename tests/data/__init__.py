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

"""Helpful datasets for configuring individual unit tests.
"""
# Standard
import os

### Constants used for data
DATA_DIR = os.path.join(os.path.dirname(__file__))
TWITTER_COMPLAINTS_DATA = os.path.join(DATA_DIR, "twitter_complaints_small.json")
TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT = os.path.join(
    DATA_DIR, "twitter_complaints_input_output.json"
)
TWITTER_COMPLAINTS_JSON_FORMAT = os.path.join(DATA_DIR, "twitter_complaints_json.json")
EMPTY_DATA = os.path.join(DATA_DIR, "empty_data.json")
MALFORMATTED_DATA = os.path.join(DATA_DIR, "malformatted_data.json")
