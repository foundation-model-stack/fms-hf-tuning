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
JSON_DATA_DIR = os.path.join(os.path.dirname(__file__), "json")
JSONL_DATA_DIR = os.path.join(os.path.dirname(__file__), "jsonl")
ARROW_DATA_DIR = os.path.join(os.path.dirname(__file__), "arrow")
PARQUET_DATA_DIR = os.path.join(os.path.dirname(__file__), "parquet")

TWITTER_COMPLAINTS_DATA_DIR_JSON = os.path.join(DATA_DIR, "datafolder")

TWITTER_COMPLAINTS_DATA_JSON = os.path.join(
    JSON_DATA_DIR, "twitter_complaints_small.json"
)
TWITTER_COMPLAINTS_DATA_JSONL = os.path.join(
    JSONL_DATA_DIR, "twitter_complaints_small.jsonl"
)
TWITTER_COMPLAINTS_DATA_ARROW = os.path.join(
    ARROW_DATA_DIR, "twitter_complaints_small.arrow"
)
TWITTER_COMPLAINTS_DATA_PARQUET = os.path.join(
    PARQUET_DATA_DIR, "twitter_complaints_small.parquet"
)
TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSON = os.path.join(
    JSON_DATA_DIR, "twitter_complaints_input_output.json"
)
TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_JSONL = os.path.join(
    JSONL_DATA_DIR, "twitter_complaints_input_output.jsonl"
)
TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_ARROW = os.path.join(
    ARROW_DATA_DIR, "twitter_complaints_input_output.arrow"
)
TWITTER_COMPLAINTS_DATA_INPUT_OUTPUT_PARQUET = os.path.join(
    PARQUET_DATA_DIR, "twitter_complaints_input_output.parquet"
)
TWITTER_COMPLAINTS_TOKENIZED_JSON = os.path.join(
    JSON_DATA_DIR, "twitter_complaints_tokenized_with_maykeye_tinyllama_v0.json"
)
TWITTER_COMPLAINTS_TOKENIZED_ONLY_INPUT_IDS_JSON = os.path.join(
    JSON_DATA_DIR,
    "twitter_complaints_tokenized_with_maykeye_tinyllama_v0_only_input_ids.json",
)
TWITTER_COMPLAINTS_TOKENIZED_JSONL = os.path.join(
    JSONL_DATA_DIR, "twitter_complaints_tokenized_with_maykeye_tinyllama_v0.jsonl"
)
TWITTER_COMPLAINTS_TOKENIZED_ARROW = os.path.join(
    ARROW_DATA_DIR, "twitter_complaints_tokenized_with_maykeye_tinyllama_v0.arrow"
)
TWITTER_COMPLAINTS_TOKENIZED_PARQUET = os.path.join(
    PARQUET_DATA_DIR, "twitter_complaints_tokenized_with_maykeye_tinyllama_v0.parquet"
)
CHAT_DATA_SINGLE_TURN = os.path.join(JSONL_DATA_DIR, "single_turn_chat.jsonl")
CHAT_DATA_MULTI_TURN = os.path.join(JSONL_DATA_DIR, "multi_turn_chat.jsonl")
CHAT_DATA_MULTI_TURN_GRANITE_3_1B = os.path.join(
    JSONL_DATA_DIR, "multi_turn_chat_granite_instruct.jsonl"
)
EMPTY_DATA = os.path.join(JSON_DATA_DIR, "empty_data.json")
MALFORMATTED_DATA = os.path.join(JSON_DATA_DIR, "malformatted_data.json")

# Other constants
CUSTOM_TOKENIZER_TINYLLAMA = os.path.join(
    DATA_DIR, "tinyllama_tokenizer_special_tokens"
)
MODEL_NAME = "Maykeye/TinyLLama-v0"
