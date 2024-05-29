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

# SPDX-License-Identifier: Apache-2.0
# https://spdx.dev/learn/handling-license-info/

# Third Party
import datasets
import pytest

# First Party
from tests.data import TWITTER_COMPLAINTS_DATA

# Local
from tuning.utils import data_utils


def test_apply_custom_formatting_template():
    json_dataset = datasets.load_dataset("json", data_files=TWITTER_COMPLAINTS_DATA)
    template = "### Input: {{Tweet text}} \n\n ### Response: {{text_label}}"
    # First response from the data file that is read.
    expected_response = (
        "### Input: @HMRCcustomers No this is my first job"
        + " \n\n ### Response: no complaint"
    )
    formatted_dataset, dataset_text_field = data_utils.apply_custom_formatting_template(
        json_dataset, template
    )
    # a new dataset_text_field is created in Dataset
    assert dataset_text_field in formatted_dataset["train"][0]
    assert formatted_dataset["train"][0][dataset_text_field] == expected_response


def test_apply_custom_formatting_template_adds_eos_token():
    json_dataset = datasets.load_dataset("json", data_files=TWITTER_COMPLAINTS_DATA)
    template = "### Input: {{Tweet text}} \n\n ### Response: {{text_label}}"
    # First response from the data file that is read.
    expected_response = (
        "### Input: @HMRCcustomers No this is my first job"
        + " \n\n ### Response: no complaintEOS"
    )
    formatted_dataset, dataset_text_field = data_utils.apply_custom_formatting_template(
        json_dataset, template, "EOS"
    )
    # a new dataset_text_field is created in Dataset
    assert dataset_text_field in formatted_dataset["train"][0]
    assert formatted_dataset["train"][0][dataset_text_field] == expected_response


def test_apply_custom_formatting_template_gives_error_with_wrong_keys():
    """Tests that the formatting function will throw error if wrong keys are passed to template"""
    json_dataset = datasets.load_dataset("json", data_files=TWITTER_COMPLAINTS_DATA)
    template = "### Input: {{not found}} \n\n ### Response: {{text_label}}"
    with pytest.raises(KeyError):
        data_utils.apply_custom_formatting_template(json_dataset, template, "EOS")
