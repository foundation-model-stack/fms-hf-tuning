# Copyright The IBM Tuning Team
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
from transformers import TrainerControl

# Local
from tuning.trainercontroller.operations import Operation


class CustomOperation(Operation):
    """Implements a custom operation for testing"""

    def __init__(self, **_):
        """Initializes the custom operation class.
        Args:
            kwargs: List of arguments (key, value)-pairs
        """
        super().__init__()

    def should_perform_action_xyz(self, control: TrainerControl, **_):
        """This method performs a set training stop flag action.

        Args:
            control: TrainerControl. Data class for controls.
            kwargs: List of arguments (key, value)-pairs
        """
        control.should_training_stop = True
