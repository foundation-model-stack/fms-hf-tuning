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


# Generic Tracker API
class Tracker:
    """
    Generic interface for a Tracker Object.
    """

    def __init__(self, name=None, tracker_config=None) -> None:
        if tracker_config is not None:
            self.config = tracker_config
        if name is None:
            self._name = "None"
        else:
            self._name = name

    # we use args here to denote any argument.
    def get_hf_callback(self):
        return None

    def track(self, metric, name, stage):
        pass

    # Object passed here is supposed to be a KV object
    # for the parameters to be associated with a run
    def set_params(self, params, name):
        pass
