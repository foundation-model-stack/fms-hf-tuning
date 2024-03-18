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

# Local
from .aimstack_tracker import AimStackTracker
from .tracker import Tracker

REGISTERED_TRACKERS = {"aim": AimStackTracker}


def get_tracker(tracker_name, tracker_config):
    if tracker_name in REGISTERED_TRACKERS:
        T = REGISTERED_TRACKERS[tracker_name]
        return T(tracker_config)
    return Tracker()
