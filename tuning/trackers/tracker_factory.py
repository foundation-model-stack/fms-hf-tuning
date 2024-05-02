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

#General
import dataclasses

# Local
from .tracker import Tracker
from tuning.utils.import_utils import is_package_available

# Third party
from transformers.utils import logging
logger = logging.get_logger("tracker_factory")

REGISTERED_TRACKERS = {}

if is_package_available("aim"):
    from tuning.config.tracker_configs import AimConfig
    from .aimstack_tracker import AimStackTracker

    AimTracker = { "tracker": AimStackTracker,
                   "config": AimConfig}

    REGISTERED_TRACKERS['aim'] = AimTracker

def get_tracker_config(name, super_configs):
    if name in REGISTERED_TRACKERS:
        meta = REGISTERED_TRACKERS[name]
        C = meta['config']
        config = C(**dataclasses.asdict(super_configs))
    else:
        config = None
    return config

def get_tracker(name, tracker_config):
    if name in REGISTERED_TRACKERS:
        meta = REGISTERED_TRACKERS[name]
        T = meta['tracker']
        return T(tracker_config)
    else:
        logger.warn("Tracker "+name+" requested but package "+name+" not found.\n"
                         "Please install tracker package before proceeding.")
    return Tracker()
