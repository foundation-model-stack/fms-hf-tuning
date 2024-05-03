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

# Standard
import dataclasses

# Third Party
from transformers.utils import logging

# Local
from .tracker import Tracker
from tuning.utils.import_utils import is_package_available

logger = logging.get_logger("tracker_factory")

REGISTERED_TRACKERS = {}

# pylint: disable=import-outside-toplevel
def register_trackers():
    if is_package_available("aim"):
        # Local
        from .aimstack_tracker import AimStackTracker
        from tuning.config.tracker_configs import AimConfig

        AimTracker = {"tracker": AimStackTracker, "config": AimConfig}

        REGISTERED_TRACKERS["aim"] = AimTracker
    else:
        logger.info(
            "Not registering Aimstack tracker due to unavailablity of package.\n"
            "Please install aim if you intend to use it.\n"
            "\t pip install aim"
        )


def get_tracker(name, super_configs):
    # a one time step.
    if not REGISTERED_TRACKERS:
        register_trackers()

    if name in REGISTERED_TRACKERS:
        meta = REGISTERED_TRACKERS[name]
        C = meta["config"]
        T = meta["tracker"]
        tracker_config = C(**dataclasses.asdict(super_configs))
        if tracker_config is not None:
            return T(tracker_config)
    else:
        logger.warning(
            "Requested Tracker %s not found.\n"
            "Please check the argument before proceeding."
        )
    return Tracker()
