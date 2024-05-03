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

# Standard
import dataclasses

# Third Party
from transformers.utils import logging
from transformers.utils.import_utils import _is_package_available

# Local
from .tracker import Tracker

# Information about all registered trackers
REGISTERED_TRACKERS = {}

# One time package check for list of trackers.
_is_aim_available = _is_package_available("aim")

logger = logging.get_logger("tracker_factory")


def _register_aim_tracker():
    # pylint: disable=import-outside-toplevel
    if _is_aim_available:
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


def register_trackers():
    if "aim" not in REGISTERED_TRACKERS:
        _register_aim_tracker()


def get_tracker(name, super_configs):
    # a one time step.
    if not REGISTERED_TRACKERS:
        register_trackers()

    if name in REGISTERED_TRACKERS:
        meta = REGISTERED_TRACKERS[name]
        C = meta["config"]
        T = meta["tracker"]
        tracker_config = C(**dataclasses.asdict(super_configs))
        return T(tracker_config)

    logger.warning(
        "Requested Tracker %s not found.\n"
        "Please check the argument before proceeding."
    )
    return Tracker()
