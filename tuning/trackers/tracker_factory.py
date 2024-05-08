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
from .filelogging_tracker import FileLoggingTracker
from .tracker import Tracker
from tuning.config.tracker_configs import FileLoggingTrackerConfig, TrackerConfigFactory

logger = logging.get_logger("tracker_factory")

# Information about all registered trackers
AVAILABLE_TRACKERS = {}

AIMSTACK_TRACKER_NAME = "aim"
FILE_LOGGING_TRACKER_NAME = "file_logger"

# One time package check for list of external trackers.
_is_aim_available = _is_package_available("aim")


def _get_tracker_class(T, C):
    return {"tracker": T, "config": C}


def _register_aim_tracker():
    # pylint: disable=import-outside-toplevel
    if _is_aim_available:
        # Local
        from .aimstack_tracker import AimStackTracker
        from tuning.config.tracker_configs import AimConfig

        AimTracker = _get_tracker_class(AimStackTracker, AimConfig)

        AVAILABLE_TRACKERS[AIMSTACK_TRACKER_NAME] = AimTracker
        logger.info("Registered aimstack tracker")
    else:
        logger.info(
            "Not registering Aimstack tracker due to unavailablity of package.\n"
            "Please install aim if you intend to use it.\n"
            "\t pip install aim"
        )


def _register_file_logging_tracker():
    FileTracker = _get_tracker_class(FileLoggingTracker, FileLoggingTrackerConfig)
    AVAILABLE_TRACKERS[FILE_LOGGING_TRACKER_NAME] = FileTracker
    logger.info("Registered file logging tracker")


# List of Available Trackers
# file_logger - Logs loss to a file
# aim - Aimstack Tracker
def _register_trackers():
    logger.info("Registering trackers")
    if AIMSTACK_TRACKER_NAME not in AVAILABLE_TRACKERS:
        _register_aim_tracker()
    if FILE_LOGGING_TRACKER_NAME not in AVAILABLE_TRACKERS:
        _register_file_logging_tracker()


def _get_tracker_config_by_name(name: str, tracker_configs: TrackerConfigFactory):
    if tracker_configs is None:
        return
    c_name = name + "_config"
    d = dataclasses.asdict(tracker_configs)
    if c_name in d:
        return d[c_name]
    return


def get_tracker(name: str, tracker_configs: TrackerConfigFactory):
    """
    Returns an instance of the tracker object based on the requested `name`.
    Expects tracker config to be present as part of the TrackerConfigFactory
    object passed as `tracker_configs` argument.
    If a valid tracker config is not found this function tries tracker with
    default config else returns an empty Tracker()
    """
    if not AVAILABLE_TRACKERS:
        # a one time step.
        _register_trackers()

    if name in AVAILABLE_TRACKERS:
        meta = AVAILABLE_TRACKERS[name]
        C = meta["config"]
        T = meta["tracker"]

        if tracker_configs is not None:
            _conf = _get_tracker_config_by_name(name, tracker_configs)
            if _conf is not None:
                config = C(**_conf)
            else:
                config = C()
        return T(config)

    logger.warning(
        "Requested Tracker %s not found. Please check the argument before proceeding.",
        name,
    )
    return Tracker()
