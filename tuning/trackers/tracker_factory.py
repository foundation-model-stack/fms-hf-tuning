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
from tuning.config.tracker_configs import FileLoggingTrackerConfig, TrackerConfigFactory

logger = logging.get_logger("tracker_factory")


# Information about all registered trackers
AIMSTACK_TRACKER = "aim"
FILE_LOGGING_TRACKER = "file_logger"

AVAILABLE_TRACKERS = [AIMSTACK_TRACKER, FILE_LOGGING_TRACKER]


# Trackers which can be used
REGISTERED_TRACKERS = {}

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

        REGISTERED_TRACKERS[AIMSTACK_TRACKER] = AimTracker
        logger.info("Registered aimstack tracker")
    else:
        logger.info(
            "Not registering Aimstack tracker due to unavailablity of package.\n"
            "Please install aim if you intend to use it.\n"
            "\t pip install aim"
        )


def _is_tracker_installed(name):
    if name == "aim":
        return _is_aim_available
    return False


def _register_file_logging_tracker():
    FileTracker = _get_tracker_class(FileLoggingTracker, FileLoggingTrackerConfig)
    REGISTERED_TRACKERS[FILE_LOGGING_TRACKER] = FileTracker
    logger.info("Registered file logging tracker")


# List of Available Trackers
# file_logger - Logs loss to a file
# aim - Aimstack Tracker
def _register_trackers():
    logger.info("Registering trackers")
    if AIMSTACK_TRACKER not in REGISTERED_TRACKERS:
        _register_aim_tracker()
    if FILE_LOGGING_TRACKER not in REGISTERED_TRACKERS:
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
    """Returns an instance of the tracker object based on the requested name.

    Args:
        name (str): name of the tracker requested.
        tracker_configs (tuning.config.tracker_configs.TrackerConfigFactory):
            An instance of TrackerConfigFactory passed which contains a
            non None instance of config for the requested tracker
    Raises:
        ValueError: If a valid tracker config is not found this function raises a ValueError
        ValueError: If a valid tracker is found but its config is not passed the tracker might
            raise a ValueError. See tuning.trackers.tracker.aimstack_tracker.AimStackTracker

    Returns:
        tuning.trackers.tracker.Tracker: A subclass of tuning.trackers.tracker.Tracker
            Valid classes available are,
            tuning.trackers.tracker.aimstack_tracker.AimStackTracker,
            tuning.trackers.tracker.filelogging_tracker.FileLoggingTracker

    Examples:
        file_logging_tracker = get_tracker("file_logger", TrackerConfigFactory(
                                    file_logger_config=FileLoggingTrackerConfig(
                                        training_logs_filename=logs_file
                                    )
                                ))
        aim_tracker = get_tracker("aim", TrackerConfigFactory(
                            aim_config=AimConfig(
                                experiment="unit_test",
                                aim_repo=tempdir + "/"
                            )
                    ))
    """
    if not REGISTERED_TRACKERS:
        # a one time step.
        _register_trackers()

    if name not in REGISTERED_TRACKERS:
        if name in AVAILABLE_TRACKERS and (not _is_tracker_installed(name)):
            e = "Requested tracker {} is not installed. Please install before proceeding".format(
                name
            )
        else:
            available = ", ".join(str(t) for t in AVAILABLE_TRACKERS)
            e = "Requested Tracker {} not found. List trackers available for use is - {} ".format(
                name, available
            )
        logger.error(e)
        raise ValueError(e)

    meta = REGISTERED_TRACKERS[name]
    C = meta["config"]
    T = meta["tracker"]

    _conf = _get_tracker_config_by_name(name, tracker_configs)
    if _conf is not None:
        config = C(**_conf)
    else:
        config = C()
    return T(config)
