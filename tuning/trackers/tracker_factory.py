# Copyright The FMS HF Tuning Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, aim_reposoftware
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
import logging

# Third Party
from transformers.utils.import_utils import _is_package_available

# Local
from .filelogging_tracker import FileLoggingTracker
from tuning.config.tracker_configs import TrackerConfigs

logger = logging.getLogger(__name__)

# Information about all registered trackers
AIMSTACK_TRACKER = "aim"
FILE_LOGGING_TRACKER = "file_logger"
MLFLOW_TRACKER = "mlflow"
HF_RESOURCE_SCANNER_TRACKER = "hf_resource_scanner"
CLEARML_TRACKER = "clearml"

AVAILABLE_TRACKERS = [
    AIMSTACK_TRACKER,
    FILE_LOGGING_TRACKER,
    HF_RESOURCE_SCANNER_TRACKER,
    MLFLOW_TRACKER,
    CLEARML_TRACKER,
]

# Trackers which can be used
REGISTERED_TRACKERS = {}

# One time package check for list of external trackers.
_is_aim_available = _is_package_available("aim")
_is_mlflow_available = _is_package_available("mlflow")
_is_hf_resource_scanner_available = _is_package_available("HFResourceScanner")
_is_clearml_available = _is_package_available("clearml")


def _is_tracker_installed(name):
    if name == AIMSTACK_TRACKER:
        return _is_aim_available
    if name == HF_RESOURCE_SCANNER_TRACKER:
        return _is_hf_resource_scanner_available
    if name == MLFLOW_TRACKER:
        return _is_mlflow_available
    if name == CLEARML_TRACKER:
        return _is_clearml_available
    return False


def _register_aim_tracker():
    # pylint: disable=import-outside-toplevel
    if _is_aim_available:
        # Local
        from .aimstack_tracker import AimStackTracker

        REGISTERED_TRACKERS[AIMSTACK_TRACKER] = AimStackTracker
        logger.info("Registered aimstack tracker")
    else:
        logger.warning(
            "Not registering Aimstack tracker due to unavailablity of package.\n"
            "Please install aim if you intend to use it.\n"
            "\t pip install aim"
        )


def _register_mlflow_tracker():
    # pylint: disable=import-outside-toplevel
    if _is_mlflow_available:
        # Local
        from .mlflow_tracker import MLflowTracker

        REGISTERED_TRACKERS[MLFLOW_TRACKER] = MLflowTracker
        logger.info("Registered mlflow tracker")
    else:
        logger.warning(
            "Not registering mlflow tracker due to unavailablity of package.\n"
            "Please install mlflow if you intend to use it.\n"
            "\t pip install mlflow"
        )


def _register_hf_resource_scanner_tracker():
    # pylint: disable=import-outside-toplevel
    if _is_hf_resource_scanner_available:
        # Local
        from .hf_resource_scanner_tracker import HFResourceScannerTracker

        REGISTERED_TRACKERS[HF_RESOURCE_SCANNER_TRACKER] = HFResourceScannerTracker
        logger.info("Registered HFResourceScanner tracker")
    else:
        logger.warning(
            "Not registering HFResourceScanner tracker due to unavailablity of package.\n"
            "Please install HFResourceScanner if you intend to use it.\n"
            "\t pip install HFResourceScanner"
        )


def _register_file_logging_tracker():
    REGISTERED_TRACKERS[FILE_LOGGING_TRACKER] = FileLoggingTracker
    logger.info("Registered file logging tracker")


def _register_clearml_tracker():
    # pylint: disable=import-outside-toplevel
    if _is_clearml_available:
        # Local
        from .clearml_tracker import ClearMLTracker

        REGISTERED_TRACKERS[CLEARML_TRACKER] = ClearMLTracker
        logger.info("Registered clearml tracker")
    else:
        logger.warning(
            "Not registering clearml tracker due to unavailablity of package.\n"
            "Please install clearml if you intend to use it.\n"
            "\t pip install clearml"
        )


# List of Available Trackers
# file_logger - Logs loss to a file
# aim - Aimstack Tracker
# mlflow - MLflow Tracking
def _register_trackers():
    logger.info("Registering trackers")
    if AIMSTACK_TRACKER not in REGISTERED_TRACKERS:
        _register_aim_tracker()
    if FILE_LOGGING_TRACKER not in REGISTERED_TRACKERS:
        _register_file_logging_tracker()
    if MLFLOW_TRACKER not in REGISTERED_TRACKERS:
        _register_mlflow_tracker()
    if HF_RESOURCE_SCANNER_TRACKER not in REGISTERED_TRACKERS:
        _register_hf_resource_scanner_tracker()
    if CLEARML_TRACKER not in REGISTERED_TRACKERS:
        _register_clearml_tracker()


def get_tracker(name: str, tracker_configs: TrackerConfigs):
    """Returns an instance of the tracker object based on the requested name.

    Args:
        name (str): name of the tracker requested.
        tracker_configs (tuning.config.tracker_configs.TrackerConfigs):
            An instance of TrackerConfigs passed which contains
            all config for the requested trackers
    Raises:
        ValueError: If a valid tracker config is not found this function raises a ValueError
        ValueError: If a valid tracker is found but its config is not passed the tracker might
            raise a ValueError.

    Returns:
        tuning.trackers.tracker.Tracker: A subclass of tuning.trackers.tracker.Tracker
            Valid classes available are in tuning.trackers.tracker/*_tracker.py

    Examples:
        file_logging_tracker = get_tracker("file_logger", TrackerConfigs(
                                        training_logs_filename=logs_file
                                    )))
    """
    if not REGISTERED_TRACKERS:
        # a one time step.
        _register_trackers()

    if name not in REGISTERED_TRACKERS:
        if name in AVAILABLE_TRACKERS and (not _is_tracker_installed(name)):
            e = f"Requested tracker {name} is not installed.\
                  Please install before proceeding"
        else:
            available = ", ".join(str(t) for t in AVAILABLE_TRACKERS)
            e = f"Requested Tracker {name} not found.\
                  List trackers available for use is - {available} "
        logger.error(e)
        raise ValueError(e)

    T = REGISTERED_TRACKERS[name]
    return T(tracker_configs)
