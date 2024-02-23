from .tracker import Tracker
from .aimstack_tracker import AimStackTracker

REGISTERED_TRACKERS = {
    "aim" : AimStackTracker
}

def get_tracker(tracker_name, tracker_config):
    if tracker_name in REGISTERED_TRACKERS:
        T = REGISTERED_TRACKERS[tracker_name]
        return T(tracker_config)
    else:
        return Tracker()