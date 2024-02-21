# Generic Tracker API

from tuning.tracker.aimstack_tracker import AimStackTracker

class Tracker:
    def __init__(self, tracker_config) -> None:
        self.config = tracker_config

    def get_hf_callback():
        return None

    def track(self, metric, name, stage):
        pass

    # Metadata passed here is supposed to be a KV object
    # Key being the name and value being the metric to track.
    def track_metadata(self, metadata=None):
        if metadata is None or not isinstance(metadata, dict):
            return
        for k, v in metadata.items():
            self.track(name=k, metric=v)

def get_tracker(tracker_name, tracker_config):
    if tracker_name == 'aim':
        if tracker_config is not None:
            tracker = AimStackTracker(tracker_config)
    else:
        tracker = Tracker()
    return tracker