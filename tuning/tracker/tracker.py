# Generic Tracker API

class Tracker:
    def __init__(self, name=None, tracker_config=None) -> None:
        if tracker_config is not None:
            self.config = tracker_config
        if name is None:
            self._name = "None"
        else:
            self._name = name

    def get_hf_callback():
        return None

    def track(self, metric, name, stage):
        pass

    # Object passed here is supposed to be a KV object
    # for the parameters to be associated with a run
    def set_params(self, params, name):
        pass

class TrackerFactory:
    def get_tracker(tracker_name, tracker_config):
        for T in Tracker.__subclasses__():
            if T._name == tracker_name:
                return T(tracker_config)
        else:
            return Tracker()