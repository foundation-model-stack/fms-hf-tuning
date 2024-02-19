# Standard
import os

from tuning.tracker.tracker import Tracker

# Third Party
from aim.hugging_face import AimCallback

class AimStackTracker(Tracker):

    def __init__(self, tracker_config):
        super().__init__(tracker_config)
        c = self.config
        if (c.remote_server_ip is not None and
            c.remote_server_port is not None):
            aim_callback = AimCallback(repo="aim://" + c.remote_server_ip+":"+ c.remote_server_port+ "/",
                                       experiment=c.experiment)
        if c.repo:
            aim_callback = AimCallback(repo=c.repo, experiment=c.experiment)
        else:
            aim_callback = AimCallback(experiment=c.experiment)

        run = aim_callback.experiment # Initialize Aim run
        run_hash = run.hash # Extract the hash

        # store the run hash
        if c.run_hash_export_location:
            with open(c.run_hash_export_location, 'w') as f:
                f.write(str(run_hash)+'\n')

        # Save Internal State
        self.hf_callback = aim_callback
        self.run = run

    def get_hf_callback(self):
        return self.hf_callback

    def track(self, metric, name, stage='additional_metrics'):
        context={'subset' : stage}
        self.run.track(metric, name=name, context=context)