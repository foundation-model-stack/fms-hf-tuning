# Standard
import os

from tuning.tracker.tracker import Tracker

# Third Party
from aim.hugging_face import AimCallback

class AimStackTracker(Tracker):

    def __init__(self, tracker_config):
        super().__init__(tracker_config)
        c = self.config
        exp = c.experiment
        ip = c.aim_remote_server_ip
        port = c.aim_remote_server_port
        repo = c.aim_repo
        hash_export_path = c.aim_run_hash_export_path

        if (ip is not None and port is not None):
            aim_callback = AimCallback(
                                repo="aim://" + ip +":"+ port + "/",
                                experiment=exp
                            )
        if repo:
            aim_callback = AimCallback(repo=repo, experiment=exp)
        else:
            aim_callback = AimCallback(experiment=exp)

        run = aim_callback.experiment # Initialize Aim run
        run_hash = run.hash # Extract the hash

        # store the run hash
        if hash_export_path:
            with open(hash_export_path, 'w') as f:
                f.write(str(run_hash)+'\n')

        # Save Internal State
        self.hf_callback = aim_callback
        self.run = run

    def get_hf_callback(self):
        return self.hf_callback

    def track(self, metric, name, stage='additional_metrics'):
        context={'subset' : stage}
        self.run.track(metric, name=name, context=context)