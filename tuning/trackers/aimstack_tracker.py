# Standard
import os

from .tracker import Tracker
from tuning.config.tracker_configs import AimConfig

# Third Party
from aim.hugging_face import AimCallback

class CustomAimCallback(AimCallback):

    # A path to export run hash generated by Aim
    # This is used to link back to the expriments from outside aimstack
    run_hash_export_path = None

    def on_init_end(self, args, state, control, **kwargs):

        if state and not state.is_world_process_zero:
            return

        self.setup() # initializes the run_hash

        # Store the run hash
        # Change default run hash path to output directory
        if self.run_hash_export_path is None:
            if args and args.output_dir:
                # args.output_dir/.aim_run_hash
                self.run_hash_export_path = os.path.join(
                                                args.output_dir,
                                                '.aim_run_hash'
                                            )

        if self.run_hash_export_path:
            with open(self.run_hash_export_path, 'w') as f:
                f.write('{\"run_hash\":\"'+str(self._run.hash)+'\"}\n')

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # call directly to make sure hyper parameters and model info is recorded.
        self.setup(args=args, state=state, model=model)

    def track_metrics(self, metric, name, context):
        if self._run is not None:
            self._run.track(metric, name=name, context=context)

    def set_params(self, params, name):
        if self._run is not None:
            for key, value in params.items():
                self._run.set((name, key), value, strict=False)

class AimStackTracker(Tracker):

    def __init__(self, tracker_config: AimConfig):
        super().__init__(name='aim', tracker_config=tracker_config)

    def get_hf_callback(self):
        c = self.config
        exp = c.experiment
        ip = c.aim_remote_server_ip
        port = c.aim_remote_server_port
        repo = c.aim_repo
        hash_export_path = c.aim_run_hash_export_path

        if (ip is not None and port is not None):
            aim_callback = CustomAimCallback(
                                repo="aim://" + ip +":"+ port + "/",
                                experiment=exp)
        if repo:
            aim_callback = CustomAimCallback(repo=repo, experiment=exp)
        else:
            aim_callback = CustomAimCallback(experiment=exp)

        aim_callback.run_hash_export_path = hash_export_path
        self.hf_callback = aim_callback
        return self.hf_callback

    def track(self, metric, name, stage='additional_metrics'):
        context={'subset' : stage}
        self.hf_callback.track_metrics(metric, name=name, context=context)

    def set_params(self, params, name='extra_params'):
        try:
            self.hf_callback.set_params(params, name)
        except:
            pass