# Copyright The IBM Tuning Team
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

# Third Party
from aim.hugging_face import AimCallback

# Local
from .tracker import Tracker
from tuning.config.tracker_configs import AimConfig

# Aimstack does not initialise the callback at init hence we cannot track
# additional metrics before the training ends, hence using custom callback
# We have an open PR in Aim - https://github.com/aimhubio/aim/pull/3116
# Which fixes this issue.
class CustomAimCallback(AimCallback):
    def on_init_end(self, args, state, control, **kwargs):
        self.setup()  # initializes the run_hash

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # call directly to make sure hyper parameters and model info is recorded.
        self.setup(args=args, state=state, model=model)


class AimStackTracker(Tracker):
    def __init__(self, tracker_config: AimConfig):
        super().__init__(name="aim", tracker_config=tracker_config)

    def get_hf_callback(self):
        c = self.config
        exp = c.experiment
        ip = c.aim_remote_server_ip
        port = c.aim_remote_server_port
        repo = c.aim_repo

        if ip is not None and port is not None:
            aim_callback = CustomAimCallback(
                repo="aim://" + ip + ":" + port + "/", experiment=exp
            )
        if repo:
            aim_callback = CustomAimCallback(repo=repo, experiment=exp)
        else:
            self.logger.warn("Aim tracker requested but repo or server is not specified\n"
                             "Please specify either aim repo or aim server ip and port for using aim")
            aim_callback = None

        self.hf_callback = aim_callback
        return self.hf_callback

    def track(self, metric, name, stage="additional_metrics"):
        context = {"subset": stage}
        callback = self.hf_callback
        run = callback.experiment
        if run is not None:
            run.track(metric, name=name, context=context)

    def set_params(self, params, name="extra_params"):
        callback = self.hf_callback
        run = callback.experiment
        if run is not None:
            for key, value in params.items():
                run.set((name, key), value, strict=False)
