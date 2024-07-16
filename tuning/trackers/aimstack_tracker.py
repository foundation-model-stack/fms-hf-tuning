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

# Third Party
from aim.hugging_face import AimCallback  # pylint: disable=import-error
from transformers.utils import logging

# Local
from .tracker import Tracker
from tuning.config.tracker_configs import AimConfig


class AimStackTracker(Tracker):
    def __init__(self, tracker_config: AimConfig):
        """Tracker which uses Aimstack to collect and store metrics.

        Args:
            tracker_config (AimConfig): A valid AimConfig which contains either
            information about the repo or the server and port where aim db is present.
        """
        super().__init__(name="aim", tracker_config=tracker_config)
        self.logger = logging.get_logger("aimstack_tracker")

    def get_hf_callback(self):
        """Returns the aim.hugging_face.AimCallback object associated with this tracker.

        Raises:
            ValueError: If the config passed at initialise does not contain one of
                aim_repo or server and port where aim db is present.

        Returns:
            aim.hugging_face.AimCallback: The Aimcallback initialsed with the config
            provided at init time.
        """
        c = self.config
        exp = c.experiment
        url = c.aim_url
        repo = c.aim_repo

        if url is not None:
            aim_callback = AimCallback(repo=url, experiment=exp)
        if repo:
            aim_callback = AimCallback(repo=repo, experiment=exp)
        else:
            self.logger.error(
                "Aim tracker requested but repo or server is not specified. "
                + "Please specify either aim repo or aim server ip and port for using Aim."
            )
            raise ValueError(
                "Aim tracker requested but repo or server is not specified."
            )

        self.hf_callback = aim_callback
        return self.hf_callback

    def track(self, metric, name, stage="additional_metrics"):
        """Track any additional metric with name under Aimstack tracker.

        Args:
            metric (int/float): Expected metrics to be tracked by Aimstack.
            name (str): Name of the metric being tracked.
            stage (str, optional): Can be used to pass the namespace/metadata to
                associate with metric, e.g. at the stage the metric was generated like train, eval.
                Defaults to "additional_metrics".

        Raises:
            ValueError: If the metric or name are passed as None.
        """
        if metric is None or name is None:
            raise ValueError(
                "aimstack track function should not be called with None metric value or name"
            )
        context = {"subset": stage}
        callback = self.hf_callback
        run = callback.experiment
        if run is not None:
            run.track(metric, name=name, context=context)

    def set_params(self, params, name="extra_params"):
        """Attach any extra params with the run information stored in Aimstack tracker.

        Args:
            params (dict): A dict of k:v pairs of parameters to be storeed in tracker.
            name (str, optional): represents the namespace under which parameters
                will be associated in Aim. Defaults to "extra_params".

        Raises:
            ValueError: the params passed is None or not of type dict
        """
        if params is None or (not isinstance(params, dict)):
            raise ValueError(
                "set_params passed to aimstack should be called with a dict of params"
            )
        callback = self.hf_callback
        run = callback.experiment
        if run is not None:
            for key, value in params.items():
                run.set((name, key), value, strict=False)
