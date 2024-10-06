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
import json
import os

# Third Party
import wandb
from transformers.integrations import WandbCallback
from transformers.utils import logging

# Local
from .tracker import Tracker
from tuning.config.tracker_configs import WandBConfig

class WandBTracker(Tracker):
    def __init__(self, tracker_config: WandBConfig):
        """Tracker which uses Wandb to collect and store metrics.
        """
        super().__init__(name="aim", tracker_config=tracker_config)
        self.logger = logging.get_logger("wandb_tracker")

    def get_hf_callback(self):
        """Returns the WandBCallback object associated with this tracker.
        """
        c = self.config
        project = c.project
        entity = c.entity

        run = wandb.init(project=project, entity=entity)
        WandbCallback = WandbCallback()

        self.run = run
        self.hf_callback = WandbCallback
        return self.hf_callback

    def _wandb_log(self, data, name):
        self.run.log({name: data})

    def track(self, metric, name, stage):
        """Track any additional metric with name under Aimstack tracker.
        """
        if metric is None or name is None:
            raise ValueError(
                "wandb track function should not be called with None metric value or name"
            )
        self._wandb_log(metric, name)

    def set_params(self, params, name="extra_params"):
        """Attach any extra params with the run information stored in Aimstack tracker.
        """
        self.run.log(params)
