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

# Local
from tuning.config.tracker_configs import TrackerConfigs


# Generic Tracker API
class Tracker:
    """
    Generic interface for a Tracker Object.
    """

    additional_metrics: dict = {}
    additional_metadata: dict = {}

    def __init__(self, name: str, tracker_config: TrackerConfigs) -> None:
        self.config = tracker_config
        self.name = name
        self.run_uri_export_path = self.config.run_uri_export_path

    def export_run_info(self, train_args, run_info: dict, filename: str = None):
        # Change default uri path to output directory if not specified
        if self.run_uri_export_path is None:
            if train_args is None or train_args.output_dir is None:
                self.logger.warning(
                    "To export run uri either output_dir \
                                    or run_uri_export_path should be set"
                )
                return
            self.run_uri_export_path = train_args.output_dir

        if not os.path.exists(self.run_uri_export_path):
            os.makedirs(self.run_uri_export_path, exist_ok=True)

        if not filename:
            filename = "{}_tracker.json".format(self.name)

        export_path = os.path.join(self.run_uri_export_path, filename)
        with open(export_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(run_info))
            self.logger.info("%s tracker run info dumped to %s", self.name, export_path)

    # we use args here to denote any argument.
    def get_hf_callback(self):
        return None

    def track(self, metrics, stage="additional_metrics"):
        """Track any additional metric with name under any tracker.

        Args:
            Dict of metrics to be tracked containing
                metrics(int/float): the metric
                name (str): Name of the metric being tracked.
            stage (str, optional): Can be used to pass the namespace/metadata to
                associate with metric, e.g. at the stage the metric was generated
                like train, eval. Defaults to None.
                If not None the metric is saved as { state.name : metric }
        Raises:
            ValueError: If the metric or name are passed as None.
        """
        if metrics is None or not isinstance(metrics, dict):
            raise ValueError(
                "tracker.track function should not be called with None metrics value or stage"
            )

        # Just save it internally for the callback to pick and track
        self.additional_metrics[stage] = metrics

    # Object passed here is supposed to be a KV object
    # for the parameters to be associated with a run
    def set_params(self, params, name="extra_params"):
        """Attach any extra params with the run information stored in tracker.

        Args:
            params (dict): A dict of k:v pairs of parameters to be storeed in tracker.
            name (str, optional): represents the namespace under which parameters
                will be associated in the tracker. e.g. {name: params}
                 Defaults to None.

        Raises:
            ValueError: the params passed is None or not of type dict
        """
        if params is None:
            return
        if not isinstance(params, dict):
            raise ValueError(
                "set_params passed to trackers should be called with a dict of params"
            )
        if name and not isinstance(name, str):
            raise ValueError("name passed to tracker.set_params should be a string")

        # Just save it internally for the callback to pick and track
        self.additional_metadata[name] = params
