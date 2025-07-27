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
import logging
import os

# Third Party
from transformers.integrations import MLflowCallback  # pylint: disable=import-error

# Local
from .tracker import Tracker
from tuning.config.tracker_configs import TrackerConfigs


class RunURIExporterMlflowCallback(MLflowCallback):
    """
    Custom MlflowCallBack callback is used to export run uri from Mlflow
    as soon as it is created, which is on setup().
    """

    _use_mlflow = True
    run_uri_export_path: str = None
    tracker: Tracker
    client = None
    logger = None

    # Override ml flow callback setup function
    # Initialise mlflow callback and export Mlflow's run url.
    # Export location is MlflowConfig.mlflow_run_uri_export_path if it is passed
    # or, training_args.output_dir/mlflow_tracker.json if output_dir is present
    # Exported url looks like '{"run_uri":"<uri>"}' in the file.
    # url is not exported if both paths are invalid
    def setup(self, args, state, model):
        """Override the `setup` function in the `MLflowCallback` callback.

            This function performs the following steps:
            1. Calls `MLFlowCallBack.setup` to
                initialize internal `mlflow` structures.
            2. Exports the run uri to `run_uri_export_path` at mlflow_tracker.json
                - If `run_uri_export_path` is not provided but
                    `args.output_dir` is specified, the uri is exported to
                    `args.output_dir/mlflow_tracker.json`
                - If neither path is valid, the uri is not exported.

            The exported uri is formatted as '{"run_uri":"<uri>"}'.

        Args:
            For the arguments see reference to transformers.TrainingCallback
        """
        super().setup(args, state, model)

        if not self._use_mlflow or not state.is_world_process_zero:
            return

        if not self._initialized:
            self.logger.warning(
                "mlflow tracker was requested but did not get initialized;"
                + " Please check the config"
            )
            self._use_mlflow = False
            return

        self.client = self._ml_flow

        active_run = self.client.active_run()
        if not active_run:
            return

        active_run_info = active_run.info
        if active_run_info:
            experiment_id = active_run_info.experiment_id
            experiment_url = f"<host_url>/#/experiments/{experiment_id}"
            run_id = active_run_info.run_id
            run_name = active_run_info.run_name
            run_uri = f"{experiment_url}/runs/{run_id}"

        if run_uri is None:
            return

        if self.tracker:
            run_info = {"run_name": run_name, "run_uri": run_uri}
            self.tracker.export_run_info(args, run_info)

        # Track any additional metadata and metrics requested
        if self.client is not None:
            # Handle metrics
            if self.tracker.additional_metrics is not None:
                for stage, metrics in self.tracker.additional_metrics.items():
                    for name, value in metrics.items():
                        self.client.log_metric(key=f"{stage}.{name}", value=value)

            # Handle metadata
            if self.tracker.additional_metadata is not None:
                self.client.log_params(self.tracker.additional_metadata)


class MLflowTracker(Tracker):
    def __init__(self, tracker_config: TrackerConfigs):
        """Tracker which uses mlflow to collect and store metrics.

        Args:
            tracker_config (MLflowConfig): A valid MLflowConfig which contains
            information on where the mlflow tracking uri is present.
        """
        super().__init__(name="mlflow", tracker_config=tracker_config)
        # Get logger with root log level
        self.logger = logging.getLogger(__name__)

    def get_hf_callback(self):
        """Returns the MLFlowCallBack object associated with this tracker.

        Raises:
            ValueError: If the config passed at initialise does not contain
            the uri where the mlflow tracking server is present

        Returns:
            MLFlowCallBack: The MLFlowCallBack initialsed with the config
            provided at init time.
        """
        exp = self.config.mlflow_experiment
        uri = self.config.mlflow_tracking_uri
        run_uri_path = self.config.run_uri_export_path

        if uri is None:
            self.logger.error(
                "mlflow tracker requested but mlflow_uri is not specified. "
                + "Please specify mlflow uri for using mlflow."
            )
            raise ValueError(
                "mlflow tracker requested but mlflow_uri is not specified."
            )

        # Modify the environment expected by mlflow
        os.environ["MLFLOW_TRACKING_URI"] = uri
        os.environ["MLFLOW_EXPERIMENT_NAME"] = exp

        cb = RunURIExporterMlflowCallback()

        if cb is not None:
            cb.run_uri_export_path = run_uri_path
            cb.logger = self.logger
            cb.tracker = self

        self.hf_callback = cb
        return self.hf_callback
