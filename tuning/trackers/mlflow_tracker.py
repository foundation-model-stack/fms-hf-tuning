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
import logging
import os

# Third Party
from transformers.integrations import MLflowCallback  # pylint: disable=import-error

# Local
from .tracker import Tracker
from tuning.config.tracker_configs import MLflowConfig

MLFLOW_RUN_URI_EXPORT_FILENAME = "mlflow_tracker.json"


class RunURIExporterMlflowCallback(MLflowCallback):
    """
    Custom MlflowCallBack callback is used to export run uri from Mlflow
    as soon as it is created, which is on setup().
    """

    run_uri_export_path: str = None
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
            2. Exports the `Mlflow` run uri:
                - If `MLflowConfig.mlflow_run_uri_export_path` is provided, the uri
                  is exported to `mlflow_run_uri_export_path/mlflow_tracker.json`
                - If `MLflowConfig.mlflow_run_uri_export_path` is not provided but
                    `args.output_dir` is specified, the uri is exported to
                    `args.output_dir/mlflow_tracker.json`
                - If neither path is valid, the uri is not exported.

            The exported uri is formatted as '{"run_uri":"<uri>"}'.

        Args:
            For the arguments see reference to transformers.TrainingCallback
        """
        super().setup(args, state, model)

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

        # Change default uri path to output directory if not specified
        if self.run_uri_export_path is None:
            if args is None or args.output_dir is None:
                self.logger.warning(
                    "To export mlflow uri either output_dir \
                                    or mlflow_run_id_export_path should be set"
                )
                return

            self.run_uri_export_path = args.output_dir

        if not os.path.exists(self.run_uri_export_path):
            os.makedirs(self.run_uri_export_path, exist_ok=True)

        export_path = os.path.join(
            self.run_uri_export_path, MLFLOW_RUN_URI_EXPORT_FILENAME
        )
        with open(export_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"run_name": run_name, "run_uri": run_uri}))
            self.logger.info("Mlflow tracker run uri id dumped to %s", export_path)


class MLflowTracker(Tracker):
    def __init__(self, tracker_config: MLflowConfig):
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
        c = self.config
        exp = c.mlflow_experiment
        uri = c.mlflow_tracking_uri
        run_uri_path = c.mlflow_run_uri_export_path

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

        mlflow_callback = RunURIExporterMlflowCallback()

        if mlflow_callback is not None:
            mlflow_callback.run_uri_export_path = run_uri_path
            mlflow_callback.logger = self.logger

        self.hf_callback = mlflow_callback
        return self.hf_callback

    def track(self, metric, name, stage=None):
        """Track any additional metric with name under mlflow tracker.

        Args:
            metric (int/float): Expected metrics to be tracked by mlflow.
            name (str): Name of the metric being tracked.
            stage (str, optional): Can be used to pass the namespace/metadata to
                associate with metric, e.g. at the stage the metric was generated
                like train, eval. Defaults to None.
                If not None the metric is saved as { state.name : metric }
        Raises:
            ValueError: If the metric or name are passed as None.
        """
        if metric is None or name is None:
            raise ValueError(
                "mlflow track function should not be called with None metric value or name"
            )

        if stage is not None:
            name = f"{stage}.{name}"

        mlflow = self.hf_callback.client
        if mlflow is not None:
            mlflow.log_metric(key=name, value=metric)

    def set_params(self, params, name=None):
        """Attach any extra params with the run information stored in mlflow tracker.

        Args:
            params (dict): A dict of k:v pairs of parameters to be storeed in tracker.
            name (str, optional): represents the namespace under which parameters
                will be associated in mlflow. e.g. {name: params}
                 Defaults to None.

        Raises:
            ValueError: the params passed is None or not of type dict
        """
        if params is None:
            return
        if not isinstance(params, dict):
            raise ValueError(
                "set_params passed to mlflow should be called with a dict of params"
            )
        if name and not isinstance(name, str):
            raise ValueError("name passed to mlflow set_params should be a string")

        if name:
            tolog = {name: params}
        else:
            tolog = params

        mlflow = self.hf_callback.client
        if mlflow is not None:
            mlflow.log_params(tolog)
