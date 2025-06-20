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
from transformers.integrations import ClearMLCallback  # pylint: disable=import-error

# Local
from .tracker import Tracker
from tuning.config.tracker_configs import ClearMLConfig

CLEARML_RUN_URI_EXPORT_FILENAME = "clearml_tracker.json"


class RunURIExporterClearMLCallback(ClearMLCallback):
    """
    Custom ClearMLCallBack callback is used to export run uri from ClearML
    as soon as it is created, which is on setup().
    """

    run_uri_export_path: str = None
    task = None
    logger = None

    # Override ml flow callback setup function
    # Initialise ClearML callback and export ClearML's run url.
    # Export location is ClearMLConfig.ClearML_run_uri_export_path if it is passed
    # or, training_args.output_dir/ClearML_tracker.json if output_dir is present
    # Exported url looks like '{"task_id":"<task-url>"}' in the file.
    # url is not exported if both paths are invalid
    def setup(self, args, state, model):
        """Override the `setup` function in the `ClearMLCallback` callback.

            This function performs the following steps:
            1. Calls `ClearMLCallBack.setup` to
                initialize internal `ClearML` structures.
            2. Exports the `ClearML` run uri:
                - If `ClearMLConfig.ClearML_run_uri_export_path` is provided, the uri
                  is exported to `ClearML_run_uri_export_path/ClearML_tracker.json`
                - If `ClearMLConfig.ClearML_run_uri_export_path` is not provided but
                    `args.output_dir` is specified, the uri is exported to
                    `args.output_dir/ClearML_tracker.json`
                - If neither path is valid, the uri is not exported.

            The exported uri is formatted as '{"task_id":"<task-url>"}'.

        Args:
            For the arguments see reference to transformers.TrainingCallback
        """
        super().setup(args, state, model)

        if not self._initialized:
            raise RuntimeError("Clearml tracker was requested but did not get initialized;"+
                             " Please check the config")

        self.task = self._clearml.Task.current_task()

        task_id = self.task.id
        task_url = self.task.get_output_log_web_page()
        if not task_url:
            return

        # Change default uri path to output directory if not specified
        if self.run_uri_export_path is None:
            if args is None or args.output_dir is None:
                self.logger.warning(
                    "To export ClearML uri either output_dir \
                                    or ClearML_run_id_export_path should be set"
                )
                return
            self.run_uri_export_path = args.output_dir

        if not os.path.exists(self.run_uri_export_path):
            os.makedirs(self.run_uri_export_path, exist_ok=True)

        export_path = os.path.join(
            self.run_uri_export_path, CLEARML_RUN_URI_EXPORT_FILENAME
        )
        with open(export_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"task_id": task_id, "task_url": task_url}))
            self.logger.info("ClearML tracker run url id dumped to %s", export_path)


class ClearMLTracker(Tracker):
    def __init__(self, tracker_config: ClearMLConfig):
        """Tracker which uses ClearML to collect and store metrics.

        Args:
            tracker_config (ClearMLConfig): A valid ClearMLConfig which contains
            information on where the ClearML tracking uri is present.
        """
        super().__init__(name="ClearML", tracker_config=tracker_config)
        # Get logger with root log level
        self.logger = logging.getLogger(__name__)

    def get_hf_callback(self):
        """Returns the ClearMLCallBack object associated with this tracker.

        Raises:
            ValueError: If the config passed at initialise does not contain
            the uri where the ClearML tracking server is present

        Returns:
            ClearMLCallBack: The ClearMLCallBack initialsed with the config
            provided at init time.
        """
        c = self.config
        project = c.clearml_project
        task = c.clearml_task
        run_uri_path = c.clearml_run_uri_export_path

        # Modify the environment expected by clearml
        os.environ["CLEARML_PROJECT"] = project
        os.environ["CLEARML_TASK"] = task

        clearml_callback = RunURIExporterClearMLCallback()

        if clearml_callback is not None:
            clearml_callback.run_uri_export_path = run_uri_path
            clearml_callback.logger = self.logger

        self.hf_callback = clearml_callback
        return self.hf_callback

    def track(self, metric, name, stage=None):
        """Track any additional metric with name under ClearML tracker.

        Args:
            metric (int/float): Expected metrics to be tracked by ClearML.
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
                "ClearML track function should not be called with None metric value or name"
            )

        if stage is not None:
            name = f"{stage}.{name}"

        clearml_logger = self.hf_callback.task.get_logger()
        if clearml_logger is not None:
            clearml_logger.report_single_value(name=name, value=metric)

    def set_params(self, params, name=None):
        """Attach any extra params with the run information stored in ClearML tracker.

        Args:
            params (dict): A dict of k:v pairs of parameters to be storeed in tracker.
            name (str, optional): represents the namespace under which parameters
                will be associated in ClearML. e.g. {name: params}
                 Defaults to None.

        Raises:
            ValueError: the params passed is None or not of type dict
        """
        if params is None:
            return
        if not isinstance(params, dict):
            raise ValueError(
                "set_params passed to ClearML should be called with a dict of params"
            )
        if name and not isinstance(name, str):
            raise ValueError("name passed to ClearML set_params should be a string")

        task = self.hf_callback.task
        if task is not None:
            for key, value in params.items():
                task.set_parameter(key, value, description=name)