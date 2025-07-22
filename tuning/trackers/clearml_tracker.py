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
from clearml.backend_api.session.defs import (  # pylint: disable=import-error
    MissingConfigError,
)
from transformers.integrations import ClearMLCallback  # pylint: disable=import-error

# Local
from .tracker import Tracker
from tuning.config.tracker_configs import TrackerConfigs


class RunURIExporterClearMLCallback(ClearMLCallback):
    """
    Custom ClearMLCallBack callback is used to export run uri from ClearML
    as soon as it is created, which is on setup().
    """

    _use_clearml = True
    run_uri_export_path: str = None
    logger = None
    tracker: Tracker = None

    def __init__(self):
        super().__init__()
        self._log_model = False

    # Override ClearML callback setup function
    # Initialise ClearML callback and export ClearML's run uri.
    # Export location is ClearMLConfig.ClearML_run_uri_export_path if it is passed
    # or, training_args.output_dir/ClearML_tracker.json if output_dir is present
    # Exported uri looks like '{"task_id":"<task-uri>"}' in the file.
    # uri is not exported if both paths are invalid
    def setup(self, args, state, model, processing_class, **kwargs):
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

            The exported uri is formatted as '{"task_id":"<task-uri>"}'.

        Args:
            For the arguments see reference to transformers.TrainingCallback
        """
        try:
            super().setup(args, state, model, processing_class, **kwargs)
        except MissingConfigError:
            self.logger.warning(
                "ClearML Setup FAILED!! No configuraion found. "
                + "Before using clearml please perform `clearml-init`, "
                + "See clearml docs at - "
                + "https://clear.ml/docs/latest/docs/clearml_sdk/clearml_sdk_setup"
            )
            # Disable clearml so it doesn't fail
            ClearMLCallback._should_close_on_train_end = False
            self._clearml = None
            return

        if not self._use_clearml or not state.is_world_process_zero:
            return

        if not self._initialized or self._clearml_task is None:
            self.logger.warning(
                "ClearMLtracker was requested but did not get initialized;"
                + " Please check the config"
            )
            self._use_clearml = False
            return

        task = self._clearml.Task.current_task()

        task_id = task.id
        task_uri = task.get_output_log_web_page()
        if not task_uri:
            return

        if self.tracker:
            run_info = {"task_id": task_id, "task_uri": task_uri}
            self.tracker.export_run_info(args, run_info)

        # Track any additional metadata and metrics requested
        clearml_logger = task.get_logger()
        if self.tracker.additional_metrics is not None:
            for stage, metrics in self.tracker.additional_metrics.items():
                for name, value in metrics.items():
                    clearml_logger.report_single_value(
                        name=f"{stage}.{name}", value=value
                    )

        if self.tracker.additional_metadata is not None:
            for name, params in self.tracker.additional_metadata.items():
                if isinstance(params, dict):
                    for key, value in params.items():
                        task.set_parameter(key, value, description=name)
                else:
                    task.set_parameter(name, params, description="additional_metadata")

    def on_save(self, args, state, control, **kwargs):
        pass


class ClearMLTracker(Tracker):
    def __init__(self, tracker_config: TrackerConfigs):
        """Tracker which uses ClearML to collect and store metrics.

        Args:
            tracker_config: A valid TrackerConfigs which contains
            information on where the ClearML tracking uri is present.
        """
        super().__init__(name="clearml", tracker_config=tracker_config)
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
        project = self.config.clearml_project
        task = self.config.clearml_task

        # Modify the environment expected by clearml
        os.environ["CLEARML_PROJECT"] = project
        os.environ["CLEARML_TASK"] = task

        cb = RunURIExporterClearMLCallback()

        if cb is not None:
            cb.run_uri_export_path = self.config.run_uri_export_path
            cb.logger = self.logger
            cb.tracker = self

        self.hf_callback = cb
        return self.hf_callback
