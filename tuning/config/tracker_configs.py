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
from dataclasses import dataclass

DEFAULT_EXP_NAME = "fms-hf-tuning"


@dataclass
class TrackerConfigs:
    # Name of the experiment
    experiment: str = DEFAULT_EXP_NAME
    # Location of where trackers run uri is to be exported.
    # If run_uri_export_path is set the run uri will be output in a json format
    # to the location pointed to by `run_uri_export_path/{name}_tracker.json`
    # for all the trackers requested by --tracker argument.
    # If this is not set then the default location where run uri will be exported
    # is training_args.output_dir/{name}_tracker.json
    run_uri_export_path: str = None

    ################## AimStack Related ######################
    # aim_repo can point to a locally accessible directory
    #    or a remote repository hosted on a server.
    # When 'aim_remote_server_ip' or 'aim_remote_server_port' is set,
    #    it designates a remote aim repo.
    # Otherwise, 'repo' specifies the directory, with default of None meaning '.aim'.
    #
    # See https://aimstack.readthedocs.io/en/latest/using/remote_tracking.html
    #     for documentation on Aim remote server tracking.
    aim_experiment: str = None
    aim_repo: str = None
    aim_remote_server_ip: str = None
    aim_remote_server_port: int = None
    aim_url: str = None

    ################## MLFlow Related ######################
    mlflow_experiment: str = None
    mlflow_tracking_uri: str = None

    ################## Clearml Related ######################
    clearml_project: str = DEFAULT_EXP_NAME
    clearml_task: str = "SFTTrainer"

    ################## Scanner Related ######################
    scanner_output_filename: str = "scanner_output.json"

    ############ FileLoggingTracker Related #################
    training_logs_filename: str = "training_logs.jsonl"

    def __post_init__(self):
        if self.experiment is not None:
            if not self.aim_experiment:
                self.aim_experiment = self.experiment
            if not self.mlflow_experiment:
                self.mlflow_experiment = self.experiment
            if not self.clearml_task:
                self.clearml_task = self.experiment

        if (
            self.aim_remote_server_ip is not None
            and self.aim_remote_server_port is not None
        ):
            self.aim_url = (
                "aim://"
                + self.aim_remote_server_ip
                + ":"
                + self.aim_remote_server_port
                + "/"
            )
