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


@dataclass
class HFResourceScannerConfig:
    scanner_output_filename: str = "scanner_output.json"


@dataclass
class FileLoggingTrackerConfig:
    training_logs_filename: str = "training_logs.jsonl"


@dataclass
class AimConfig:
    # Name of the experiment
    experiment: str = "fms-hf-tuning"
    # aim_repo can point to a locally accessible directory
    #    or a remote repository hosted on a server.
    # When 'aim_remote_server_ip' or 'aim_remote_server_port' is set,
    #    it designates a remote aim repo.
    # Otherwise, 'repo' specifies the directory, with default of None meaning '.aim'.
    #
    # See https://aimstack.readthedocs.io/en/latest/using/remote_tracking.html
    #     for documentation on Aim remote server tracking.
    aim_repo: str = None
    aim_remote_server_ip: str = None
    aim_remote_server_port: int = None
    aim_url: str = None
    # Location of where aimstack's run hash is to be exported.
    # If aim_run_id_export_path is set the run hash will be output in a json format
    # to the location pointed to by `aim_run_id_export_path/aimstack_tracker.json`
    # If this is not set then the default location where run hash will be exported
    # is training_args.output_dir/aimstack_tracker.json
    # Hash is not exported if aim_run_id_export_path variable is not set
    # and output_dir is not specified.
    aim_run_id_export_path: str = None

    def __post_init__(self):
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


@dataclass
class MLflowConfig:
    # Name of the experiment
    mlflow_experiment: str = "fms-hf-tuning"
    mlflow_tracking_uri: str = None
    # Location of where mlflow's run uri is to be exported.
    # If mlflow_run_uri_export_path is set the run uri will be output in a json format
    # to the location pointed to by `mlflow_run_uri_export_path/mlflow_tracker.json`
    # If this is not set then the default location where run uri will be exported
    # is training_args.output_dir/mlflow_tracker.json
    # Run uri is not exported if mlflow_run_uri_export_path variable is not set
    # and output_dir is not specified.
    mlflow_run_uri_export_path: str = None


@dataclass
class TrackerConfigFactory:
    file_logger_config: FileLoggingTrackerConfig = None
    aim_config: AimConfig = None
    mlflow_config: MLflowConfig = None
    hf_resource_scanner_config: HFResourceScannerConfig = None
