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
class AimConfig:
    # Name of the experiment
    experiment: str = None
    # aim_repo can point to a locally accessible directory (e.g., '~/.aim')
    #    or a remote repository hosted on a server.
    # When 'aim_remote_server_ip' or 'aim_remote_server_port' is set,
    #    it designates a remote aim repo.
    # Otherwise, 'repo' specifies the directory, with default of None meaning '.aim'.
    #
    # See https://aimstack.readthedocs.io/en/latest/using/remote_tracking.html
    #     for documentation on Aim remote server tracking.
    aim_repo: str = ".aim"
    aim_remote_server_ip: str = None
    aim_remote_server_port: int = None
