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
import os

# Local
from tuning.utils.import_utils import is_aim_available

if is_aim_available():
    # Third Party
    from aim.hugging_face import AimCallback  # pylint: disable=import-error


def get_aimstack_callback():
    # Initialize a new run
    aim_server = os.environ.get("AIMSTACK_SERVER")
    aim_db = os.environ.get("AIMSTACK_DB")
    aim_experiment = os.environ.get("AIMSTACK_EXPERIMENT")
    if aim_experiment is None:
        aim_experiment = ""

    if aim_server:
        aim_callback = AimCallback(
            repo="aim://" + aim_server + "/", experiment=aim_experiment
        )
    if aim_db:
        aim_callback = AimCallback(repo=aim_db, experiment=aim_experiment)
    else:
        aim_callback = AimCallback(experiment=aim_experiment)

    return aim_callback
