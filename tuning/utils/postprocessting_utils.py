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


def get_highest_checkpoint(dir_path):
    """Given full path to directory, returns name of highest checkpoint directory.
    Expects checkpoint directories to be formatted 'checkpoint-<number>'

    Args:
        dir_path: str
    Returns:
        str
    """
    checkpoint_dir = ""
    for curr_dir in os.listdir(dir_path):
        if curr_dir.startswith("checkpoint"):
            if checkpoint_dir:
                curr_dir_num = int(checkpoint_dir.rsplit("-", maxsplit=1)[-1])
                new_dir_num = int(curr_dir.split("-")[-1])
                if new_dir_num > curr_dir_num:
                    checkpoint_dir = curr_dir
            else:
                checkpoint_dir = curr_dir

    return os.path.join(dir_path, checkpoint_dir)
