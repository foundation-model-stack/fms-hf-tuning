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

# Third Party
from HFResourceScanner import Scanner  # pylint: disable=import-error

# Local
from .tracker import Tracker
from tuning.config.tracker_configs import HFResourceScannerConfig


class HFResourceScannerTracker(Tracker):
    def __init__(self, tracker_config: HFResourceScannerConfig):
        """Tracker which encodes callback to scan for resources using HFResourceScanner

        Args:
            tracker_config (HFResourceScannerConfig): An instance of HFResourceScanner
              tracker config which contains the location of output file.
        """
        super().__init__(name="hf_resource_scanner", tracker_config=tracker_config)
        # Get logger with root log level
        self.logger = logging.getLogger()

    def get_hf_callback(self):
        """Returns the HFResourceScanner object associated with this tracker.

        Returns:
            HFResourceScanner: The file logging callback which inherits
                transformers.TrainerCallback and records the metrics to a file.
        """
        output_filename = self.config.scanner_output_filename
        self.hf_callback = Scanner(output_fmt=output_filename)
        return self.hf_callback
