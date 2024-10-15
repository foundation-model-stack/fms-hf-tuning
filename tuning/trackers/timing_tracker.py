# Standard
from datetime import datetime
import json
import logging
import os

# Third Party
from transformers import TrainerCallback

# Local
from .tracker import Tracker
from tuning.config.tracker_configs import TimingTrackerConfig


class TimingCallback(TrainerCallback):
    """Logs the start time, end time, and runtime of the training to a file."""

    def __init__(self, logs_filename="timing_logs.jsonl"):
        self.logs_filename = logs_filename
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        """Record the start time when training begins."""
        self.start_time = datetime.now()

    def on_train_end(self, args, state, control, **kwargs):
        """Record the end time and calculate total training time when training ends."""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()

        log_obj = {
            "name": "training_timing",
            "data": {
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_duration_seconds": total_duration,
                "train_runtime": state.log_history[-1].get("train_runtime", "N/A"),
                "timestamp": datetime.now().isoformat(),
            },
        }

        # Write the log to a JSONL file
        log_file_path = os.path.join(args.output_dir, self.logs_filename)
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"{json.dumps(log_obj, sort_keys=True)}\n")


class TimingTracker(Tracker):
    def __init__(self, tracker_config: TimingTrackerConfig):
        """Tracker to log training start and end times along with train_runtime."""
        super().__init__(name="timing", tracker_config=tracker_config)
        self.logger = logging.getLogger()

    def get_hf_callback(self):
        """Returns the TimingCallback object for this tracker."""
        file = self.config.timing_logs_filename
        self.hf_callback = TimingCallback(logs_filename=file)
        return self.hf_callback
