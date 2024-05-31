# Third Party
from transformers import TrainerControl
from transformers.utils import logging
import torch

# Local
from .operation import Operation


class LogControl(Operation):
    """Implements the control actions for the HuggingFace controls in
    transformers.TrainerControl class."""

    def __init__(self, **kwargs):
        """Initializes the HuggingFace controls. In this init, the fields with `should_` of the
        transformers.TrainerControl data class are extracted, and for each of those fields, the
        control_action() method's pointer is set, and injected as a class member function.

        Args:
            kwargs: List of arguments (key, value)-pairs
        """
        self.kwargs = kwargs
        self.logger = logging.get_logger("__main__")
        self.logger.setLevel(level=kwargs["logging_level"])
        super().__init__()

    def should_log(
        self, control: TrainerControl, **kwargs
    ):  # pylint: disable=unused-argument
        """This method peeks into the stack-frame of the caller to get the action the triggered
        a call to it. Using the name of the action, the value of the control is set.

        Args:
            control: TrainerControl. Data class for controls.
            kwargs: List of arguments (key, value)-pairs
        """
        event_name = kwargs.get("event_name")
        metrics = kwargs.get("metrics")
        state = kwargs.get("state")
        train_loss = None
        if state is not None and len(state.log_history) > 0:
            train_loss = state.log_history[-1]["loss"]
        epoch_int = 0 if state is None else int(state.epoch)
        log_format_string = self.kwargs.get("log-format")
        log_level = self.kwargs.get("log-level")
        rank = torch.distributed.get_rank()
        log_msg = log_format_string.format(
            epoch_int=epoch_int,
            event_name=event_name,
            metrics=metrics,
            state=state,
            train_loss=train_loss,
            rank=rank,
        )
        if log_level == "ERROR":
            self.logger.error(log_msg)
        elif log_level == "WARNING":
            self.logger.warning(log_msg)
        elif log_level == "INFO":
            self.logger.info(log_msg)
        elif log_level == "DEBUG":
            self.logger.debug(log_msg)
