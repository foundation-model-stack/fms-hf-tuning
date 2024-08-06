# Third Party
from transformers import TrainingArguments
from transformers.utils import logging

# Local
from .operation import Operation

logger = logging.get_logger(__name__)
logger.setLevel(level=logging.DEBUG)


class LogControl(Operation):
    """Operation that can be used to log useful information on specific events."""

    def __init__(self, log_format: str, log_level: str, **kwargs):
        """Initializes the HuggingFace controls. In this init, the fields with `should_` of the
        transformers.TrainerControl data class are extracted, and for each of those fields, the
        control_action() method's pointer is set, and injected as a class member function.

        Args:
            kwargs: List of arguments (key, value)-pairs
        """
        log_levels = logging.get_log_levels_dict()
        if log_level not in log_levels:
            raise ValueError(
                "Specified log_level [%s] is invalid for LogControl" % (log_level)
            )
        self.log_level = log_levels[log_level]
        self.log_format = log_format
        super().__init__(**kwargs)

    def should_log(
        self,
        event_name: str = None,
        control_name: str = None,
        args: TrainingArguments = None,
        **kwargs,
    ):
        """This method peeks into the stack-frame of the caller to get the action the triggered
        a call to it. Using the name of the action, the value of the control is set.

        Args:
            control: TrainerControl. Data class for controls.
            kwargs: List of arguments (key, value)-pairs
        """
        log_msg = self.log_format.format(
            event_name=event_name,
            control_name=control_name,
            args=args,
            **kwargs,
        )
        logger.log(
            self.log_level,
            log_msg,
        )
