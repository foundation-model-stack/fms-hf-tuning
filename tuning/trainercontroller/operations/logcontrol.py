# Standard
from dataclasses import fields
import inspect
import re

# Third Party
from transformers import TrainerControl
from transformers.utils import logging

# Local
from .operation import Operation

logger = logging.get_logger(__name__)


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
        super().__init__()

    def should_log(self, control: TrainerControl, **kwargs):
        """This method peeks into the stack-frame of the caller to get the action the triggered
        a call to it. Using the name of the action, the value of the control is set.

        Args:
            control: TrainerControl. Data class for controls.
            kwargs: List of arguments (key, value)-pairs
        """
        event_name = kwargs.get('event_name')
        metrics = kwargs.get('metrics')
        state = kwargs.get('state')
        log_format_string = self.kwargs.get('log-format')
        log_level = self.kwargs.get('log-level')
        log_msg = log_format_string.format(event_name=event_name, metrics=metrics, state=state)
        if log_level == 'ERROR':
            logger.error(log_msg)
        elif log_level == 'WARNING':
            logger.warning(log_msg)
        elif log_level == 'INFO':
            logger.info(log_msg)
        elif log_level == 'DEBUG':
            logger.debug(log_msg)

