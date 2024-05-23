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
NoPatience = -1

class HFControls(Operation):
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
        self._patience_counter = 0
        self._patience_threshold = kwargs.get("patience-threshold", NoPatience)
        self.kwargs = kwargs
        for control_field in fields(TrainerControl):
            if re.search(r"^should_.+", control_field.name) is not None:
                setattr(self, control_field.name, self.control_action)
        super().__init__()

    def control_action(self, control: TrainerControl, rule_outcome: bool, **kwargs):
        """This method peeks into the stack-frame of the caller to get the action the triggered
        a call to it. Using the name of the action, the value of the control is set.

        Args:
            control: TrainerControl. Data class for controls.
            kwargs: List of arguments (key, value)-pairs
        """
        if self._patience_threshold != NoPatience:
            if not rule_outcome:
                self._patience_counter = 0
            else:
                self._patience_counter += 1
                if self._patience_counter <= self._patience_threshold:
                    logger.info(
                        "Patience counter %d is within patience threshold of %d",
                        self._patience_counter,
                        self._patience_threshold,
                    )
                    return
                logger.info("Patience counter %d exceeded threshold %d", 
                            self._patience_counter,
                            self._patience_threshold)
        elif rule_outcome:
            logger.debug("Arguments passed to control_action: %s", repr(kwargs))
            frame_info = inspect.currentframe().f_back
            arg_values = inspect.getargvalues(frame_info)
            setattr(control, arg_values.locals["action"], True)
