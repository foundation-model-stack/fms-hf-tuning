# Standard
import abc
import inspect
import logging
import re

logger = logging.getLogger(__name__)


class Operation(metaclass=abc.ABCMeta):
    """Base class for operations"""

    def __init__(self, name: str, **kwargs):
        """Initializes the HuggingFace controls. In this init, we follow the convention that
        every action should preceed with prefix `should_`. If so, it is treated as a valid
        action.
        """
        self._name = name
        self.kwargs = kwargs
        self.valid_actions = {}
        self.name = name
        self.kwargs = kwargs
        for action_name, action_method in inspect.getmembers(
            self, predicate=inspect.ismethod
        ):
            if re.search(r"^should_.+", action_name) is not None:
                self.valid_actions[action_name] = action_method

    def get_name(self) -> str:
        """Returns the name of the operation.

        Returns:
            str
        """
        return self._name

    def validate(self, action: str) -> bool:
        """Validates the action by checking if it valid action or not.

        Args:
            action: str. String depicting the action.

        Returns:
            bool. Indicates True if valid. If not, returns False.
        """
        return action in self.valid_actions

    def act(
        self,
        action: str,
        log_level: int,
        event_name: str = None,
        control_name: str = None,
        **kwargs,
    ):
        """Validates the action and invokes it.

        Args:
            action: str. String depicting the action.
            event_name: str. Event name triggering the act.
            control_name: str. Name of the controller defining the act.
            log_level: int. Log level for triggering the log.
            kwargs: List of arguments (key, value)-pairs.
        """
        if not self.validate(action):
            raise ValueError(f"Invalid operation {action}")
        logger.log(
            log_level,
            "Taking [%s] action in controller [%s] triggered at event [%s]",
            action,
            control_name,
            event_name,
        )
        kwargs["event_name"] = event_name
        kwargs["control_name"] = control_name
        self.valid_actions[action](**kwargs)

    def get_actions(self) -> list[str]:
        """Gets the list of all valid actions."""
        return self.valid_actions.keys()
