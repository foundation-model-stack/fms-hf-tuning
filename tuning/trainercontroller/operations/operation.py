# Standard
import abc
import inspect
import re


class Operation(metaclass=abc.ABCMeta):
    """Base class for operations"""

    def __init__(self):
        """Initializes the HuggingFace controls. In this init, we follow the convention that
        every action should preceed with prefix `should_`. If so, it is treated as a valid
        action.
        """
        self.valid_actions = {}
        for action_name, action_method in inspect.getmembers(
            self, predicate=inspect.ismethod
        ):
            if re.search(r"^should_.+", action_name) is not None:
                self.valid_actions[action_name] = action_method

    def validate(self, action: str) -> bool:
        """Validates the action by checking if it valid action or not.

        Args:
            action: str. String depicting the action.

        Returns:
            bool. Indicates True if valid. If not, returns False.
        """
        return action in self.valid_actions

    def act(self, action: str, **kwargs):
        """Validates the action and invokes it.

        Args:
            action: str. String depicting the action.
            kwargs: List of arguments (key, value)-pairs.
        """
        if not self.validate(action):
            raise ValueError(f"Invalid operation {action}")
        self.valid_actions[action](**kwargs)

    def get_actions(self) -> list[str]:
        """Gets the list of all valid actions."""
        return self.valid_actions.keys()
