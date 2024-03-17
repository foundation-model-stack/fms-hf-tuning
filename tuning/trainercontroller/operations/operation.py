import abc
import re,inspect

class Operation(metaclass=abc.ABCMeta):
    """Base class for the Operations"""

    def __init__(self):
        self.valid_actions = {}       
        for action_name, action_method in inspect.getmembers(self, predicate=inspect.ismethod):
            if re.search(r'^should_', action_name) != None:
                self.valid_actions[action_name] = action_method

    def validate(self, action: str) -> bool:
        if action not in self.valid_actions:
            return False
    
    def act(self, action: str, **kwargs):
        if self.validate(action):
            raise ValueError(f"Invalid operation {action}")
        self.valid_actions[action](**kwargs)

    def get_actions(self):
        return self.valid_actions.keys()