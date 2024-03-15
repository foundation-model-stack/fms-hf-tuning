from transformers import TrainerControl
from dataclasses import fields
import inspect

class Validator:
    """Implements a validator for control fields"""

    def __init__(self, class_types):
        """Initializes the validator with the control classes and their fields.

        Args:
            control_classes: List of control classes
        """
        self._class_field_map = {}
        for class_key, metadata in class_types.items():
            filter_regex = metadata['filter']
            class_type = metadata['class']
            if metadata['type_of_validation'] == 'methods':
                candidates_list = inspect.getmembers(class_type, predicate=inspect.ismethod)
            elif metadata['type_of_validation'] == 'fields':
                candidates_list = fields(class_type)
            filtered_items = []
            for candidate in candidates_list:
                if re.search(filter_regex, candidate) != None:
                    filtered_items.append(candidate)
                self._class_field_map[class_key] = set(filtered_items)

    def validate(self, class_type, item_to_search) -> bool:
        return item_to_search in self._class_field_map[class_type.__name__]



        
