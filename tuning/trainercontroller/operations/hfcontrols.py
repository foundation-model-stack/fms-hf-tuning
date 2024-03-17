
import re, inspect
from dataclasses import fields

from transformers import TrainerControl

from tuning.trainercontroller.operations import Operation

class HFControls(Operation):

    def __init__(self, **kwargs):
        for control_field in fields(TrainerControl):
            if re.search(r'^should_', control_field.name) != None:
                setattr(self, control_field.name, self.control_action)
        super().__init__()
    
    def control_action(self, control: TrainerControl, **kwargs):
        frame_info = inspect.currentframe().f_back
        arg_values = inspect.getargvalues(frame_info)
        setattr(control, arg_values.locals["action"], True)
