from tuning.trainercontroller.operations.operation import Operation
from tuning.trainercontroller.operations.hfcontrols import HFControls

from typing import Type

operation_handlers = []

def register(cl: Type):
    operation_handlers.append(cl)

register(HFControls)
