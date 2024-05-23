# Standard
from typing import Type

# Local
from .hfcontrols import HFControls
from .operation import Operation

# List of operation handlers
operation_handlers = []


def register(cl: Type):
    """Registers the list of operation handlers by adding to the handler list.

    Args:
        cl: Class type of the handler
    """
    operation_handlers.append(cl)


# Register the default operation handlers in this package here
register(HFControls)
