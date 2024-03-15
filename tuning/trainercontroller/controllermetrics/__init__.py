from .metricshandler import MetricHandler
from .metricshandlerwithcache import MetricHandlerWithCache
from .epochloss import EpochLoss
from .steploss import StepLoss

from typing import Class

handlers = {}

def register(cl: Class):
    handlers[cl.__name__] = cl

register(EpochLoss)
register(StepLoss)
