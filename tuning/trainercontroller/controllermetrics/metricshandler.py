import abc

class MetricHandler(metaclass=abc.ABCMeta):
    """Base class for the controller-metrics"""
    def __init__(self, name):
        self.__name = name

    def get_name(self):
        return self.__name

    @abc.abstractmethod 
    def validate(self, training_args) -> bool:
        pass
    
    @abc.abstractmethod 
    def compute(self, training_state, training_args=None, metrics=None) -> dict:
        pass   