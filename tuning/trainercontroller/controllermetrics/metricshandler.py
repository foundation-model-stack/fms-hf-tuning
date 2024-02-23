class MetricHandler:
    """Base class for the controller-metrics"""
    def __init__(self, name):
        self.__name = name

    def get_name(self):
        return self.__name

    def validate(self, training_args) -> bool:
        pass
    
    def compute(self, training_state, training_args=None, metrics=None) -> dict:
        pass   