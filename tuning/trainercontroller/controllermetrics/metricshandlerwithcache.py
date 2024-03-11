import abc
from . import MetricHandler

class MetricHandlerWithCache(MetricHandler):
    """Base class for the controller-metrics which use cache"""
    def __init__(self, name, cache):
        super().__init__(name)
        self.__cache = cache
    
    @abc.abstractmethod
    def validate(self, training_args) -> bool:
        pass
    
    @abc.abstractmethod
    def compute(self, training_state, training_args=None, metrics=None) -> dict:
        pass   

    def get_cache(self):
        return self.__cache

    def add_to_cache(self, **kwargs):
        for k, v in kwargs.items():
            self.__cache[k].append(v)

    def slide_the_window(self, window_size):
        keys = list(self.__cache.keys())
        if len(self.__cache[keys[0]]) < window_size:
            return False        
        if len(self.__cache[keys[0]]) == window_size:
            return True
        for k in keys:
            self.__cache[k].popleft()
        return True
    
    @abc.abstractmethod
    def compute_metrics_on_cache(self) -> dict:
        pass