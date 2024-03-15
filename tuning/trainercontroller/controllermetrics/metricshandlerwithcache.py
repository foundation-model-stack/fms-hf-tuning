import abc
from . import MetricHandler
from transformers import TrainerState, TrainingArguments
from typing import Any
from tuning.trainercontroller.validator import Validator

class MetricHandlerWithCache(MetricHandler):
    """Base class for the controller-metrics which use cache"""
    def __init__(self, name: str, validator: Validator, cache: dict):
        super().__init__(name, validator)
        self._cache = cache
    
    @abc.abstractmethod
    def validate(self, training_args: TrainingArguments, **kwargs) -> bool:
        pass
    
    @abc.abstractmethod
    def compute(self, training_state: TrainerState, training_args: TrainingArguments=None, metrics=None, **kwargs) -> Any:
        pass   

    def get_cache(self):
        return self._cache

    def add_to_cache(self, **kwargs):
        for k, v in kwargs.items():
            self._cache[k].append(v)

    def slide_the_window(self, window_size: int):
        keys = list(self._cache.keys())
        if len(self._cache[keys[0]]) < window_size:
            return False        
        if len(self._cache[keys[0]]) == window_size:
            return True
        for k in keys:
            self._cache[k].popleft()
        return True
    
    @abc.abstractmethod
    def compute_metrics_on_cache(self) -> Any:
        pass