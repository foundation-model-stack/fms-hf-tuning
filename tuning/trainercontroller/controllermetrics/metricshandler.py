import abc
from transformers import TrainerCallback, TrainerState, TrainingArguments
from typing import Any
from tuning.trainercontroller.validator import Validator

class MetricHandler(metaclass=abc.ABCMeta):
    """Base class for the controller-metrics"""
    def __init__(self, name, validator: Validator):
        self._name = name
        self._events = []
        self._validator = validator

    def register_events(self, event_names):
        for event_name in event_names:
            if not self._validator.validate(TrainerCallback, event_name):
                raise KeyError(f'Metric {self._name} is attempting to register an invalid event name {event_name}')
        self._events.extend(event_names)

    def get_events(self):
        return self._events

    @abc.abstractmethod 
    def validate(self, training_args: TrainingArguments, **kwargs) -> bool:
        pass
    
    def _compute(self, training_state: TrainerState, event_name: str, training_args: TrainingArguments=None, metrics=None, **kwargs) -> dict:
        return {self._name: self.compute(training_state, event_name, training_args, metrics, kwargs)}

    @abc.abstractmethod 
    def compute(self, training_state: TrainerState, event_name: str, training_args: TrainingArguments=None, metrics=None, **kwargs) -> Any:
        pass   