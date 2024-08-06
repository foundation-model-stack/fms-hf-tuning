# Standard
from dataclasses import dataclass

# Local
from .utils import ensure_nested_dataclasses_initialized, parsable_dataclass


@parsable_dataclass
@dataclass
class PaddingFree:

    method: str = "huggingface"

    def __post_init__(self):
        if self.method != "huggingface":
            raise ValueError("only 'huggingface' method currently supported.")


@dataclass
class AttentionAndDistributedPackingConfig:

    padding_free: PaddingFree = None

    def __post_init__(self):
        # ensure nested dataclasses initialized
        ensure_nested_dataclasses_initialized(self)
