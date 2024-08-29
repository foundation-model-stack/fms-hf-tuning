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


@parsable_dataclass
@dataclass
class MultiPack:

    num_processes: int = 16


@dataclass
class AttentionAndDistributedPackingConfig:

    padding_free: PaddingFree = None

    multipack: MultiPack = None

    def __post_init__(self):
        # ensure nested dataclasses initialized
        ensure_nested_dataclasses_initialized(self)
