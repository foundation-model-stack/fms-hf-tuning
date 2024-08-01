# Standard
from dataclasses import dataclass
from typing import List

# Local
from .utils import ensure_nested_dataclasses_initialized, parsable_dataclass


@parsable_dataclass
@dataclass
class MultipackConfig:

    # effective batch size of the packing
    effective_batch_size: int = 3840

    # aka max_batch_len
    # https://github.com/instructlab/training/blob/d9237f8df779c737982acc9bfd9e965ccd83cb77/src/instructlab/training/config.py#L126
    max_number_tokens: int = 60000


@parsable_dataclass
@dataclass
class LossAcrossGPUsConfig:

    # how the losses are reduced
    reduction: str = "mean"

    # resolution in which losses are reduced.
    # - currently this plugin only supports by token reduction
    resolution: str = "token"


@parsable_dataclass
@dataclass
class PaddingFreeConfig:

    # the method we use to enable padding free on the models
    # - the huggingface injected method is change the varlen function
    #   this allows to access padding free methods before the transformers
    #   0.43 on huggingface models
    method: str = "huggingface-injected"

    dropout_method: str = "none"

    dropout_value: float = 0.0


@parsable_dataclass
@dataclass
class MLPDropoutConfig:

    # this plugin allows to modify the MLP behaviors, e.g.
    # adding extra dropouts
    method: str = "residual"

    # percentage of the dropout
    value: float = 0.1


@parsable_dataclass
@dataclass
class EmbeddingDropoutConfig:

    # this plugin allows to modify the Embedding behaviors, e.g.
    # adding extra dropouts
    method: str = "inputs"

    # percentage of the dropout
    value: float = 0.1


@dataclass
class FastAttentionConfig:

    # to access the multipack dataloader
    multipack: MultipackConfig = None

    # for activating padding-free methods
    padding_free: PaddingFreeConfig = None

    # for different flavours of loss reduction across GPUs
    loss_across_gpus: LossAcrossGPUsConfig = None

    # for mlp modifications
    mlp_dropout: MLPDropoutConfig = None

    # for embedding modifications
    emb_dropout: EmbeddingDropoutConfig = None

    def __post_init__(self):
        # ensure nested dataclasses initialized
        ensure_nested_dataclasses_initialized(self)
