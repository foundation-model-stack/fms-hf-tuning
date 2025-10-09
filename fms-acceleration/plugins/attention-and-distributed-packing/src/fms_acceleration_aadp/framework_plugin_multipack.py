# Copyright The FMS HF Tuning Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from types import MethodType
from typing import Dict, Tuple
import warnings

# Third Party
from accelerate import Accelerator
from fms_acceleration import AccelerationPlugin
from peft import LoraConfig
from torch.utils.data import DataLoader
from transformers import TrainingArguments

# from accelerate.data_loader import DataLoaderShard
import torch


class MultipackDataloaderAccelerationPlugin(AccelerationPlugin):

    require_packages = {"numba"}

    def __init__(
        self,
        configurations: Dict[str, Dict],
        seed: int = 42,
    ):
        super().__init__(configurations)

        self.num_processes = self._check_config_and_maybe_check_values(
            key="training.dataloader.multipack.num_processes",
        )

        # see about the collator
        attention = self._check_config_and_maybe_check_values(
            key="training.attention",
        )

        # internal flags
        self._seed = seed
        self._padding_free = False
        self._pad_token_id = None

        if "padding_free" in attention:
            # for padding free the multipack preparation will ignore the padding tokens
            self._padding_free = True
        else:
            # NOTE: need to get this from somewhere
            assert self._pad_token_id is not None, "need to get pad token id"

    @property
    def requires_augmentation(self):
        return True

    def augmentation(
        self,
        model,
        train_args: TrainingArguments,
        modifiable_args: Tuple[LoraConfig],
    ):

        # guarded because multipack has numba dependencies
        # Third Party
        # pylint: disable=import-outside-toplevel
        from fms_acceleration.accelerator_patcher import (
            AcceleratorPatcher,
            AcceleratorPatcherComponent,
        )

        # Local
        from .aadp_utils import (  # pylint: disable=import-outside-toplevel
            calculate_token_lengths,
        )
        from .multipack_sampler import (  # pylint: disable=import-outside-toplevel
            MultipackDistributedBatchSampler,
        )

        rank, num_bins = 0, 1
        if torch.distributed.is_initialized():
            num_bins = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            # NOTE: or should we do a silent fallback
            raise AssertionError(
                "Multipack dataloader only works for distributed training."
            )

        # some checks
        def _prereq(dataloader: DataLoader):
            return hasattr(dataloader, "dataset")

        def _build_multipack_dataloader(
            dataloader: DataLoader, accelerator: Accelerator
        ):

            # NOTE: for now we disable support for deepspeed, but can be added in
            # future if needed
            assert (
                not accelerator.state.deepspeed_plugin
            ), "Currently, multipack not supported for deepspeed"

            # get the dataset
            dataset = dataloader.dataset
            if torch.distributed.get_rank() > 0:
                warnings.warn(
                    "Waiting for main process to perform the mapping."
                    "If the dataset is large, some processes might time out,"
                    "You may need to increase the timeout limit or the number "
                    f"of workers processing the dataset > {self.num_processes}."
                )
                torch.distributed.barrier()

            lengths = calculate_token_lengths(dataset, num_processes=self.num_processes)

            if torch.distributed.get_rank() == 0:
                torch.distributed.barrier()

            self._max_number_tokens = (
                train_args.per_device_train_batch_size * lengths.mean()
            )

            # prepare the multipack distributed batch sampler
            sampler = MultipackDistributedBatchSampler(
                batch_max_length=self._max_number_tokens,
                lengths=lengths,
                num_replicas=num_bins,
                rank=rank,
                seed=self._seed,
                padding=not self._padding_free,
            )

            # wanted to use this but its abit annoying,
            # from accelerate.data_loader import DataLoaderShard
            # - so will just patch for now, but lets have a better
            #   solution later
            dataloader = DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=dataloader.num_workers,
                collate_fn=dataloader.collate_fn,
            )

            # patch a set epoch function to delegate the call to the
            # batch_sampler
            def _set_epoch(self, epoch: int):
                self.batch_sampler.set_epoch(epoch)

            dataloader.set_epoch = MethodType(_set_epoch, dataloader)
            return dataloader

        AcceleratorPatcher.replace(
            "multipack",
            AcceleratorPatcherComponent.data_loader,
            replacement_builder=_build_multipack_dataloader,
            pre_requisite_check=_prereq,
            skip_prepare=True,
        )

        # take a pointer to train args
        self._train_args = train_args
        return model, modifiable_args


# register
AccelerationPlugin.register_plugin(
    MultipackDataloaderAccelerationPlugin,
    configuration_and_paths=[
        "training.dataloader.multipack",  # activate if multipack config
        "training.attention",  # currently we require multipack to work with padding free
    ],
)
