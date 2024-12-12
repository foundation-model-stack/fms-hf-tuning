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
from typing import Callable, Dict, List, Union
import logging
import os

# Third Party
from datasets import Dataset, DatasetDict, IterableDataset
from datasets.exceptions import DatasetNotFoundError
from transformers import AutoTokenizer
import datasets
import torch

# Local
from tuning.data.data_config import DataConfig, DataPreProcessorConfig, DataSetConfig
from tuning.data.data_handlers import AVAILABLE_DATA_HANDLERS
from tuning.utils.utils import get_extension, get_loader_for_filepath


class DataPreProcessor:

    tokenizer = None
    data_config: DataConfig = None
    processor_config: DataPreProcessorConfig = None
    registered_handlers: Dict[str, Callable] = None

    def __init__(
        self, processor_config: DataPreProcessorConfig, tokenizer: AutoTokenizer
    ):
        self.tokenizer = tokenizer
        self.processor_config = processor_config

        # Initialize other objects
        self.registered_handlers = {}

        # Auto register available data handlers
        for k, v in AVAILABLE_DATA_HANDLERS.items():
            self.registered_handlers[k] = v

    def register_data_handler(self, name: str, func: Callable):
        if not isinstance(name, str) or not callable(func):
            raise ValueError("Handlers should be of type Dict, str to callable")
        if name in self.registered_handlers:
            logging.warning(
                "Handler name '%s' already exists and will be overwritten", name
            )
        self.registered_handlers[name] = func
        logging.info("Registered new handler %s", name)

    def register_data_handlers(self, handlers: Dict[str, Callable]):
        if handlers is None:
            return
        if not isinstance(handlers, Dict):
            raise ValueError("Handlers should be of type Dict, str to callable")
        for k, v in handlers.items():
            self.register_data_handler(name=k, func=v)

    def load_dataset(
        self,
        datasetconfig: DataSetConfig,
        splitName: str,
        datafile: str = None,
        **kwargs,
    ):

        if datafile and datasetconfig:
            raise ValueError("Both datafile and datasetconfig should not be set")
        if (not datafile) and (not datasetconfig):
            raise ValueError("Either datafile or datasetconfig must be set")

        if datafile:
            files = [datafile]
            loader = get_loader_for_filepath(file_path=datafile)
        elif datasetconfig:
            files = datasetconfig.data_paths
            name = datasetconfig.name
            # simple check to make sure all files are of same type.
            extns = [get_extension(f) for f in files]
            assert extns.count(extns[0]) == len(
                extns
            ), f"All files in the dataset {name} should have the same extension"
            loader = get_loader_for_filepath(file_path=files[0])

        if loader in (None, ""):
            raise ValueError(f"data path is invalid [{', '.join(files)}]")

        try:
            return datasets.load_dataset(
                loader,
                data_files=files,
                split=splitName,
                **kwargs,
            )
        except DatasetNotFoundError as e:
            raise e
        except FileNotFoundError as e:
            raise ValueError(f"data path is invalid [{', '.join(files)}]") from e

    def _process_dataset_configs(
        self, dataset_configs: List[DataSetConfig], **extra_kwargs
    ) -> Union[Dataset, IterableDataset]:

        splitName = "train"  # default

        all_datasetdicts = []
        sampling_probabilities = []

        # quick check to see if we are sampling and if we need to throw error.
        sampling_probabilities = [d.sampling for d in dataset_configs if d.sampling]

        if len(sampling_probabilities) > 0:
            if len(sampling_probabilities) != len(dataset_configs):
                raise ValueError(
                    "Sampling probabilities should be provided for all datasets"
                )
            if sum(p for p in sampling_probabilities) != 1:
                raise ValueError("Sampling probabilities don't sum to 1")
            sample_datasets = True
            logging.info(
                "Sampling ratios are specified; given datasets will be interleaved."
            )
        else:
            logging.info(
                "Sampling is not specified; if multiple datasets are provided,"
                " the given datasets will be concatenated."
            )
            sample_datasets = False

        logging.info("Starting DataPreProcessor...")
        # Now Iterate over the multiple datasets provided to us to process
        for d in dataset_configs:
            logging.info("Loading %s", d.name)

            # In future the streaming etc go as kwargs of this function
            raw_dataset = self.load_dataset(d, splitName)

            logging.info("Loaded raw dataset : %s", str(raw_dataset))

            raw_datasets = DatasetDict()

            # Assume all is train split
            if isinstance(raw_dataset, Dataset):
                raw_datasets[splitName] = raw_dataset
            else:
                raw_datasets = raw_dataset

            if d.data_handlers:  # Execute the datahandlers
                for data_handler in d.data_handlers:
                    handler_name: str = data_handler.name
                    handler: callable = self.registered_handlers[handler_name]
                    kwargs: Dict = data_handler.arguments

                    if "batched" not in kwargs:
                        kwargs["batched"] = False

                    column_names = raw_datasets[splitName].column_names

                    # remove __content__ from all processing
                    if "__content__" in column_names:
                        column_names.remove("__content__")

                    if "remove_columns" not in kwargs:
                        kwargs["remove_columns"] = None
                    if kwargs["remove_columns"] == "all":
                        kwargs["remove_columns"] = column_names

                    if "num_proc" not in kwargs:
                        kwargs["num_proc"] = os.cpu_count()

                    if "fn_kwargs" not in kwargs:
                        kwargs["fn_kwargs"] = {}

                    kwargs["fn_kwargs"]["tokenizer"] = self.tokenizer
                    kwargs["fn_kwargs"]["column_names"] = column_names

                    kwargs["fn_kwargs"] = dict(kwargs["fn_kwargs"], **extra_kwargs)

                    logging.info("Applying Handler: %s Args: %s", data_handler, kwargs)

                    raw_datasets = raw_datasets.map(handler, **kwargs)

            # Append the processed datasets to the final dict
            all_datasetdicts.append(raw_datasets)

        # This is a dict of { split: list[datasets] }
        final_datasets = {}
        for d in all_datasetdicts:
            for k, v in d.items():
                if k not in final_datasets:
                    final_datasets[k] = [v]
                else:
                    final_datasets[k].append(v)

        if sample_datasets:
            strategy = self.processor_config.sampling_stopping_strategy
            seed = self.processor_config.sampling_seed
            logging.info(
                "Interleaving datasets: strategy[%s] seed[%d] probabilities[%s]",
                strategy,
                seed,
                str(sampling_probabilities),
            )
            for k, v in final_datasets.items():
                interleaved = datasets.interleave_datasets(
                    datasets=v,
                    probabilities=sampling_probabilities,
                    stopping_strategy=strategy,
                    seed=seed,
                )
                final_datasets[k] = interleaved
        else:
            for k, v in final_datasets.items():
                final_datasets[k] = (
                    v[0] if len(v) == 1 else datasets.concatenate_datasets(v)
                )

        train_dataset = final_datasets.get("train", None)

        return train_dataset

    def process_dataset_configs(
        self, dataset_configs: List[DataSetConfig], **kwargs
    ) -> Union[Dataset, IterableDataset]:
        train_dataset = None

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                logging.info("Processing data on rank 0...")
                train_dataset = self._process_dataset_configs(dataset_configs, **kwargs)
            else:
                train_dataset = None

            # Use broadcast_object_list to share the dataset object across ranks
            # TODO: Check if torch.distributed.barrier() is called in broadcast_object_list()
            # See https://github.com/pytorch/pytorch/issues/56142
            # for why the list is shared like this
            to_share = [train_dataset]
            torch.distributed.broadcast_object_list(to_share, src=0)
            train_dataset = to_share[0]
        else:
            logging.info("Processing data...")
            train_dataset = self._process_dataset_configs(dataset_configs, **kwargs)

        return train_dataset


def get_datapreprocessor(
    processor_config: DataPreProcessorConfig,
    tokenizer: AutoTokenizer,
    additional_data_handlers: Dict[str, Callable] = None,
) -> DataPreProcessor:
    processor = DataPreProcessor(
        processor_config=processor_config,
        tokenizer=tokenizer,
    )
    processor.register_data_handlers(additional_data_handlers)
    return processor
