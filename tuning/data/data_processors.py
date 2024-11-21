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
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import logging
import os

# Third Party
from datasets import Dataset, DatasetDict, IterableDataset
from datasets.exceptions import DatasetNotFoundError
from transformers import AutoTokenizer
import datasets

# Local
from tuning.data.data_config import DataConfig, DataLoaderConfig, DataSetConfig
from tuning.data.data_handlers import AVAILABLE_DATA_HANDLERS
from tuning.utils.utils import get_extension, get_loader_for_filepath


class DataPreProcessor(ABC):

    tokenizer = None
    data_config: DataConfig = None
    dataloaderconfig: DataLoaderConfig = None
    registered_handlers: Dict[str, callable] = None

    def __init__(
        self,
        dataloaderconfig: DataLoaderConfig,
        tokenizer: AutoTokenizer,
        accelerator=None,
    ):
        self.tokenizer = tokenizer
        self.dataloaderconfig = dataloaderconfig
        self.accelerator = None

        # Initialize other objects
        self.registered_handlers = {}

    def register_data_handler(self, name: str, func: callable):
        self.registered_handlers[name] = func

    @abstractmethod
    def process_dataset_configs(
        self, dataset_cofigs: List[DataSetConfig], **extra_kwargs
    ) -> Union[Dataset, IterableDataset]:
        NotImplementedError("Needs to be implemented")


class HFBasedDataPreProcessor(DataPreProcessor):
    def __init__(
        self,
        dataloaderconfig: DataLoaderConfig,
        tokenizer: AutoTokenizer,
        accelerator=None,
    ):
        super().__init__(
            dataloaderconfig=dataloaderconfig,
            tokenizer=tokenizer,
            accelerator=accelerator,
        )

    def _load_dataset(self, datasetconfig, splitName, **kwargs):
        files = datasetconfig.data_paths
        name = datasetconfig.name
        extns = []
        for f in files:
            e = get_extension(f)
            extns.append(e)
        # simple check to make sure all files are of same type.
        # Do we need this assumption?
        assert (
            extns.count(extns[0]) == len(extns),
            f"all files in a dataset {name} should have same extension",
        )

        loader = get_loader_for_filepath(file_path=files[0])
        try:
            return datasets.load_dataset(
                loader,
                data_files=files,
                split=splitName,
                **kwargs,
            )
        except DatasetNotFoundError as e:
            raise e

    def _process_dataset_configs(
        self, dataset_cofigs: List[DataSetConfig], **extra_kwargs
    ) -> Union[Dataset, IterableDataset]:
        train_dataset = None
        final_datasets = None
        splitName = "train"  # default

        logging.info("Starting HFBasedDataPreProcessor...")
        # Iterate over the multiple datasets provided to us
        for d in dataset_cofigs:
            logging.info("Loading %s" % (d.name))

            # In future the streaming etc go as kwargs of this function
            raw_dataset = self._load_dataset(d, splitName)

            logging.info("Loaded raw dataset : {raw_datasets}")

            raw_datasets = DatasetDict()

            # Assume all is train split
            if isinstance(raw_dataset, Dataset):
                raw_datasets[splitName] = raw_dataset
            else:
                raw_datasets = raw_dataset

            if d.sampling:
                logging.warning("Sampling multiple datasets is not supported yet")

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

                    logging.info(f"Applying Handler : {data_handler} Args : {kwargs}")

                    raw_datasets = raw_datasets.map(handler, **kwargs)

            if final_datasets is None:
                final_datasets = raw_datasets
            else:
                for k in raw_datasets.keys():
                    if k in final_datasets:
                        final_datasets[k] = datasets.concatenate_datasets(
                            [final_datasets[k], raw_datasets[k]]
                        )
                    else:
                        final_datasets[k] = raw_datasets[k]

        if "train" in final_datasets:
            train_dataset = final_datasets["train"]

        return train_dataset

    def process_dataset_configs(
        self, dataset_cofigs: List[DataSetConfig], **kwargs
    ) -> Union[Dataset, IterableDataset]:
        train_dataset = None
        if self.accelerator:
            with self.accelerator.main_process_first(desc="Processing data..."):
                train_dataset = self._process_dataset_configs(dataset_cofigs, **kwargs)
        else:
            train_dataset = self._process_dataset_configs(dataset_cofigs, **kwargs)

        return train_dataset


def autoregister_available_handlers(processor: DataPreProcessor):
    if processor is None:
        return
    for name, func in AVAILABLE_DATA_HANDLERS.items():
        processor.register_data_handler(name=name, func=func)


def get_dataprocessor(
    dataloaderconfig: DataLoaderConfig,
    tokenizer: AutoTokenizer,
    accelerator: Optional[Any] = None,
) -> DataPreProcessor:
    loader = dataloaderconfig.type
    if loader == "default":
        processor = HFBasedDataPreProcessor(
            dataloaderconfig=dataloaderconfig,
            tokenizer=tokenizer,
            accelerator=accelerator,
        )
    else:
        processor = None
    autoregister_available_handlers(processor)
    return processor
