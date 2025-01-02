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
from tuning.utils.utils import get_loader_for_filepath, validate_mergeable_datasets

logger = logging.getLogger(__name__)


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
            logger.warning(
                "Handler name '%s' already exists and will be overwritten", name
            )
        self.registered_handlers[name] = func
        logger.info("Registered new handler %s", name)

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

        def _load_dataset(data_path=None, builder=None, data_files=None, data_dir=None):
            """
            Helper function to load a dataset using datasets.load_dataset
            with standardized exception handling.

            Args:
                data_path: The path argument for load_dataset (directory, file, pattern, dataset_id)
                builder: Optional builder to use if provided.
                data_files: Optional data_files list if loading from files.
                data_dir: Optional data_dir if loading from a directory with a builder.
            Returns: dataset
            """

            load_kwargs = {**kwargs, "split": splitName}
            if data_dir is not None:
                load_kwargs["data_dir"] = data_dir
            if data_files is not None:
                load_kwargs["data_files"] = data_files

            # Determine the `path` parameter for load_dataset
            load_path = builder if builder else data_path

            try:
                return datasets.load_dataset(path=load_path, **load_kwargs)
            except DatasetNotFoundError as e:
                # Reraise with a more context-specific message if needed
                raise e
            except FileNotFoundError as e:
                # Handle file/directory not found
                context = (
                    f"path {data_path} with builder {builder}"
                    if builder
                    else f"path {data_path}"
                )
                raise ValueError(f"Data loading failed: invalid {context}.") from e
            except datasets.exceptions.DatasetGenerationError as e:
                context = (
                    f"builder {builder} and data_dir {data_dir}"
                    if builder and data_dir
                    else f"builder {builder}"
                    if builder
                    else f"path {data_path}"
                )
                raise ValueError(
                    f"Failed to generate the dataset from the provided {context}."
                ) from e

        if datafile:
            loader = get_loader_for_filepath(file_path=datafile)
            if loader in (None, ""):
                raise ValueError(f"data path is invalid [{datafile}]")
            return _load_dataset(builder=loader, data_files=[datafile])

        data_paths = datasetconfig.data_paths
        builder = datasetconfig.builder
        all_datasets = []

        for data_path in data_paths:
            # CASE 1: User passes directory
            if os.path.isdir(data_path):  # Checks if path exists and isdirectory
                # Directory case
                if builder:
                    # Load using a builder with a data_dir
                    dataset = _load_dataset(builder=builder, data_dir=data_path)
                else:
                    # Load directly from the directory
                    dataset = _load_dataset(data_path=data_path)
            else:
                # Non-directory (file, pattern, HF dataset name)
                # If no builder provided, attempt to infer one
                effective_builder = (
                    builder if builder else get_loader_for_filepath(data_path)
                )

                if effective_builder:
                    # CASE 2: Files passed with builder. Load using the builder and specific files
                    dataset = _load_dataset(
                        builder=effective_builder, data_files=[data_path]
                    )
                else:
                    # CASE 3: User passes files/folder/pattern/HF_dataset which has no builder
                    dataset = _load_dataset(data_path=data_path)

            all_datasets.append(dataset)

        # Logs warning if datasets have different columns
        validate_mergeable_datasets(all_datasets)

        # Concatenate all datasets
        try:
            if len(all_datasets) == 1:
                return all_datasets[0]

            raw_datasets = datasets.concatenate_datasets(all_datasets)
            logger.info(
                "Datasets concatenated from %s .Concatenated dataset columns: %s",
                datasetconfig.name,
                list(raw_datasets.features.keys()),
            )
            return raw_datasets

        except Exception as e:
            raise ValueError(
                f"An error occurred while concatenating datasets from {datasetconfig.name}: {e}"
            ) from e

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
            logger.info(
                "Sampling ratios are specified; given datasets will be interleaved."
            )
        else:
            logger.info(
                "Sampling is not specified; if multiple datasets are provided,"
                " the given datasets will be concatenated."
            )
            sample_datasets = False

        logger.info("Starting DataPreProcessor...")
        # Now Iterate over the multiple datasets provided to us to process
        for d in dataset_configs:
            logger.info("Loading %s", d.name)

            # In future the streaming etc go as kwargs of this function
            raw_dataset = self.load_dataset(d, splitName)

            logger.info("Loaded raw dataset : %s", str(raw_dataset))

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

                    logger.info("Applying Handler: %s Args: %s", data_handler, kwargs)

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
            logger.info(
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
                logger.info("Processing data on rank 0...")
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
            logger.info("Processing data...")
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
