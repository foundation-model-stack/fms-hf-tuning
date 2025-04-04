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
from typing import Dict, List, Union
import logging
import os

# Third Party
from accelerate.state import PartialState
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from datasets.exceptions import DatasetNotFoundError
from transformers import AutoTokenizer
import datasets

# Local
from tuning.data.data_config import DataConfig, DataPreProcessorConfig, DataSetConfig
from tuning.data.data_handlers import (
    AVAILABLE_DATA_HANDLERS,
    DataHandler,
    DataHandlerType,
)
from tuning.utils.utils import (
    get_loader_for_filepath,
    resolve_iterable_dataset_features,
    validate_mergeable_datasets,
)

logger = logging.getLogger(__name__)


class DataPreProcessor:

    tokenizer = None
    data_config: DataConfig = None
    processor_config: DataPreProcessorConfig = None
    registered_handlers: Dict[str, DataHandler] = None

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

    def register_data_handler(self, name: str, handler: DataHandler):
        if not isinstance(name, str) or not isinstance(handler, DataHandler):
            raise ValueError(
                "Handler should be of type tuning.data_handler.DataHandler, and name of str"
            )
        if name in self.registered_handlers:
            logger.warning(
                "Handler name '%s' already exists and will be overwritten", name
            )
        self.registered_handlers[name] = handler
        logger.info("Registered new handler %s", name)

    def register_data_handlers(self, handlers: Dict[str, DataHandler]):
        if handlers is None:
            return
        if not isinstance(handlers, Dict):
            raise ValueError(
                "Handler should be of type tuning.data_handler.DataHandler, and name of str"
            )
        for k, v in handlers.items():
            self.register_data_handler(name=k, handler=v)

    def _get_registered_datahandler(self, handler_name):
        try:
            return self.registered_handlers[handler_name]
        except KeyError as e:
            raise ValueError(
                f"Data handler requested {handler_name} is "
                "not registed. Registered handlers are "
                + ", ".join(self.registered_handlers.keys())
            ) from e

    def load_dataset(
        self,
        datasetconfig: DataSetConfig,
        streaming: bool,
        splitName: str,
        datafile: str = None,
        **kwargs,
    ):

        if datafile and datasetconfig:
            raise ValueError("Both datafile and datasetconfig should not be set")
        if (not datafile) and (not datasetconfig):
            raise ValueError("Either datafile or datasetconfig must be set")

        def _load_dataset(
            data_path=None,
            builder=None,
            data_files=None,
            data_dir=None,
            streaming=False,
        ):
            """
            Helper function to load a dataset using datasets.load_dataset
            with standardized exception handling.

            Args:
                data_path: The path argument for load_dataset (directory, file, pattern, dataset_id)
                builder: Optional builder to use if provided.
                data_files: Optional data_files list if loading from files.
                data_dir: Optional data_dir if loading from a directory with a builder.
                streaming: Optional bool if using IterableDataset.
            Returns: dataset
            """

            load_kwargs = {**kwargs, "split": splitName}
            if data_dir is not None:
                load_kwargs["data_dir"] = data_dir
            if data_files is not None:
                load_kwargs["data_files"] = data_files
            if streaming is not None:
                load_kwargs["streaming"] = streaming

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

        def _try_load_dataset(dataset_path, dataset_builder, streaming):
            """
            Helper function to call load dataset on case by case basis to ensure we handle
            directories and files (with or without builders) and hf datasets.

            Args:
                dataset_path: Path of directory/file, pattern, or hf dataset id.
                dataset_builder: Optional builder to use if provided.
            Returns: dataset
            """
            if not dataset_path:
                raise ValueError("Invalid dataset path")

            # CASE 1: User passes directory
            if os.path.isdir(dataset_path):  # Checks if path exists and it is a dir
                # Directory case
                if dataset_builder:
                    # Load using a builder with a data_dir
                    return _load_dataset(
                        builder=dataset_builder,
                        data_dir=dataset_path,
                        streaming=streaming,
                    )

                # If no builder then load directly from the directory
                return _load_dataset(data_path=dataset_path, streaming=streaming)

            # Non-directory (file, pattern, HF dataset name)
            # If no builder provided, attempt to infer one
            effective_builder = (
                dataset_builder
                if dataset_builder
                else get_loader_for_filepath(dataset_path)
            )

            if effective_builder:
                # CASE 2: Files passed with builder. Load using the builder and specific files
                return _load_dataset(
                    builder=effective_builder,
                    data_files=[dataset_path],
                    streaming=streaming,
                )

            # CASE 3: User passes files/folder/pattern/HF_dataset which has no builder
            # Still no builder, try if this is a dataset id
            return _load_dataset(data_path=dataset_path, streaming=streaming)

        if datafile:
            return _try_load_dataset(datafile, None, streaming=False)

        data_paths = datasetconfig.data_paths
        builder = datasetconfig.builder
        all_datasets = []

        for data_path in data_paths:
            dataset = _try_load_dataset(data_path, builder, streaming)
            if isinstance(dataset, IterableDataset):
                dataset = resolve_iterable_dataset_features(dataset)
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

    def __execute_rename_data_handler(self, raw_datasets, handler, **kwargs):
        """
        Rename columns in the dataset using the provided column mapping.
        Uses Huggingface {DatasetDict/IterableDatasetDict}.rename_columns() API
        """
        fn_kwargs = kwargs.get("fn_kwargs", {})
        if not isinstance(fn_kwargs, Dict) or "column_mapping" not in fn_kwargs:
            raise ValueError(
                "Please pass fn_kwargs dict with key column_mapping for "
                + " %s data handler" % handler.handler_type.name
            )

        mapping = fn_kwargs["column_mapping"]
        if mapping is None or not isinstance(mapping, Dict):
            raise ValueError(
                f"column mapping {mapping} passed to {handler.handler_type.name} data handler "
                "should be a Dict of str:str"
            )
        logger.info("Renaming %s columns", str(mapping))
        return raw_datasets.rename_columns(column_mapping=mapping)

    def __execute_select_data_handler(self, raw_datasets, handler, **kwargs):
        """
        Selects specific columns from the dataset.
        Uses HuggingFace {DatasetDict/IterableDatasetDict}.select_columns() API
        """
        fn_kwargs = kwargs.get("fn_kwargs", {})
        if not isinstance(fn_kwargs, Dict) or "column_names" not in fn_kwargs:
            raise ValueError(
                "Please pass fn_kwargs dict with key column_names for "
                + " %s data handler" % handler.handler_type.name
            )

        columns = fn_kwargs["column_names"]
        if columns is None or not isinstance(columns, List):
            raise ValueError(
                f"column names {columns} passed to {handler.handler_type.name} data handler "
                "should be a List of columns to select"
            )
        logger.info("Selecting only %s columns", str(columns))
        return raw_datasets.select_columns(column_names=columns)

    def __execute_remove_data_handler(self, raw_datasets, handler, **kwargs):
        """
        Removes specified columns from the dataset.
        Uses HuggingFace {DatasetDict/IterableDatasetDict}.remove_columns() API
        """
        fn_kwargs = kwargs.get("fn_kwargs", {})
        if not isinstance(fn_kwargs, Dict) or "column_names" not in fn_kwargs:
            raise ValueError(
                "Please pass fn_kwargs dict with key column_names for "
                + " %s data handler" % handler.handler_type.name
            )
        columns = fn_kwargs["column_names"]
        if columns is None or not isinstance(columns, List):
            raise ValueError(
                f"column names {columns} passed to {handler.handler_type.name} data handler "
                "should be a List of columns to remove"
            )
        logger.info("Removing %s columns", str(columns))
        return raw_datasets.remove_columns(column_names=columns)

    def __execute_filter_data_handler(self, raw_datasets, handler, **kwargs):
        if "fn_kwargs" not in kwargs:
            kwargs["fn_kwargs"] = {}
        return raw_datasets.filter(handler.op, **kwargs)

    def __execute_map_data_handler(
        self, raw_datasets, handler, splitName, datasetName, **kwargs
    ):
        column_names = None
        if hasattr(raw_datasets[splitName], "column_names"):
            column_names = raw_datasets[splitName].column_names

        # remove __content__ from all processing
        if column_names and "__content__" in column_names:
            column_names.remove("__content__")

        if "remove_columns" not in kwargs:
            kwargs["remove_columns"] = None
        if kwargs["remove_columns"] == "all":
            if column_names is None:
                logger.warning(
                    "Could not infer column names from the dataset %s \
                    unable to set `remove_columns` to all.\n \
                    Please explicitly specify the column list or \
                    use remove/select data handlers.",
                    datasetName,
                )
            else:
                kwargs["remove_columns"] = column_names

        if "fn_kwargs" not in kwargs:
            kwargs["fn_kwargs"] = {}

        kwargs["fn_kwargs"]["tokenizer"] = self.tokenizer
        kwargs["fn_kwargs"]["column_names"] = column_names

        return raw_datasets.map(handler.op, **kwargs)

    def _execute_data_handlers(
        self, raw_datasets, data_handler_config, splitName, datasetName
    ):
        handler_name: str = data_handler_config.name
        kwargs: Dict = data_handler_config.arguments
        handler: DataHandler = self._get_registered_datahandler(handler_name)

        # Check batching and num_proc for multiprocessing of handlers.
        if "batched" in kwargs:
            # If batching is requested but not allowed throw error
            if kwargs["batched"] and not handler.allows_batching:
                raise ValueError(
                    f"DataHandler {handler} does not support batching\
                        but was called with batched=True in data config"
                )
        else:
            # If batching is not requested set the batching to allows_batching
            kwargs["batched"] = handler.allows_batching

        if isinstance(raw_datasets, IterableDatasetDict):
            if "num_proc" in kwargs:
                del kwargs["num_proc"]
                logger.warning(
                    "num_proc is not applicable for \
                                IterableDatasets and has been removed."
                )
        else:
            if "num_proc" not in kwargs:
                kwargs["num_proc"] = os.cpu_count()
                logger.info("setting num_proc to %s", os.cpu_count())

        logger.info("Applying Handler: %s Args: %s", handler, kwargs)

        handler_type = handler.handler_type

        if handler_type is not DataHandlerType.MAP:
            if "remove_columns" in kwargs:
                logger.warning(
                    "remove_columns passed to handler {} "
                    "will not be used as underlying API doesn't support it".format(
                        handler
                    )
                )
                kwargs.pop("remove_columns")

        if handler_type is DataHandlerType.REMOVE:
            return self.__execute_remove_data_handler(raw_datasets, handler, **kwargs)
        if handler_type is DataHandlerType.SELECT:
            return self.__execute_select_data_handler(raw_datasets, handler, **kwargs)
        if handler_type is DataHandlerType.RENAME:
            return self.__execute_rename_data_handler(raw_datasets, handler, **kwargs)
        if handler_type is DataHandlerType.FILTER:
            return self.__execute_filter_data_handler(raw_datasets, handler, **kwargs)
        if handler_type is DataHandlerType.MAP:
            return self.__execute_map_data_handler(
                raw_datasets, handler, splitName, datasetName, **kwargs
            )

        raise ValueError(
            f'Unknown data handler type {handler.handler_type} \
              supported types are - \
              {",".join([e.name for e in DataHandlerType])}'
        )

    def _process_dataset_configs(
        self, dataset_configs: List[DataSetConfig]
    ) -> Union[Dataset, IterableDataset]:

        splitName = "train"  # default
        all_datasetdicts = []

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
            raw_dataset = self.load_dataset(
                d, self.processor_config.streaming, splitName
            )
            if isinstance(raw_dataset, IterableDataset):
                raw_dataset = resolve_iterable_dataset_features(raw_dataset)

            logger.info("Loaded raw dataset : %s", str(raw_dataset))

            if isinstance(raw_dataset, IterableDataset):
                raw_datasets = IterableDatasetDict()
            else:
                raw_datasets = DatasetDict()

            # Assume all is train split
            if isinstance(raw_dataset, (Dataset, IterableDataset)):
                raw_datasets[splitName] = raw_dataset
            else:
                raw_datasets = raw_dataset

            if d.data_handlers:  # Execute the datahandlers
                for data_handler_config in d.data_handlers:
                    raw_datasets = self._execute_data_handlers(
                        raw_datasets=raw_datasets,
                        data_handler_config=data_handler_config,
                        splitName=splitName,
                        datasetName=d.name,
                    )

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
        if isinstance(train_dataset, IterableDataset):
            train_dataset = resolve_iterable_dataset_features(train_dataset)

        return train_dataset

    def process_dataset_configs(
        self, dataset_configs: List[DataSetConfig]
    ) -> Union[Dataset, IterableDataset]:
        train_dataset = None

        # Use partial state as recommended by HF documentation for process control
        # https://huggingface.co/docs/accelerate/v1.0.0rc1/en/package_reference/state#accelerate.PartialState
        # and is used similarly in trainer.sft_trainer
        # https://github.com/huggingface/trl/blob/e3244d/trl/trainer/sft_trainer.py#L367
        state = PartialState()

        # The main_process_first context ensures that the main process runs first
        # as we want to reuse HF cache and not redo computation on all nodes
        # For rationale see https://github.com/huggingface/trl/pull/3106
        with state.main_process_first():
            train_dataset = self._process_dataset_configs(dataset_configs)

        return train_dataset


def get_datapreprocessor(
    processor_config: DataPreProcessorConfig,
    tokenizer: AutoTokenizer,
    additional_data_handlers: Dict[str, DataHandler] = None,
) -> DataPreProcessor:
    processor = DataPreProcessor(
        processor_config=processor_config,
        tokenizer=tokenizer,
    )
    processor.register_data_handlers(additional_data_handlers)
    return processor
