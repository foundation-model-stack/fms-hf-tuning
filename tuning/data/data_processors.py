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
from typing import Dict, List, Tuple, Union
import logging
import os

# Third Party
from accelerate.state import PartialState
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from datasets.exceptions import DatasetNotFoundError
from transformers import AutoProcessor, AutoTokenizer
import datasets

# Local
from tuning.data.data_config import DataConfig, DataPreProcessorConfig, DataSetConfig
from tuning.data.data_handlers import (
    AVAILABLE_DATA_HANDLERS,
    DataHandler,
    DataHandlerType,
)
from tuning.data.utils import (
    get_loader_for_filepath,
    maybe_align_datasets,
    resolve_iterable_dataset_features,
    try_concatenate_datasets,
)

logger = logging.getLogger(__name__)


class DataPreProcessor:

    tokenizer = None
    processor = None
    data_config: DataConfig = None
    processor_config: DataPreProcessorConfig = None
    registered_handlers: Dict[str, DataHandler] = None

    def __init__(
        self,
        processor_config: DataPreProcessorConfig,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
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
        splitName: str = None,
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

            load_kwargs = {**kwargs}
            if splitName is not None:
                load_kwargs["split"] = splitName
            if data_dir is not None:
                load_kwargs["data_dir"] = data_dir
            if data_files is not None:
                load_kwargs["data_files"] = data_files
            if streaming is not None:
                load_kwargs["streaming"] = streaming

            # Determine the `path` parameter for load_dataset
            load_path = builder if builder else data_path

            try:
                return datasets.load_dataset(load_path, **load_kwargs)
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
            all_datasets.append(dataset)

        raw_datasets = try_concatenate_datasets(all_datasets)

        return raw_datasets

    def __execute_rename_data_handler(self, raw_datasets, handler, **kwargs):
        """
        Rename columns in the dataset using the provided column mapping.
        Uses Huggingface {DatasetDict/IterableDatasetDict}.rename_columns() API
        """
        mapping = kwargs["column_mapping"]
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
        columns = kwargs["column_names"]
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
        columns = kwargs["column_names"]
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
        # IterableDatasets doesn't support any description
        if not isinstance(raw_datasets, (IterableDatasetDict or IterableDataset)):
            kwargs["desc"] = handler.desc
        return raw_datasets.filter(handler.op, **kwargs)

    def __execute_map_data_handler(self, raw_datasets, handler, datasetName, **kwargs):
        """Apply handler.op to all splits in raw_datasets instead of just a single split."""
        processed_ds = (
            IterableDatasetDict()
            if isinstance(raw_datasets, IterableDatasetDict)
            else DatasetDict()
        )

        # set up kwargs for map
        if not isinstance(raw_datasets, (IterableDatasetDict, IterableDataset)):
            kwargs["desc"] = handler.desc
        if "remove_columns" not in kwargs:
            kwargs["remove_columns"] = None
        if "fn_kwargs" not in kwargs:
            kwargs["fn_kwargs"] = {}
        kwargs["fn_kwargs"]["tokenizer"] = self.tokenizer

        for split_name, ds in raw_datasets.items():

            if kwargs["remove_columns"] == "all":
                column_names = getattr(ds, "column_names", None)
                if column_names is None:
                    raise ValueError(
                        f"Could not infer column names from the split '{split_name}' in "
                        f"'{datasetName}'. Unable to set `remove_columns` to all.\n"
                        f"Please explicitly specify the column list or use remove/select handlers."
                    )
                kwargs["remove_columns"] = column_names

            processed_ds[split_name] = ds.map(handler.op, **kwargs)

        return processed_ds

    def _execute_data_handlers(self, raw_datasets, data_handler_config, datasetName):
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
                raw_datasets, handler, datasetName, **kwargs
            )

        raise ValueError(
            f'Unknown data handler type {handler.handler_type} \
              supported types are - \
              {",".join([e.name for e in DataHandlerType])}'
        )

    def split_dataset(
        self,
        dataset_config: DataSetConfig,
        dataset: Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict],
    ) -> Union[DatasetDict, IterableDatasetDict]:

        train_split = "train"
        eval_split = "test"

        seed = self.processor_config.seed

        # TODO: This is a problem.
        # The HF function expects a test key but outside we take "validation" from data config.
        train_size = dataset_config.split.get("train", 0.0)
        eval_size = dataset_config.split.get("validation", 0.0)

        total = train_size + eval_size
        if total > 1.0 or total <= 0:
            raise ValueError(
                f"Sum of dataset split definitions (train + validation) must be in (0,1]. "
                f"Got {total}"
            )

        if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
            splits = dataset.keys()
            if len(splits) == 1 and train_split in splits:
                d = dataset[train_split]
            else:
                logger.warning(
                    "Loaded dataset has multiple splits or no train split.\
                    For splitting train to train and validate a train split is required\n\
                    This dataset will be used as is without splitting"
                )
                return dataset
        else:
            d = dataset

        container = (
            IterableDatasetDict if isinstance(d, IterableDataset) else DatasetDict
        )

        # HF API doesn't handle this so we handle this outside.
        if train_size == 1.0:  # validation has to be 0.0 as sum is 1.0
            return container({train_split: d})
        if eval_size == 1.0:  # train has to be 0.0
            return container({eval_split: d})

        if isinstance(d, IterableDataset):
            raise NotImplementedError(
                "Partial splits for IterableDataset(streaming=True) are not supported;"
                + "Either specify full split or set streaming to False\n"
            )

        # Try to split the dataset now.
        split_datasets = d.train_test_split(
            train_size=train_size if train_size > 0.0 else None,
            test_size=eval_size if eval_size > 0.0 else None,
            shuffle=True,
            seed=seed,
        )

        # Again this is not handeled by hugging face API.
        # We have to handle this outside
        if train_size == 0.0:
            del split_datasets[train_split]
        elif eval_size == 0.0:
            del split_datasets[eval_split]

        logger.info(
            "Split dataset {} to create {}".format(
                dataset_config.name, str(split_datasets)
            )
        )
        return split_datasets

    def _prepare_processed_datasets(
        self, dataset_configs: List[DataSetConfig]
    ) -> List[Tuple[DataSetConfig, Union[IterableDataset, Dataset]]]:
        if not dataset_configs:
            raise ValueError(
                "No dataset configs provided. Provided Dataset configs is None."
            )

        train_split = "train"  # default
        eval_split = "test"

        processed_datasets = []

        logger.info("Starting DataPreProcessor...")
        # Now Iterate over the multiple datasets provided to us to process
        for d in dataset_configs:
            logger.info("Loading the dataset - %s", d.name)

            # In future the streaming etc go as kwargs of this function
            loaded_dataset = self.load_dataset(d, self.processor_config.streaming)
            logger.info("Loaded raw dataset : %s", str(loaded_dataset))

            if d.split is not None:
                loaded_dataset = self.split_dataset(d, loaded_dataset)

            # Create a raw dataset which is a Dict container to house all Datasets
            raw_datasets = (
                IterableDatasetDict()
                if isinstance(loaded_dataset, (IterableDataset, IterableDatasetDict))
                else DatasetDict()
            )

            splits_to_keep = [train_split, eval_split]
            if isinstance(loaded_dataset, (Dataset, IterableDataset)):
                # Assume all is train split
                raw_datasets[train_split] = loaded_dataset
            else:
                for k, v in loaded_dataset.items():
                    if k in splits_to_keep:
                        raw_datasets[k] = v

            if d.data_handlers:  # Execute the datahandlers
                for data_handler_config in d.data_handlers:
                    raw_datasets = self._execute_data_handlers(
                        raw_datasets=raw_datasets,
                        data_handler_config=data_handler_config,
                        datasetName=d.name,
                    )

            # Append the processed datasets to the final dict
            processed_datasets.append((d, raw_datasets))
        return processed_datasets

    def _validate_sampling_ratios(self, sampling_ratios: List[float], train_datasets):
        if len(sampling_ratios) > 0:
            if len(sampling_ratios) < len(train_datasets):
                raise ValueError(
                    "Sampling probability should be specified for all datasets with train split"
                )
            if len(sampling_ratios) > len(train_datasets):
                raise ValueError(
                    "Sampling probability should only be specified for datasets with train split"
                )
            if sum(p for p in sampling_ratios) != 1:
                raise ValueError(
                    "Sampling probabilities for train datasets don't sum to 1"
                )
            return True

    def _process_dataset_configs(
        self, dataset_configs: List[DataSetConfig]
    ) -> Tuple[
        Union[Dataset, IterableDataset],
        Union[Dataset, IterableDataset],
        Dict[str, float],
    ]:
        train_split = "train"  # default
        eval_split = "test"
        processed_datasets = self._prepare_processed_datasets(dataset_configs)

        train_datasets = []
        train_sampling_probabilities = []
        validation_datasets = []

        for d, raw in processed_datasets:
            if train_split in raw:
                logger.info("Taking train split from dataset {}".format(d.name))
                train_datasets.append(raw[train_split])
            if eval_split in raw:
                logger.info("Taking validation split from dataset {}".format(d.name))
                validation_datasets.append(raw[eval_split])
            if d.sampling and d.sampling > 0.0:
                train_sampling_probabilities.append(d.sampling)

        if len(train_datasets) == 0:
            raise ValueError(
                "Processed datasets do not contain train split. Check your split ratios"
            )

        # quick check to see if we are sampling and if we need to throw error.
        sample_datasets = self._validate_sampling_ratios(
            train_sampling_probabilities, train_datasets
        )

        # Ensure again datasets are aligned before interleaving or concatenating
        maybe_align_datasets(train_datasets)
        maybe_align_datasets(validation_datasets)

        train_dataset = None
        eval_dataset = None

        if len(train_datasets) == 1:
            train_dataset = train_datasets[0]
        elif sample_datasets:
            strategy = self.processor_config.sampling_stopping_strategy
            seed = self.processor_config.seed
            logger.info(
                "Interleaving train datasets: strategy[%s] seed[%d] probabilities[%s]",
                strategy,
                seed,
                str(train_sampling_probabilities),
            )
            train_dataset = datasets.interleave_datasets(
                datasets=train_datasets,
                probabilities=train_sampling_probabilities,
                stopping_strategy=strategy,
                seed=seed,
            )
        else:
            train_dataset = datasets.concatenate_datasets(train_datasets)

        if len(validation_datasets) > 0:
            logger.info(
                "Validataion splits are present and will always be concatenated",
            )
            eval_dataset = (
                validation_datasets[0]
                if len(validation_datasets) == 1
                else datasets.concatenate_datasets(validation_datasets)
            )

        # Just a failsafe in case this is required later.
        if isinstance(train_dataset, IterableDataset):
            train_dataset = resolve_iterable_dataset_features(train_dataset)
        if eval_dataset and isinstance(eval_dataset, IterableDataset):
            eval_dataset = resolve_iterable_dataset_features(eval_dataset)

        return train_dataset, eval_dataset, None

    def process_dataset_configs(
        self, dataset_configs: List[DataSetConfig]
    ) -> Tuple[
        Union[Dataset, IterableDataset],
        Union[Dataset, IterableDataset],
        Dict[str, float],
    ]:
        train_dataset = eval_dataset = None

        # Use partial state as recommended by HF documentation for process control
        # https://huggingface.co/docs/accelerate/v1.0.0rc1/en/package_reference/state#accelerate.PartialState
        # and is used similarly in trainer.sft_trainer
        # https://github.com/huggingface/trl/blob/e3244d/trl/trainer/sft_trainer.py#L367
        state = PartialState()

        # The main_process_first context ensures that the main process runs first
        # as we want to reuse HF cache and not redo computation on all nodes
        # For rationale see https://github.com/huggingface/trl/pull/3106
        with state.main_process_first():
            (
                train_dataset,
                eval_dataset,
                sampling_weights,
            ) = self._process_dataset_configs(dataset_configs)

        logger.info("Processed train dataset {}".format(train_dataset))
        logger.info("Processed eval dataset {}".format(eval_dataset))

        return train_dataset, eval_dataset, sampling_weights


class ODMDataPreProcessor(DataPreProcessor):
    def _process_dataset_configs(
        self, dataset_configs: List[DataSetConfig]
    ) -> Tuple[
        Dict[str, Union[Dataset, IterableDataset]],
        Dict[str, Union[Dataset, IterableDataset]],
        Dict[str, float],
    ]:
        processed_datasets = self._prepare_processed_datasets(dataset_configs)
        train_split = "train"
        eval_split = "test"
        train_datasets_dict = {}
        eval_datasets_dict = {}
        sampling_weights_dict = {}
        for d, raw in processed_datasets:
            if d.sampling is not None and d.sampling > 0.0:
                sampling_weights_dict[d.name] = d.sampling
            if train_split in raw:
                train_datasets_dict[d.name] = raw[train_split]
            if eval_split in raw:
                eval_datasets_dict[d.name] = raw[eval_split]
        self._validate_sampling_ratios(
            sampling_weights_dict.values(), train_datasets_dict.values()
        )
        return train_datasets_dict, eval_datasets_dict, sampling_weights_dict


def get_datapreprocessor(
    processor_config: DataPreProcessorConfig,
    tokenizer: AutoTokenizer,
    processor: AutoProcessor = None,
    additional_data_handlers: Dict[str, DataHandler] = None,
) -> DataPreProcessor:
    data_processor_cls = DataPreProcessor
    if processor_config.type == "odm":
        data_processor_cls = ODMDataPreProcessor
    data_processor = data_processor_cls(
        processor_config=processor_config,
        tokenizer=tokenizer,
        processor=processor,
    )
    data_processor.register_data_handlers(additional_data_handlers)
    return data_processor
