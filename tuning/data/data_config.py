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
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging
import os

# Local
from tuning.utils.utils import load_yaml_or_json

logger = logging.getLogger(__name__)


@dataclass
class DataHandlerConfig:
    name: str
    arguments: Optional[Dict]


@dataclass
class DataSetConfig:
    name: str
    data_paths: List[str]
    builder: Optional[str] = None  # Referring to Hugging Face dataset builder
    sampling: Optional[float] = None
    data_handlers: Optional[List[DataHandlerConfig]] = None


@dataclass
class DataPreProcessorConfig:
    type: Optional[str] = "default"
    sampling_stopping_strategy: Optional[str] = "all_exhausted"
    # Default seed is not none to ensure reproducability
    sampling_seed: Optional[float] = 42
    streaming: Optional[bool] = False
    chat_template: Optional[str] = None


@dataclass
class DataConfig:
    dataprocessor: DataPreProcessorConfig
    datasets: List[DataSetConfig]


def _validate_data_handler_config(data_handler) -> DataHandlerConfig:
    kwargs = data_handler
    assert isinstance(kwargs, dict), "data_handlers in data_config needs to be a dict"
    assert "name" in kwargs and isinstance(
        kwargs["name"], str
    ), "data_handlers need to have a name with type str"
    assert "arguments" in kwargs, "data handlers need to have arguments"
    assert isinstance(
        kwargs["arguments"], dict
    ), "data handler arguments should be of the type dict"
    return DataHandlerConfig(**kwargs)


def _validate_dataset_config(dataset_config) -> DataSetConfig:
    kwargs = dataset_config
    assert isinstance(kwargs, dict), "dataset_config in data_config needs to be a dict"

    c = DataSetConfig(name=kwargs.get("name", ""), data_paths=[])

    if "name" in kwargs:
        assert isinstance(kwargs["name"], str), "dataset name should be string"
    if "data_paths" not in kwargs:
        raise ValueError("data_paths should be specified for each dataset")
    data_paths = kwargs["data_paths"]
    # TODO: Support that data_paths can be a directory or directories
    assert isinstance(data_paths, list), "data_paths should be an array of files"
    c.data_paths = []
    for p in data_paths:
        assert isinstance(p, str), f"path {p} should be of the type string"
        if not os.path.isabs(p):
            _p = os.path.abspath(p)
            logger.warning(" Provided path %s is not absolute changing it to %s", p, _p)
            p = _p
        c.data_paths.append(p)
    if "builder" in kwargs and kwargs["builder"] is not None:
        builder = kwargs["builder"]
        assert isinstance(
            builder, str
        ), f"builder should be a string representing a supported \
        Hugging Face dataset builder, but got: {builder}"
        c.builder = builder
    if "sampling" in kwargs and kwargs["sampling"] is not None:
        ratio = kwargs["sampling"]
        assert isinstance(ratio, float) and (
            0 <= ratio <= 1.0
        ), f"sampling ratio: {ratio} should be float and in range [0.0,1.0]"
        c.sampling = ratio
    if "data_handlers" in kwargs:
        c.data_handlers = []
        for handler in kwargs["data_handlers"]:
            c.data_handlers.append(_validate_data_handler_config(handler))
    return c


def _validate_dataprocessor_config(dataprocessor_config) -> DataPreProcessorConfig:
    kwargs = dataprocessor_config
    c = DataPreProcessorConfig()
    assert isinstance(kwargs, dict), "dataprocessor in data_config needs to be a dict"
    if "type" in kwargs:
        assert isinstance(kwargs["type"], str), "dataprocessor type must be a string"
        c.type = kwargs["type"]
    if "sampling_stopping_strategy" in kwargs:
        strategy = kwargs["sampling_stopping_strategy"]
        assert isinstance(
            strategy, str
        ), "dataset sampling stopping strategy must be a string"
        assert strategy in [
            "first_exhausted",
            "all_exhausted",
        ], "allowed sampling stopping strategies are all_exhausted(default) or first_exhausted"
        c.sampling_stopping_strategy = strategy
    if "sampling_seed" in kwargs:
        seed = kwargs["sampling_seed"]
        assert isinstance(seed, int), "sampling seed should be int"
        c.sampling_seed = seed
    if "streaming" in kwargs:
        streaming = kwargs["streaming"]
        assert isinstance(streaming, bool), f"streaming: {streaming} should be a bool"
        c.streaming = streaming
    if "chat_template" in kwargs:
        chat_template = kwargs["chat_template"]
        assert isinstance(chat_template, str), "chat_template should be a string"
        c.chat_template = chat_template
    return c


def validate_data_config(dataconfig: DataConfig):
    _validate_dataprocessor_config(dataconfig.dataprocessor)
    for d in dataconfig.datasets:
        _validate_dataset_config(d)


def load_and_validate_data_config(data_config_file: str) -> DataConfig:
    raw_data = load_yaml_or_json(data_config_file)
    assert isinstance(
        raw_data, dict
    ), f"The provided data_config file is invalid: {data_config_file}"
    assert "datasets" in raw_data, "datasets should be provided in data config"
    assert isinstance(
        raw_data["datasets"], list
    ), "datasets should be provided as a list"
    datasets = []
    for d in raw_data["datasets"]:
        datasets.append(_validate_dataset_config(d))
    if "dataprocessor" in raw_data:
        dataprocessor = _validate_dataprocessor_config(raw_data["dataprocessor"])

    data_config = DataConfig(dataprocessor=dataprocessor, datasets=datasets)
    return data_config
