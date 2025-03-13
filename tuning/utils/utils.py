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
import json
import logging
import os

# Third Party
from datasets import IterableDataset
import yaml

logger = logging.getLogger(__name__)


def get_extension(file_path: str) -> str:
    _, ext = os.path.splitext(file_path)
    return ext.lower()


def get_loader_for_filepath(file_path: str) -> str:
    ext = get_extension(file_path)
    if ext in (".txt", ".md"):
        return "text"
    if ext in (".json", ".jsonl"):
        return "json"
    if ext in (".arrow",):
        return "arrow"
    if ext in (".parquet",):
        return "parquet"
    return ext


def load_yaml_or_json(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        ext = get_extension(file_path)
        if ext in (".yaml", ".yml"):
            return yaml.safe_load(f)
        if ext == ".json":
            return json.load(f)
    return None


def resolve_iterable_dataset_features(data: IterableDataset):
    if data.column_names is None:
        if isinstance(data, IterableDataset):
            if hasattr(data, "_resolve_features"):
                data = data._resolve_features()
            else:
                raise ValueError(
                    "_resolve_features API is not available to fetch column names"
                )
        else:
            raise ValueError(
                f"not possible to fetch column names for the loaded dataset of type {type(data)}"
            )
    return data


def get_dataset_text_field_config(data_config):
    dataset_text_fields = []

    # data_config.datasets will be a List[DataSetConfig]
    for dataset in data_config.datasets or []:
        # data_handlers is Optional[List[DataHandlerConfig]]
        if not dataset.data_handlers:
            continue
        for handler in dataset.data_handlers:
            # handler.arguments is Optional[Dict]. Handle if it's None
            fn_kwargs = (
                handler.arguments.get("fn_kwargs", {}) if handler.arguments else {}
            )
            dataset_text_field = fn_kwargs.get("dataset_text_field")
            if dataset_text_field:
                dataset_text_fields.append(dataset_text_field)

    # If multiple unique fields, log a warning
    if len(set(dataset_text_fields)) > 1:
        logger.warning(
            "Multiple dataset_text_field values found in data_config. "
            "Only the first value will be used: %s",
            dataset_text_fields[0],
        )

    # Return the first field if any, else None
    return dataset_text_fields[0] if dataset_text_fields else None


def validate_mergeable_datasets(datasets):
    """Given list of datasets, validate if all datasets have same type and number of columns."""
    if len(datasets) > 1:
        ref_columns = datasets[0].features
        ref_column_names = list(ref_columns.keys())
        ref_column_types = {col: feat.dtype for col, feat in ref_columns.items()}

        # Check all other datasets
        for i, ds in enumerate(datasets[1:], start=2):
            ds_column_names = list(ds.features.keys())
            ds_column_types = {col: feat.dtype for col, feat in ds.features.items()}

            # Check same set of columns
            if set(ds_column_names) != set(ref_column_names):
                logger.warning(
                    "Dataset %d has different columns: %s. Columns in Dataset 1: %s",
                    i,
                    ds_column_names,
                    ref_column_names,
                )

            # Check column data types
            for col in ref_column_names:
                if (col in ds_column_types) and (
                    ds_column_types[col] != ref_column_types[col]
                ):
                    logger.warning(
                        "Column '%s' in dataset %d has type %s, expected %s",
                        col,
                        i,
                        ds_column_types[col],
                        ref_column_types[col],
                    )
