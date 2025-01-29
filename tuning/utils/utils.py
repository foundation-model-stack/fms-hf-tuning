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
from datasets import Dataset, IterableDataset
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


def get_iterable_dataset_schema(dataset):
    """Infer column names and types from the first sample of an IterableDataset."""
    try:
        first_example = next(iter(dataset))
        if not isinstance(first_example, dict):
            raise ValueError("IterableDataset does not yield dictionaries.")
        column_names = list(first_example.keys())
        column_types = {col: type(first_example[col]).__name__ for col in first_example}
        return column_names, column_types
    except StopIteration:
        raise ValueError("IterableDataset is empty and cannot be validated.")


def validate_mergeable_datasets(datasets):
    """Given list of datasets, validate if all datasets have same type and number of columns."""
    if len(datasets) <= 1:
        return

    # Determine reference dataset type
    first_ds = datasets[0]
    if isinstance(first_ds, Dataset):
        ref_column_names = list(first_ds.features.keys())
        ref_column_types = {col: feat.dtype for col, feat in first_ds.features.items()}
    elif isinstance(first_ds, IterableDataset):
        ref_column_names, ref_column_types = get_iterable_dataset_schema(first_ds)
    else:
        raise TypeError("Unsupported dataset type")

    # Check all other datasets
    for i, ds in enumerate(datasets[1:], start=2):
        if isinstance(ds, Dataset):
            ds_column_names = list(ds.features.keys())
            ds_column_types = {col: feat.dtype for col, feat in ds.features.items()}
        elif isinstance(ds, IterableDataset):
            ds_column_names, ds_column_types = get_iterable_dataset_schema(ds)
        else:
            raise TypeError(f"Dataset {i} has an unsupported type: {type(ds)}")

        # Check same set of columns
        if set(ds_column_names) != set(ref_column_names):
            logger.warning(
                "Dataset %d has different columns: %s. Expected columns: %s",
                i,
                ds_column_names,
                ref_column_names,
            )

        # Check column data types
        for col in ref_column_names:
            if (col in ds_column_types) and (ds_column_types[col] != ref_column_types[col]):
                logger.warning(
                    "Column '%s' in dataset %d has type %s, expected %s",
                    col,
                    i,
                    ds_column_types[col],
                    ref_column_types[col],
                )
