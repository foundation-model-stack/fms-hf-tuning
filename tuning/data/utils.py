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
from typing import List, Union
import io
import json
import logging
import os

# Third Party
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    IterableDataset,
    IterableDatasetDict,
    concatenate_datasets,
)
from PIL import Image
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


def __get_dataset_features(d, default_split: str = "train") -> Features:
    return (
        d[default_split].features
        if isinstance(d, (DatasetDict, IterableDatasetDict))
        else d.features
    )


def _maybe_cast_columns(datasets: list, default_split: str = "train") -> None:
    """
    Given list of datasets, try casting datasets to same features.
    Assumes that the datasets are aligned in terms of columns which
    could be ensure by calling validate_mergeable_datasets
    """
    if len(datasets) <= 1:
        return

    # pick the first dataset as the reference
    features = __get_dataset_features(datasets[0], default_split)

    # Cast remaining datasets according to this
    for i in range(1, len(datasets)):
        datasets[i] = datasets[i].cast(features)


def _validate_mergeable_datasets(datasets: list, default_split: str = "train") -> None:
    """Given list of datasets, validate if all datasets have same type and number of columns."""
    if len(datasets) <= 1:
        return

    ref_columns = __get_dataset_features(datasets[0], default_split)
    ref_column_names = list(ref_columns.keys())

    # Check all other datasets
    mismatching_ds = []
    for _, ds in enumerate(datasets[1:], start=1):
        ds_features = __get_dataset_features(ds, default_split)
        ds_column_names = list(ds_features.keys())

        # Check same set of columns
        if set(ds_column_names) != set(ref_column_names):
            mismatching_ds.append([ds])

    if len(mismatching_ds) > 0:
        raise ValueError(
            "Datasets passed should have same column names. "
            + "Found {} datasets with mismatching column names".format(
                len(mismatching_ds)
            ),
        )


def maybe_align_datasets(datasets: list) -> None:
    """
    Given list of datasets
     1. validate if all datasets have same type and number of columns.
     2. try casting dataset columns to same value to ensure mergability
    """
    try:
        for i, d in enumerate(datasets):
            if isinstance(d, IterableDataset):
                datasets[i] = resolve_iterable_dataset_features(d)

        _validate_mergeable_datasets(datasets)
        _maybe_cast_columns(datasets)
    except Exception as e:  # pylint: disable=broad-exception-raised
        raise ValueError("Failed to align datasets " + str(datasets)) from e


def try_convert_bytes_dict_to_pil(image):
    """
    Convert image data (in various shapes) where the data may be stored as:
    1) A list of lists of dicts containing bytes,
    2) A list of dicts containing bytes,
    3) A single dict containing bytes.

    Args:
        image (Union[list[list[dict]], list[dict], dict]):
            The input image data to be converted. Each dict should contain a "bytes" key.

    Returns:
        Union[list[list[Image.Image]], list[Image.Image], Image.Image]:
            The converted image data as PIL Image objects, maintaining the original structure.
    """
    # Case 1: List of lists of dicts
    if image and isinstance(image, list) and isinstance(image[0], list):
        # We have something like [[{bytes: ...}, {bytes: ...}], [{bytes: ...}]]
        for i, sub_list in enumerate(image):
            for j, item in enumerate(sub_list):
                if isinstance(item, dict) and "bytes" in item:
                    pil_image = Image.open(io.BytesIO(item["bytes"]))
                    image[i][j] = pil_image

    # Case 2: List of dicts
    elif image and isinstance(image, list) and isinstance(image[0], dict):
        # We have something like [{bytes: ...}, {bytes: ...}, ...]
        for i, item in enumerate(image):
            if "bytes" in item:
                pil_image = Image.open(io.BytesIO(item["bytes"]))
                image[i] = pil_image

    # Case 3: Single dict
    elif isinstance(image, dict):
        # We have a single dict {bytes: ...}
        if "bytes" in image:
            image = Image.open(io.BytesIO(image["bytes"]))

    return image


def try_convert_image_to_rgb(image):
    """
    Converts image data to RGB format if it is not already in RGB mode.
    The input image data can be in one of the following formats:

    1. A list of lists of PIL Image objects.
    2. A list of PIL Image objects.
    3. A single PIL Image object.

    Args:
        image (Union[list[list[Image.Image]], list[Image.Image], Image.Image]):
            The input image data to be converted.

    Returns:
        Union[list[list[Image.Image]], list[Image.Image], Image.Image]:
            The image data converted to RGB format, maintaining the original structure.
    """
    # Case 1: List of lists of PIL images
    if image and isinstance(image, list) and isinstance(image[0], list):
        image = [
            [_img.convert("RGB") if _img.mode != "RGB" else _img for _img in img]
            for img in image
        ]
    # Case 2: List of PIL images
    elif isinstance(image, list) and image and isinstance(image[0], Image.Image):
        image = [img.convert("RGB") if img.mode != "RGB" else img for img in image]
    # Case 3: Single PIL image
    elif image and isinstance(image, Image.Image):
        image = image.convert("RGB") if image.mode != "RGB" else image

    return image


def _concatenate_datasets(
    all_datasets: List[Union[Dataset, IterableDataset]]
) -> Union[Dataset, IterableDataset]:
    """
    Concatenates a list of Dataset or IterableDataset objects into one.
    Aligns datasets before concatenation and resolves features if needed
    for IterableDataset.
    """
    if len(all_datasets) == 1:
        return all_datasets[0]
    maybe_align_datasets(all_datasets)
    concatenated = concatenate_datasets(all_datasets)
    if isinstance(concatenated, IterableDataset):
        concatenated = resolve_iterable_dataset_features(concatenated)
    return concatenated


def try_concatenate_datasets(
    all_datasets: List[
        Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]
    ],
) -> Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]:
    """
    Attempts to concatenate a list of datasets into a unified structure.
    Supports both flat and dict-style datasets.

    This function handles:
    - Flat datasets (`Dataset` or `IterableDataset`):
        - Concatenates all datasets directly using row-wise alignment.
    - Dict-style datasets (`DatasetDict` or `IterableDatasetDict`):
        - Concatenates keys that appear in more than one dictionary.
        - Preserves keys that are unique to a single dictionary.
        - Returns a new dictionary-style dataset (`DatasetDict` or `IterableDatasetDict`).

    Args:
        all_datasets (List[Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]]):
            A list of datasets to concatenate. Must be homogeneous in type
            (all flat or all dict-style, and all either streaming or non-streaming).

    Returns:
        Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]:
            Single concatenated dataset.
            Return type matches the input structure and streaming mode.

    Raises:
        ValueError: If datasets are of mixed or incompatible types, or if an error occurs during
        concatenation.
    """
    if len(all_datasets) == 1:
        return all_datasets[0]

    try:
        # Case 1: Flat datasets
        if all(isinstance(d, (Dataset, IterableDataset)) for d in all_datasets):
            return _concatenate_datasets(all_datasets)

        # Case 2: Dict-style datasets
        if all(isinstance(d, (DatasetDict, IterableDatasetDict)) for d in all_datasets):
            unique_keys = set(key for d in all_datasets for key in d.keys())
            merged_dict = {}
            for key in unique_keys:
                to_concat = [d[key] for d in all_datasets if key in d]
                merged_dict[key] = _concatenate_datasets(to_concat)

            if all(isinstance(d, IterableDatasetDict) for d in all_datasets):
                return IterableDatasetDict(merged_dict)
            return DatasetDict(merged_dict)

        raise ValueError(
            f"Cannot concatenate mixed types of datasets: {[type(d) for d in all_datasets]}"
        )

    except Exception as e:
        raise ValueError(f"An error occurred while concatenating datasets: {e}") from e
