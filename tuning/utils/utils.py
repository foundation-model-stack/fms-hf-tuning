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
import io
import json
import logging
import os

# Third Party
from datasets import IterableDataset
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
