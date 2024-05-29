"""Pulls and formats data into alpaca format. Not that each specific dataset needs its own
formatter, since it is dependent on the format of the raw dataset being processed.
"""

# Standard
from shutil import rmtree
from typing import Any, Optional
import json
import os
import random

# Third Party
import boto3

S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY")
S3_ENDPOINT = os.getenv("S3_ENDPOINT")
if S3_ACCESS_KEY_ID is None or S3_SECRET_ACCESS_KEY is None or S3_ENDPOINT is None:
    raise ValueError(
        "Error - must set env vars: S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, and S3_ENDPOINT"
    )

##### data formatters
def format_and_export_cc_tone_file(
    file_path: str, export_path: str, num_samples: Optional[int]
):
    """Formats the tone dataset by comma separated labels in the output.

    Args:
        file_path: str
            Path to tone file to be formatted.
        export_path: str
            Path to export the formatted data to.
        num_samples: Optional[int]
            Number of samples to be included in the formatted file.
    """
    with open(file_path, "r", encoding="utf-8") as tone_file:
        data = [json.loads(line) for line in tone_file.readlines() if line]

    export_path = export_path.split(".")[0] + ".json"
    if num_samples:
        data = random.sample(data, num_samples)
        # Update the file name to prepend num samples
        base_path, export_name = os.path.split(export_path)
        export_path = os.path.join(base_path, f"{num_samples}_{export_name}")

    formatted_data = [
        {
            "instruction": "",
            "input": datum["text"],
            "output": ", ".join(datum["labels"]),
        }
        for datum in data
    ]
    with open(export_path, "w", encoding="utf-8") as export_file:
        json.dump(formatted_data, export_file, sort_keys=True, indent=4)


def format_and_export_entities_file(
    file_path: str, export_path: str, num_samples: Optional[int]
):
    """Formats the entites/TSA datasets by setting the output to literal "None"
    if no target is extracted, and a comma separated list in the format
                                {text} : {type}
    for each extracted object.
    Example for entities: "waitress: JobTitle"
    Example for TSA: "waitress: positive"

    Args:
        file_path: str
            Path to tone file to be formatted.
        export_path: str
            Path to export the formatted data to.
        num_samples: Optional[int]
            Number of samples to be included in the formatted file.
    """

    def get_entites_output_text(datum):
        mentions = datum["mentions"]
        if not mentions:
            return "None"
        mention_strs = [
            f"{mention['text']}: {mention['type']}".replace(",", "\\,")
            for mention in mentions
        ]
        return ", ".join(mention_strs)

    with open(file_path, "r", encoding="utf-8") as entities_file:
        data = json.load(entities_file)
    if num_samples:
        data = random.sample(data, num_samples)
        # Update the file name to prepend num samples
        base_path, export_name = os.path.split(export_path)
        export_path = os.path.join(base_path, f"{num_samples}_{export_name}")

    formatted_data = [
        {
            "instruction": "",
            "input": datum["text"],
            "output": get_entites_output_text(datum),
        }
        for datum in data
    ]
    with open(export_path, "w", encoding="utf-8") as export_file:
        json.dump(formatted_data, export_file, sort_keys=True, indent=4)


# Where we will put the downloaded data
DOWNLOAD_DIR = "unformatted_data"
# Where we will put the formatted data files to
EXPORT_DIR = "formatted_data"

COS_LOCATION_KEY = "cos_location"
FORMAT_FUNC_KEY = "format_func"
SUBSAMPLE_KEY = "subsample_info"
DATASET_INFOS = [
    {
        COS_LOCATION_KEY: "fm-validation-staging-models-and-datasets/datasets/unitxt/cc_tone",
        FORMAT_FUNC_KEY: format_and_export_cc_tone_file,
        SUBSAMPLE_KEY: {
            "train.jsonl": 1000,
            "test.jsonl": 500,
        },
    },
    {
        # pylint: disable=line-too-long
        COS_LOCATION_KEY: "fm-validation-staging-models-and-datasets/datasets/unitxt/en/Extraction/Entities",
        FORMAT_FUNC_KEY: format_and_export_entities_file,
        SUBSAMPLE_KEY: {
            "IBM_NET_2019_DELIVERABLE_VERSION_C_train.json": 1000,
            "IBM_NET_2019_DELIVERABLE_VERSION_C_test.json": 500,
        },
    },
    {
        COS_LOCATION_KEY: "fm-validation-staging-models-and-datasets/datasets/unitxt/tsa_mams",
        FORMAT_FUNC_KEY: format_and_export_entities_file,
        SUBSAMPLE_KEY: {
            "train.json": 1000,
            "test.json": 500,
        },
    },
]


def create_data_dirs():
    """Create the directories to contain formatted/unformatted data."""
    print("Creating data directories...")
    if os.path.exists(DOWNLOAD_DIR):
        rmtree(DOWNLOAD_DIR)
    if os.path.exists(EXPORT_DIR):
        rmtree(EXPORT_DIR)
    os.mkdir(DOWNLOAD_DIR)
    os.mkdir(EXPORT_DIR)


def download_datasets(dataset_infos: list[dict[str, Any]]):
    """Download the datasets to local disk.

    Args:
        dataset_infos: list[dict[str, Any]]
            Structure containing information about each dataset we need to download
            and where it lives in the connected S3 instance.
    """
    s3 = boto3.resource(
        "s3",
        aws_access_key_id=S3_ACCESS_KEY_ID,
        aws_secret_access_key=S3_SECRET_ACCESS_KEY,
        endpoint_url=S3_ENDPOINT,
    )
    for dataset_info in dataset_infos:
        first_slash_idx = dataset_info[COS_LOCATION_KEY].find("/")
        bucket_name = dataset_info[COS_LOCATION_KEY][:first_slash_idx]
        cos_path = dataset_info[COS_LOCATION_KEY][first_slash_idx + 1 :]
        # Make the subdir to download files into...
        download_subdir = os.path.join(DOWNLOAD_DIR, cos_path.split(os.sep)[-1])
        os.mkdir(download_subdir)
        print(f"Downloading files for {download_subdir}")
        # Download the unformatted data files...
        bucket = s3.Bucket(bucket_name)
        for obj in bucket.objects.filter(Prefix=cos_path):
            if obj.key[-1] == "/":
                continue
            target_path = os.path.join(download_subdir, obj.key.split("/")[-1])
            bucket.download_file(obj.key, target_path)


def apply_data_formatters(dataset_infos: list[dict[str, Any]]):
    """Formats each of the downloaded datasets.

    Args:
        dataset_infos: list[dict[str, Any]]
            Structure containing information about each dataset we need to format.
    """
    for dataset_info in dataset_infos:
        subdir = dataset_info[COS_LOCATION_KEY].split(os.sep)[-1]
        download_subdir = os.path.join(DOWNLOAD_DIR, subdir)
        # Make the dir to export files to
        export_subdir = os.path.join(EXPORT_DIR, subdir)
        os.mkdir(export_subdir)

        data_files = [
            os.path.join(download_subdir, filename)
            for filename in os.listdir(download_subdir)
        ]
        for data_file in data_files:
            # Apply this datasets formatter to the data file
            export_path = os.path.join(
                EXPORT_DIR, data_file[data_file.index(os.sep) + 1 :]
            )
            dataset_info[SUBSAMPLE_KEY]
            # This is silly :)
            filename = os.path.split(data_file)[-1]
            num_samples = (
                dataset_info[SUBSAMPLE_KEY][filename]
                if filename in dataset_info[SUBSAMPLE_KEY]
                else None
            )
            if num_samples:
                print(f"--> File: {data_file} will be subsampled to {num_samples}")
                # make sure the split is deterministic
                random.seed(42)
            dataset_info[FORMAT_FUNC_KEY](data_file, export_path, num_samples)


if __name__ == "__main__":
    create_data_dirs()
    download_datasets(DATASET_INFOS)
    apply_data_formatters(DATASET_INFOS)
