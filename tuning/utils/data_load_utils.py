# Standard
from typing import Union
import os

# Third Party
import datasets


def load_dataset(data_path: str) -> Union[datasets.Dataset, None]:
    """loads datasets given as either a file or a directory

    Args:
        data_path (str): path to the dataset file or directory

    Returns:
        datasets.Dataset: loaded dataset
    """
    if not data_path:
        return None

    if os.path.isdir(data_path):
        return datasets.load_dataset(path=data_path, split="train")
    return datasets.load_dataset(
        path=os.path.dirname(data_path), data_files=data_path, split="train"
    )
