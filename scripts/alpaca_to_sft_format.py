# Standard
import argparse
import os

# Third Party
import datasets

# Prompt template to be used by default
SIMPLE_PROMPT = "Input:\n{input}\n\n### Response:"

# Prompt template to be used if --verbose is provided
VERBOSE_PROMPT_INPUT = (
    # pylint: disable=line-too-long
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
)
VERBOSE_PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


def parse_args() -> argparse.Namespace:
    """Parse the arguments and ensure everything is valid.

    Returns:
        argparse.Namespace
            Parsed arguments object.
    """
    parser = argparse.ArgumentParser(
        description="Converts Alpaca formatted data files into SFT formatted files."
    )
    parser.add_argument(
        "--verbose",
        help="Indicates whether or not the verbose format should be used.",
        action="store_true",
    )
    parser.add_argument(
        "--files",
        help="Alpaca formatted files to be converted to SFT format.",
        nargs="+",
        required=True,
    )
    return parser.parse_args()


def format_alpaca_fn_simple(example: str) -> dict[str, str]:
    """Format a single example using the simple template format.

    Args:
        example: str
            Example to be formatted.

    Returns:
        dict[str, str]
            Dictionary containing the formatted example.
    """
    output = SIMPLE_PROMPT.format_map(example)
    output = f"{output} {example['output']}"
    return {"output": output}


def format_alpaca_fn_verbose(example: str) -> dict[str, str]:
    """Format a single example using the verbose template format.

    Args:
        example: str
            Example to be formatted.

    Returns:
        dict[str, str]
            Dictionary containing the formatted example.
    """
    output = (
        VERBOSE_PROMPT_INPUT.format_map(example)
        if example.get("input", "") != ""
        else VERBOSE_PROMPT_NO_INPUT.format_map(example)
    )
    output = f"{output} {example['output']}"
    return {"output": output}


if __name__ == "__main__":
    parsed_args = parse_args()
    format_alpaca_fn = (
        format_alpaca_fn_verbose if parsed_args.verbose else format_alpaca_fn_simple
    )

    for file_path in parsed_args.files:
        if not os.path.isfile(file_path):
            raise ValueError(f"alpaca dataset f{file_path} does not exist!")
        base_path, file_name = os.path.split(file_path)
        export_path = os.path.join(base_path, f"sft_format_{file_name}")
        print(
            f"Converting alpaca format file: {file_path} to SFT trainer training format"
        )
        ds = datasets.load_dataset("json", data_files=file_path)
        alpaca_ds = ds["train"].map(
            format_alpaca_fn, remove_columns=["instruction", "input"]
        )
        alpaca_ds.to_json(export_path)
        print(f"Exported SFT format data to {export_path}")
