# Standard
import logging
import os
import sys
import traceback

# Third Party
from transformers import AutoTokenizer

# Local
from tuning.config import configs
from tuning.data.setup_dataprocessor import process_dataargs
from tuning.data.tokenizer_utils import setup_tokenizer
from tuning.sft_trainer import get_parser
from tuning.utils.error_logging import USER_ERROR_EXIT_CODE, write_termination_log
from tuning.utils.logging import set_log_level


def save_dataset_shards(
    dataset, output_dir: str, num_shards: int, dataset_name: str
) -> None:
    """
    Saves the given dataset in the specified number of shards.

    Args:
        dataset: The dataset to shard and save.
        output_dir (str): Directory to save the dataset shards.
        num_shards (int): Number of shards to create.
        dataset_name (str): Name of the dataset (used for logging).
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.info(
        "Dumping processesd dataaset %s at %s in %d shards",
        dataset_name,
        output_dir,
        num_shards,
    )
    for shard_idx in range(num_shards):
        shard = dataset.shard(index=shard_idx, num_shards=num_shards)
        shard_path = os.path.join(output_dir, f"ds_{shard_idx:05d}.parquet")
        shard.to_parquet(shard_path)
    logging.info("Dumped %d shards", num_shards)


def process_datasets_offline(
    model_args: configs.ModelArguments,
    data_args: configs.DataArguments,
    train_args: configs.TrainingArguments,
):
    """
    Processes the dataset based on data config yaml.

    Args:
        model_args (configs.ModelArguments): Model configuration arguments.
        data_args (configs.DataArguments): Data configuration arguments.
        train_args (configs.TrainingArguments): Training configuration arguments.

    Returns:
        tuple: A tuple containing the formatted training dataset and validation dataset.
    """
    # Set log level for this function
    train_args, logger = set_log_level(train_args, "process_datasets_offline")

    logger.info(
        "Starting offline dataset processing with \n\
         model_args: %s, \n\
         data_args: %s, \n\
         training_args: %s",
        model_args,
        data_args,
        train_args,
    )

    # Load tokenizer for the model
    tokenizer_path = model_args.tokenizer_name_or_path or model_args.model_name_or_path
    logger.debug("Loading tokenizer from %s", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        cache_dir=train_args.cache_dir,
        use_fast=True,
        legacy=True,
    )
    logger.debug("Tokenizer loaded successfully.")

    _ = setup_tokenizer(tokenizer, data_args, model_args, None)

    # Process data using the provided arguments and tokenizer
    logger.info("Calling process_dataargs to format datasets.")
    (
        formatted_train_dataset,
        formatted_validation_dataset,
        _,
        _,
        _,
        _,
    ) = process_dataargs(data_args, tokenizer, train_args)
    logger.info("Dataset processing completed successfully.")

    formatted_train_dataset = formatted_train_dataset.flatten_indices()
    if formatted_validation_dataset:
        formatted_validation_dataset = formatted_validation_dataset.flatten_indices()

    return formatted_train_dataset, formatted_validation_dataset


def main():
    """
    Main function that parses arguments, processes datasets, and saves the output.
    """
    logger = logging.getLogger()
    logger.info("Starting Data Processing script execution.")

    parser = get_parser()
    parser.add_argument(
        "--num_dataset_shards",
        type=int,
        default=1,
        help="Number of shards to be used for saving the dataset.",
    )

    try:
        parsed_output = parser.parse_args_into_dataclasses()
        # Extract arguments based on type
        arg_types = {
            configs.ModelArguments: "model_args",
            configs.DataArguments: "data_args",
            configs.TrainingArguments: "training_args",
        }
        args = {key: None for key in arg_types.values()}
        for item in parsed_output:
            for arg_class, key in arg_types.items():
                if isinstance(item, arg_class):
                    args[key] = item

        # Extract additional namespace argument
        num_dataset_shards = next(
            (
                item.num_dataset_shards
                for item in parsed_output
                if hasattr(item, "num_dataset_shards")
            ),
            1,
        )

        if None in args.values():
            raise ValueError(
                "One of the arguments is None. Please check the arguments passed."
            )

        logger.debug(
            "Input args parsed:\n model_args: %s\n data_args: %s\n training_args: %s\n Shards: %d",
            args["model_args"],
            args["data_args"],
            args["training_args"],
            num_dataset_shards,
        )
        args["training_args"], logger = set_log_level(args["training_args"], __name__)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error parsing arguments: %s", traceback.format_exc())
        write_termination_log(f"Exception raised during argument parsing: {e}")
        sys.exit(USER_ERROR_EXIT_CODE)

    try:
        logger.info("Processing dataset.")
        train_dataset, validation_dataset = process_datasets_offline(
            model_args=args["model_args"],
            data_args=args["data_args"],
            train_args=args["training_args"],
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error processing dataset: %s", traceback.format_exc())
        write_termination_log(f"Exception raised during dataset processing: {e}")
        sys.exit(USER_ERROR_EXIT_CODE)

    # Save train dataset shards
    train_dataset_dir = os.path.join(args["training_args"].output_dir, "train_dataset")
    logging.info(
        "Trying to dump %d shards of train dataset at %s",
        num_dataset_shards,
        train_dataset_dir,
    )
    if train_dataset is not None:
        save_dataset_shards(
            train_dataset,
            train_dataset_dir,
            num_dataset_shards,
            "train_dataset",
        )
    else:
        logging.warning("Train dataset is None. Not saving train dataset.")

    # Save validation dataset shards
    validation_dataset_dir = os.path.join(
        args["training_args"].output_dir, "validation_dataset"
    )
    logging.info(
        "Trying to dump %d shards of validation dataset at %s",
        num_dataset_shards,
        validation_dataset_dir,
    )
    if validation_dataset is not None:
        save_dataset_shards(
            validation_dataset,
            validation_dataset_dir,
            num_dataset_shards,
            "validation_dataset",
        )
    else:
        logging.warning("Validation dataset is None. Not saving validation dataset.")

    logger.info(
        "Data Processing script execution completed. Data saved in %s directory",
        args["training_args"].output_dir,
    )


if __name__ == "__main__":
    main()
