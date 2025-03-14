# Standard
import logging
import os
import sys
import traceback

# Third Party
from transformers import (
    AutoTokenizer,
    GPT2Tokenizer,
    GPTNeoXTokenizerFast,
    LlamaTokenizer,
    LlamaTokenizerFast,
)

# Local
from tuning.config import configs
from tuning.data.setup_dataprocessor import process_dataargs
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
    for shard_idx in range(num_shards):
        shard = dataset.shard(index=shard_idx, num_shards=num_shards)
        shard_path = os.path.join(output_dir, f"ds_{shard_idx:05d}.parquet")
        shard.to_parquet(shard_path)
    logging.info("Dumped %d shards of %s at %s", num_shards, dataset_name, output_dir)


def get_processed_dataset(
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
    train_args, logger = set_log_level(train_args, "get_processed_dataset")

    logger.info(
        "Starting dataset processing with model_args: %s, data_args: %s, training_args: %s",
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

    # Add chat_template to the tokenizer if provided
    if data_args.chat_template:
        data_args.chat_template = data_args.chat_template.replace(r"\n", "\n")

        logger.info("Adding chat_template to the tokenizer")
        if tokenizer.chat_template:
            logger.warning(
                "replacing existing chat_template %s with the given chat_template %s",
                tokenizer.chat_template,
                data_args.chat_template,
            )
        tokenizer.chat_template = data_args.chat_template

    # Prepare special tokens dictionary
    special_tokens_dict = {}
    if not model_args.tokenizer_name_or_path:
        if isinstance(tokenizer, (LlamaTokenizer, LlamaTokenizerFast)):
            special_tokens_dict["bos_token"] = "<s>"
            special_tokens_dict["eos_token"] = "</s>"
            special_tokens_dict["unk_token"] = "<unk>"
            special_tokens_dict["pad_token"] = "<pad>"
        elif isinstance(tokenizer, (GPT2Tokenizer, GPTNeoXTokenizerFast)):
            special_tokens_dict["pad_token"] = "<pad>"

        if tokenizer.pad_token is None:
            logger.warning(
                "PAD token not found in tokenizer; setting PAD token to default."
            )
            special_tokens_dict["pad_token"] = configs.DEFAULT_PAD_TOKEN
        if tokenizer.eos_token is None:
            logger.warning(
                "EOS token not found in tokenizer; setting EOS token to default."
            )
            special_tokens_dict["eos_token"] = configs.DEFAULT_EOS_TOKEN
        if tokenizer.pad_token == tokenizer.eos_token:
            logger.warning(
                "PAD token and EOS token are the same. Overriding accordingly."
            )
            if tokenizer.eos_token != configs.DEFAULT_PAD_TOKEN:
                tokenizer.pad_token = configs.DEFAULT_PAD_TOKEN
                special_tokens_dict["pad_token"] = configs.DEFAULT_PAD_TOKEN
            else:
                tokenizer.eos_token = configs.DEFAULT_EOS_TOKEN
                special_tokens_dict["eos_token"] = configs.DEFAULT_EOS_TOKEN

    # adds user specified special tokens to vocab
    if data_args.add_special_tokens:
        logger.info(
            "Adding user-defined special tokens: %s ", data_args.add_special_tokens
        )
        special_tokens_dict["additional_special_tokens"] = data_args.add_special_tokens

    if special_tokens_dict:
        logger.info("Adding special tokens: %s", special_tokens_dict)
        tokenizer.add_special_tokens(special_tokens_dict)

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
        formatted_train_dataset, formatted_validation_dataset = get_processed_dataset(
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
    if formatted_train_dataset is not None:
        save_dataset_shards(
            formatted_train_dataset,
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
    if formatted_validation_dataset is not None:
        save_dataset_shards(
            formatted_validation_dataset,
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
