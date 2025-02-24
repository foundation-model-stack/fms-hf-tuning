# Standard
from typing import Callable, Dict, Optional
import logging
import os
import sys
import traceback

# Third Party
from transformers import AutoTokenizer, LlamaTokenizer, LlamaTokenizerFast

# Local
from tuning.config import configs
from tuning.data.setup_dataprocessor import process_dataargs
from tuning.sft_trainer import get_parser
from tuning.utils.error_logging import USER_ERROR_EXIT_CODE, write_termination_log
from tuning.utils.logging import set_log_level


def get_processed_dataset(
    model_args: configs.ModelArguments,
    data_args: configs.DataArguments,
    train_args: configs.TrainingArguments,
    additional_data_handlers: Optional[Dict[str, Callable]] = None,
):
    """Process dataset based on config yaml

    Args:
        model_args: tuning.config.configs.ModelArguments
        data_args: tuning.config.configs.DataArguments
        train_args: tuning.config.configs.TrainingArguments
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
    tokenizer_path = (
        model_args.tokenizer_name_or_path
        if model_args.tokenizer_name_or_path
        else model_args.model_name_or_path
    )
    logger.debug("Loading tokenizer from %s", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        cache_dir=train_args.cache_dir,
        use_fast=True,
        legacy=True,
    )
    logger.debug("Tokenizer loaded successfully.")

    # Add chat_template to the tokenizer
    if data_args.chat_template:
        logger.info("Adding chat_template to the tokenizer.")
        if tokenizer.chat_template:
            logger.warning(
                "Replacing existing chat_template %s with the given chat_template %s",
                tokenizer.chat_template,
                data_args.chat_template,
            )
        tokenizer.chat_template = data_args.chat_template

    # Prepare special tokens dictionary
    special_tokens_dict = {}
    if not model_args.tokenizer_name_or_path:
        if isinstance(tokenizer, (LlamaTokenizer, LlamaTokenizerFast)):
            logger.debug(
                "Using a Llama tokenizer—setting eos_token to </s> by default."
            )
            special_tokens_dict["eos_token"] = "</s>"

    if not model_args.tokenizer_name_or_path:
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
                "PAD token and EOS token are currently the same. Overriding PAD token or EOS token."
            )
            if tokenizer.eos_token != configs.DEFAULT_PAD_TOKEN:
                tokenizer.pad_token = configs.DEFAULT_PAD_TOKEN
                special_tokens_dict["pad_token"] = configs.DEFAULT_PAD_TOKEN
            else:
                tokenizer.eos_token = configs.DEFAULT_EOS_TOKEN
                special_tokens_dict["eos_token"] = configs.DEFAULT_EOS_TOKEN

    if special_tokens_dict:
        logger.info("Adding special tokens: %s", special_tokens_dict)
        tokenizer.add_special_tokens(special_tokens_dict)

    # Process data with the provided arguments and tokenizer
    logger.info("Calling process_dataargs to format datasets.")
    (
        formatted_train_dataset,
        formatted_validation_dataset,
        _,
        _,
        _,
        _,
    ) = process_dataargs(
        data_args,
        tokenizer,
        train_args,
        additional_data_handlers,
    )

    logger.info("Dataset processing completed successfully.")
    return formatted_train_dataset, formatted_validation_dataset


def main():
    """Main function to parse arguments and process datasets."""
    logger = logging.getLogger()
    logger.info("Starting Data Processing script execution.")

    parser = get_parser()
    parser.add_argument(
        "--num_datasets_shard",
        type=int,
        default=1,
        help="Number of shards to be used for saving the dataset.",
    )

    # Parse arguments and set log level
    try:
        parsed_output = parser.parse_args_into_dataclasses()
        model_args = next(
            (
                item
                for item in parsed_output
                if isinstance(item, configs.ModelArguments)
            ),
            None,
        )
        data_args = next(
            (item for item in parsed_output if isinstance(item, configs.DataArguments)),
            None,
        )
        training_args = next(
            (
                item
                for item in parsed_output
                if isinstance(item, configs.TrainingArguments)
            ),
            None,
        )
        namespace_args = next(
            (item for item in parsed_output if hasattr(item, "num_datasets_shard")),
            None,
        )
        num_datasets_shard = namespace_args.num_datasets_shard if namespace_args else 1

        # If any of the arguments are None, raise an error
        if None in [model_args, data_args, training_args]:
            raise ValueError(
                "One of the arguments is None. Please check the arguments passed."
            )

        logger.debug(
            "Input args parsed:\n"
            "  model_args: %s\n"
            "  data_args: %s\n"
            "  training_args: %s\n",
            model_args,
            data_args,
            training_args,
            num_datasets_shard,
        )
        training_args, logger = set_log_level(training_args, __name__)
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error parsing arguments: %s", traceback.format_exc())
        write_termination_log(
            f"Exception raised during argument parsing. This may be a problem with your input: {e}"
        )
        sys.exit(USER_ERROR_EXIT_CODE)

    # Process dataset
    try:
        logger.info("Processing dataset with get_processed_dataset.")
        formatted_train_dataset, formatted_validation_dataset = get_processed_dataset(
            model_args=model_args,
            data_args=data_args,
            train_args=training_args,
        )
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error processing dataset: %s", traceback.format_exc())
        write_termination_log(
            f"Exception raised during dataset processing.\
            This may be a problem with your input: {e}"
        )
        sys.exit(USER_ERROR_EXIT_CODE)

    # Save train dataset
    train_dataset_dir = os.path.join(training_args.output_dir, "train_dataset")

    logging.info(
        "Trying to dump %d shards of train dataset at %s",
        num_datasets_shard,
        train_dataset_dir,
    )
    if formatted_train_dataset is not None:
        os.makedirs(train_dataset_dir, exist_ok=True)
        for shard_idx in range(num_datasets_shard):
            shard = formatted_train_dataset.shard(
                index=shard_idx, num_shards=num_datasets_shard
            )
            shard.to_parquet(f"{train_dataset_dir}/ds_{shard_idx:05d}.parquet")
        logging.info(
            "Dumped %d shards of train_dataset at %s",
            num_datasets_shard,
            train_dataset_dir,
        )
    else:
        logging.warning("Train dataset is None. Not saving train dataset.")

    # Save validation dataset
    validation_dataset_dir = os.path.join(
        training_args.output_dir, "validation_dataset"
    )
    logging.info(
        "Trying to dump %d shards of validation dataset at %s",
        num_datasets_shard,
        validation_dataset_dir,
    )
    if formatted_validation_dataset is not None:
        os.makedirs(validation_dataset_dir, exist_ok=True)
        for shard_idx in range(num_datasets_shard):
            shard = formatted_validation_dataset.shard(
                index=shard_idx, num_shards=num_datasets_shard
            )
            shard.to_parquet(f"{validation_dataset_dir}/ds_{shard_idx:05d}.parquet")

        logging.info(
            "Dumped %d shards of validation_dataset at %s",
            num_datasets_shard,
            validation_dataset_dir,
        )
    else:
        logging.warning("Validation dataset is None. Not saving validation dataset.")

    logger.info("Data Processing script execution completed.")


if __name__ == "__main__":
    main()
