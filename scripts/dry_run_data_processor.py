# Standard
from typing import Callable, Dict, Optional
import logging
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
        "--save_train_dataset",
        type=str,
        default=None,
        help="Path to JSON file for saving the processed train dataset.",
    )
    parser.add_argument(
        "--save_validation_dataset",
        type=str,
        default=None,
        help="Path to JSON file for saving the processed validation dataset.",
    )

    # Parse arguments and set log level
    try:
        (
            model_args,
            data_args,
            training_args,
            save_train_dataset,
            save_validation_dataset,
        ) = parser.parse_args_into_dataclasses()

        training_args, logger = set_log_level(training_args, __name__)
        logger.debug(
            "Input args parsed:\n"
            "  model_args: %s\n"
            "  data_args: %s\n"
            "  training_args: %s\n"
            "  save_train_dataset: %s\n"
            "  save_validation_dataset: %s",
            model_args,
            data_args,
            training_args,
            save_train_dataset,
            save_validation_dataset,
        )
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
    if save_train_dataset:
        logger.info("Saving processed train dataset to %s", save_train_dataset)
        formatted_train_dataset.to_json(save_train_dataset)

    # Save validation dataset
    if save_validation_dataset:
        logger.info(
            "Saving processed validation dataset to %s", save_validation_dataset
        )
        formatted_validation_dataset.to_json(save_validation_dataset)

    logger.info("Data Processing script execution completed.")


if __name__ == "__main__":
    main()
