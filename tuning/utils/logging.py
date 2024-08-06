# Standard
import logging
import os


def set_log_level(train_args, logger_name=None):
    """Set log level of python native logger and TF logger via argument from CLI or env variable.

    Args:
        train_args
            Training arguments for training model.

    Returns:
        train_args
            Updated training arguments for training model.
        train_logger
            Logger with updated effective log level
    """

    # Clear any existing handlers if necessary
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure Python native logger and transformers log level
    # If CLI arg is passed, assign same log level to python native logger
    log_level = "WARNING"
    if train_args.log_level != "passive":
        log_level = train_args.log_level

    # If CLI arg not is passed and env var LOG_LEVEL is set,
    # assign same log level to both logger
    elif os.environ.get("LOG_LEVEL"):
        log_level = os.environ.get("LOG_LEVEL")
        train_args.log_level = (
            log_level.lower()
            if not os.environ.get("TRANSFORMERS_VERBOSITY")
            else os.environ.get("TRANSFORMERS_VERBOSITY")
        )

    logging.basicConfig(level=log_level.upper())

    if logger_name:
        train_logger = logging.getLogger(logger_name)
    else:
        train_logger = logging.getLogger()
    return train_args, train_logger
