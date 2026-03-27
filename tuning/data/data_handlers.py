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

# Definition of some predefined data preprocessing functions that we need.

# Standard
from enum import Enum
from typing import Any, Dict, List, Union

# import copy
import logging

# Third Party
from jinja2 import StrictUndefined, TemplateSyntaxError, UndefinedError
from jinja2.sandbox import SandboxedEnvironment, SecurityError
from PIL import Image
from transformers import GPT2TokenizerFast, LlavaNextProcessor, LlavaProcessor
import torch

# Local
from tuning.data.utils import try_convert_bytes_dict_to_pil, try_convert_image_to_rgb

logger = logging.getLogger(__name__)


class DataHandlerType(Enum):
    """
    ENUM which represents the type of data handlers supported.
    """

    # Map:
    #   https://huggingface.co/docs/datasets/en/process#map
    MAP = 1
    # Filter:
    #   https://huggingface.co/docs/datasets/en/process#select-and-filter
    FILTER = 2
    # Remove and Select:
    #   https://huggingface.co/docs/datasets/en/process#remove
    REMOVE = 3
    SELECT = 4
    # Rename:
    #   https://huggingface.co/docs/datasets/en/process#rename
    RENAME = 5


class DataHandler:
    """
    A class which represents a data processing handler internally.

    Args:
        op (callable): The data handler callable function which operates on the data
                       in case of MAP or FILTER type handlers.
                       For other handles like REMOVE/SELECT/RENAME use Native API so
                       op can be None.
        handler_type (DataHandlerType): Indicates whether the handler is for mapping or filtering.
                                        One out of the supported types in DataHandlerType
        allows_batching (bool): Flag to indicate if the handler supports batched operations.
                                See https://huggingface.co/docs/datasets/en/about_map_batch
    """

    op: callable = None  # the actual handler function
    handler_type: DataHandlerType = None  # either map or filter
    allows_batching: bool = False  # supports batched mode or not
    desc: str = None

    def __init__(
        self,
        op: callable = None,
        handler_type: DataHandlerType = None,
        allows_batching: bool = False,
        desc: str = None,
    ):
        self.op = op
        self.handler_type = handler_type
        self.allows_batching = allows_batching
        if desc is None:
            self.desc = self.__str__()

    def __str__(self):
        o = self.op.__name__ if hasattr(self.op, "__name__") else str(self.op)
        n = self.handler_type.name
        b = self.allows_batching
        return f"DataHandler(op={o}, handler_type={n}, allows_batching={b})"


def tokenize_and_apply_input_masking(
    element: Dict[str, str],
    input_column_name: str,
    output_column_name: str,
    add_eos_token: bool = True,
    **kwargs,
):
    """Function to tokenize and apply instruction masking on dataset
       Expects to be run as a HF Map API function.
    Args:
        element: the HF Dataset element.
        input_column_name: Name of the input (instruction) field in dataset
        output_column_name: Name of the output field in dataset
        add_eos_token: should add tokenizer.eos_token to text or not, defaults to True
        **kwargs: Any additional args passed to the handler
    Returns:
        Formatted Dataset element with input_ids, labels and attention_mask columns
    """

    # These are made available by the data preprocessor framework
    try:
        tokenizer = kwargs["tokenizer"]
    except KeyError as e:
        raise RuntimeError(
            "Data processor failed to pass default args to data handlers"
        ) from e

    if (input_column_name or output_column_name) not in element:
        raise ValueError(
            f"Dataset should contain {input_column_name} \
                and {output_column_name} field if \
                no dataset_text_field or data_formatter_template specified"
        )

    input_text = element[input_column_name]
    output_text = element[output_column_name]

    eos_token = ""
    if add_eos_token:
        eos_token = tokenizer.eos_token

    if not input_text.endswith((" ", "\n", "\t")) and not output_text.startswith(
        (" ", "\n", "\t")
    ):
        combined = input_text + " " + output_text + eos_token
    else:
        combined = input_text + output_text + eos_token

    tokenizer_kwargs = kwargs.get("tokenizer_kwargs", {})

    tokenized_comb_seqs = tokenizer(combined, **tokenizer_kwargs)
    tokenized_input = tokenizer(input_text, **tokenizer_kwargs)

    masked_labels = [-100] * len(
        tokenized_input.input_ids
    ) + tokenized_comb_seqs.input_ids[len(tokenized_input.input_ids) :]

    # Any benefit of retaining the old columns?
    return {
        "input_ids": tokenized_comb_seqs.input_ids,
        "labels": masked_labels,
        "attention_mask": tokenized_comb_seqs.attention_mask,
    }


def __wrap_jinja_rendering_with_exception_handling(render_template: callable, **kwargs):
    base_err = (
        "Failed to render provided chat/custom template on dataset."
        + "Please check your dataset and the template. Likely failure cause - "
    )
    try:
        return render_template(**kwargs)  # <-------- Actual function call
    except TemplateSyntaxError as e:
        raise ValueError(
            f"{base_err}Provided jinja template syntax is invalid. {e}"
        ) from e
    except UndefinedError as e:
        raise KeyError(
            f"{base_err}Jinja template keys are not present in the dataset. {e}"
        ) from e
    except SecurityError as e:
        raise ValueError(
            f"{base_err}Unsafe operation detected in the Jinja template. {e}"
        ) from e
    except Exception as e:
        raise ValueError(f"{base_err}{e}") from e


def apply_custom_jinja_template(
    element: Dict[str, str],
    formatted_text_column_name: str,
    template: str,
    **kwargs,
):
    """Function to format datasets with Alpaca style / any other jinja templates.
       Expects to be run as a HF Map API function.
    Args:
        element: the HF Dataset element
        formatted_text_column_name: Name of the dataset column where formatted
                                    text is to be saved. If doesn't exist a new
                                    column will be created.
        template: Template to format data with. Features of Dataset
            should be referred to by {{key}}.
    Returns:
        Formatted HF Dataset element by formatting dataset with provided jinja template
        Saves the result to formatted_text_column_name argument.
    """

    # These are made available by the data preprocessor framework
    try:
        tokenizer = kwargs["tokenizer"]
    except KeyError as e:
        raise RuntimeError(
            "Data processor failed to pass default args to data handlers"
        ) from e

    def render():
        env = SandboxedEnvironment(undefined=StrictUndefined)
        jinja_template = env.from_string(template)
        template_kwargs = {**tokenizer.special_tokens_map, **element}
        return jinja_template.render(element=element, **template_kwargs)

    return {
        f"{formatted_text_column_name}": __wrap_jinja_rendering_with_exception_handling(
            render
        )
    }


def apply_tokenizer_chat_template(
    element: Dict[str, str],
    formatted_text_column_name: str,
    conversation_column_name: str = None,
    **kwargs,
):
    """Function to apply tokenizers chat template to dataset elements.
       Does not tokenize the dataset.
       Expects to be run as a HF Map API function.
    Args:
        element: the HF Dataset element.
        formatted_text_column_name: Name of the column where the rendered text is to
                                    be stored post applying chat template.
        conversation_column_name: If chat template is to be run on full sample pass this as None
                                  if chat template expects to be run on a specific column of the
                                  data sample pass the column name here.
    Returns:
        Formatted HF Dataset element by formatting dataset with tokenizer's chat template
        Saves the result to formatted_text_column_name argument.
    """

    # These are made available by the data preprocessor framework
    try:
        tokenizer = kwargs["tokenizer"]
    except KeyError as e:
        raise RuntimeError(
            "Data processor failed to pass default args to data handlers"
        ) from e

    processor = kwargs.get("processor", None)
    if processor is not None:
        tokenizer = processor
    if tokenizer.chat_template is None:
        raise ValueError(
            "Tokenizer does not contain tokenizer.chat_template\
                          please pass data_args.chat_template"
        )
    if conversation_column_name:
        if conversation_column_name not in element:
            raise ValueError(
                "conversation_column_name %s is not present in data sample"
                % conversation_column_name
            )
        conversation = element[conversation_column_name]
    else:
        conversation = element

    tools = element["tools"] if "tools" in element else None
    documents = element["documents"] if "documents" in element else None

    return __wrap_jinja_rendering_with_exception_handling(
        lambda: {
            f"{formatted_text_column_name}": tokenizer.apply_chat_template(
                conversation, tools=tools, documents=documents, tokenize=False
            )
        }
    )


def prepare_multimodal_data_processor(
    element: Dict[str, str],
    **kwargs,
):
    """Function (data handler) to apply processor to multimodal dataset elements.
       Expects to be run as a HF Map API function.
    Args:
        element: the HF Dataset element.
    Returns:
        Formatted HF Dataset element by formatting dataset with processor
    """

    processor = kwargs.get("processor", None)
    if processor is None:
        raise ValueError(
            "Processor is missing. Please provide a processor when initializing the handler."
        )

    processor_kwargs = kwargs.get("processor_kwargs", {})
    fields_name = kwargs.get("fields_name", {})
    try:
        text_field = fields_name["dataset_text_field"]
        image_field = fields_name["dataset_image_field"]
    except KeyError as e:
        raise ValueError(f"Missing required field in fields_name: {e}") from e

    text = element.get(text_field)
    image = element.get(image_field)

    if text is None or image is None:
        raise ValueError("Missing text or image data in element.")

    image = try_convert_bytes_dict_to_pil(image)  # Needed for below image processing

    # We need to pick first image from the Image list for LlavaProcessor and
    # LlavaNextProcessor (Granite Vision Model)
    if isinstance(processor, LlavaProcessor) or (
        isinstance(processor, LlavaNextProcessor)
        and isinstance(processor.tokenizer, GPT2TokenizerFast)
    ):

        if (
            image and isinstance(image, list) and isinstance(image[0], list)
        ):  # FOR BATCHED = TRUE
            image = [img[0] for img in image]
            logger.warning(
                "LlavaProcessor and LlavaNextProcessor (tokenizer GPT2TokenizerFast)  \
                expects a single image, picking the first image from the list."
            )
        elif (
            image and isinstance(image, list) and isinstance(image[0], Image.Image)
        ):  # FOR BATCHED = FALSE
            image = image[0]
            logger.warning(
                "LlavaProcessor and LlavaNextProcessor (tokenizer GPT2TokenizerFast) \
                expects a single image, picking the first image from the list."
            )

    # Convert image to RGB if it is not in RGB format
    if isinstance(processor, (LlavaProcessor, LlavaNextProcessor)):
        image = try_convert_image_to_rgb(image)

    element = {
        text_field: text,
        image_field: image,
        "fields_name": fields_name,
        "processor_kwargs": processor_kwargs,
    }
    return element


def tokenize(
    element: Union[Dict[str, str], Dict[str, List]],
    text_column_name: str,
    truncation: Union[bool, str] = True,
    max_length: int = None,
    **kwargs,
):
    """Function to tokenize dataset columns.
       Expects to be run as a HF Map API function.
    Args:
        element: the HF Dataset element.
        text_column_name: the text column name to tokenize
        truncation: Truncation strategy to use, refer the link
                    (https://huggingface.co/docs/transformers/en/pad_truncation)
                    Defaults to True.
        max_length: Max length to truncate the samples to.
        kwargs: Any additional kwargs that need to be passed to the tokenizer can be passed as
                kwargs['tokenizer_kwargs']
    Returns:
        sample with tokenized text_column_name
    """

    # These are made available by the data preprocessor framework
    try:
        tokenizer = kwargs["tokenizer"]
    except KeyError as e:
        raise RuntimeError(
            "Data processor failed to pass default args to data handlers"
        ) from e

    tokenizer_kwargs = kwargs.get("tokenizer_kwargs", {})
    return tokenizer(
        element[text_column_name],
        truncation=truncation,
        max_length=max_length,
        **tokenizer_kwargs,
    )


def duplicate_columns(
    element: Union[Dict[str, str], Dict[str, List]],
    existing_column_name: str,
    new_column_name: str,
    **kwargs,
):
    """Function to duplicate one columne of a dataset to another.
       Expects to be run as a HF Map API function.
    Args:
        element: the HF Dataset element
        existing_column_name: Name of the column which is to be duplicated
        new_column_name: Name of the new column where duplicated column is to be saved
    Returns:
        Formatted HF Dataset element with
        {"new_column_name": element["existing_column_name"]}.
    """
    if not existing_column_name or not new_column_name:
        raise ValueError(
            "for duplicating columns both old and new column name must be specified"
        )
    if existing_column_name not in element:
        raise ValueError(
            "Cannot duplicate %s to %s as column %s doesn't exist"
            % (existing_column_name, new_column_name, existing_column_name)
        )
    if new_column_name in element:
        raise ValueError(
            "Cannot duplicate %s to %s as column %s already exist"
            % (existing_column_name, new_column_name, existing_column_name)
        )

    return {
        f"{existing_column_name}": element[existing_column_name],
        f"{new_column_name}": element[existing_column_name],
    }


def skip_samples_with_large_columns(
    element: Dict[str, Any], column_name: str, max_allowed_length: int
):
    """Function to skip samples which contains certain columns {column_name}
       larger than the passed {max_allowed_length} in the dataset.
       i.e if samples[column_name] <= max_allowed_length its allowed else skipped.
       raises ValueError if
          1) column_name is None
          2) max_allowed_length is None
          3) samples[column_name] is None
       Expects to be run as a HF Filter API function.
    Args:
        element: the HF Dataset samples
        column_name: Name of the column
        max_allowed_length: Max allowed length of the column.
                    If passing "input_ids" as column name this will be tokens
                    else this can be characters for text column
    Returns:
        Filtered dataset which contains elements with column {column_name}
                 having length shorter than {max_allowed_length}
    """
    if column_name not in element or max_allowed_length is None:
        raise ValueError(
            "Please provide correct column name and max_allowed_length"
            "to skip samples with large columns"
        )
    if element[column_name] is None:
        raise ValueError(
            f"Column {column_name} value in dataset element {element} is None"
        )
    return len(element[column_name]) <= max_allowed_length


def tokenize_and_apply_chat_template_with_masking(
    element: Dict[str, str],
    max_seq_length: int = None,
    conversation_column: str = "messages",
    **kwargs,
):
    """Function to apply chat template to the dataset elements and
       perform masking to ensure model is trained only on completions.
       Assumes the dataset is modelled according to ChatML style format
       like,
       { messages: {'role': 'user', 'content': 'blah'}

       Tokenizes the dataset and returns a tokenized element.
       Requires that max_seq_length is passed to ensure truncation of
       extra large samples. If samples are to be skipped truncated please
       use filter data handler before using this to ensure skipping
       of samples.

       Expects to be run as a HF Map function.
       Ensures that element contains `input_ids`, `labels` and
       `attention_mask`
       If used with `remove_columns=all` the dataset can be used
       directly to train.
    Args:
        element: the HF Dataset samples
        max_seq_length: Max seq length of the tokens allowed.
                        Required argument.
        conversation_column: Name of the column which contains conversations
                        Typically `messages`
        kwargs: Unused by this function.
    Returns:
        Tokenized element which contains `input_ids` `labels` and `attention_mask`
        with labels properly masked to train only on completions.
    """

    # These are made available by the data preprocessor framework
    try:
        tokenizer = kwargs["tokenizer"]
    except KeyError as e:
        raise RuntimeError(
            "Data processor failed to pass default args to data handlers"
        ) from e

    # This function is taken from OpenInstruct
    # https://github.com/allenai/open-instruct/blob/\
    #   d208aa371976a09152f61991951e981573e7582f/open_instruct/\
    #   dataset_transformation.py#L632

    messages = element[conversation_column]

    if len(messages) == 0:
        raise ValueError(
            f"Contents of the column {conversation_column} must not be empty."
        )

    tools = element["tools"] if "tools" in element else None
    documents = element["documents"] if "documents" in element else None

    # Tokenize the whole sample
    input_ids = __wrap_jinja_rendering_with_exception_handling(
        lambda: tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=True,
            padding=False,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_length,
            add_generation_prompt=False,
            tools=tools,
            documents=documents,
        )
    )

    # clone labels from input ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            # we calculate the start index of this non-assistant message
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = __wrap_jinja_rendering_with_exception_handling(
                    lambda idx=message_idx: tokenizer.apply_chat_template(
                        # here marks the end of the previous messages
                        conversation=messages[:idx],
                        tokenize=True,
                        padding=False,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_seq_length,
                        add_generation_prompt=False,
                        tools=tools,
                        documents=documents,
                    ).shape[1]
                )
            # next, we calculate the end index of this non-assistant message
            if (
                message_idx < len(messages) - 1
                and messages[message_idx + 1]["role"] == "assistant"
            ):
                # for intermediate messages that follow with an assistant message,
                # we need to set `add_generation_prompt=True` to avoid the assistant
                # generation prefix being included in the loss (e.g., `<|assistant|>`)
                message_end_idx = __wrap_jinja_rendering_with_exception_handling(
                    lambda idx=message_idx: tokenizer.apply_chat_template(
                        conversation=messages[: idx + 1],
                        tokenize=True,
                        return_tensors="pt",
                        padding=False,
                        truncation=True,
                        max_length=max_seq_length,
                        add_generation_prompt=True,
                        tools=tools,
                        documents=documents,
                    ).shape[1]
                )
            else:
                # for the last message or the message that doesn't follow with
                # an assistant message, we don't need to add the assistant generation prefix
                message_end_idx = __wrap_jinja_rendering_with_exception_handling(
                    lambda idx=message_idx: tokenizer.apply_chat_template(
                        conversation=messages[: idx + 1],
                        tokenize=True,
                        return_tensors="pt",
                        padding=False,
                        truncation=True,
                        max_length=max_seq_length,
                        add_generation_prompt=False,
                        tools=tools,
                        documents=documents,
                    ).shape[1]
                )
            # set the label to -100 for the non-assistant part
            labels[:, message_start_idx:message_end_idx] = -100
            if max_seq_length and message_end_idx >= max_seq_length:
                break
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


AVAILABLE_DATA_HANDLERS = {
    "remove_columns": DataHandler(
        # Native function
        handler_type=DataHandlerType.REMOVE,
    ),
    "select_columns": DataHandler(
        # Native function
        handler_type=DataHandlerType.SELECT,
    ),
    "rename_columns": DataHandler(
        # Native function
        handler_type=DataHandlerType.RENAME,
    ),
    "tokenize": DataHandler(
        op=tokenize,
        handler_type=DataHandlerType.MAP,
        allows_batching=True,
        desc="Tokenizing the dataset",
    ),
    "apply_custom_jinja_template": DataHandler(
        op=apply_custom_jinja_template,
        handler_type=DataHandlerType.MAP,
        allows_batching=False,
        desc="Formatting dataset with given formatting template",
    ),
    "tokenize_and_apply_input_masking": DataHandler(
        op=tokenize_and_apply_input_masking,
        handler_type=DataHandlerType.MAP,
        allows_batching=False,
        desc="Combining and tokenizing instruction and response, masking instructions",
    ),
    "apply_tokenizer_chat_template": DataHandler(
        op=apply_tokenizer_chat_template,
        handler_type=DataHandlerType.MAP,
        allows_batching=False,
        desc="Applying chat template to dataset",
    ),
    "tokenize_and_apply_chat_template_with_masking": DataHandler(
        op=tokenize_and_apply_chat_template_with_masking,
        handler_type=DataHandlerType.MAP,
        allows_batching=False,
        desc="Applying chat template to dataset and tokenizing",
    ),
    "duplicate_columns": DataHandler(
        op=duplicate_columns,
        handler_type=DataHandlerType.MAP,
        allows_batching=True,
        desc="Duplicating columns",
    ),
    "skip_samples_with_large_columns": DataHandler(
        op=skip_samples_with_large_columns,
        handler_type=DataHandlerType.FILTER,
        allows_batching=False,
        desc="Skipping large samples",
    ),
    "prepare_multimodal_data_processor": DataHandler(
        op=prepare_multimodal_data_processor,
        handler_type=DataHandlerType.MAP,
        allows_batching=False,
        desc="Processing multimodal data",
    ),
}
