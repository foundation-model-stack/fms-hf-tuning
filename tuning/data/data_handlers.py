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
import copy
import re

# Third Party
from jinja2 import StrictUndefined, TemplateSyntaxError, UndefinedError
from jinja2.sandbox import SandboxedEnvironment, SecurityError
from transformers import AutoTokenizer

# Local
from tuning.utils.config_utils import process_jinja_placeholders


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

    def __init__(
        self,
        op: callable = None,
        handler_type: DataHandlerType = None,
        allows_batching: bool = False,
    ):
        self.op = op
        self.handler_type = handler_type
        self.allows_batching = allows_batching

    def __str__(self):
        o = self.op.__name__ if hasattr(self.op, "__name__") else str(self.op)
        n = self.handler_type.name
        b = self.allows_batching
        return f"DataHandler(op={o}, handler_type={n}, allows_batching={b})"


### Utils for custom masking / manipulating input / output strs, etc
def combine_sequence(input_element: str, output_element: str, eos_token: str = ""):
    """Combines / concatenates input & output element.

    Args:
        input_element: str
            Input component of the combined sequence.
        output_element: str
            Output component of the combined sequence.
        eos_token: str
            EOS token associated with the tokenizer. \
            If passed, it will be concatenated at end

    Returns:
        str
            Sequence combined with whitespace.
    """
    if not input_element.endswith((" ", "\n", "\t")) and not output_element.startswith(
        (" ", "\n", "\t")
    ):
        return input_element + " " + output_element + eos_token
    return input_element + output_element + eos_token


def tokenize_and_apply_input_masking(
    element: Dict[str, str],
    tokenizer: AutoTokenizer,
    column_names: List[str],
    input_field_name: str,
    output_field_name: str,
    add_eos_token: bool = True,
    **kwargs,
):
    """Function (data handler) to tokenize and apply instruction masking on dataset
       Expects to be run as a HF Map API function.
    Args:
        element: the HF Dataset element.
        tokenizer: Tokenizer to be used for tokenization.
        column_names: Name of all the columns in the dataset.
        input_field_name: Name of the input (instruction) field in dataset
        output_field_name: Name of the output field in dataset
        add_eos_token: should add tokenizer.eos_token to text or not, defaults to True
        **kwargs: Any additional args passed to the handler
    Returns:
        Formatted Dataset element with input_ids, labels and attention_mask columns
    """

    if column_names and (input_field_name or output_field_name) not in column_names:
        raise ValueError(
            f"Dataset should contain {input_field_name} \
                and {output_field_name} field if \
                no dataset_text_field or data_formatter_template specified"
        )

    input_text = element[input_field_name]
    output_text = element[output_field_name]

    eos_token = ""
    if add_eos_token:
        eos_token = tokenizer.eos_token

    combined = combine_sequence(input_text, output_text, eos_token=eos_token)

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


def add_tokenizer_eos_token(
    element: Dict[str, str],
    tokenizer: AutoTokenizer,
    dataset_text_field: str,
    **kwargs,
):
    """Function (data handler) to add tokenizer's EOS token to text field of an element
       Expects to be run as a HF Map API function.
    Args:
        element: the HF Dataset element.
        tokenizer: Tokenizer to be used for the EOS token, which will be appended
            when formatting the data into a single sequence. Defaults to empty.
        dataset_text_field: Text column name of the dataset where EOS is to be added.
    Returns:
        Formatted Dataset element with EOS added to dataset_text_field of the element.
    """

    if dataset_text_field not in element:
        raise KeyError(f"Dataset should contain {dataset_text_field} field.")
    return {
        f"{dataset_text_field}": element[f"{dataset_text_field}"] + tokenizer.eos_token
    }


def apply_custom_data_formatting_template(
    element: Dict[str, str],
    tokenizer: AutoTokenizer,
    dataset_text_field: str,
    template: str,
    add_eos_token: bool = True,
    **kwargs,
):
    """Function (data handler) to format datasets with Alpaca style / other templates.
       Expects to be run as a HF Map API function.
    Args:
        element: the HF Dataset element.
        tokenizer: Tokenizer to be used for the EOS token, which will be appended
            when formatting the data into a single sequence. Defaults to empty.
        dataset_text_field: Text column name of the dataset where formatted text is saved.
        template: Template to format data with. Features of Dataset
            should be referred to by {{key}}
        add_eos_token: should add tokenizer.eos_token to text or not, defaults to True
    Returns:
        Formatted Dataset element by formatting dataset with template+tokenizer.EOS_TOKEN
        Saves the result to dataset_text_field argument.
    """

    if add_eos_token:
        template += tokenizer.eos_token

    def replace_text(match_obj):
        captured_groups = match_obj.groups()
        if len(captured_groups) != 1:
            raise ValueError(
                "Unexpectedly captured multiple groups in template formatting"
            )

        index_object = captured_groups[0]
        if index_object not in element:
            raise KeyError("Requested template string is not a valid key in dict")

        return str(element[index_object])

    return {
        f"{dataset_text_field}": re.sub(
            r"{{([\s0-9a-zA-Z_\-\.]+)}}", replace_text, template
        )
    }


def apply_custom_jinja_template(
    element: Dict[str, str],
    tokenizer: AutoTokenizer,
    dataset_text_field: str,
    template: str,
    add_eos_token: bool = True,
    **kwargs,
):
    """Function (data handler) to format datasets with jinja templates.
       Expects to be run as a HF Map API function.
    Args:
        element: the HF Dataset element
        tokenizer: Tokenizer to be used for the EOS token, which will be appended
            when formatting the data into a single sequence. Defaults to empty.
        dataset_text_field: formatted_dataset_field.
        template: Template to format data with. Features of Dataset
            should be referred to by {{key}}.
        add_eos_token: should add tokenizer.eos_token to text or not, defaults to True
    Returns:
        Formatted HF Dataset element by formatting dataset with provided jinja template
        Saves the result to dataset_text_field argument.
    """
    if add_eos_token:
        template += tokenizer.eos_token

    template = process_jinja_placeholders(template)
    env = SandboxedEnvironment(undefined=StrictUndefined)

    try:
        jinja_template = env.from_string(template)
    except TemplateSyntaxError as e:
        raise ValueError(
            f"Invalid template syntax in provided Jinja template. {e.message}"
        ) from e

    try:
        rendered_text = jinja_template.render(element=element, **element)
    except UndefinedError as e:
        raise KeyError(
            f"The dataset does not contain the key used in the provided Jinja template. {e.message}"
        ) from e
    except SecurityError as e:
        raise ValueError(
            f"Unsafe operation detected in the provided Jinja template. {e.message}"
        ) from e
    except Exception as e:
        raise ValueError(
            f"Error occurred while rendering the provided Jinja template. {e.message}"
        ) from e

    return {f"{dataset_text_field}": rendered_text}


def apply_tokenizer_chat_template(
    element: Dict[str, str],
    tokenizer: AutoTokenizer,
    dataset_text_field: str,
    conversation_column: str = None,
    **kwargs,
):
    """Function (data handler) to apply tokenizers chat template to dataset elements.
       Does not tokenize the dataset.
       Expects to be run as a HF Map API function.
    Args:
        element: the HF Dataset element.
        tokenizer: Tokenizer to be used.
        dataset_text_field: the field in which to store the rendered text.
        conversation_column: column name where the chat template expects the conversation
    Returns:
        Formatted HF Dataset element by formatting dataset with tokenizer's chat template
        Saves the result to dataset_text_field argument.
    """
    if tokenizer.chat_template is None:
        raise ValueError(
            "Tokenizer does not contain tokenizer.chat_template\
                          please pass data_args.chat_template"
        )
    if conversation_column:
        converation = element[conversation_column]
    else:
        converation = element

    tools = element["tools"] if "tools" in element else None
    documents = element["documents"] if "documents" in element else None

    return {
        f"{dataset_text_field}": tokenizer.apply_chat_template(
            converation, tools=tools, documents=documents, tokenize=False
        )
    }


def tokenize(
    element: Union[Dict[str, str], Dict[str, List]],
    tokenizer: AutoTokenizer,
    dataset_text_field: str,
    truncation: Union[bool, str] = None,
    max_length: int = None,
    **kwargs,
):
    """Function (data handler) to tokenize dataset columns.
       Expects to be run as a HF Map API function.
    Args:
        element: the HF Dataset element.
        tokenizer: Tokenizer to be used.
        dataset_text_field: the dataset field to tokenize
        truncation: Truncation strategy to use, refer the link
                    (https://huggingface.co/docs/transformers/en/pad_truncation)
        max_length: Max length to truncate the samples to.
        kwargs: Any additional kwargs that need to be passed to the tokenizer can be passed as
                kwargs['tokenizer_kwargs']
    Returns:
        tokenized dataset elemenent field "dataset_text_field"
    """
    tokenizer_kwargs = kwargs.get("tokenizer_kwargs", {})
    return tokenizer(
        element[dataset_text_field],
        truncation=truncation,
        max_length=max_length,
        **tokenizer_kwargs,
    )


def duplicate_columns(
    element: Union[Dict[str, str], Dict[str, List]],
    old_column: str,
    new_column: str,
    **kwargs,
):
    """Function (data handler) to duplicate one columne of a dataset to another.
       Expects to be run as a HF Map API function.
    Args:
        element: the HF Dataset element
        old_column: Name of the column which is to be duplicated
        new_column: Name of the new column where duplicated column is to be saved
    Returns:
        Formatted HF Dataset element with new_column where existing_columns content is deep copied.
    """
    if not old_column or not new_column:
        raise ValueError(
            "for duplicating columns both old and new column name must be specified"
        )
    if old_column not in element:
        raise ValueError(
            f"Cannot duplicate {old_column} to {new_column} as column {old_column} doesn't exist"
        )
    if new_column in element:
        raise ValueError(
            f"Cannot duplicate {old_column} to f{new_column} as column {new_column} already exists"
        )

    return {
        f"{old_column}": element[old_column],
        f"{new_column}": copy.deepcopy(element[old_column]),
    }


def skip_large_columns(element: Dict[str, Any], column_name: str, max_length: int):
    """Function (data handler) to skip elements which contains certain columns {column_name}
       larger than the passed {max_length} in the dataset.
       i.e if element[column_name] <= max_length its allowed else skipped.
       raises ValueError if
          1) column_name is None
          2) max_length is None
          3) element[column_name] is None
       Expects to be run as a HF Filter API function.
    Args:
        element: the HF Dataset element
        column_name: Name of the column
        max_length: Max allowed length of the column.
                    If passing "input_ids" as column name this will be tokens
                    else this can be characters for text column
    Returns:
        Filtered dataset which contains elements with column {column_name}
                 having length shorter than {max_length}
    """
    if column_name not in element or max_length is None:
        raise ValueError(
            "Please provide correct column name and max_length to skip large columns"
        )
    if element[column_name] is None:
        raise ValueError(
            f"Column {column_name} value in dataset element {element} is None"
        )
    return len(element[column_name]) <= max_length


AVAILABLE_DATA_HANDLERS = {
    "tokenize_and_apply_input_masking": DataHandler(
        op=tokenize_and_apply_input_masking,
        handler_type=DataHandlerType.MAP,
        allows_batching=False,
    ),
    "add_tokenizer_eos_token": DataHandler(
        op=add_tokenizer_eos_token,
        handler_type=DataHandlerType.MAP,
        allows_batching=False,
    ),
    "apply_custom_data_formatting_template": DataHandler(
        op=apply_custom_data_formatting_template,
        handler_type=DataHandlerType.MAP,
        allows_batching=False,
    ),
    "apply_custom_jinja_template": DataHandler(
        op=apply_custom_jinja_template,
        handler_type=DataHandlerType.MAP,
        allows_batching=False,
    ),
    "apply_tokenizer_chat_template": DataHandler(
        op=apply_tokenizer_chat_template,
        handler_type=DataHandlerType.MAP,
        allows_batching=False,
    ),
    "duplicate_columns": DataHandler(
        op=duplicate_columns,
        handler_type=DataHandlerType.MAP,
        allows_batching=True,
    ),
    "tokenize": DataHandler(
        op=tokenize,
        handler_type=DataHandlerType.MAP,
        allows_batching=True,
    ),
    "skip_large_columns": DataHandler(
        op=skip_large_columns,
        handler_type=DataHandlerType.FILTER,
        allows_batching=False,
    ),
    "remove_columns": DataHandler(
        handler_type=DataHandlerType.REMOVE,
    ),
    "select_columns": DataHandler(
        handler_type=DataHandlerType.SELECT,
    ),
    "rename_columns": DataHandler(
        handler_type=DataHandlerType.RENAME,
    ),
}
