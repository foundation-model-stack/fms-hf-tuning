# Standard
from typing import Callable, Dict, List
import warnings

# Third Party
from transformers import PreTrainedTokenizer
from trl import DataCollatorForCompletionOnlyLM

DEFAULT_FIELDS = ["input_ids", "attention_mask", "labels"]


def build_data_formatting_func(
    tokenizer: PreTrainedTokenizer = None,
    formatting: str = "instruct",
    tokenize: bool = False,
    input_field: str = "input",
    dataset_text_field: str = "output",
    features: List = None,
    response_template: str = None,
    response_field: str = None,
    chat_template: str = None,
):
    if tokenizer is None or chat_template is None:
        return _build_data_formatting_func_without_chat_template(
            tokenizer,
            formatting,
            tokenize,
            input_field,
            dataset_text_field,
            features,
            response_template,
        )

    return _build_data_formatting_func(
        tokenizer,
        tokenize,
        chat_template,
        dataset_text_field,
        features,
        response_template,
        response_field,
    )


# this one uses the chat template and tokenizer
def _build_data_formatting_func(
    tokenizer: PreTrainedTokenizer,
    tokenize: bool = False,
    chat_template: str = None,
    dataset_text_field: str = "output",
    features: List = None,
    response_template: str = None,
    response_field: str = None,
    ignore_index: int = -100,
):

    tokenizer.chat_template = chat_template

    loss_masking = None
    if tokenize and response_template is not None:
        loss_masking = instruction_mask_loss(tokenizer, response_template)
    elif tokenize and response_template is None:
        assert response_field is not None, \
            "response_field must be specified if tokenize=True and response_template=None."

    def _format(example):
        # `nonlocal` is used because the format_fn will be passed to dataset.map and
        # `loss_masking` needs to be bounded by `nonlocal` otherwise the spawned
        # processes will have no reference to it
        nonlocal loss_masking 
        formatted_and_maybe_tokenized = tokenizer.apply_chat_template(
            [example], tokenize=tokenize
        )
        key = "input_ids" if tokenize else dataset_text_field

        if tokenize and response_template is None and response_field:
            # in this case we need to use the response field to tokenize
            warnings.warn(
                "chat_template passed in with tokenize=True and "
                "response_template was None. To ensure loss masking is "
                f"correct, please do not put reponse_field '{response_field}' "
                "in the chat template."
            )
            # NOTE: in this case not handling attention mask
            response = tokenizer(example[response_field])['input_ids']
            return {
                key: formatted_and_maybe_tokenized + response,
                'labels': [ ignore_index ] * len(formatted_and_maybe_tokenized) + response
            }

        if not loss_masking:
            return {key: formatted_and_maybe_tokenized}
        return loss_masking(formatted_and_maybe_tokenized)

    return _format, {"remove_columns": features.difference(set(DEFAULT_FIELDS))}


# ---- NOTE: remove this eventually and move to check templates ----
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

# combine functions
# c = combine(a, b) then c(i) = b(a(i))
FUNC = Callable[[Dict], Dict]


def combine_functions(*funcs: FUNC) -> FUNC:
    def _combine(x):
        for f in funcs:
            x = f(x)
        return x

    return _combine


def _build_data_formatting_func_without_chat_template(
    tokenizer: PreTrainedTokenizer = None,
    formatting: str = "instruct",
    tokenize: bool = False,
    input_field: str = "input",
    dataset_text_field: str = "output",
    features: List = None,
    response_template: str = None,
):
    # FIFO
    funcs = []

    if features is None:
        features = set()

    if formatting == "instruct":
        funcs.append(
            instruction_formatter(
                input_field=input_field, dataset_text_field=dataset_text_field
            )
        )

    if tokenize:
        funcs.append(tokenization(tokenizer, dataset_text_field=dataset_text_field))

        if formatting == "instruct" and response_template:
            funcs.append(instruction_mask_loss(tokenizer, response_template))

    if len(funcs) == 0:
        raise ValueError("Unable to build a data formatting recipe")

    return combine_functions(*funcs), {
        "remove_columns": features.union(
            set([input_field, dataset_text_field])
        ).difference(set(DEFAULT_FIELDS))
    }


def instruction_formatter(
    input_field: str = "input", dataset_text_field: str = "output"
):
    def format_fn(example: Dict):
        prompt_input, prompt_no_input = (
            PROMPT_DICT["prompt_input"],
            PROMPT_DICT["prompt_no_input"],
        )
        output = (
            prompt_input.format_map(example)
            if example.get(input_field, "") != ""
            else prompt_no_input.format_map(example)
        )
        output = f"{output} {example[dataset_text_field]}"
        return {dataset_text_field: output}

    return format_fn


def tokenization(tokenizer: PreTrainedTokenizer, dataset_text_field: str = "output"):
    def _tokenize(example):
        text_field = example[dataset_text_field] + tokenizer.eos_token
        return tokenizer(text_field)

    return _tokenize


# ---- NOTE: remove this eventually and move to check templates ----


def instruction_mask_loss(
    tokenizer: PreTrainedTokenizer,
    response_template: str,
    take_from_index: int = 2,
):

    print(f"Applying loss masking to reponse template '{response_template}'")

    # cheat, use the data collator to mask the loss tokens
    response_template_ids = tokenizer.encode(
        response_template, add_special_tokens=False
    )

    # this ignores the first
    if len(response_template_ids) > take_from_index:
        response_template_ids = response_template_ids[take_from_index:]
        print(
            f"Taking response_ids[{take_from_index}:] from '{len(response_template_ids)}' response tokens"
        )

    collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer, ignore_index=-100
    )

    def collate_example(example):
        # single example
        collated_example = collator([example], return_tensors="pt")
        # flatten the additional dim
        return {k: v.view(-1) for k, v in collated_example.items()}

    return collate_example