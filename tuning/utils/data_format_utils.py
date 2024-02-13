from datasets import Dataset
from collections.abc import Mapping
from transformers import AutoTokenizer
from transformers.utils import logging
import torch
from typing import Union
import json
from datasets import Dataset, IterableDataset

from torch.utils.data import Dataset as DatasetTorch
 
## this is needed to convert HFDataset to torch.utils.Dataset
## torch.utils.Dataset is needed by SFT TRainer to bypass prepare_dataset
##  https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py#L370C67-L370C91
class HFDataset(DatasetTorch):
    def __init__(self, dset):
        self.dset = dset

    def __getitem__(self, idx):
        return self.dset[idx]

    def __len__(self):
        return len(self.dset)

logger = logging.get_logger("sft_trainer")
def preprocess_function(
        data_path: str,
        tokenizer: AutoTokenizer,
        use_iterable_dataset: bool = False
    ):
        """Pre-process each example to get it prepared for training."""
        fn_kwargs = {
            "tokenizer": tokenizer,
        }
        dataset_type = IterableDataset if use_iterable_dataset else Dataset
        dataset = dataset_type.from_generator(
            get, gen_kwargs={"train_file": data_path}
        )

        mapped_dataset = dataset.map(
            tokenize_function,
            fn_kwargs=fn_kwargs,
            batched=True,
            # Drop the input / output columns; we need to do this for dimensions to play
            # happily when operating on batched inputs for causal language modeling.
            remove_columns=["input", "output"],
        )
        if dataset_type == IterableDataset:
            return mapped_dataset
        else:
            return HFDataset(mapped_dataset)

def tokenize_function(
        example: Mapping,
        tokenizer: AutoTokenizer,
    ) ->  "BatchEncoding":
        """Tokenization function to be used for causallm training; this function consumes a
        GenerationTrainRecord object and applies the verbalizer to it followed by
        the model tokenizer. Due to the nature of our training data with src/target seqs,
        each sample yields one example per token in the target sequence.

        Args:
            example: Mapping
                Training data model object to convert a form we can learn on, or a Mapping
                that has keys input/output.
            tokenizer: AutoTokenizer
                Tokenizer object to be applied to input records.
        Returns:
            Union[DataStream[BatchEncoding], BatchEncoding]
                or a single batch encoding object containing 1+ tokenized results.
        """
        ### Things common to all Causal LM tokenization approaches
        # Extract the source & target from our provided inputs
        #print(example)
        if 'input' not in example or 'output' not in example:
            logger.error("Input and output fields must be present in JSON.")
        source, target = example["input"], example["output"]
        #print(source, target)

        # Treat this as a seq2seq type problem. Note that this implementation is different
        # from the seq2seq tokenization function even though it is conceptually similar due
        # to sequence length / padding requirements assumed internally by causal LMs.

        return causal_lm_padding_as_seq2seq(
                tokenizer=tokenizer,
                source=source,
                target=target,
            )

def causal_lm_padding_as_seq2seq(
        tokenizer: "AutoTokenizer",
        source: str,
        target: str,
    ) -> "BatchEncoding":
        """Tokenize the example as a seq2seq type problem; this is conceptually similar to
        what seq2seq tokenization is doing, but some care needs be taken to ensure the labels
        are the same length as the input sequence because of the shifting mechanism implemented
        in most causal language models.

        Collator compatability is extremely important here; because we are setting the labels
        directly, we should NOT use the causal lm collator, otherwise it will clobber it with a
        shifted input sequence.

        Args:
            tokenizer: AutoTokenizer
                Tokenizer object to be applied to input records.
            source: str
                Raw source string.
            target: str
                Raw target string.
        Returns:
            BatchEncoding
                BatchEncoding object corresponding to this example, where the input_ids,
                attention_mask, and labels all have the same length, i.e.,
                [max_source_length + max_target_length + 1].
        """
        IGNORE_ID = -100

        # ID of the token to append after our target string; this should generally be pad / EOS
        if tokenizer.eos_token_id is not None:
            FINAL_TOK_ID = tokenizer.eos_token_id
        # Fall back to using pad token id no EOS token is defined
        else:
            FINAL_TOK_ID = tokenizer.pad_token_id
        # TODO: Handle models/tokenizers that produce token_type_ids
        model_inputs = tokenizer(source)
        labels = tokenizer(target)
        # NOTE: It doesn't matter what key we inspect in model_inputs, since they
        # all have dicts of the same length when operating in batch mode.
        batch_size = len(model_inputs['attention_mask'])
        logger.info(f"Batch size: {batch_size}")

        # Infer the max tok length based on the source and target
        max_concat_length = get_max_seq_len(model_inputs, labels)
        logger.info(f"Inferred max concat length: {max_concat_length}")

        # Now that we know the max length of the batch, form the model/label input ids, attention mask
        for idx in range(batch_size):

            # Combine the source + target strings into the source input IDs
            # This makes the source and target the same length, and then masks the source out of the
            # target IDs, and updates the length of the attention vector to be evenly spread on the
            # whole combined sequence
            sample_input_ids = model_inputs["input_ids"][idx]
            label_input_ids = labels["input_ids"][idx] + [FINAL_TOK_ID]

            model_inputs["input_ids"][idx] = sample_input_ids + label_input_ids
            labels["input_ids"][idx] = [IGNORE_ID] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][idx] = [1] * len(model_inputs["input_ids"][idx])

            # Now we have to update everything to be the max length of the tokenizer, then pad &
            # ensure all of the padded stuff we have added has attention weights of 0.
            sample_input_ids = model_inputs[
                "input_ids"
            ][idx]  # NOTE - combined source + target + <FINAL_TOK_ID>

            label_input_ids = labels["input_ids"][idx]

            if tokenizer.padding_side.lower() == "left":
                labels["input_ids"][idx] = [IGNORE_ID] * (
                    max_concat_length - len(sample_input_ids)
                ) + label_input_ids
            else:
                labels["input_ids"][idx] = label_input_ids + [IGNORE_ID] * (
                    max_concat_length - len(sample_input_ids)
                )


            model_inputs["input_ids"][idx] = model_inputs["input_ids"][idx][:max_concat_length]
            model_inputs["attention_mask"][idx] = model_inputs["attention_mask"][idx][:max_concat_length]
                

            labels["input_ids"][idx] = labels["input_ids"][idx][:max_concat_length]

        # Pad all of the model inputs back up to the max length as a batch & update the labels
        # in our model inputs to point back to the batch of updated labels
        model_inputs = tokenizer.pad(
            model_inputs, padding="max_length", max_length=max_concat_length
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

def get(train_file):
    f = open (train_file, "r")
    train_stream = [json.loads(line) for line in f]
    for data in train_stream:
        yield {"input": data["input"], "output": data["output"]}

def infer_max_steps(
    num_epochs: int,
    batch_size: int,
    training_dataset: Union[Dataset, IterableDataset],
):
    # Calculate the number of samples that we have
    if isinstance(training_dataset, Dataset):
        data_len = len(training_dataset)
    else:
        data_len = 0
        for _ in training_dataset:
            data_len += 1
    # Figure out how many batches we'll have per epoch
    num_batches = data_len // batch_size
    # Assume drop_last=False; in general, this doesn't really matter.
    # We mostly do this to avoid strange behavior when the dataset
    # size is smaller than the batch size.
    if num_batches != (data_len * batch_size):
        num_batches += 1
    num_steps = num_batches * num_epochs
    return num_steps

def get_max_seq_len(model_inputs, labels):
    """Gets the max length of the concat sequence. Assumes that we are operating in batch mode.
    """
    max_seq_len = 0
    for source_ids, target_ids in zip(model_inputs.input_ids, labels.input_ids):
        # Combined seq adds one for seq end
        concat_len = len(source_ids) + len(target_ids) + 1
        if concat_len > max_seq_len:
            max_seq_len = concat_len
    return max_seq_len

# tokenizer = AutoTokenizer.from_pretrained(
#         '/Users/sukritisharma/workspace/fms-hf-tuning/tuned_models/llama_float32_50_epochs',
#         use_fast = True
#     )
# stream = preprocess_function('/Users/sukritisharma/workspace/fms-hf-tuning/dataset_twitter.json', tokenizer, 100000, use_iterable_dataset=True)
# print(type(stream))
# #print(stream.dset['labels'])




