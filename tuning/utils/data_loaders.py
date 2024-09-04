# Standard
from typing import List, Union
import logging

# Third Party
from datasets import Dataset
from datasets import IterableDataset as HFIterableDataset
from datasets import interleave_datasets
from torch.utils.data import IterableDataset
import torch

logger = logging.getLogger(__name__)


class ConstantLengthHybridDataset(
    IterableDataset
):  # pylint: disable=too-many-instance-attributes
    def __init__(  # pylint: disable=super-init-not-called
        self,
        datasets: List[Union[HFIterableDataset, Dataset]],
        sampling_probs: List[float],
        seq_length=1024,
        num_of_sequences=1024,
        tokenizer=None,
        tokens_field="input_ids",
        text_field="contents",
        add_bos_token=True,
        add_eos_token=True,
        infinite=False,
    ):
        """packing for pretokenized datasets for pretraining only
        since all tokens are attended upon packing.

        Args:
            datasets (List[Union[HFIterableDataset, Dataset]]): list of datasets to be packed
            sampling_probs (List[float]): sampling probs for each of the dataset
            seq_length (int, optional): sequence length. Defaults to 1024.
            num_of_sequences (int, optional): max number of sequences can be
            kept in memory for packing.
            Defaults to 1024.
            tokenizer (_type_, optional): tokenizer.
            Defaults to None.
            tokens_field (str, optional): data field having tokens.
            Defaults to "input_ids".
            text_field (str, optional): data field having text content.
            Defaults to "contents".
            add_bos_token (bool, optional): add bos token at the start of each sample.
            Defaults to True.
            add_eos_token (bool, optional): add eos token at the end of each sample.
            Defaults to True.
            infinite (`bool`, *optional*, defaults to `False`):
                If True the iterator is reset after dataset reaches end else stops.
        """
        self.datasets = datasets
        self.sampling_probs = sampling_probs
        self.seq_length = seq_length
        self.current_size = 0
        self.max_buffer_size = seq_length * num_of_sequences
        self.tokenizer = tokenizer
        logger.warning("tokenizer: {}".format(self.tokenizer))
        logger.warning(self.tokenizer.encode("hi hello"))
        self.tokens_field = tokens_field
        self.text_field = text_field
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.dataset = interleave_datasets(datasets=self.datasets, split="train")
        self.column_names = self.dataset.column_names
        self.infinite = infinite
        if self.infinite:
            logger.warning(
                "samples will be provided infinitely.\
                Datasets that are exhausted will be reiterated from start."
            )
        # self._info = self.dataset._info
        # self._epoch = 0
        logger.warning("add_bos_token: {}".format(self.add_bos_token))
        logger.warning("add_eos_token: {}".format(self.add_eos_token))

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):

        # iterator = iter(self.dataset)
        iterators = [iter(dataset) for dataset in self.datasets]
        tokens_seen_so_far = [0] * len(iterators)

        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    need_more_tokens = [
                        self.sampling_probs[i]
                        - (tokens_seen_so_far[i] / (sum(tokens_seen_so_far) + 1e-9))
                        for i in range(len(iterators))
                    ]
                    dataset_id_which_needs_more_tokens = need_more_tokens.index(
                        max(need_more_tokens)
                    )
                    iterator = iterators[dataset_id_which_needs_more_tokens]
                    sample = next(iterator)
                    # when interleaved some datasets though having
                    # tokens field column might not have any data associated
                    if self.tokens_field not in sample or not sample[self.tokens_field]:
                        try:
                            sample[self.tokens_field] = self.tokenizer.encode(
                                sample[self.text_field]
                            )
                        except Exception as e:  # pylint: disable=broad-exception-caught
                            logger.warning(
                                "failed to tokenize the data {} of type {}.".format(
                                    sample[self.text_field][:10],
                                    type(sample[self.text_field]),
                                )
                            )
                            logger.debug(e)
                            sample[self.tokens_field] = []
                            continue
                        # llama3 tokenzier adds bos token by default upon tokenization
                        # we check if there is any such token we remove for consistency
                        # with user interface allowing for adding eos and bos tokens
                        if self.tokenizer.bos_token_id == sample[self.tokens_field][0]:
                            sample[self.tokens_field] = sample[self.tokens_field][1:]
                    if self.add_bos_token:
                        buffer.append(self.tokenizer.bos_token_id)
                    buffer.extend(sample[self.tokens_field])
                    if self.add_eos_token:
                        buffer.append(self.tokenizer.eos_token_id)
                    tokens_seen_so_far[dataset_id_which_needs_more_tokens] += len(
                        sample[self.tokens_field]
                    )
                    buffer_len = len(buffer)
                except StopIteration:
                    if self.infinite:
                        iterators[dataset_id_which_needs_more_tokens] = iter(
                            self.datasets[dataset_id_which_needs_more_tokens]
                        )
                        logger.warning(
                            "iterator is reset for one of the datasets since it is exhausted."
                        )
                    else:
                        more_examples = False
                        break
            all_token_ids = buffer
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }
