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

# Third Party
from torch.utils.data import Dataset, IterableDataset
import numpy as np
import torch

# Local
from tuning.odm.mixers import RLAgent


class SimpleTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.inputs = [
            tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            for text in texts
        ]
        self.texts = [
            text
            for text, ip in zip(self.texts, self.inputs)
            if ip["input_ids"][0][-1] == tokenizer.pad_token_id
        ]
        self.inputs = [
            ip for ip in self.inputs if ip["input_ids"][0][-1] == tokenizer.pad_token_id
        ]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = self.inputs[idx]
        input_ids = input["input_ids"].squeeze(0)
        attention_mask = input["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # <== mask pad tokens

        ###### Learn on last token only
        labels = torch.full_like(input_ids, -100)  # Initialize all labels as -100

        # Find the last non-padding token
        last_token_index = attention_mask.nonzero(as_tuple=True)[0][-1]
        labels[last_token_index] = input_ids[
            last_token_index
        ]  # Only train on last token
        ####################

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "text": self.texts[idx],
        }


class OnlineDataset(IterableDataset):
    def __init__(
        self, datasets, model, sample_interval=10, update_interval=10, **kwargs
    ):
        super().__init__()
        self.datasets = datasets
        self.model = model
        self.dataset_iters = [iter(dataset) for dataset in datasets]
        self.current_domain = None
        self.ready_for_iteration = (
            True  # Flag to indicate if the dataset is ready for iteration
        )
        self.iteration = 0
        self.domain_logs = []
        # sampling interval for the mixer
        self.sample_interval = sample_interval
        # update interval for the mixer. Here the interval is in the units of steps / batches
        self.update_interval = update_interval
        print("sample interval", self.sample_interval)
        print("update interval", self.update_interval)


    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __iter__(self):
        # Since the individual datasets are iterated infinitely by resetting the iter pointer
        # we ideally should use infinite loop here.
        self.current_iter_sequential = -1
        self.sample_iter = 0
        self.update_iter = 0

        self.yielded = 0

        while True:
            # call to mixer's sample functionality should be done at every sample_interval
            # so that the frequency of the next domain or sample from the mixer is controlled
            if self.sample_iter % self.sample_interval == 0:
                self.current_domain = self.sample()
            self.sample_iter += 1
            self.domain_logs.append(self.current_domain)
            try:
                sample = next(self.dataset_iters[self.current_domain])
            except StopIteration:
                # If the dataset is exhausted, reset the iterator
                self.dataset_iters[self.current_domain] = iter(
                    self.datasets[self.current_domain]
                )
                sample = next(self.dataset_iters[self.current_domain])
            if self.yielded < 20:
                sample['yield_num'] = self.yielded
                with open("outputs/samples.jsonl", "a") as f:
                    f.write(str(sample) + "\n")
            self.yielded += 1
            # TODO: fix first batch missing error
            # we only return input_ids and labels
            # we let the tuning stack just do causal modeling on top of these
            yield {
                "input_ids": torch.LongTensor(sample["input_ids"]),
                "labels": torch.LongTensor(sample["labels"]),
            }

    def update_mixer(self, batch, train_loss, eval_loss):
        """
        This method should be called to feed the mixer with training signals in this case it is batch,
        training and eval loss
        :param batch: The current batch of data (from data loader).
        :param train_loss: Training loss
        :param eval_loss: Eval loss
        :return:
        """
        # mixer updates should happen in controlled manner through update_interval
        if self.update_iter % self.update_interval == 0:
            batch["metadata"] = {"domain_index": self.current_domain}
            self.update(batch, train_loss, eval_loss)
        self.update_iter += 1

    def sample(self):
        """
        This method should be implemented by subclasses to determine the next item.
        Item can be a domain (dataset corpus) to choose from or an exact sample.
        :return:
            int: The index of the next domain to sample from or directly the exact sample.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def update(self, batch, train_loss, eval_loss):
        """
        This method should be implemented by subclasses to handle training signals
        and update the internal data mixer algorithm.
        :param batch: The current batch of data (with metadata).
        :param loss: The loss or reward signal to process.
        """
        raise NotImplementedError("This method should be implemented by subclasses")


class UniformDataMixing(OnlineDataset):
    def __init__(self, datasets, model, sample_interval=10, update_interval=10, **kwargs):
        super().__init__(datasets, model, sample_interval, update_interval, **kwargs)
        self.num_domains = len(self.datasets)
        self.rl_agent = RLAgent([1 for _ in range(self.num_domains)])

    def sample(self):
        return self.rl_agent.sample()

    def update(self, batch, train_loss, eval_loss):
        pass


class StaticDataMixing(OnlineDataset):
    def __init__(self, datasets, model, sample_interval=10, update_interval=10, weights=None, **kwargs):
        super().__init__(datasets, model, sample_interval, update_interval, **kwargs)
        self.num_domains = len(self.datasets)
        assert len(weights) == self.num_domains
        self.rl_agent = RLAgent(weights)

    def sample(self):
        return self.rl_agent.sample()

    def update(self, batch, train_loss, eval_loss):
        pass


class OnlineDataMixing(OnlineDataset):
    def __init__(self, datasets, model, sample_interval=10, update_interval=10, alpha=1.0, **kwargs):
        super().__init__(datasets, model, sample_interval, update_interval, **kwargs)
        self.num_domains = len(self.datasets)
        self.rl_agent = RLAgent([1 for _ in range(self.num_domains)], alpha=alpha)

    def sample(self):
        return self.rl_agent.sample()

    def update(self, batch, train_loss, eval_loss):
        domain_index = batch["metadata"]["domain_index"]
        # prevents overflow while calculating the exponential of the reward
        reward = np.clip(train_loss, 0, 5)
        self.rl_agent.update(domain_index, reward=reward)


class SequentialDataMixing(OnlineDataset):
    def __init__(self, datasets, model, sample_interval=10, update_interval=10,  sequence=None, **kwargs):
        super().__init__(datasets, model,sample_interval, update_interval, **kwargs)
        self.num_domains = len(self.datasets)
        self.total_iter = kwargs['max_steps']
        print("Total iteration (from dataset):", self.total_iter)
        self.sequence = sequence if sequence is not None else np.arange(self.num_domains)
        if self.total_iter % len(self.sequence) != 0:
            raise RuntimeError("Total iteration should be a multiple of sequence length.")
        self.iter_per_mix = self.total_iter // len(self.sequence)
        print("Iteration per mix (from dataset):", self.iter_per_mix)
        self.rl_agent = RLAgent([1.0 for _ in range(self.num_domains)])

    def sample(self):
        current_index = self.current_iter_sequential // self.iter_per_mix
        current_domains = self.sequence[current_index]
        probabilities = np.zeros(self.num_domains)
        if isinstance(current_domains, (list, tuple, np.ndarray)):
            probabilities[current_domains] = 1.0 / len(current_domains)
        else:
            probabilities[current_domains] = 1.0
        self.rl_agent._probabilities = probabilities
        self.current_iter_sequential += 1
        return self.rl_agent.sample()

    def update(self, batch, loss, eval_loss):
        pass

