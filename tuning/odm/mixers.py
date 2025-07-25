# Standard
from typing import List
import math
import random

# Third Party
import numpy as np


class RLAgent:
    def __init__(
        self,
        weights: List[float],
        smoothing_factor: float = 0.9,
        alpha: float = 1.0,
    ):
        self.num_domains = len(weights)
        self.weights = weights
        self._estimated_reward = [0.0] * self.num_domains
        total_weights = np.sum(weights)
        self._probabilities = [weight / total_weights for weight in weights]
        self.eps = 1 / self.num_domains
        self.prev_eps = None
        self.smoothing_factor = smoothing_factor
        self.vars_to_log = ["_probabilities", "_estimated_reward"]
        self.all_done = False
        self.iteration = 0
        self.alpha = alpha

    def sample(self):
        index = random.choices(
            np.arange(self.num_domains), weights=self._probabilities
        )[0]
        return index

    def mark_done(self, index: int):
        """
        Marks the domain as done, which means it will not be sampled again.
        """
        self._probabilities[index] = 0
        total_weights = sum(self._probabilities)
        if total_weights > 0:
            self._probabilities = [p / total_weights for p in self._probabilities]
        else:
            self.all_done = True

    def update(self, index: int, reward: float) -> List[float]:
        """
        Updates the weights based on the provided reward.
        """
        self.iteration += 1

        # update cumulative estimated reward
        self._estimated_reward[index] = self.smoothing_factor * self._estimated_reward[
            index
        ] + (1 - self.smoothing_factor) * math.exp(reward)

        # calculate epsilons
        self.prev_eps = self.eps
        self.eps = min(
            1 / self.num_domains,
            math.sqrt(math.log(self.num_domains) / (self.num_domains * self.iteration)),
        )

        # calculate scaling factor
        total_estimated_rewards = sum(
            [math.exp(r * self.prev_eps) for r in self._estimated_reward]
        )
        scaling_factor = (1 - self.num_domains * self.eps) / total_estimated_rewards

        # update weights
        for i in range(self.num_domains):
            self.weights[i] = (
                math.exp(self._estimated_reward[i] * self.prev_eps) * scaling_factor
                + self.eps
            )

        # update probabilities
        total_weights = sum(self.weights)
        for i in range(self.num_domains):
            self._probabilities[i] = self.weights[i] / total_weights

        ############### Exaggerate differences in probabilities
        for i in range(self.num_domains):
            self._probabilities[i] = self._probabilities[i] ** self.alpha
        total_probabilities = sum(self._probabilities)
        for i in range(self.num_domains):
            self._probabilities[i] /= total_probabilities
        ##################

        return self._probabilities

    def reset(self):
        """
        Resets the agent's state.
        """
        self._estimated_reward = [0] * self.num_domains
        total_weights = np.sum(self.weights)
        self._probabilities = [weight / total_weights for weight in self.weights]
        self.eps = 1 / self.num_domains
        self.prev_eps = None
        self.all_done = False
        self.iteration = 0

    def group_update(self, idx: List[int], rewards: List):
        self.iteration += 1
        # calculate epsilons
        self.prev_eps = self.eps
        self.eps = min(
            1 / self.num_domains,
            math.sqrt(math.log(self.num_domains) / (self.num_domains * self.iteration)),
        )

        # update cumulative estimated reward
        for index, reward in zip(idx, rewards):
            # smoothed mean
            # self._estimated_reward[name] = self.smoothing_factor*self._estimated_reward[name] + (1-self.smoothing_factor)*reward
            # smoothed exponentiated mean
            self._estimated_reward[
                index
            ] = self.smoothing_factor * self._estimated_reward[index] + (
                1 - self.smoothing_factor
            ) * math.exp(
                reward
            )
        # print(f"Rank: {torch.distributed.get_rank()} -- estimated_reward {self._estimated_reward}")

        # calculate normalized scaling factor
        total_estimated_rewards = sum(
            (r * self.prev_eps) for r in self._estimated_reward
        )
        scaling_factor = (1 - self.num_domains * self.eps) / total_estimated_rewards

        # update weights
        for i in range(self.num_domains):
            # self.weights[self.dataset_map[name]] = math.exp(self._estimated_reward[name]*self.prev_eps)*scaling_factor + self.eps
            self.weights[i] = (
                self._estimated_reward[i] * self.prev_eps * scaling_factor + self.eps
            )

        # update probabilities
        total_weights = sum(self.weights)
        for i in range(self.num_domains):
            self._probabilities[i] = self.weights[i] / total_weights

        return self._probabilities


if __name__ == "__main__":
    agent = RLAgent([1, 1])
    for i in range(100):
        index = agent.sample()
        print(f"Sampled index: {index}")
        reward = 5 * index
        print(f"Reward: {reward}")
        agent.update(index, reward)
        print(f"Updated probabilities: {agent._probabilities}")
        if agent.all_done:
            print("All domains done.")
            break
