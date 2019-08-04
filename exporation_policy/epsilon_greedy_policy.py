from exporation_policy.exporation_policy import ExplorationPolicy
import random


class EpsilonGreedyPolicy(ExplorationPolicy):
    def __init__(self, max_epochs):
        self._max_epochs = max_epochs
        self._current_epoch = 0

    def should_explore(self) -> bool:
        return random.random() > self._current_epoch / self._max_epochs

    def increment_epoch(self):
        self._current_epoch += 1

    def current_epoch(self) -> int:
        return self._current_epoch

    def reset(self):
        self._current_epoch = 0

    def end_exploration(self):
        return self._current_epoch >= self._max_epochs