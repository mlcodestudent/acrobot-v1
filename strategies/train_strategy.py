import numpy as np
from exporation_policy.exporation_policy import ExplorationPolicy
from memory import Memory, MemoryElement
from strategies.strategy import Strategy
from gym.spaces import Box, Discrete
from keras import Model


class TrainStrategy(Strategy):
    def __init__(self, model: Model, exploration_policy: ExplorationPolicy):
        self._exploration_policy = exploration_policy
        self._model = model

    def action(self, state: Box, action_space: Discrete) -> int:
        if self._exploration_policy.should_explore():
            return action_space.sample()
        else:
            predict = self._model.predict(np.array([state]))
            return np.argmax(predict).item()

    def update_weights(self, memory: Memory):
        gamma = 0.95

        elements = memory.sample(200)
        x = []
        y = []

        for element in elements:
            x.append(element.old_state)
            new_reward = element.reward + gamma * np.max(self._predict(element.new_state))
            scores = self._predict(element.old_state)
            scores[element.action] = new_reward
            y.append(scores)

        x = np.asarray(x)
        y = np.asarray(y)

        self._model.fit(x=x, y=y, verbose=0)

    def _predict(self, state):
        return self._model.predict(np.asarray([state]))[0]