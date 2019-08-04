from strategies.strategy import Strategy
from gym.spaces import Discrete, Box


class RandomStrategy(Strategy):
    def action(self, state: Box, action_space: Discrete) -> int:
        return action_space.sample()
