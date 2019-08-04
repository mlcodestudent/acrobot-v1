import gym
from gym.spaces import Box, Discrete
from strategies.strategy import Strategy
from memory import Memory
from typing import List, Optional


class Game:
    def __init__(self):
        self._env = gym.make("Acrobot-v1")

    def action_space(self) -> Discrete:
        return self._env.action_space

    def observation_space(self) -> Box:
        return self._env.observation_space

    def play(self, strategy: Strategy, memory: Optional[Memory], render: bool) -> int:
        state = self._env.reset()
        done = False
        reward_sum = 0

        while not done:
            action = strategy.action(state=state, action_space=self._env.action_space)
            new_state, reward, done, info = self._env.step(action)
            reward_sum += reward

            if memory:
                memory.add_element(old_state=state, action=action, reward=reward, new_state=new_state)

            if render:
                self._env.render()

            state = new_state

        return reward_sum

    def play_several_games(self, strategy: Strategy, memory: Optional[Memory], games_count: int) -> List[int]:
        results: List[int] = []

        for i in range(games_count):
            result = self.play(strategy=strategy, memory=memory, render=False)
            results.append(result)

        return results
