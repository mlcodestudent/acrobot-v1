from typing import List
from gym.spaces import Box
import random


class MemoryElement:
    def __init__(self, old_state: Box, action: int, reward: int, new_state: Box):
        self.old_state = old_state
        self.action = action
        self.reward = reward
        self.new_state = new_state


class Memory:
    def __init__(self):
        self._memory: List[MemoryElement] = []

    def add_element(self, old_state: Box, action: int, reward: int, new_state: Box):
        element = MemoryElement(old_state=old_state, action=action, reward=reward, new_state=new_state)
        self._memory.append(element)

    def all_elements(self) -> List[MemoryElement]:
        return self._memory

    def sample(self, count) -> List[MemoryElement]:
        return random.sample(self._memory, count)
