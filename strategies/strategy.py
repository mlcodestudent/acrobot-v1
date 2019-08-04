from gym.spaces import Box, Discrete


class Strategy:
    def action(self, state: Box, action_space: Discrete) -> int:
        raise NotImplemented
