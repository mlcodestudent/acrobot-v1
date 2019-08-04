from exporation_policy.exporation_policy import ExplorationPolicy


class GreedyPolicy(ExplorationPolicy):
    def should_explore(self):
        return False