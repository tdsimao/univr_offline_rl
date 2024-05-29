from .base import Agent


class RandomAgent(Agent):
    def __init__(self, env, seed=None, **kwargs):
        self.action_space = env.action_space
        super().__init__(env, seed)

    def act(self, observation, training):
        return self.random_action()

    def random_action(self):
        return self.action_space.sample()

    def _seed(self, seed):
        self.action_space.seed(seed)
