from .base import Agent
from policy import FSC
from gym import Env


class FSCAgent(Agent):
    """
    this class acts as a wrapper around an FSC
    """
    def __init__(self, env: Env, fsc: FSC, seed=None):
        super().__init__(env=env, seed=seed)
        self.policy = fsc

    def end_episode(self):
        self.policy.reset()

    def _seed(self, seed):
        super()._seed(seed)
        self.policy.seed(seed)
