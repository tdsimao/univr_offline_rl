import numpy as np

from gym.utils import seeding
from utils import EncoderDecoder


class Policy:

    def __init__(
            self,
            action_distribution,
            seed=None
    ):
        self.nO, self.nA = action_distribution.shape
        self.action_distribution = action_distribution
        self.rng = None
        self.init_seed = seed
        self.seed(seed)

    def get_action(self, observation, step=True):
        return self.sample(self.action_distribution[observation, :])

    def seed(self, seed):
        self.rng, seed = seeding.np_random(seed)

    def sample(self, dist):
        return self.rng.multinomial(1, dist).argmax()

    def set_action_distribution(self, new_action_map):
        self.action_distribution = new_action_map

    @classmethod
    def make_uniform_policy(cls, num_observations, num_actions, seed=None):
        action_distribution = np.ones((num_observations, num_actions)) / num_actions
        return cls(action_distribution, seed)

    def copy(self):
        return Policy(self.action_distribution.copy(), self.init_seed)
