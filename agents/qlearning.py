import numpy as np
from gym.utils import seeding

from policy import Policy

from .random_agent import RandomAgent


class QLearning(RandomAgent):
    rng: np.random.Generator

    def __init__(self, env, seed=None, **kwargs):
        super().__init__(env, seed)
        self.Q_table = np.zeros([self.num_observations, self.num_actions])
        params = self.parse_parameters(kwargs)
        self.discount = params['discount']
        self.alpha, self.alpha_init = params["alpha"], params["alpha"]
        self.epsilon, self.epsilon_init = params["epsilon"], params["epsilon"]
        self.decaying_rate = params["decaying_rate"]
        self.episode = 0
        self.step = 0

    @staticmethod
    def default_parameters():
        return dict(
            alpha=0.9,
            discount=0.95,
            epsilon=1,
            decaying_rate=0.001
        )

    def parse_parameters(self, params):
        default_params = self.default_parameters()
        if params is not None:
            default_params.update(params)
        params = default_params
        return params

    def _seed(self, seed):
        self.rng, seed = seeding.np_random(seed)
        super()._seed(seed)

    def act(self, observation, training):
        if not training:
            return self._greedy_action(observation)
        if self.rng.uniform(0, 1) < self.epsilon:
            return self.random_action()
        else:
            return self._greedy_action(observation)

    def _greedy_action(self, observation):
        choices = self._greedy_actions(observation)
        return self.rng.choice(choices)

    def _greedy_actions(self, observation):
        q_max = self.Q_table[observation].max()
        greedy_actions = np.isclose(self.Q_table[observation], q_max)
        return np.flatnonzero(greedy_actions)

    def _anneal(self):
        self.epsilon = self.epsilon_init * np.exp(-self.decaying_rate * self.episode)
        self.alpha = self.alpha_init * np.exp(-self.decaying_rate * self.episode)

    def update(self, observation, action, reward, next_observation, done, info):
        old = self.Q_table[observation, action]
        estimate_future_value = 0 if done else self.Q_table[next_observation, :].max()
        target = reward + self.discount * estimate_future_value
        self.Q_table[observation, action] = (1 - self.alpha) * old + self.alpha * target
        self.step += 1

    def end_episode(self):
        self.step = 0
        self.episode += 1
        self._anneal()

    def export_policy(self, beta=None):
        fsc = self.new_policy()
        if beta is None:
            fsc.action_distribution = self.greedy_policy()
        else:
            fsc.action_distribution = self.softmax_policy(beta)
        return fsc

    def new_policy(self):
        return Policy.make_uniform_policy(self.num_observations, self.num_actions)

    def greedy_policy(self):
        policy = np.zeros((self.num_observations, self.num_actions))
        for o in range(self.num_observations):
            policy[o] = self._greedy_policy(o)
        return policy

    def _greedy_policy(self, o):
        policy_o = np.zeros(self.num_actions)
        policy_o[self._greedy_action(o)] = 1.0
        return policy_o

    def softmax_policy(self, beta=0.2):
        assert beta > 0
        policy = np.zeros((self.num_observations, self.num_actions))
        for o in range(self.num_observations):
            policy[o] = self._softmax_policy(o, beta)
        return policy

    def _softmax_policy(self, o, beta):
        q_values = self.Q_table[o]
        exp = np.exp(beta * q_values)
        return exp / exp.sum()
