import numpy as np
from policy import FSC
from gym import Env

from utils import EncoderDecoder
from .qlearning import QLearning


class QLearningFSCAgent(QLearning):
    """
    this agent computes a policy based on the memory of an FSC
    """
    def __init__(self, env: Env, fsc: FSC, seed, **kwargs):
        self.fsc = fsc
        super().__init__(env=env, seed=seed, **kwargs)
        self.encoder = EncoderDecoder([range(self.fsc.nM), range(self.num_observations)])
        self.Q_table = np.zeros([self.encoder.size, self.num_actions])

    def act(self, obs, training):
        return super().act(self._get_memory_observation(obs), training)

    def _get_memory_observation(self, obs):
        return self.encoder.encode(self.fsc.current_memory, obs)

    def update(self, obs, action, reward, next_obs, done, info):
        mem_obs = self._get_memory_observation(obs)
        self.fsc.update_memory(obs, action)
        nex_mem_obs = self._get_memory_observation(next_obs)
        super().update(mem_obs, action, reward, nex_mem_obs, done, info)

    def end_episode(self):
        super().end_episode()
        self.fsc.reset()

    def _seed(self, seed):
        super()._seed(seed)
        self.fsc.seed(seed)

    def new_policy(self):
        return self.fsc.copy()

    def _softmax_policy(self, o, beta, memory):
        q_values = self.Q_table[self.encoder.encode(memory, o)]
        exp = np.exp(beta * q_values)
        return exp / exp.sum()

    def _greedy_policy(self, o, memory):
        policy_o = np.zeros(self.num_actions)
        policy_o[self._greedy_action(self.encoder.encode(memory, o))] = 1.0
        return policy_o
