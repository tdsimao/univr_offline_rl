import numpy as np
from operator import xor

from utils import EncoderDecoder
from .policy import Policy


class FSC(Policy):
    current_memory = None

    def __init__(
            self,
            memory_transition,
            action_distribution,
            initial_memory_distribution,
            seed=None
    ):
        super().__init__(action_distribution, seed)
        self.nM, self.nO, self.nA = action_distribution.shape
        assert memory_transition.shape == (self.nM, self.nA, self.nO, self.nM)
        assert initial_memory_distribution.shape == (self.nM,)
        self.m0 = initial_memory_distribution
        self.memory_transition = memory_transition

        self.seed(seed)

        self.reset()

    def reset(self):
        self.current_memory = self.sample(self.m0)
        return self.current_memory

    def get_action(self, observation, step=True):
        action = self.sample(self.action_distribution[self.current_memory, observation, :])
        if step:
            self._step(observation, action)
        return action

    def _step(self, observation, action):
        return self.update_memory(observation, action)

    def update_memory(self, observation, action):
        self.current_memory = self.sample(self.memory_transition[self.current_memory, action, observation, :])
        return self.current_memory

    @classmethod
    def make_uniform_fsc(cls, num_observations, num_actions, memory_size, seed=None):
        memory_transition = np.ones((memory_size, num_actions, num_observations, memory_size)) / memory_size
        action_distribution = np.ones((memory_size, num_observations, num_actions)) / num_actions
        m0 = np.ones(memory_size) / memory_size
        return FSC(memory_transition, action_distribution, m0, seed)

    def copy(self):
        return FSC(self.memory_transition.copy(), self.action_distribution.copy(), self.m0.copy(), self.init_seed)


class FiniteHistoryFSC(FSC):
    def __init__(self, num_observations, num_actions, k, encoder_decoder, **kwargs):
        self.num_observations, self.num_actions, self.k = num_observations, num_actions, k
        self.encoder_decoder = encoder_decoder
        super().__init__(**kwargs)

    @classmethod
    def make_uniform_fsc(cls, num_observations, num_actions, k, seed=None):
        encoder_decoder = EncoderDecoder(cls.get_history_domain(num_observations, num_actions, k))
        memory_size = encoder_decoder.size

        memory_transition = np.zeros((memory_size, num_actions, num_observations, memory_size))
        for m in range(memory_size):
            h = encoder_decoder.decode(m)
            if not cls.valid_history(h, num_observations, num_actions):
                continue
            for o in range(num_observations):
                for a in range(num_actions):
                    new_h = cls.update_history(h, o, a)
                    new_m = encoder_decoder.encode(*new_h)
                    memory_transition[m, a, o, new_m] = 1

        action_distribution = np.ones((memory_size, num_observations, num_actions)) / num_actions
        initial_memory_distribution = np.zeros(memory_size)
        initial_memory_distribution[memory_size - 1] = 1
        return cls(
            num_observations, num_actions, k, encoder_decoder,
            memory_transition=memory_transition,
            action_distribution=action_distribution,
            initial_memory_distribution=initial_memory_distribution,
            seed=seed
        )

    @staticmethod
    def get_history_domain(num_observations, num_actions, k):
        return [range(num_observations + 1), range(num_actions + 1)] * k

    @staticmethod
    def update_history(h, o, a):
        new_h = h[2:] + [o, a]
        return new_h

    def encode(self, *args):
        return self.encoder_decoder.encode(*args)

    def decode(self, i):
        return self.encoder_decoder.decode(i)

    @staticmethod
    def valid_history(h, num_observations, num_actions):
        """
        observation is empty iff the next action is empty
        """
        for o, a in pairwise(h):
            if xor((o == num_observations), (a == num_actions)):
                return False
        return True

    def copy(self):
        return self.__class__(
            self.num_observations, self.num_actions, self.k, self.encoder_decoder,
            memory_transition=self.memory_transition.copy(),
            action_distribution=self.action_distribution.copy(),
            initial_memory_distribution=self.m0.copy(),
            seed=self.init_seed
        )

    def expand_memory(self, new_history_size):
        assert new_history_size >= self.k
        if new_history_size == self.k:
            return self.copy()
        else:
            fsc = self.__class__.make_uniform_fsc(self.num_observations, self.num_actions, new_history_size)
            for m in range(fsc.nM):
                history = fsc.decode(m)
                original_h = self.get_history(history)
                original_m = self.encode(*original_h)
                fsc.action_distribution[m] = self.action_distribution[original_m]
            return fsc

    def get_history(self, history):
        return history[-self.k * 2:][:self.k * 2]


class FiniteObservationHistoryFSC(FiniteHistoryFSC):

    @staticmethod
    def get_history_domain(num_observations, num_actions, k):
        return [range(num_observations + 1)] * k

    @staticmethod
    def update_history(h, o, a):
        new_h = h[1:] + [o]
        return new_h

    @staticmethod
    def valid_history(h, num_observations, num_actions):
        return True

    def get_history(self, history):
        return history[-self.k:][:self.k]


def pairwise(t):
    it = iter(t)
    return zip(it, it)
