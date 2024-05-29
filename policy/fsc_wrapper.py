from gym import Wrapper
from gym.spaces import Discrete

from policy import FSC
from utils import EncoderDecoder


class FSCWrapper(Wrapper):
    def __init__(self, env, fsc: FSC, **kwargs):
        assert isinstance(env.observation_space, Discrete)
        super().__init__(env, **kwargs)
        self.env = env
        self.fsc = fsc
        self.encoder = EncoderDecoder([range(self.fsc.nM), range(self.env.observation_space.n)])
        self.observation_space = Discrete(self.encoder.size)

    def step(self, action):
        next_observation, reward, terminated, truncated, info = self.env.step(action)
        memory = self.fsc.update_memory(next_observation, action)
        return self.encode(memory, next_observation), reward, terminated, truncated, info

    def reset(self, seed=None, **kwargs):
        obs, info = super().reset(seed=seed, **kwargs)
        if seed is not None:
            self.fsc.seed(seed)
        memory = self.fsc.reset()
        return self.encode(memory, obs), info

    def encode(self, memory, next_observation):
        return self.encoder.encode(memory, next_observation)

    def decode(self, i):
        return self.encoder.decode(i)
