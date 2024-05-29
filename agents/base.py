from gym import Env
from gym.spaces import Discrete



class Agent:
    def __init__(self, env: Env, policy, seed=None):
        self.policy = policy
        assert isinstance(env.action_space, Discrete)
        assert isinstance(env.observation_space, Discrete)
        self.num_actions: int = env.action_space.n
        self.num_observations: int = env.observation_space.n
        self._seed(seed)

    def act(self, obs, training):
        return self.policy.get_action(obs, step=True)

    def update(self, observation, action, reward, next_observation, done, info):
        pass

    def _seed(self, seed):
        pass

    def end_episode(self):
        pass
