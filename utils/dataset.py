from agents import Agent
from gym import Env

from .training import run_episode


def new_dataset(agent: Agent, env: Env, size):
    dataset = list()
    for _ in range(size):
        episode = get_new_episode(agent, env)
        dataset.append(episode)
    return dataset


def get_new_episode(agent: Agent, env: Env):
    episode = list()
    for observation, action, reward, next_observation, terminated, _ in run_episode(agent, env):
        episode.append((observation, action, reward, next_observation, terminated))
    return episode
