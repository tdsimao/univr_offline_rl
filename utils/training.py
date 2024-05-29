import numpy as np
from tqdm import trange
from gym import Env

from agents import Agent


def run_episode(agent, env, render=False, training=False):
    terminated, truncated = False, False
    observation, info = env.reset()
    if render:
        env.render()
    while not (terminated or truncated):
        action = agent.act(observation, training=training)
        next_observation, reward, terminated, truncated, info = env.step(action)
        if training:
            agent.update(observation, action, reward, next_observation, terminated, info)
        if render:
            env.render()
        yield observation, action, reward, next_observation, terminated, info
        observation = next_observation
    agent.end_episode()


def run(agent: Agent, env: Env, num_episodes: int, disc: float, verbose=False, label="", training=True):
    results = []
    ret = 0
    with trange(num_episodes, desc=label, postfix=f"episode {ret}", disable=not verbose) as progress_bar:
        for _ in progress_bar:
            rewards = get_rewards(run_episode(agent, env, render=False, training=training))
            ret = discounted_return(disc, rewards)
            results.append(ret)
            progress_bar.set_postfix(ret=ret)
    return results


def evaluate_agent(agent: Agent, env: Env, num_episodes: int, disc: float, **kwargs) -> float:
    episodic_returns = np.array(run(agent, env, num_episodes, disc, training=False, **kwargs))
    return float(episodic_returns.mean())


def discounted_return(disc, rewards):
    return sum(r * disc ** t for t, r in enumerate(rewards))


def get_rewards(episode):
    for _, _, reward, _, _, _ in episode:
        yield reward
