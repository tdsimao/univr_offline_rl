import argparse
import os
import pickle
from collections import namedtuple
from multiprocessing import Pool

import pandas as pd
import yaml
from gym import Env

import gym
from agents import Agent
from mdp import MDP
from mdp import estimate_mdp
from utils import evaluate_agent
from utils import new_dataset



# # run full experiments
# DEFAULT_SEEDS = [1]
# DEFAULT_N_WEDGES = [5, 7, 10, 15, 20, 30, 50, 70, 100]
# DEFAULT_DATASET_SIZES = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]


# run fewer experiments during development
DEFAULT_SEEDS = [1]
DEFAULT_N_WEDGES = [1, 10, 100]
DEFAULT_DATASET_SIZES = [1,  10,  100,  1000,  10000]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--behavior_policy_path",
        default="data/Taxi-v3.pkl",
        help="file containing the behavior policy"
    )
    parser.add_argument(
        "-s", "--seeds",
        default=DEFAULT_SEEDS,
        type=int,
        nargs="+",
        help="list of seeds"
    )
    parser.add_argument(
        "-c", "--cpus",
        default=1,
        type=int,
        help="number of cpus to par"
    )
    parser.add_argument(
        "--dataset_sizes",
        default=DEFAULT_DATASET_SIZES,
        type=int,
        nargs="+",
        help="number of worker processes"
    )
    parser.add_argument(
        "--n_wedges",
        default=DEFAULT_N_WEDGES,
        type=int,
        nargs="+",
        help="list of hyper-parameters to be evaluated"
    )
    parser.add_argument(
        "--output_path",
        default="results/",
        type=str,
        help="output directory (default=results)"
    )
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="runs on verbose mode"
    )

    return parser.parse_args()


Result = namedtuple(
    'Result',
    'algorithm, expected_return, seed, dataset_size, n_wedge'
)


def experiment(
        seed: int,
        behavior_policy_path: str,
        output_path: str,
        verbose: bool,
        n_wedges: list,
        dataset_sizes: list
):
    baseline_policy, discount, env_id = get_baseline_policy(behavior_policy_path)

    eval_env: Env = gym.make(env_id)
    eval_env.reset(seed=seed)

    out_dir = os.path.join(output_path, env_id, f"behavior_policy")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'{seed}.yaml'), 'w') as f:
        yaml.dump(data={
            "behavior_policy_path": behavior_policy_path,
            "output_path": output_path,
            "verbose": verbose,
            "seed": seed,
            "discount": discount,
            "n_wedges": n_wedges,
            "env_id": env_id,
            "dataset_sizes": dataset_sizes,
        }, stream=f)

    data_collection_env: Env = gym.make(env_id)
    data_collection_env.reset(seed=seed)
    baseline_agent = Agent(data_collection_env, baseline_policy, seed)
    results = []

    for dataset_size in dataset_sizes:
        # data collection
        dataset = new_dataset(baseline_agent, data_collection_env, dataset_size)
        # estimate model
        mdp: MDP = estimate_mdp(
            dataset,
            baseline_agent.num_observations,
            baseline_agent.num_actions,
            discount,
            baseline_policy.copy(),
            n_wedge=0
        )
        # compute new policy (BASIC RL)
        basic_rl_fsc = mdp.basic().solve(verbose=verbose)
        basic_rl_eval = evaluate(discount, eval_env, basic_rl_fsc, seed, verbose=verbose, label="evaluate basic rl")
        basic_rl_result = Result("Basic RL", basic_rl_eval, seed, dataset_size, 0)
        results.append(basic_rl_result._asdict())

        print(f"seed {seed:4}: {basic_rl_result}")

        for n_wedge in n_wedges:
            mdp.set_n_wedge_and_update_mask(n_wedge)
            # compute new policy (SPIBB)
            spi_fsc = mdp.spibb().solve(verbose=verbose)
            spibb_eval = evaluate(discount, eval_env, spi_fsc, seed, verbose=verbose, label="evaluate spibb")
            spibb_result = Result("SPIBB", spibb_eval, seed, dataset_size, n_wedge)
            results.append(spibb_result._asdict())
            print(f"seed {seed:4}: {spibb_result}")

    df = pd.DataFrame.from_records(results)
    df.to_csv(os.path.join(out_dir, f"{seed}.csv"))


def get_baseline_policy(behavior_policy_path):
    with open(behavior_policy_path, 'rb') as f:
        data = pickle.load(f)
    return (
        data["baseline_policy"],
        data["discount"],
        data["env_id"],
    )


def evaluate(disc, env, policy, seed, num_episodes=100, **kwargs):
    agent = Agent(env, policy, seed=123)
    env.reset(seed=seed)

    return evaluate_agent(agent, env, num_episodes=num_episodes, disc=disc, **kwargs)


if __name__ == '__main__':
    args = vars(parse_args())
    seeds = args.pop("seeds")
    cpus = args.pop("cpus")

    def f(s):
        experiment(s, **args)

    if cpus > 1:
        with Pool(cpus) as p:
            p.map(f, seeds)
    else:
        for s in seeds:
            f(s)
