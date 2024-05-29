import tqdm
import numpy as np
import numpy.ma as ma
from numpy.testing import assert_approx_equal

from policy import Policy
from utils import EncoderDecoder


class MDP:
    def __init__(
        self,
        ns: int,
        na: int,
        discount: float,
        transition: np.array,
        reward: np.array,
        initial_state_distribution: np.array,
        terminate_prob: np.array,
        encoder: EncoderDecoder,
        policy: Policy = None,
        sa_counter: np.array = None,
        n_wedge: int = 0
    ):
        self.ns, self.na = ns, na
        self.discount = discount
        self.transition = transition
        self.reward = reward
        self.initial_state_distribution = initial_state_distribution
        self.non_terminate = 1 - terminate_prob
        self.encoder = encoder
        self.policy = policy
        self.sa_counter = sa_counter
        self.n_wedge = n_wedge
        self.mask = self.compute_mask()

        self.value_update = self.value_update_basic
        self.get_greedy_action_distribution = self.get_greedy_action_distribution_basic

    def solve(self, residual_tolerance=0.1, verbose=False) -> Policy:
        v_ = np.zeros(self.ns)
        residual = 1
        with tqdm.trange(1, 0, 1, disable=not verbose, desc=f"solving mdp with {self.ns} states") as pbar:
            while residual > residual_tolerance:
                v = v_.copy()
                residual = 0
                for s in range(self.ns):
                    v_[s] = self.value_update(v, s)
                    residual = max(residual, abs(v_[s] - v[s]))
                pbar.set_postfix(dict(residual=residual))
                pbar.update(1)
        return self.compute_policy(v_)

    def value_update_basic(self, v, s):
        return max(
            self.q_values(v, s)
        )

    def q_values(self, v, s):
        return np.array([
            self.reward[s, a] +
            self.non_terminate[s, a] * self.discount * sum(p * v[s_] for s_, p in self.successors(s, a))
            for a in range(self.na)
        ])

    def successors(self, s, a):
        state_action_transition = self.transition[s, a]
        for ind in np.nonzero(state_action_transition)[0]:
            yield ind, state_action_transition[ind]

    def compute_policy(self, v_):
        action_distribution = self.get_greedy_action_distribution(v_)
        return Policy(action_distribution)

    def get_greedy_action_distribution_basic(self, v_):
        action_distribution = np.zeros((self.ns, self.na))
        for s in range(self.ns):
            q = self.q_values(v_, s)
            action_distribution[s, np.argmax(q)] = 1.0
        return action_distribution

    def compute_mask(self):
        return self.sa_counter >= self.n_wedge

    def set_n_wedge_and_update_mask(self, n_wedge):
        self.n_wedge = n_wedge
        self.mask = self.compute_mask()

    def spibb(self) -> 'MDP':
        self.value_update = self.value_update_spibb
        self.get_greedy_action_distribution = self.get_greedy_action_distribution_spibb
        return self

    def value_update_spibb(self, v, s):
        q_values = self.q_values(v, s)
        new_pi = self.get_bootstrapping_policy(q_values, s)
        return q_values @ new_pi

    def get_greedy_action_distribution_spibb(self, v):
        policy = np.zeros((self.ns, self.na))
        for s in range(self.ns):
            q_values = self.q_values(v, s)
            policy[s] = self.get_bootstrapping_policy(q_values, s)
        return policy

    def get_bootstrapping_policy(self, q_values, s):
        pib = self.policy.action_distribution[s]
        mask = self.mask[s]
        new_pi = np.zeros(self.na)

        # TODO implement the policy improvement step for the state s given the SPIBB constraints

        assert (new_pi >= 0).all()
        assert_approx_equal(new_pi.sum(), 1)
        return new_pi

    def basic(self) -> 'MDP':
        self.value_update = self.value_update_basic
        self.get_greedy_action_distribution = self.get_greedy_action_distribution_basic
        return self


def estimate_mdp(
    dataset: list,
    ns: int,
    na: int,
    discount,
    policy: Policy = None,
    n_wedge: int = 0
) -> 'MDP':
    if policy is None:
        policy = Policy.make_uniform_policy(ns, na)

    transition = np.zeros((ns, na, ns))
    reward = np.zeros((ns, na))
    probability_terminate = np.zeros((ns, na))

    counter_t = np.zeros(transition.shape)
    acc_reward = np.zeros(reward.shape)
    termination_counter = np.zeros(reward.shape)
    initial_state_counter = np.zeros(ns)
    for episode in dataset:
        for t, (s, a, r, next_s, terminated) in enumerate(episode):
            if t == 0:
                initial_state_counter[s] += 1
            counter_t[s, a, next_s] += 1
            acc_reward[s, a] += r
            termination_counter[s, a] += int(terminated)
    sa_counter = np.sum(counter_t, 2)

    np.divide(counter_t, sa_counter[:, :, np.newaxis], out=transition, where=sa_counter[:, :, np.newaxis] > 0)
    np.divide(acc_reward, sa_counter, out=reward, where=sa_counter > 0)
    np.divide(termination_counter, sa_counter, probability_terminate, where=sa_counter > 2)

    isd = initial_state_counter / len(dataset)

    return MDP(ns, na, discount, transition, reward, isd, probability_terminate, None, policy, sa_counter, n_wedge)
