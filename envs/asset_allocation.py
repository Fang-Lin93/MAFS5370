import random
import numpy as np
from agents.agent import Agent
from collections import deque


class AssetAllocationEnv(object):
    """
    A modified version of the asset allocation example in Section 8.4 of R&J's textbook
    for t <= 5: proportion return (t -> t+1)  = a * (Uniform < p) + b * (Uniform > p)
    for 5 < t < T-1: proportion return (t -> t+1) = Normal(mu. var)
    Assume a > r > b
    Assume discount factor = 1 as used in R&J's textbook
    """

    def __init__(self,
                 W0=1,  # the initial wealth
                 r=0.01,  # riskless return rate
                 a=0.03,  # risky asset return rate: with prob = p
                 b=-0.02,  # risky asset return rate: with prob = 1-p
                 p=0.5,  # risky asset prob = 1-p with t <= 5
                 mu=0.02,  # normal risky mean return
                 sigma=0.1,  # normal risky return std
                 change_t=5,  # risky asset return change from binomial to normal
                 T=10,  # horizon
                 utility_factor=1,  # utility_factor for the utility function ('a' in the textbook, page 223)
                 lr=0.01,
                 num_s=20,  # granularity of states
                 num_a=20,  # granularity of actions
                 greedy_max: float = 1,
                 greedy_min: float = 0.1,
                 total_training_steps: int = 1000000,
                 ):

        # interest rate for the riskless asset
        self.r = r
        # parameters for the risky asset
        assert a > r > b
        self.a = a
        self.b = b
        self.p = p
        self.mu = mu
        self.sigma = sigma
        self.policy_term = self.p * (self.a - self.r) / (1 - self.p) / (self.r - self.b)  # simplifying the calculation
        self.binomial_mean = p * a + (1 - p) * b
        # current wealth
        self.W0 = W0
        self.Wt = W0
        # dynamics
        self.t = 0
        self.T = T
        self.change_t = change_t
        self.u = utility_factor

        # Q function approximation
        self.action_space = np.linspace(-10, 10, num=num_a)
        self.action_indices = range(num_a)
        self.state_space = np.linspace(-5, 5, num=num_s)[:-1]
        self.Q_table = - np.random.uniform(size=(num_s, T, num_a))
        self.lr = lr
        self.eps_greedy_fn = lambda t: max(greedy_max - (greedy_max - greedy_min)*t/total_training_steps, greedy_min)
        self.total_training_steps = total_training_steps

    def reset(self):
        self.t = 0
        self.Wt = self.W0
        return (self.Wt, self.t), 0, False, {'Wt': self.Wt}

    def step(self, xt: float) -> tuple[(float, int), float, bool, dict]:
        # action = xt = the amount of allocation to risky asset
        reward = 0
        # xt = np.clip(xt, -10, 10)
        if self.t < self.T:
            if self.t <= self.change_t:  # categorical risky asset
                risky_r = self.a * (random.random() < self.p) + self.b * (random.random() >= self.p)
            else:  # normal risky asset
                risky_r = random.normalvariate(self.mu, self.sigma)
            self.Wt = xt * (1 + risky_r) + (self.Wt - xt) * (1 + self.r)
            self.t += 1
            if self.t == self.T:  # the final step
                reward = - np.exp(-self.u * self.Wt) / self.u
        return (self.Wt, self.t), reward, self.t >= self.T, {'Wt': self.Wt}  # (Wt, t), reward, done, info

    def Q_greedy_action(self, obs: (float, int)):
        Wt, t = obs
        w_id = sum(Wt > self.state_space)
        return self.action_space[self.Q_table[w_id, t].argmax()]

    def q_learning(self, obs, action, reward, next_obs):
        Wt, t = obs
        q_tar = reward
        if t < self.T-1:
            greedy_action = self.Q_greedy_action(next_obs)
            nx_Wt, nx_t = next_obs
            # self.Q_fn(*next_obs, greedy_action)
            q_tar += self.Q_table[sum(nx_Wt > self.state_space), nx_t,  sum(greedy_action > self.action_space)]
        act_id = sum(action > self.action_space)
        w_id = sum(Wt > self.state_space)
        td = q_tar - self.Q_table[w_id, t, act_id]
        self.Q_table[w_id, t, act_id] += self.lr * td
        return td ** 2

    def train(self):
        loss = []
        td_loss = deque(maxlen=100)
        num_steps = 0
        log_interval = self.total_training_steps // 10
        while num_steps < self.total_training_steps:
            obs, reward, done, info = self.reset()
            while not done:
                if random.random() < self.eps_greedy_fn(num_steps):
                    action = self.action_space[np.random.choice(self.action_indices)]
                else:
                    action = self.Q_greedy_action(obs)

                next_obs, reward, done, info = self.step(action)
                td_loss.append(self.q_learning(obs, action, reward, next_obs))
                obs = next_obs
                if num_steps % log_interval == 0:
                    print(f"td_loss={np.mean(td_loss)}")
                    loss.append(np.mean(td_loss))
                num_steps += 1
        return loss

    def optimal_action(self, obs):
        Wt, t = obs
        if t > self.change_t:
            return (self.mu - self.r) / (self.sigma ** 2 * self.u * (1 + self.r) ** (self.T - t - 1))
        c = self.u * (1 + self.r) ** (self.T - t - 1)  # c_t+1

        return np.log(self.policy_term) / (c * (self.a - self.b))

    def suboptimal_action(self, obs):
        """
        An example of suboptimal action, for comparison purposes
        """
        Wt, t = obs
        if t > self.change_t:
            return (self.mu - self.r) / (self.sigma ** 2 * self.u * (1 + self.r) ** (self.T - t - 1))

        return self.Wt if self.binomial_mean > 0 else 0

    def optimal_V(self):
        if self.t > self.change_t:
            return (- np.exp(-(self.mu - self.r) ** 2 * (self.T - self.t) / (2 * self.sigma ** 2)) / self.u *
                    np.exp(- self.u * (1 + self.r) ** (self.T - self.t) * self.Wt))


def evaluate_asset_allocation(agent: Agent, env: AssetAllocationEnv, num_episodes: int) -> (list, list, list[list]):
    returns, Wt, acts = [], [], []
    for _ in range(num_episodes):
        act_records = []
        obs, epi_return, done, info = env.reset()
        while not done:
            action = agent.sample_actions(obs)
            obs, r, done, info = env.step(action)
            epi_return += r
            act_records.append(action)
        returns.append(epi_return)
        Wt.append(info['Wt'])
        acts.append(act_records)
    return returns, Wt, acts
