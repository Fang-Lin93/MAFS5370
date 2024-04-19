
import random
import numpy as np
from envs import AssetAllocationEnv
from envs.asset_allocation import evaluate_asset_allocation
from agents import RandomAgent
from matplotlib import pyplot as plt


def run(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)

    env = AssetAllocationEnv(total_training_steps=1000000)
    env.train()
    rand_returns, rand_wealth, rand_a = evaluate_asset_allocation(RandomAgent(sample_fn=
                                                                              lambda x: random.uniform(-2, 2)),
                                                                  AssetAllocationEnv(), 10000)
    print(f'random: return={np.mean(rand_returns)}, wealth={np.mean(rand_wealth)}')

    sub_returns, sub_wealth, sub_a = evaluate_asset_allocation(
        RandomAgent(sample_fn=lambda x: env.suboptimal_action(x)),
        AssetAllocationEnv(), 10000)
    print(f'suboptimal: return={np.mean(sub_returns)}, wealth={np.mean(sub_wealth)}')

    opt_returns, opt_wealth, opt_a = evaluate_asset_allocation(RandomAgent(sample_fn=lambda x: env.optimal_action(x)),
                                                               AssetAllocationEnv(),
                                                               10000)
    print(f'optimal: return={np.mean(opt_returns)}, wealth={np.mean(opt_wealth)}')

    q_returns, q_wealth, q_a = evaluate_asset_allocation(RandomAgent(sample_fn=lambda x: env.Q_greedy_action(x)),
                                                         AssetAllocationEnv(),
                                                         10000)
    print(f'Q-learning: return={np.mean(q_returns)}, wealth={np.mean(q_wealth)}')

    rand_a = np.array(rand_a).mean(axis=0)
    sub_a = np.array(sub_a).mean(axis=0)
    opt_a = np.array(opt_a).mean(axis=0)
    q_a = np.array(q_a).mean(axis=0)

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].hist(rand_returns, bins=100, label=f'rand ({np.mean(rand_returns):.3f})', range=(-1, 0), alpha=0.7)
    ax[0].hist(sub_returns, bins=100, label=f'sub opt ({np.mean(sub_returns):.3f})', range=(-1, 0), alpha=0.7)
    ax[0].hist(opt_returns, bins=100, label=f'opt ({np.mean(opt_returns):.3f})', range=(-1, 0), alpha=0.7)
    ax[0].hist(q_returns, bins=100, label=f'Q_agent ({np.mean(q_returns):.3f})', range=(-1, 0), alpha=0.7)
    ax[0].set_title("Utilities")
    ax[0].legend()

    ax[1].hist(rand_wealth, bins=100, label=f'rand ({np.mean(rand_wealth):.3f})', range=(-1, 5), alpha=0.7)
    ax[1].hist(sub_wealth, bins=100, label=f'sub opt ({np.mean(sub_wealth):.3f})', range=(-1, 5), alpha=0.7)
    ax[1].hist(opt_wealth, bins=100, label=f'opt ({np.mean(opt_wealth):.3f})', range=(-1, 5), alpha=0.7)
    ax[1].hist(q_wealth, bins=100, label=f'Q_agent ({np.mean(q_wealth):.3f})', range=(-1, 5), alpha=0.7)
    ax[1].set_title("Wealth")
    ax[1].legend()

    ax[2].plot(rand_a, label='random')
    ax[2].plot(sub_a, label='suboptimal')
    ax[2].plot(opt_a, label='optimal')
    ax[2].plot(q_a, label='Q-learning')
    ax[2].set_title("Mean Actions (xt)")
    ax[2].legend()
    fig.show()


if __name__ == '__main__':
    run()
