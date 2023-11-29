# %%
import random
import numpy as np
from matplotlib import pyplot as plt
from main import Environment

def plot_sampled_rewards(num_samples=5,
                         k=10,
                         q_star_mean=0,
                         q_star_std=1,
                         reward_std=1,
                         seed=None):
    """Plots rewards sampled for each of k actions, with a line for the long term 
    mean reward.
    
    The long term means for each action are normally distributed with mean of
    q_star_mean and standard deviation of q_star_std. Rewards are normally
    distributed around the long term mean with standard deviation of reward_std."""
    
    random.seed(seed)
    env = Environment(k, q_star_mean, q_star_std, reward_std)

    sampled_rewards = []
    for action in range(k):
        sampled_rewards.append([])
        for i in range(num_samples):
            sampled_rewards[action].append(env.reward(action))

    for action, rewards in enumerate(sampled_rewards):
        plt.plot([action-0.3, action+0.3], [env.q_star[action]]*2)
        plt.plot([action]*num_samples, sampled_rewards[action], "o")
    plt.show()
    return env.q_star, sampled_rewards
# %%
q_star, sampled_rewards = plot_sampled_rewards()
print("q_star", q_star)
for i, q_star in enumerate(q_star):
    print(q_star, sampled_rewards[i])
# %%
def mean_maximum_q_star(num_samples=1000, 
                        k=10,
                        q_star_mean=0,
                        q_star_std=1,
                        seed=None):
    """Estimates the mean value of the maximum of k normally distributed random
    variables, each with a mean of q_star_mean and standard deviation of
    q_star_std."""

    random.seed(seed)
    env = Environment(k, q_star_mean, q_star_std)

    max_q_stars = []
    for i in range(num_samples):
        max_q_stars.append(max(env.q_star))
        env.reset()
    mean = sum(max_q_stars) / num_samples
    return mean
    
# %%
ks = list(range(1, 11))
means = []

for k in ks:
    means.append(mean_maximum_q_star(k=k))
    print(k, means[-1])
# %%
plt.plot(ks, means)
plt.show()
# %%
