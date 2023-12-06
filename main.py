#%%
import random
import numpy as np
from matplotlib import pyplot as plt

class Environment:
    """Creates a k-bandit environment.
    
    Environment contains only 1 state, in which there are k choices for actions.
    Each action has a mean reward given by self.q_star[action], whose values are
    normally distributed with a mean of q_star_mean and standard deviation of
    q_star_std, and are fixed unless reset. When an action is selected the reward
    given is normally distributed with a mean of self.q_star[action] and standard
    deviation of reward_std.
    """

    def __init__(self, k=10, q_star_mean=0, q_star_std=1, reward_std=1):
        self.k = k
        self.q_star_mean = q_star_mean
        self.q_star_std = q_star_std
        self.reward_std = reward_std

        self.reset()

    def reward(self, action):
        mean = self.q_star[action]
        return random.gauss(mean, self.reward_std)
    
    def reset(self):
        self.q_star = []
        for i in range(self.k):
            self.q_star.append(random.gauss(self.q_star_mean, self.q_star_std))

        self.optimal_action = np.argmax(self.q_star)

# %%
class Agent:
    """An epsilon-greedy agent that estimates action-values based on mean sampled rewards.
    
    Agent chooses a random action with probability eps, otherwise acts greedily,
    choosing the action with the highest mean sampled reward. In the case of a tie
    the action with the smallest index is chosen. Actions that have never been
    sampled are estimated to have default_value."""
    
    def __init__(self, name, k, eps=0, default_value=0, stepsize=None):
        self.name = name
        self.k = k
        self.eps = eps
        self.default_value = default_value
        if stepsize is None:
            self.stepsize = lambda action_count: 1 / action_count  # Equivalent to equal-weight sample averaging
        else:
            self.stepsize = stepsize

        self.reset()

    def receive_reward(self, action, reward):
        self.action_counts[action] += 1
        n = self.action_counts[action]
        q = self.action_values[action]
        self.action_values[action] += self.stepsize(n) * (reward - q)

        self.cumu_reward += reward

    def select_eps_greedy_action(self):
        if random.random() < self.eps:
            # Explore
            action = random.randrange(self.k)
        else:
            # Greedy
            action = np.argmax(self.action_values)
        return action

    def reset(self):
        self.action_counts = [0] * self.k
        self.action_values = [self.default_value] * self.k
        self.cumu_reward = 0

# %%
def run_simulation(k, max_rollouts, max_time_steps, environment, agents):
    cumu_reward_results = {agent.name: np.zeros((max_rollouts, max_time_steps)) for agent in agents}
    optimal_action_results = {agent.name: np.zeros((max_rollouts, max_time_steps)) for agent in agents}

    for rollout in range(max_rollouts):
        environment.reset()
        for agent in agents:
            agent.reset()

        for time_step in range(max_time_steps):
            for agent in agents:           
                action = agent.select_eps_greedy_action()
                reward = environment.reward(action)
                agent.receive_reward(action, reward)

                cumu_reward_results[agent.name][rollout, time_step] = agent.cumu_reward
                if action == environment.optimal_action:
                    optimal_action_results[agent.name][rollout, time_step] = 1
    
    return cumu_reward_results, optimal_action_results
# %%
k = 10  # Number of actions the Agent can choose from
max_rollouts = 2000  # Number of rollouts. Each rollout the Agent and Environment are reset
max_time_steps = 1000  # Number of time steps per rollout

environment = Environment(k)
# agent00_0 = Agent("agent00_0", k, eps=0.00, default_value=0)
# agent01_0 = Agent("agent01_0", k, eps=0.01, default_value=0)
agent10_0 = Agent("agent10_0", k, eps=0.10, default_value=0)
# erwa = exponential recency-weighted averaging
agent10_0_erwa01 = Agent("agent10_0_erwa01", k, eps=0.10, default_value=0, stepsize=lambda n: 0.01)
agent10_0_erwa05 = Agent("agent10_0_erwa05", k, eps=0.10, default_value=0, stepsize=lambda n: 0.05)
agent10_0_erwa20 = Agent("agent10_0_erwa20", k, eps=0.10, default_value=0, stepsize=lambda n: 0.20)
agent10_0_erwa50 = Agent("agent10_0_erwa50", k, eps=0.10, default_value=0, stepsize=lambda n: 0.50)
agent10_0_erwa100 = Agent("agent10_0_erwa100", k, eps=0.10, default_value=0, stepsize=lambda n: 1.00)
agents = [agent10_0, agent10_0_erwa01, agent10_0_erwa05, agent10_0_erwa20, agent10_0_erwa50, agent10_0_erwa100]


cumu_reward_results, optimal_action_results = run_simulation(k, max_rollouts, max_time_steps, environment, agents)
# %%
colors = ["green", "red", "blue", "purple", "orange", "brown"]
color_idx = 0

for agent_name, cumu_reward in cumu_reward_results.items():
    reward_time_mean = cumu_reward / (np.arange(max_time_steps) + 1)
    reward_rollout_mean = reward_time_mean.mean(axis=0)
    reward_rollout_std = reward_time_mean.std(axis=0, ddof=1)
    
    print(agent_name, reward_rollout_mean[-1])
    color = colors[color_idx]
    color_idx += 1
    plt.plot(reward_rollout_mean, label=agent_name, color=color)
    # plt.errorbar(np.arange(max_time_steps), reward_rollout_mean, reward_rollout_std, label=agent_name, color=color)

plt.legend()
plt.show()
# %%
color_idx = 0
for agent_name, optimal_action in optimal_action_results.items():
    color = colors[color_idx]
    color_idx += 1
    optimal_rollout_average = optimal_action.mean(axis=0)
    plt.plot(optimal_rollout_average, label=agent_name, color=color)

plt.legend()
plt.show()

# %%
