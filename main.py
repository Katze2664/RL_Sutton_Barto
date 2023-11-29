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
    
    def __init__(self, name, k, eps=0, default_value=0):
        self.name = name
        self.k = k
        self.eps = eps
        self.default_value = default_value

        self.reset()

    def receive_reward(self, action, reward):
        self.action_count[action] += 1
        self.action_value_sum[action] += reward
        self.action_value[action] = self.action_value_sum[action] / self.action_count[action]
        self.cumu_reward += reward

    def select_eps_greedy_action(self):
        if random.random() < self.eps:
            # Explore
            action = random.randrange(self.k)
            print(self.name, "explore", action)
        else:
            # Greedy
            action = np.argmax(self.action_value)
        return action

    def reset(self):
        self.action_count = [0] * self.k
        self.action_value_sum = [0] * self.k
        self.action_value = [self.default_value] * self.k
        self.cumu_reward = 0


# %%
k = 10  # Number of actions the Agent can choose from
max_rollouts = 2  # Number of rollouts. Each rollout the Agent and Environment are reset
max_time_steps = 5  # Number of time steps per rollout

env = Environment(k)
agent10_0 = Agent("agent10_0", k, eps=0.1, default_value=0)
agent50_0 = Agent("agent50_0", k, eps=0.5, default_value=0)
agents = [agent10_0, agent50_0]

cumu_reward_results = {agent.name: np.zeros((max_rollouts, max_time_steps)) for agent in agents}
optimal_action_results = {agent.name: np.zeros((max_rollouts, max_time_steps)) for agent in agents}

for rollout in range(max_rollouts):
    env.reset()
    for agent in agents:
        agent.reset()

    for time_step in range(max_time_steps):
        for agent in agents:           
            action = agent.select_eps_greedy_action()
            reward = env.reward(action)

            print((f"rollout= {rollout}\n"
                   f"time=    {time_step}\n"
                   f"name=    {agent.name}\n"
                   f"sum      {np.round(agent.action_value_sum, 2)}\n"
                   f"count    {agent.action_count}\n"
                   f"value    {np.round(agent.action_value, 2)}\n"
                   f"q_star   {np.round(env.q_star, 2)}\n"
                   f"action   {action}\n"
                   f"optimal  {env.optimal_action}\n"
                   f"reward   {np.round(reward, 2)}"))

            agent.receive_reward(action, reward)

            cumu_reward_results[agent.name][rollout, time_step] = agent.cumu_reward
            if action == env.optimal_action:
                optimal_action_results[agent.name][rollout, time_step] = 1
            print(np.round(cumu_reward_results[agent.name], 2))
            print(optimal_action_results[agent.name], "\n")





# %%
