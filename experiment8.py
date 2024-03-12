
# %%
import time
from environments import EnvironmentBanditManual
from agents import AgentActionValuerPolicy
from actionvaluers import SampleAverager
from policies import EpsilonGreedy
from simulators import run_simulation
from plotters import plot_episode_mean, plot_action_values

start_time = time.time()

n = 2
k = 2  # Number of actions the Agent can choose from
max_episodes = 1000  # Number of episodes. Each episode the Agent and Environment are reset
max_time_steps = 1000  # Number of time steps per episode

# Independent variable: reveal_state

# Based on Chapter 2.9 - Associative Search (Contextual Bandits) (page 41) of
# Reinforcement Learning: An Introduction (2nd ed) by Sutton and Barto

# This "contextual bandit" contains 2 states. In state 0, action 1 is optimal (mean value of 20 vs 10). Whereas in state
# 1, action 0 is optimal (mean value of 90 vs 80). When the environment reveals the state to the agent, the agent learns
# separate action values for each state, and so can learn to behave differently depending on the state, and so can
# achieve a mean reward of 20 in state 0 and 90 in state 1, leading to an overall mean reward of 55, and acting
# optimally in each state (except when exploring). When the environment hides the state from the agent, action 0 has a
# value of 0.5*10 + 0.5*90 = 50, and action 1 has a value of 0.5*20 + 0.5*80 = 50. Choosing either action results in a
# mean reward of only 50, and either action is optimal only 50% of the time.

environment_reveal = EnvironmentBanditManual(q_stars=[[10, 20], [90, 80]], n=n, k=k, reward_std=[10, 10], reveal_state=True)
agent10_0 = AgentActionValuerPolicy("agent10_0", SampleAverager(k, default_value=0), EpsilonGreedy(eps=0.1))
agents = [agent10_0]

results8a = run_simulation(max_episodes, max_time_steps, environment_reveal, agents)

print(time.time() - start_time) # 1000 episodes, 1000 time steps = 82 seconds

plot_episode_mean(results8a, "rewards")
plot_episode_mean(results8a, "optimal_actions")
plot_action_values(results8a, "agent10_0", 0)
# %%
start_time = time.time()

environment_hide = EnvironmentBanditManual(q_stars=[[10, 20], [90, 80]], n=n, k=k, reward_std=[10, 10], reveal_state=False)
agent10_0 = AgentActionValuerPolicy("agent10_0", SampleAverager(k, default_value=0), EpsilonGreedy(eps=0.1))
agents = [agent10_0]

results8b = run_simulation(max_episodes, max_time_steps, environment_hide, agents)

print(time.time() - start_time) # 1000 episodes, 1000 time steps = 82 seconds

plot_episode_mean(results8b, "rewards")
plot_episode_mean(results8b, "optimal_actions")
plot_action_values(results8b, "agent10_0", 0)
# %%
