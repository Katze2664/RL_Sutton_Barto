
# %%
import time
from environments import EnvironmentBanditNonstationary
from agents import AgentActionValuerPolicy
from actionvaluers import SampleAverager
from policies import EpsilonGreedy
from simulators import run_simulation
from plotters import plot_rollout_mean, plot_action_values

start_time = time.time()

# Exercise 2.5, page 33 of Reinforcement Learning: An Introduction (2nd ed) by Sutton and Barto

k = 10  # Number of actions the Agent can choose from
max_rollouts = 2000  # Number of rollouts. Each rollout the Agent and Environment are reset
max_time_steps = 1000  # Number of time steps per rollout

# Independent variables: calc_stepsize

# The environment is non-stationary (q_star values perform a random walk starting at zero). Compared to an
# equally-weighted sample average (stepsize = 1/n), an exponential recency-weighted sample average (stepsize = constant)
# adapts more slowly initially, but over longer time horizons it is able to adapt more quickly to a non-stationary
# target. Though it is susceptible to overcorrection with noisy reward signals for stationary targets.

environment = EnvironmentBanditNonstationary(q_stars_std=0, rand_walk_std=0.01, k=k)

# Equally weighted sameple average uses harmonic stepsize
agent_equally_weighted = AgentActionValuerPolicy("agent_equally_weighted", SampleAverager(k, calc_stepsize=lambda count: 1/count), EpsilonGreedy(eps=0.10))

# Exponential recency-weighted averaging (erwa) uses constant stepsize
agent_recency_weighted = AgentActionValuerPolicy("agent_recency_weighted", SampleAverager(k, calc_stepsize=lambda count: 0.1), EpsilonGreedy(eps=0.10))
agents = [agent_equally_weighted, agent_recency_weighted]

results4 = run_simulation(max_rollouts, max_time_steps, environment, agents)

print(time.time() - start_time) # 2000 rollouts, 1000 steps = 99 seconds
# %%
plot_rollout_mean(results4, "rewards")
plot_rollout_mean(results4, "optimal_actions")
# %%
rollout = 0
plot_action_values(results4, "agent_equally_weighted", rollout)
plot_action_values(results4, "agent_recency_weighted", rollout)
# %%
