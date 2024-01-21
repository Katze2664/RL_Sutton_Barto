
# %%
from environments import EnvironmentBanditNonstationary
from agents import AgentActionValuerPolicy
from actionvaluers import SampleAverager
from policies import EpsilonGreedy
from simulators import run_simulation
from plotters import plot_rollout_mean, plot_action_values

k = 10  # Number of actions the Agent can choose from
max_rollouts = 200  # Number of rollouts. Each rollout the Agent and Environment are reset
max_time_steps = 2000  # Number of time steps per rollout

# Independent variables: eps, calc_stepsize

# The environment is non-stationary (q_star values perform a random walk). Compared to an equally-weighted sample
# average (stepsize = 1/n), an exponential recency-weighted sample average (stepsize = constant) adapts more slowly
# initially, but over longer time horizons it is able to adapt more quickly to a non-stationary target. Though it is
# susceptible to overcorrection with noisy reward signals for stationary targets.

environment = EnvironmentBanditNonstationary(rand_walk_std=0.1, k=k)
agent00_0 = AgentActionValuerPolicy("agent00_0", SampleAverager(k, default_value=0), EpsilonGreedy(eps=0))
agent10_0 = AgentActionValuerPolicy("agent10_0", SampleAverager(k, default_value=0), EpsilonGreedy(eps=0.10))
# erwa = exponential recency-weighted averaging
agent00_0_erwa10 = AgentActionValuerPolicy("agent00_0_erwa10", SampleAverager(k, default_value=0, calc_stepsize=lambda count: 0.1), EpsilonGreedy(eps=0))
agent10_0_erwa10 = AgentActionValuerPolicy("agent10_0_erwa10", SampleAverager(k, default_value=0, calc_stepsize=lambda count: 0.1), EpsilonGreedy(eps=0.10))
agents = [agent00_0, agent10_0, agent00_0_erwa10, agent10_0_erwa10]

results3 = run_simulation(max_rollouts, max_time_steps, environment, agents)

plot_rollout_mean(results3, "rewards")
plot_rollout_mean(results3, "optimal_actions")
plot_action_values(results3, "agent00_0", 0)
plot_action_values(results3, "agent10_0", 0)
plot_action_values(results3, "agent00_0_erwa10", 0)
plot_action_values(results3, "agent10_0_erwa10", 0)
# %%
