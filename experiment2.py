
# %%
import numpy as np
from environments import EnvironmentBandit
from agents import AgentActionValuerPolicy
from actionvaluers import SampleAverager
from policies import EpsilonGreedy
from simulators import run_simulation
from plotters import plot_rollout_mean, plot_action_values

k = 10  # Number of actions the Agent can choose from
max_rollouts = 200  # Number of rollouts. Each rollout the Agent and Environment are reset
max_time_steps = 1000  # Number of time steps per rollout

# Independent variable: default_value

# Using an infinite default value (with overwrite on the first sample) forces it to explore every action once before
# acting greedily. Using a high finite default value (without overwrite, and stepsize=1/(n+1) so that the default value
# acts as the first sample) causes it to explore every action multiple times until the samples outweigh the default
# value. If the default value is too high then exploration of poor actions persists for too long. If the default value
# is too low then the first high reward terminates exploration.

environment = EnvironmentBandit(k)
agent00_0 = AgentActionValuerPolicy("agent00_0", SampleAverager(k, default_value=0), EpsilonGreedy(eps=0))
agent00_inf = AgentActionValuerPolicy("agent00_inf", SampleAverager(k, default_value=np.inf, overwrite_default=True), EpsilonGreedy(eps=0))
agent00_10 = AgentActionValuerPolicy("agent00_10", SampleAverager(k, default_value=10, calc_stepsize=lambda count: 1 / (count + 1)), EpsilonGreedy(eps=0))
agents = [agent00_0, agent00_inf, agent00_10]

results2 = run_simulation(max_rollouts, max_time_steps, environment, agents)

plot_rollout_mean(results2, "rewards")
plot_rollout_mean(results2, "optimal_actions")
plot_action_values(results2, "agent00_0", rollout=0)
plot_action_values(results2, "agent00_inf", rollout=0)
plot_action_values(results2, "agent00_10", rollout=0)

# %%
