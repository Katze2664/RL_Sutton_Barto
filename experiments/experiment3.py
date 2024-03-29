
# %%
import os
import sys
from pathlib import Path

ROOT_DIR = "RL_Sutton_Barto"
path = Path(os.path.abspath(__file__))
assert ROOT_DIR in path.parts, f"{ROOT_DIR=} not found in {path=}"
for part in Path(os.path.abspath(__file__)).parents:
    if part.stem == ROOT_DIR and str(part) not in sys.path:
        sys.path.insert(0, str(part))

import time
from environments import EnvironmentBanditNonstationary
from agents import AgentActionValuerPolicy
from actionvaluers import SampleAverager
from policies import EpsilonGreedy
from simulators import run_simulation
from plotters import plot_episode_mean, plot_action_values

start_time = time.time()

k = 10  # Number of actions the Agent can choose from
max_episodes = 200  # Number of episodes. Each episode the Agent and Environment are reset
max_time_steps = 2000  # Number of time steps per episode

# Independent variables: eps, calc_stepsize

# The environment is non-stationary (q_star values perform a random walk). Compared to an equally-weighted sample
# average (stepsize = 1/n), an exponential recency-weighted sample average (stepsize = constant) adapts more slowly
# initially, but over longer time horizons it is able to adapt more quickly to a non-stationary target. Though it is
# susceptible to overcorrection with noisy reward signals for stationary targets.

environment = EnvironmentBanditNonstationary(rand_walk_std=[0.1], k=k, q_stars_std=[0.1])
agent00_0 = AgentActionValuerPolicy("agent00_0", SampleAverager(k, default_value=0), EpsilonGreedy(eps=0))
agent10_0 = AgentActionValuerPolicy("agent10_0", SampleAverager(k, default_value=0), EpsilonGreedy(eps=0.10))
# erwa = exponential recency-weighted averaging
agent00_0_erwa10 = AgentActionValuerPolicy("agent00_0_erwa10", SampleAverager(k, default_value=0, calc_stepsize=lambda count: 0.1), EpsilonGreedy(eps=0))
agent10_0_erwa10 = AgentActionValuerPolicy("agent10_0_erwa10", SampleAverager(k, default_value=0, calc_stepsize=lambda count: 0.1), EpsilonGreedy(eps=0.10))
agents = [agent00_0, agent10_0, agent00_0_erwa10, agent10_0_erwa10]

results3 = run_simulation(max_episodes, max_time_steps, environment, agents)

print(time.time() - start_time) # 200 episodes, 2000 time steps = 148 seconds

plot_episode_mean(results3, "rewards")
plot_episode_mean(results3, "optimal_actions")
plot_action_values(results3, "agent00_0", 0)
plot_action_values(results3, "agent10_0", 0)
plot_action_values(results3, "agent00_0_erwa10", 0)
plot_action_values(results3, "agent10_0_erwa10", 0)
# %%
