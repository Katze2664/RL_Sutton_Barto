
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

# Exercise 2.5, page 33 of Reinforcement Learning: An Introduction (2nd ed) by Sutton and Barto

k = 10  # Number of actions the Agent can choose from
max_episodes = 2000  # Number of episodes. Each episode the Agent and Environment are reset
max_time_steps = 1000  # Number of time steps per episode

# Independent variables: calc_stepsize

# The environment is non-stationary (q_star values perform a random walk starting near zero). Compared to an
# equally-weighted sample average (stepsize = 1/n), an exponential recency-weighted sample average (stepsize = constant)
# adapts more slowly initially, but over longer time horizons it is able to adapt more quickly to a non-stationary
# target. Though it is susceptible to overcorrection with noisy reward signals for stationary targets.

environment = EnvironmentBanditNonstationary(rand_walk_std=[0.01], k=k, q_stars_std=[0.01])

# Equally weighted sameple average uses harmonic stepsize
agent_equally_weighted = AgentActionValuerPolicy("agent_equally_weighted", SampleAverager(k, calc_stepsize=lambda count: 1/count), EpsilonGreedy(eps=0.10))

# Exponential recency-weighted averaging (erwa) uses constant stepsize
agent_recency_weighted = AgentActionValuerPolicy("agent_recency_weighted", SampleAverager(k, calc_stepsize=lambda count: 0.1), EpsilonGreedy(eps=0.10))
agents = [agent_equally_weighted, agent_recency_weighted]

results4 = run_simulation(max_episodes, max_time_steps, environment, agents)

print(time.time() - start_time) # 2000 episodes, 1000 time steps = 99 seconds

plot_episode_mean(results4, "rewards")
plot_episode_mean(results4, "optimal_actions")
episode = 0
plot_action_values(results4, "agent_equally_weighted", episode)
plot_action_values(results4, "agent_recency_weighted", episode)
# %%
