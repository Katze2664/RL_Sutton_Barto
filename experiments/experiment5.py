
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
from environments import EnvironmentBandit
from agents import AgentActionValuerPolicy
from actionvaluers import SampleAverager
from policies import EpsilonGreedy
from simulators import run_simulation
from plotters import plot_episode_mean, plot_action_values

start_time = time.time()

# Replicating Figure 2.3, page 34 of Reinforcement Learning: An Introduction (2nd ed) by Sutton and Barto

k = 10  # Number of actions the Agent can choose from
max_episodes = 2000  # Number of episodes. Each episode the Agent and Environment are reset
max_time_steps = 1000  # Number of time steps per episode

# Independent variables: default_value, eps

# An optimistic initial estimate (default_value=5) with a constant stepsize encourages substantial initial explortion.
# This results in low initial reward and low optimal action selection compared to an accurate initial estimate
# (default_value=0). However in the longer term the initial exploration results in higher reward and higher optimal
# action selection. When using an optimistic initial estimate, using eps=0.1 leads to worse performance since it causes
# non-optimal actions to be selected 10% of the time, and this exploration is not necessary for identifying the optimal
# action since the initial exploration is usually sufficient.

environment = EnvironmentBandit(k=k)

# Exponential recency-weighted averaging (erwa) uses constant stepsize
agent_default0_eps10 = AgentActionValuerPolicy("agent_default0_eps10", SampleAverager(k, default_value=0, calc_stepsize=lambda count: 0.1), EpsilonGreedy(eps=0.10))
agent_default5_eps0 = AgentActionValuerPolicy("agent_default5_eps0", SampleAverager(k, default_value=5, calc_stepsize=lambda count: 0.1), EpsilonGreedy(eps=0.0))
agent_default5_eps10 = AgentActionValuerPolicy("agent_default5_eps10", SampleAverager(k, default_value=5, calc_stepsize=lambda count: 0.1), EpsilonGreedy(eps=0.10))
agents = [agent_default0_eps10, agent_default5_eps0, agent_default5_eps10]

results4 = run_simulation(max_episodes, max_time_steps, environment, agents)

print(time.time() - start_time) # 2000 episodes, 1000 time steps = 114 seconds

plot_episode_mean(results4, "rewards")
plot_episode_mean(results4, "optimal_actions")

episode = 0
plot_action_values(results4, "agent_default0_eps10", episode)
plot_action_values(results4, "agent_default5_eps0", episode)
plot_action_values(results4, "agent_default5_eps10", episode)
# %%
