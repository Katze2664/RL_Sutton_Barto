
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

k = 10  # Number of actions the Agent can choose from
max_episodes = 20  # Number of episodes. Each episode the Agent and Environment are reset
max_time_steps = 100  # Number of time steps per episode

# Independent variable: eps

# The agent uses an epsilon-greedy strategy, where it behaves greedily with probability 1 - eps, and behaves randomly
# with probability eps. If eps = 0, the agent never explores, so can get stuck acting greedily on the first action that
# has positive expected value, even if a better action exists. If eps is a small positive number, then agent explores
# occasionally, allowing it to eventually identify the optimal action, but while still acting greedily most of the time
# to exploit best known action. If eps is too high, the agent will explore frequently and so will identify the optimal
# action, however it perform poorly overall since it does not exploit the optimal action frequently enough.
environment = EnvironmentBandit(k=k)
agent00_0 = AgentActionValuerPolicy("agent00_0", SampleAverager(k, default_value=0), EpsilonGreedy(eps=0))
agent01_0 = AgentActionValuerPolicy("agent01_0", SampleAverager(k, default_value=0), EpsilonGreedy(eps=0.01))
agent10_0 = AgentActionValuerPolicy("agent10_0", SampleAverager(k, default_value=0), EpsilonGreedy(eps=0.10))
agent50_0 = AgentActionValuerPolicy("agent50_0", SampleAverager(k, default_value=0), EpsilonGreedy(eps=0.50))
agents = [agent00_0, agent01_0, agent10_0, agent50_0]

results1 = run_simulation(max_episodes, max_time_steps, environment, agents)
# Textbook graphs at t=1000: Green = 1.05, Red = 1.31, Blue = 1.41

print(time.time() - start_time) # 200 episodes, 1000 time steps = 124 seconds

plot_episode_mean(results1, "rewards")
plot_episode_mean(results1, "optimal_actions")
plot_action_values(results1, "agent00_0", 0)
plot_action_values(results1, "agent01_0", 0)
plot_action_values(results1, "agent10_0", 0)
plot_action_values(results1, "agent50_0", 0)
# %%
