
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
from actionvaluers import UCB, SampleAverager
from policies import EpsilonGreedy
from simulators import run_simulation
from plotters import plot_episode_mean, plot_action_values

start_time = time.time()

k = 10  # Number of actions the Agent can choose from
max_episodes = 2000  # Number of episodes. Each episode the Agent and Environment are reset
max_time_steps = 1000  # Number of time steps per episode

# Independent variable: c

# Upper Confidence Bound (UCB) estimates the action values using sample averaging but adds a confidence interval to
# account for uncertainty in the estimate. The confidence interval, sqrt(ln(t) / N(a)), for an action decreases each
# time the action is tried, but increases for all actions each time step to ensure that all actions will eventually be
# tried.

agent_sa = AgentActionValuerPolicy("agent_sa", SampleAverager(k), EpsilonGreedy(eps=0.1))
environment = EnvironmentBandit(k=k)
agent_ucb1 = AgentActionValuerPolicy("agent_ucb1", UCB(k, c=2), EpsilonGreedy(eps=0))

agents = [agent_sa, agent_ucb1]

results6 = run_simulation(max_episodes, max_time_steps, environment, agents)

print(time.time() - start_time) # 2000 episodes, 1000 time steps = 312 seconds

plot_episode_mean(results6, "rewards")
plot_episode_mean(results6, "optimal_actions")
plot_action_values(results6, "agent_sa", 0)
plot_action_values(results6, "agent_ucb1", 0)
# %%
