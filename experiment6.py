
# %%
import time
from environments import EnvironmentBandit
from agents import AgentActionValuerPolicy
from actionvaluers import UCB, SampleAverager
from policies import EpsilonGreedy
from simulators import run_simulation
from plotters import plot_rollout_mean, plot_action_values

start_time = time.time()

k = 10  # Number of actions the Agent can choose from
max_rollouts = 2000  # Number of rollouts. Each rollout the Agent and Environment are reset
max_time_steps = 1000  # Number of time steps per rollout

# Independent variable: c

agent_sa = AgentActionValuerPolicy("agent_sa", SampleAverager(k), EpsilonGreedy(eps=0.1))
environment = EnvironmentBandit(k)
agent_ucb1 = AgentActionValuerPolicy("agent_ucb1", UCB(k, c=2), EpsilonGreedy(eps=0))

agents = [agent_sa, agent_ucb1]

results6 = run_simulation(max_rollouts, max_time_steps, environment, agents)

print(time.time() - start_time) # 2000 rollouts, 1000 steps = 312 seconds

plot_rollout_mean(results6, "rewards")
plot_rollout_mean(results6, "optimal_actions")
plot_action_values(results6, "agent_sa", 0)
plot_action_values(results6, "agent_ucb1", 0)
# %%
