
# %%
import time
from environments import EnvironmentBandit
from agents import AgentActionValuerPolicy
from actionvaluers import PreferenceGradientAscent
from policies import PreferenceSoftmax
from simulators import run_simulation
from plotters import plot_rollout_mean, plot_action_values

# Based on Chapter 2.8 - Gradient Bandit Algorithms (page 37) of
# Reinforcement Learning: An Introduction (2nd ed) by Sutton and Barto

start_time = time.time()

k = 10  # Number of actions the Agent can choose from
max_rollouts = 200  # Number of rollouts. Each rollout the Agent and Environment are reset
max_time_steps = 1000  # Number of time steps per rollout

# Independent variable: step_size

agent_pga01 = AgentActionValuerPolicy("agent_pga01", PreferenceGradientAscent(k, preference_step_size=0.1), PreferenceSoftmax())
agent_pga04 = AgentActionValuerPolicy("agent_pga04", PreferenceGradientAscent(k, preference_step_size=0.4), PreferenceSoftmax())
environment = EnvironmentBandit(k)

agents = [agent_pga01, agent_pga04]

results7 = run_simulation(max_rollouts, max_time_steps, environment, agents)

print(time.time() - start_time) # 200 rollouts, 1000 steps = 74 seconds

plot_rollout_mean(results7, "rewards")
plot_rollout_mean(results7, "optimal_actions")
plot_action_values(results7, "agent_pga01", 0)
plot_action_values(results7, "agent_pga04", 0)

# %%
