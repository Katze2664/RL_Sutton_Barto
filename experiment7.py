
# %%
import time
from environments import EnvironmentBandit
from agents import AgentActionValuerPolicy
from actionvaluers import PreferenceGradientAscent
from policies import PreferenceSoftmax
from simulators import run_simulation
from plotters import plot_episode_mean, plot_action_values

# Based on Chapter 2.8 - Gradient Bandit Algorithms (page 37) of
# Reinforcement Learning: An Introduction (2nd ed) by Sutton and Barto
# Recreates Figure 2.5 on page 38

start_time = time.time()

k = 10  # Number of actions the Agent can choose from
max_episodes = 2000  # Number of episodes. Each episode the Agent and Environment are reset
max_time_steps = 100  # Number of time steps per episode

# Independent variable: step_size, baseliner

# Instead of estimating action values, PreferenceGradientAscent updates preferences for each action, and actions are
# selected with a probability equal to the softmax of these preferences. Preferences are updated such that the expected
# update is in the direction of the gradient of the expected reward (with respect to the preferences). 

# H(A) += (R - B) * (1 - pi(A)) 
# H(a) -= (R - B) * pi(a)

# The preference, H, of the action taken, A, is updated in proportion to how much the reward, R, exceeds the baseline,
# B, weighted by the probability that the action wouldn't be taken, 1 - pi(A). The preference of each other action, a,
# is updated in the opposite direction, weighted by the probability that the action would be taken, pi(a). The baseline
# can be any value that does not depend on the action selected. By calculating the baseline to be the average reward
# over all previous time steps reduces the variance in the updates compared to using a constant baseline such as zero.

agent_pga01 = AgentActionValuerPolicy("agent_pga01", PreferenceGradientAscent(k, preference_step_size=0.1), PreferenceSoftmax())
agent_pga04 = AgentActionValuerPolicy("agent_pga04", PreferenceGradientAscent(k, preference_step_size=0.4), PreferenceSoftmax())
agent_pga01_0 = AgentActionValuerPolicy("agent_pga01_0", PreferenceGradientAscent(k, preference_step_size=0.1, baseliner=lambda reward: 0), PreferenceSoftmax())
agent_pga04_0 = AgentActionValuerPolicy("agent_pga04_0", PreferenceGradientAscent(k, preference_step_size=0.4, baseliner=lambda reward: 0), PreferenceSoftmax())
environment = EnvironmentBandit(k=k, q_stars_mean=[4])

agents = [agent_pga01, agent_pga04, agent_pga01_0, agent_pga04_0]

results7 = run_simulation(max_episodes, max_time_steps, environment, agents)

print(time.time() - start_time) # 200 episodes, 1000 time steps = 79 seconds

plot_episode_mean(results7, "rewards")
plot_episode_mean(results7, "optimal_actions")
plot_action_values(results7, "agent_pga01", 0)
plot_action_values(results7, "agent_pga04", 0)
plot_action_values(results7, "agent_pga01_0", 0)
plot_action_values(results7, "agent_pga04_0", 0)
# %%
