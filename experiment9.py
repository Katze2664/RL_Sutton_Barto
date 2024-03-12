
# %%
import time
import numpy as np
import matplotlib.pyplot as plt
from environments import EnvironmentBandit
from agents import AgentActionValuerPolicy
from actionvaluers import SampleAverager, PreferenceGradientAscent, UCB
from policies import EpsilonGreedy, PreferenceSoftmax
from simulators import run_simulation
from plotters import plot_episode_mean, plot_action_values

# Based on Chapter 2.10 - Summary (page 42) of
# Reinforcement Learning: An Introduction (2nd ed) by Sutton and Barto
# Recreates Figure 2.6 on page 42

k = 10  # Number of actions the Agent can choose from
max_episodes = 200  # Number of episodes. Each episode the Agent and Environment are reset
max_time_steps = 1000  # Number of time steps per episode

# Parameter study: eps, preference_step_size, c, default_value

param_ranges = [("eps", np.arange(-7, -1)),
                ("pref", np.arange(-5, 5)),
                ("ucb", np.arange(-4, 5)),
                ("default", np.arange(-2, 5))]

params = []
for param_name, param_range in param_ranges:
    for param_log_value in param_range:
        agent_name = param_name + "=2.0^" + str(param_log_value)
        params.append((param_name, agent_name, param_log_value))

agents = []
for param_name, agent_name, param_log_value in params:
    param_value = 2.0 ** param_log_value
    if param_name == "eps":
        agent = AgentActionValuerPolicy(agent_name, SampleAverager(k), EpsilonGreedy(eps=param_value))
    elif param_name == "pref":
        agent = AgentActionValuerPolicy(agent_name, PreferenceGradientAscent(k, preference_step_size=param_value), PreferenceSoftmax())
    elif param_name == "ucb":
        agent = AgentActionValuerPolicy(agent_name, UCB(k, c=param_value), EpsilonGreedy())
    elif param_name == "default":
        agent = AgentActionValuerPolicy(agent_name, SampleAverager(k, default_value=param_value), EpsilonGreedy())
    else:
        assert False, f"{param_name=} not recognised"
    agents.append(agent)
# %%
start_time = time.time()

environment = EnvironmentBandit(k=k)
results9 = run_simulation(max_episodes, max_time_steps, environment, agents)

print(time.time() - start_time) # 200 episodes, 1000 time steps = 710 seconds

# %%
# Plot Parameter Study
param_results = {}
for param_name, agent_name, param_log_value in params:
    mean_reward = results9["rewards"][agent_name][:, 1:].mean()  # rewards[:, 1:] because reward is NaN for time_step==0
    if param_name not in param_results:
        param_results[param_name] = {"param_log_values": [], "mean_rewards": []}
    param_results[param_name]["param_log_values"].append(param_log_value)
    param_results[param_name]["mean_rewards"].append(mean_reward)

colors = ["red", "green", "blue", "black"]
color_idx = 0
for param_name, results in param_results.items():
    x = results["param_log_values"]
    y = results["mean_rewards"]
    print(param_name, x, y)
    plt.plot(x, y, label=param_name, color=colors[color_idx])
    color_idx += 1
plt.legend()
plt.title("Parameter Study")
plt.xlabel("Parameter Value (log_2)")
plt.ylabel("Mean Reward")
plt.show()

# param_results:
# param_name, x=param_log_values, y=mean_rewards
# eps [-7, -6, -5, -4, -3, -2] [1.19725312159883, 1.233054244258382, 1.3404420615652575, 1.3373701202858648, 1.2849020692038038, 1.1201984674775416]
# pref [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4] [1.1145277392177715, 1.299526737931388, 1.4067677292635827, 1.4352986072754528, 1.4376034085825316, 1.3620266421232075, 1.2863169614945231, 1.2027431619580926, 0.973940545257757, 0.9947662853888716]
# ucb [-4, -3, -2, -1, 0, 1, 2, 3, 4] [1.4355616479752398, 1.4583922155002884, 1.456788574381008, 1.4863341665824827, 1.488427191744514, 1.4024638733025834, 1.1705673401268917, 0.7906636021847854, 0.42957850374616435]
# default [-2, -1, 0, 1, 2, 3, 4] [1.1444635238454484, 1.2377586095792479, 1.3967231239180242, 1.3963051600306753, 1.3901791909475145, 1.4433504682556124, 1.417506774487564]

# %%
# Plot learning curves
select_param = ["all"]
# select_param = ["esp", "pref", "ucb", "default"]
select_value = ["all"]

results_by_param = {}
for param_name, agent_name, param_log_value in params:
        if ("all" in select_param) or (param_name in select_param):
            if ("all" in select_value) or (param_log_value in select_value):
                if param_name not in results_by_param:
                    results_by_param[param_name] = {"rewards": {}, "optimal_actions": {}}
                results_by_param[param_name]["rewards"][agent_name] = results9["rewards"][agent_name]
                results_by_param[param_name]["optimal_actions"][agent_name] = results9["optimal_actions"][agent_name]

for param_name, results in results_by_param.items():
    print(param_name)
    plot_episode_mean(results, "rewards", colors=["red", "orange", "olive", "green", "blue", "purple", "cyan", "pink", "brown", "grey"])
    plot_episode_mean(results, "optimal_actions", colors=["red", "orange", "olive", "green", "blue", "purple", "cyan", "pink", "brown", "grey"])

# %%
# Plot action values
episode = 0
for action, q_star in enumerate(results9["q_stars"]["environment"][episode, 0, 0]):
    print(f"{action}: {round(q_star, 2)}")

select_param = ["all"]
# select_param = ["esp", "pref", "ucb", "default"]
select_value = ["all"]

for param_name, agent_name, param_log_value in params:
    if ("all" in select_param) or (param_name in select_param):
        if ("all" in select_value) or (param_log_value in select_value):
            plot_action_values(results9, agent_name, episode)
