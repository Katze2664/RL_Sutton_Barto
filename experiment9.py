
# %%
import time
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from environments import EnvironmentBandit
from agents import AgentActionValuerPolicy
from actionvaluers import SampleAverager, PreferenceGradientAscent, UCB
from policies import EpsilonGreedy, PreferenceSoftmax
from simulators import run_simulation
from plotters import plot_episode_mean, plot_action_values
# %%
# Based on Chapter 2.10 - Summary (page 42) of
# Reinforcement Learning: An Introduction (2nd ed) by Sutton and Barto
# Recreates Figure 2.6 on page 42

k = 10  # Number of actions the Agent can choose from
max_episodes = 200  # Number of episodes. Each episode the Agent and Environment are reset
max_time_steps = 1000  # Number of time steps per episode

# Parameter study: eps, preference_step_size, c, default_value

param_ranges = [("eps", list(range(-7, -1))),
                ("pref", list(range(-5, 3))),
                ("ucb", list(range(-4, 3))),
                ("default", list(range(-2, 3)))]

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
        agent = AgentActionValuerPolicy(agent_name, SampleAverager(k, default_value=param_value, calc_stepsize=lambda time_step: 0.1), EpsilonGreedy())
    else:
        raise Exception(f"{param_name=} not recognised")
    agents.append(agent)

start_time = time.time()

environment = EnvironmentBandit(k=k)
results9 = run_simulation(max_episodes, max_time_steps, environment, agents)

print(time.time() - start_time) # 200 episodes, 1000 time steps = 930 seconds

# Extract Parameter Study results (param_results)

param_results = {}
for param_name, agent_name, param_log_value in params:
    mean_reward = results9["rewards"][agent_name][:, 1:].mean()  # rewards[:, 1:] because reward is NaN for time_step==0
    if param_name not in param_results:
        param_results[param_name] = {"param_log_values": [], "mean_rewards": []}
    param_results[param_name]["param_log_values"].append(param_log_value)
    param_results[param_name]["mean_rewards"].append(mean_reward)


def save(obj, file_name):
    with open(file_name, "w") as f:
        f.write(json.dumps(obj))

def load(file_name):
    with open(file_name, "r") as f:
        obj = json.loads(f.read())
    return obj

# %%
file_name = "results/exp9_param_results_e200_t1000.json"

# %%
# Save param_results
if os.path.isfile(file_name):
    print(f"{file_name=}")
    user = input("Do you want to overwrite this file? (Y or N)")
    if user.lower() in ["y", "yes"]:
        save(param_results, file_name)
    else:
        print("Cancelled")
else:
    save(param_results, file_name)

# %%
# Load param_results
print(f"{file_name=}")
user = input("Do you want to load this file? (Y or N)")
if user.lower() in ["y", "yes"]:
    param_results = load(file_name)
else:
    print("Cancelled")

# %%
# Plot Parameter Study (param_results)
colors = ["red", "green", "blue", "black"]
color_idx = 0
for param_name, results in param_results.items():
    x = results["param_log_values"]
    y = results["mean_rewards"]
    print(param_name)
    print(np.vstack((x, np.round(y, 2))))
    plt.plot(x, y, label=param_name, color=colors[color_idx])
    color_idx += 1
plt.legend()
plt.title("Parameter Study")
plt.xlabel("Parameter Value (log_2)")
plt.ylabel("Mean Reward")
xticks_min = -7
xticks_max = 2
xticks = np.arange(xticks_min, xticks_max + 1)
xtick_labels = []
for xtick in xticks:
    if xtick < 0:
        xtick_labels.append("1/" + str(2**-xtick))
    else:
        xtick_labels.append(str(2**xtick))
plt.xticks(ticks=xticks, labels=xtick_labels)
plt.show()

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

# %%
