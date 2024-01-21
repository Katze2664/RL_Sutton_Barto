from matplotlib import pyplot as plt

def plot_rollout_mean(results, category_name, colors=["green", "red", "blue", "purple", "orange", "brown"]):
    color_idx = 0

    for agent_name, arr in results[category_name].items():
        arr_mean = arr.mean(axis=0)  # Averaged over rollouts
        color = colors[color_idx]
        color_idx += 1
        plt.plot(arr_mean, label=agent_name, color=color)

        # arr_std = arr.std(axis=0, ddof=1)
        # plt.errorbar(np.arange(max_time_steps), reward_mean, reward_std, label=agent_name, color=color)

    plt.legend()
    plt.title(category_name)
    plt.show()

def plot_action_values(results, agent_name, rollout, x_min=None, x_max=None, y_min=None, y_max=None, colors=["red", "orange", "olive", "green", "blue", "purple", "cyan", "pink", "brown", "grey"]):
    q_stars = results["q_stars"]["environment"][rollout]
    action_values = results["action_values"][agent_name][rollout]
    assert q_stars.shape == action_values.shape
    assert len(q_stars.shape) == 2
    max_time_steps, k = q_stars.shape

    color_idx = 0
    for col in range(k):
        plt.plot(q_stars[:, col], color=colors[color_idx], linewidth=1)
        color_idx += 1

    color_idx = 0
    for col in range(k):
        plt.plot(action_values[:, col], color=colors[color_idx], linewidth=2, label=col)
        color_idx += 1
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.title(label=f"{agent_name}, rollout: {rollout}")
    if x_min is not None:
        plt.xlim(left=x_min)
    if x_max is not None:
        plt.xlim(right=x_max)
    if y_min is not None:
        plt.ylim(bottom=y_min)
    if y_max is not None:
        plt.ylim(top=y_max)
    plt.show()