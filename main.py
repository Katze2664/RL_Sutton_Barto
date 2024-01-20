
# %%
import random
import numpy as np
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod

class Environment(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    def output_observation(self, agent_name):
        pass
    
    @abstractmethod
    def receive_action(self, agent_name, action):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update_time_step(self, time_step):
        pass

class EnvironmentBandit(Environment):
    """Creates a k-bandit environment.
    
    Environment contains only 1 state, in which there are k choices for actions.
    Each action has a mean reward given by self.q_star[action], whose values are
    normally distributed with a mean of q_star_mean and standard deviation of
    q_star_std, and are fixed unless reset. When an action is selected the reward
    given is normally distributed with a mean of self.q_star[action] and standard
    deviation of reward_std.
    """

    def __init__(self, k=10, q_stars_mean=0, q_stars_std=1, reward_std=1):
        self.k = k
        self.q_stars_mean = q_stars_mean
        self.q_stars_std = q_stars_std
        self.reward_std = reward_std

        self.reset()

    def output_observation(self, agent_name):
        state_observable = 0
        if agent_name not in self.actions_received:
            reward = None
            return state_observable, reward
        
        action = self.actions_received[agent_name]
        mean = self.state_internal["q_stars"][action]
        reward = random.gauss(mean, self.reward_std)
        return state_observable, reward

    def receive_action(self, agent_name, action):
        self.actions_received[agent_name] = action

    def reset(self):
        self.time_step = None
        self.actions_received = {}
        self.state_internal = {"q_stars": []}  # List of mean reward for each action (a.k.a. q_star)

        for i in range(self.k):
            self.state_internal["q_stars"].append(random.gauss(self.q_stars_mean, self.q_stars_std))
    
    def update_time_step(self, time_step):
        self.time_step = time_step

    def output_state_internal(self):
        return self.state_internal
    
    def output_action_optimal(self):
        action_optimal = np.argmax(self.state_internal["q_stars"])
        return action_optimal


class EnvironmentBanditNonstationary(EnvironmentBandit):
    def __init__(self, rand_walk_std=0.01, k=10, q_stars_mean=0, q_stars_std=1, reward_std=1):
        self.rand_walk_std = rand_walk_std
        super().__init__(k=k, q_stars_mean=q_stars_mean, q_stars_std=q_stars_std, reward_std=reward_std)
    
    def walk_q_star_randomly(self):
        for i in range(self.k):
            self.state_internal["q_stars"][i] += random.gauss(0, self.rand_walk_std)
    
    def update_time_step(self, time_step):
        if self.time_step is not None:
            self.walk_q_star_randomly()
        super().update_time_step(time_step)


# %%
class Agent(ABC):
    @abstractmethod
    def __init__(self, name):
        pass

    @abstractmethod
    def receive_observation(self, state_observed, reward):
        pass

    @abstractmethod
    def output_action(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class AgentActionValuerPolicy(Agent):
    def __init__(self, name, action_valuer, policy):
        self.name = name
        self.action_valuer = action_valuer
        self.policy = policy

        self.reset()
    
    def receive_observation(self, state_observed, reward):
        self.state_observed = state_observed
        self.action_valuer.update_action_values(state_observed, self.action, reward)

    def output_action(self):
        action_values = self.action_valuer.output_action_values()
        self.action = self.policy.select_action(action_values, self.state_observed)
        return self.action

    def reset(self):
        self.action = None
        self.state_observed = None
        self.action_valuer.reset()
        self.policy.reset()
    
    def output_action_values(self, state=None, action=None):
        return self.action_valuer.output_action_values(state, action)


class ActionValuer(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def update_action_values(self, state, action, reward):
        pass

    @abstractmethod
    def output_action_values(self, state=None, action=None):
        if state is None:
            return self.action_values
        elif action is None:
            return self.action_values[state]
        else:
            return self.action_values[state][action]

    @abstractmethod
    def reset(self):
        pass

class SampleAverager(ActionValuer):
    def __init__(self, k, default_value=0, calc_stepsize=None, overwrite_default=False):
        self.k = k
        self.default_value = default_value
        self.overwrite_default = overwrite_default
        if calc_stepsize is None:
            self.calc_stepsize = lambda action_count: 1 / action_count  # Equivalent to equal-weight sample averaging
        else:
            self.calc_stepsize = calc_stepsize
        self.reset()
    
    def update_action_values(self, state, action, reward):
        if state not in self.action_values:
            self.set_default_values(state)

        if action is not None:
            if self.action_counts[state][action] == 0 and self.overwrite_default:
                self.action_counts[state][action] += 1
                self.action_values[state][action] = reward
            else:
                self.action_counts[state][action] += 1
                n = self.action_counts[state][action]
                q = self.action_values[state][action]
                self.action_values[state][action] += self.calc_stepsize(n) * (reward - q)

    def output_action_values(self, state=None, action=None):
        return super().output_action_values(state, action)

    def reset(self):
        self.action_counts = {}
        self.action_values = {}

    def set_default_values(self, state):
        self.action_counts[state] = [0] * self.k
        self.action_values[state] = [self.default_value] * self.k


class Policy(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def select_action(self, action_values, state=None):
        pass
    
    @abstractmethod
    def reset(self):
        pass

class EpsilonGreedy(Policy):
    def __init__(self, eps=0):
        self.eps = eps
        self.reset()
    
    def select_action(self, action_values, state=None):
        if state is not None:
            action_values = action_values[state]
        
        if random.random() < self.eps:
            # Explore
            k = len(action_values)
            action_selected = random.randrange(k)
        else:
            # Greedy
            action_selected = np.argmax(action_values)
        return action_selected

    def reset(self):
        pass


# %%
def run_simulation(max_rollouts, max_time_steps, environment, agents):
    agent_names = []
    for agent in agents:
        assert agent.name not in agent_names, "Agent names must be unique"
        agent_names.append(agent.name)

    results = {}
    agent_categories = [("rewards", (max_rollouts, max_time_steps)),
                        ("actions", (max_rollouts, max_time_steps)),
                        ("optimal_actions", (max_rollouts, max_time_steps)),
                        ("action_values", (max_rollouts, max_time_steps, k))]
    
    environment_categories = [("q_stars", (max_rollouts, max_time_steps, k))]

    for category_name, category_shape in agent_categories:
        results[category_name] = {agent_name: np.zeros(category_shape) for agent_name in agent_names}
    
    for category_name, category_shape in environment_categories:
        results[category_name] = {"environment": np.zeros(category_shape) for agent_name in agent_names}

    for rollout in range(max_rollouts):
        environment.reset()
        for agent in agents:
            agent_name = agent.name
            agent.reset()

        for time_step in range(max_time_steps):
            environment.update_time_step(time_step)
            results["q_stars"]["environment"][rollout, time_step] = environment.output_state_internal()["q_stars"]

            for agent in agents:
                agent_name = agent.name
                state_observed, reward = environment.output_observation(agent_name)
                agent.receive_observation(state_observed, reward)

                results["action_values"][agent_name][rollout, time_step] = agent.output_action_values(state=state_observed)

                if time_step > 0:
                    results["rewards"][agent_name][rollout, time_step] = reward
                
                action = agent.output_action()
                environment.receive_action(agent_name, action)
                
                results["actions"][agent_name][rollout, time_step] = action
                if action == environment.output_action_optimal():
                    results["optimal_actions"][agent_name][rollout, time_step] = 1
    return results

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


# %%
k = 10  # Number of actions the Agent can choose from
max_rollouts = 200  # Number of rollouts. Each rollout the Agent and Environment are reset
max_time_steps = 2000  # Number of time steps per rollout

# %%
# Independent variable: eps
environment = EnvironmentBandit(k)
agent00_0 = AgentActionValuerPolicy("agent00_0", SampleAverager(k, default_value=0), EpsilonGreedy(eps=0))
agent01_0 = AgentActionValuerPolicy("agent01_0", SampleAverager(k, default_value=0), EpsilonGreedy(eps=0.01))
agent10_0 = AgentActionValuerPolicy("agent10_0", SampleAverager(k, default_value=0), EpsilonGreedy(eps=0.10))
agents = [agent00_0, agent01_0, agent10_0]

results1 = run_simulation(max_rollouts, max_time_steps, environment, agents)
# Textbook graphs at t=1000: Green = 1.05, Red = 1.31, Blue = 1.41

plot_rollout_mean(results1, "rewards")
plot_rollout_mean(results1, "optimal_actions")
plot_action_values(results1, "agent00_0", 0)
plot_action_values(results1, "agent01_0", 0)
plot_action_values(results1, "agent10_0", 0)

# %%

# Independent variable: default_value

# Using an infinite default value (with overwrite on the first sample) forces it to explore every action once before
# acting greedily. Using a high finite default value (without overwrite, and stepsize=1/(n+1) so that the default value
# acts as the first sample) causes it to explore every action multiple times until the samples outweigh the default
# value. If the default value is too high then exploration of poor actions persists for too long. If the default value
# is too low then the first high reward terminates exploration.

environment = EnvironmentBandit(k)
agent00_0 = AgentActionValuerPolicy("agent00_0", SampleAverager(k, default_value=0), EpsilonGreedy(eps=0))
agent00_inf = AgentActionValuerPolicy("agent00_inf", SampleAverager(k, default_value=np.inf, overwrite_default=True), EpsilonGreedy(eps=0))
agent00_10 = AgentActionValuerPolicy("agent00_10", SampleAverager(k, default_value=10, calc_stepsize=lambda count: 1 / (count + 1)), EpsilonGreedy(eps=0))
agents = [agent00_0, agent00_inf, agent00_10]

results2 = run_simulation(max_rollouts, max_time_steps, environment, agents)

plot_rollout_mean(results2, "rewards")
plot_rollout_mean(results2, "optimal_actions")
plot_action_values(results2, "agent00_0", rollout=0)
plot_action_values(results2, "agent00_inf", rollout=0)
plot_action_values(results2, "agent00_10", rollout=0)

# %% 

# Independent variables: eps, calc_stepsize

# The environment is non-stationary (q_star values perform a random walk). Compared to an equally-weighted sample
# average (stepsize = 1/n), an exponential recency-weighted sample average (stepsize = constant) adapts more slowly
# initially, but over longer time horizons it is able to adapt more quickly to a non-stationary target. Though it is
# susceptible to overcorrection with noisy reward signals for stationary targets.

environment = EnvironmentBanditNonstationary(rand_walk_std=0.1, k=k)
agent00_0 = AgentActionValuerPolicy("agent00_0", SampleAverager(k, default_value=0), EpsilonGreedy(eps=0))
agent10_0 = AgentActionValuerPolicy("agent10_0", SampleAverager(k, default_value=0), EpsilonGreedy(eps=0.10))
# erwa = exponential recency-weighted averaging
agent00_0_erwa10 = AgentActionValuerPolicy("agent00_0_erwa10", SampleAverager(k, default_value=0, calc_stepsize=lambda count: 0.1), EpsilonGreedy(eps=0))
agent10_0_erwa10 = AgentActionValuerPolicy("agent10_0_erwa10", SampleAverager(k, default_value=0, calc_stepsize=lambda count: 0.1), EpsilonGreedy(eps=0.10))
agents = [agent00_0, agent10_0, agent00_0_erwa10, agent10_0_erwa10]

results3 = run_simulation(max_rollouts, max_time_steps, environment, agents)

plot_rollout_mean(results3, "rewards")
plot_rollout_mean(results3, "optimal_actions")
plot_action_values(results3, "agent00_0", 0)
plot_action_values(results3, "agent10_0", 0)
plot_action_values(results3, "agent00_0_erwa10", 0)
plot_action_values(results3, "agent10_0_erwa10", 0)
