
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
    def __init__(self, k, default_value=0, calc_stepsize=None):
        self.k = k
        self.default_value = default_value
        if calc_stepsize is None:
            self.calc_stepsize = lambda action_count: 1 / action_count  # Equivalent to equal-weight sample averaging
        else:
            self.calc_stepsize = calc_stepsize
        self.reset()
    
    def update_action_values(self, state, action, reward):
        if state not in self.action_values:
            self.set_default_values(state)

        if action is not None:
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
    name_set = set()
    for agent in agents:
        assert agent.name not in name_set, "Agent names must be unique"
        name_set.add(agent.name)

    reward_results = {agent.name: np.zeros((max_rollouts, max_time_steps)) for agent in agents}
    optimal_action_results = {agent.name: np.zeros((max_rollouts, max_time_steps)) for agent in agents}
    
    for rollout in range(max_rollouts):
        environment.reset()
        for agent in agents:
            agent.reset()

        for time_step in range(max_time_steps):
            environment.update_time_step(time_step)
            for agent in agents:
                agent_name = agent.name
                state_observed, reward = environment.output_observation(agent_name)
                agent.receive_observation(state_observed, reward)

                if time_step > 0:
                    reward_results[agent.name][rollout, time_step] = reward
                
                action = agent.output_action()
                environment.receive_action(agent_name, action)
                
                if action == environment.output_action_optimal():
                    optimal_action_results[agent.name][rollout, time_step] = 1
    return reward_results, optimal_action_results
# %%
k = 10  # Number of actions the Agent can choose from
max_rollouts = 200  # Number of rollouts. Each rollout the Agent and Environment are reset
max_time_steps = 1000  # Number of time steps per rollout

environment = EnvironmentBandit(k)
# environment = EnvironmentBanditNonstationary(rand_walk_std=0.1, k=k)
agent00_0 = AgentActionValuerPolicy("agent00_0", SampleAverager(k, default_value=0), EpsilonGreedy(eps=0))
agent01_0 = AgentActionValuerPolicy("agent01_0", SampleAverager(k, default_value=0), EpsilonGreedy(eps=0.01))
agent10_0 = AgentActionValuerPolicy("agent10_0", SampleAverager(k, default_value=0), EpsilonGreedy(eps=0.10))
agents = [agent00_0, agent01_0, agent10_0]
# agents = [agent10_0]

# erwa = exponential recency-weighted averaging
# agent10_0_erwa01 = Agent("agent10_0_erwa01", k, eps=0.10, default_value=0, stepsize=lambda n: 0.01)
# agent10_0_erwa05 = Agent("agent10_0_erwa05", k, eps=0.10, default_value=0, stepsize=lambda n: 0.05)
# agent10_0_erwa20 = Agent("agent10_0_erwa20", k, eps=0.10, default_value=0, stepsize=lambda n: 0.20)
# agent10_0_erwa50 = Agent("agent10_0_erwa50", k, eps=0.10, default_value=0, stepsize=lambda n: 0.50)
# agent10_0_erwa100 = Agent("agent10_0_erwa100", k, eps=0.10, default_value=0, stepsize=lambda n: 1.00)
# agents = [agent10_0, agent10_0_erwa01, agent10_0_erwa05, agent10_0_erwa20, agent10_0_erwa50, agent10_0_erwa100]


reward_results, optimal_action_results = run_simulation(max_rollouts, max_time_steps, environment, agents)
# %%
colors = ["green", "red", "blue", "purple", "orange", "brown"]
color_idx = 0

for agent_name, rewards in reward_results.items():
    reward_mean = rewards.mean(axis=0)  # Averaged over rollouts
    reward_std = rewards.std(axis=0, ddof=1)
    
    print(agent_name, reward_mean[-1])
    color = colors[color_idx]
    color_idx += 1
    plt.plot(reward_mean, label=agent_name, color=color)
    # plt.errorbar(np.arange(max_time_steps), reward_rollout_mean, reward_rollout_std, label=agent_name, color=color)

plt.legend()
plt.show()
# %%
color_idx = 0
for agent_name, optimal_action in optimal_action_results.items():
    color = colors[color_idx]
    color_idx += 1
    optimal_rollout_average = optimal_action.mean(axis=0)
    plt.plot(optimal_rollout_average, label=agent_name, color=color)

plt.legend()
plt.show()

# %%
