import random
import numpy as np
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
            # TODO: Use numpy to generate q_stars array.
    
    def update_time_step(self, time_step):
        self.time_step = time_step

    def output_state_internal(self):
        return self.state_internal
    
    def output_action_optimal(self):
        action_optimal = np.argmax(self.state_internal["q_stars"])
        # TODO: Currently returns lowest index optimal action when tied. If action selected is a higher index optimal
        # action this results in incorrectly being judged as non-optimal. Should return a list or set of optimal actions.
        return action_optimal


class EnvironmentBanditNonstationary(EnvironmentBandit):
    def __init__(self, rand_walk_std=0.01, k=10, q_stars_mean=0, q_stars_std=1, reward_std=1):
        self.rand_walk_std = rand_walk_std
        super().__init__(k=k, q_stars_mean=q_stars_mean, q_stars_std=q_stars_std, reward_std=reward_std)
    
    def walk_q_star_randomly(self):
        for i in range(self.k):
            self.state_internal["q_stars"][i] += random.gauss(0, self.rand_walk_std)
            # TODO: Use numpy to generate q_stars random walk
    
    def update_time_step(self, time_step):
        if self.time_step is not None:
            self.walk_q_star_randomly()
        super().update_time_step(time_step)

