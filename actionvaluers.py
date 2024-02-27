from abc import ABC, abstractmethod
import numpy as np

class ActionValuer(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def update_action_values(self, state, action, reward):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_action_values(self):
        return self.action_values

    @abstractmethod
    def get_action_values_for_state(self, state):
        return self.action_values[state]

    @abstractmethod
    def get_action_values_for_state_action(self, state, action):
        return self.action_values[state][action]

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

    def reset(self):
        self.action_counts = {}
        self.action_values = {}

    def get_action_values(self):
        return super().get_action_values()
    
    def get_action_values_for_state(self, state):
        return super().get_action_values_for_state(state)

    def get_action_values_for_state_action(self, state, action):
        return super().get_action_values_for_state_action(state, action)

    def set_default_values(self, state):
        self.action_counts[state] = [0] * self.k
        self.action_values[state] = [self.default_value] * self.k

class UCB(SampleAverager):  # UCB = Upper-Confidence-Bound
    def __init__(self, k, c=1, default_value=0, calc_stepsize=None, overwrite_default=False):
        super().__init__(k, default_value, calc_stepsize, overwrite_default)
        self.c = c
    
    def update_action_values(self, state, action, reward):
        super().update_action_values(state, action, reward)
        if action is not None:
            self.time_step += 1
            ns = np.array(self.action_counts[state])
            qs = np.array(self.action_values[state])
            upper_confidence_interval = self.c * np.sqrt(np.log(self.time_step) / ns)
            upper_confidence_interval[ns == 0] = np.inf
            self.ucbs[state] = qs + upper_confidence_interval

    def reset(self):
        super().reset()
        self.time_step = 0
        self.ucbs = {}
    
    def get_action_values(self):
        return self.ucbs
    
    def get_action_values_for_state(self, state):
        return self.ucbs[state]

    def get_action_values_for_state_action(self, state, action):
        return self.ucbs[state][action]

    def set_default_values(self, state):
        super().set_default_values(state)
        self.ucbs[state] = np.full(self.k, np.inf)
