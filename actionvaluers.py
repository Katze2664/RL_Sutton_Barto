from abc import ABC, abstractmethod

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
        # TODO: Refactor into 3 separate methods for the 3 different cases.

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
