from abc import ABC, abstractmethod
import numpy as np

class ActionValuer(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def update_action_values(self, state_previous, action, state_current, reward):
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
    
    def update_action_values(self, state_previous, action, state_current, reward):
        if state_previous is not None:
            if self.action_counts[state_previous][action] == 0 and self.overwrite_default:
                self.action_counts[state_previous][action] += 1
                self.action_values[state_previous][action] = reward
            else:
                self.action_counts[state_previous][action] += 1
                n = self.action_counts[state_previous][action]
                q = self.action_values[state_previous][action]
                self.action_values[state_previous][action] += self.calc_stepsize(n) * (reward - q)

        if state_current not in self.action_values:
            self.set_default_values(state_current)

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
    
    def update_action_values(self, state_previous, action, state_current, reward):
        super().update_action_values(state_previous, action, state_current, reward)
        if state_previous is not None:
            self.time_step += 1
            ns = np.array(self.action_counts[state_previous])
            qs = np.array(self.action_values[state_previous])
            upper_confidence_interval = self.c * np.sqrt(np.log(self.time_step) / ns)
            upper_confidence_interval[ns == 0] = np.inf
            self.ucbs[state_previous] = qs + upper_confidence_interval

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

class PreferenceGradientAscent(ActionValuer):
    def __init__(self, k, preference_step_size=1.0, baseliner=None, calc_baseliner_step_size=None):
        self.k = k
        self.preference_step_size = preference_step_size

        if calc_baseliner_step_size is None:
            calc_baseliner_step_size = lambda time_step: 1 / time_step

        if baseliner is None:
            self.baseliner = self.make_sample_average_baseliner(calc_baseliner_step_size)
        else:
            self.baseliner = baseliner
        
        self.reset()
    
    def update_action_values(self, state_previous, action, state_current, reward):
        if state_previous is not None:
            baseline = self.baseliner(reward)
            indicator = np.zeros(self.k)
            indicator[action] = 1
            self.action_preferences[state_previous] += self.preference_step_size * (reward - baseline) * (indicator - self.action_probabilities[state_previous])
            self.action_probabilities[state_previous] = self.softmax(self.action_preferences[state_previous])

        if state_current not in self.action_preferences:
            self.set_default_values(state_current)

    def reset(self):
        self.reward_cumulative = 0
        self.action_preferences = {}
        self.action_probabilities = {}
        
    def get_action_values(self):
        return self.action_preferences

    def get_action_values_for_state(self, state):
        return self.action_preferences[state]

    def get_action_values_for_state_action(self, state, action):
        return self.action_preferences[state][action]

    def set_default_values(self, state):
        self.action_preferences[state] = np.zeros(self.k)
        self.action_probabilities[state] = self.softmax(self.action_preferences[state])
    
    def softmax(self, preferences):
        preferences_normalized = preferences - np.max(preferences) # For numerical stability
        exp_preferences = np.exp(preferences_normalized)
        probabilities = exp_preferences / np.sum(exp_preferences)
        return probabilities
    
    def make_sample_average_baseliner(self, calc_step_size):
        time_step = 0
        baseline = None
        previous_reward = None
        def sample_average_baseliner(reward):
            nonlocal time_step
            nonlocal baseline
            nonlocal previous_reward
            time_step += 1
            if time_step == 1:
                baseline = reward
            else:
                baseline += calc_step_size(time_step - 1) * (previous_reward - baseline)
            
            previous_reward = reward
            return baseline
        return sample_average_baseliner
    

        
