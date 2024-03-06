import random
import numpy as np
from abc import ABC, abstractmethod

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

class PreferenceSoftmax(ABC):
    def __init__(self):
        self.rng = np.random.default_rng()
    
    def select_action(self, action_preferences, state=None):
        if state is not None:
            action_preferences = action_preferences[state]
        
        action_probabilities = self.softmax(action_preferences)
        action_selected = self.rng.choice(len(action_preferences), p=action_probabilities)
        return action_selected
    
    def reset(self):
        pass

    def softmax(self, preferences):
        preferences_normalized = preferences - np.max(preferences) # For numerical stability
        exp_preferences = np.exp(preferences_normalized)
        probabilities = exp_preferences / np.sum(exp_preferences)
        return probabilities
