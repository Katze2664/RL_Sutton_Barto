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