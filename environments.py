import random
import numpy as np
from abc import ABC, abstractmethod

class Environment(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    def get_observation(self, agent_name):
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
    """Creates a n,k-bandit environment.
    
    Environment contains n states with k actions each. Each state is selected with probability state_probability[state]
    each time step, independent of previous states. The state is made observable to the agent if reveal_state==True.
    Each action has a mean reward given by q_stars[state, action], whose values are normally distributed with a mean of
    q_star_mean[state] and standard deviation of q_star_std[state], and are fixed unless reset. When an action is
    selected the reward given is normally distributed with a mean of q_stars[state, action] and standard deviation of
    reward_std[state].
    """

    def __init__(self, n=1, k=10, q_stars_mean=None, q_stars_std=None, reward_std=None, state_probabilities=None, reveal_state=True):
        self.n = n
        self.k = k

        self.q_stars_mean = [0] * self.n if q_stars_mean == None else q_stars_mean
        assert len(self.q_stars_mean) == self.n, f"{len(self.q_stars_mean)=} must equal {self.n=}"
        self.q_stars_std = [1] * self.n if q_stars_std == None else q_stars_std
        assert len(self.q_stars_std) == self.n, f"{len(self.q_stars_std)=} must equal {self.n=}"
        self.reward_std = [1] * self.n if reward_std == None else reward_std
        assert len(self.reward_std) == self.n, f"{len(self.reward_std)=} must equal {self.n=}"
        self.state_probabilities = [1 / self.n] * self.n if state_probabilities == None else state_probabilities
        assert len(self.state_probabilities) == self.n, f"{len(self.state_probabilities)=} must equal {self.n=}"
        assert sum(self.state_probabilities) == 1, f"{self.state_probabilities=} must sum to 1, not {sum(self.state_probabilities)=}"
        self.reveal_state = reveal_state
        self.rng = np.random.default_rng()

        self.reset()

    def get_observation(self, agent_name):
        if agent_name not in self.agent_action:
            reward = None
        else:
            action = self.agent_action[agent_name]
            state_actual = self.agent_state[agent_name]
            mean = self.q_stars[state_actual, action]
            std = self.reward_std[state_actual]
            reward = random.gauss(mean, std)
        
        state_actual = self.rng.choice(np.arange(self.n), p=self.state_probabilities)
        self.agent_state[agent_name] = state_actual
        if self.reveal_state:
            state_observable = state_actual
        else:
            state_observable = 0
        
        return state_observable, reward

    def receive_action(self, agent_name, action):
        self.agent_action[agent_name] = action

    def reset(self):
        self.time_step = None
        self.agent_state = {}
        self.agent_action = {}
        # Array of mean reward for each state and action (a.k.a. q_star)
        self.q_stars = self.rng.normal(np.tile(self.q_stars_mean, (self.k, 1)).T,
                                       np.tile(self.q_stars_std, (self.k, 1)).T)
        assert self.q_stars.shape == (self.n, self.k)
    
    def update_time_step(self, time_step):
        self.time_step = time_step

    def get_q_stars(self):
        return self.q_stars

    def get_state_actual(self, agent_name):
        return self.agent_state[agent_name]
    
    def get_all_optimal_actions(self, state):
        idxs = np.arange(self.k)
        optimal_actions = idxs[self.q_stars[state, :] == np.max(self.q_stars[state, :])]
        return optimal_actions


class EnvironmentBanditNonstationary(EnvironmentBandit):
    def __init__(self, rand_walk_std=0.01, n=1, k=10, q_stars_mean=None, q_stars_std=None, reward_std=None, state_probabilities=None, reveal_state=True):
        super().__init__(n=n,
                         k=k,
                         q_stars_mean=q_stars_mean,
                         q_stars_std=q_stars_std,
                         reward_std=reward_std,
                         state_probabilities=state_probabilities,
                         reveal_state=reveal_state)
        
        self.rand_walk_std = [0.01] * self.n if rand_walk_std == None else rand_walk_std
        assert len(self.rand_walk_std) == self.n, f"{len(self.rand_walk_std)=} must equal {self.n=}"
    
    def walk_q_star_randomly(self):
        self.q_stars += self.rng.normal(0, np.tile(self.rand_walk_std, (self.k, 1)).T)
    
    def update_time_step(self, time_step):
        if self.time_step is not None:
            self.walk_q_star_randomly()
        super().update_time_step(time_step)

class EnvironmentBanditManual(EnvironmentBandit):
    """Specify q_stars manually. q_stars are not reset."""
    def __init__(self, q_stars, n=1, k=10, reward_std=None, state_probabilities=None, reveal_state=True):
        self.q_stars = np.array(q_stars)
        self.n = n
        self.k = k
        assert self.q_stars.shape == (self.n, self.k)

        self.reward_std = [1] * self.n if reward_std == None else reward_std
        assert len(self.reward_std) == self.n, f"{len(self.reward_std)=} must equal {self.n=}"
        self.state_probabilities = [1 / self.n] * self.n if state_probabilities == None else state_probabilities
        assert len(self.state_probabilities) == self.n, f"{len(self.state_probabilities)=} must equal {self.n=}"
        assert sum(self.state_probabilities) == 1, f"{self.state_probabilities=} must sum to 1, not {sum(self.state_probabilities)=}"
        self.reveal_state = reveal_state
        self.rng = np.random.default_rng()

        self.reset()
    
    def reset(self):
        self.time_step = None
        self.agent_state = {}
        self.agent_action = {}