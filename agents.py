from abc import ABC, abstractmethod

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