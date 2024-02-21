import numpy as np

def run_simulation(max_rollouts, max_time_steps, environment, agents):
    agent_names = []
    for agent in agents:
        assert agent.name not in agent_names, "Agent names must be unique"
        agent_names.append(agent.name)

    results = {}
    agent_categories = [("rewards", (max_rollouts, max_time_steps)),
                        ("actions", (max_rollouts, max_time_steps)),
                        ("optimal_actions", (max_rollouts, max_time_steps)),
                        ("action_values", (max_rollouts, max_time_steps, environment.k))]
    
    environment_categories = [("q_stars", (max_rollouts, max_time_steps, environment.k))]

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
            results["q_stars"]["environment"][rollout, time_step] = environment.get_state_internal()["q_stars"]

            for agent in agents:
                agent_name = agent.name
                state_observed, reward = environment.get_observation(agent_name)
                agent.receive_observation(state_observed, reward)

                results["action_values"][agent_name][rollout, time_step] = agent.get_action_values_for_state(state_observed)

                if time_step > 0:
                    results["rewards"][agent_name][rollout, time_step] = reward
                
                action = agent.get_action()
                environment.receive_action(agent_name, action)
                
                results["actions"][agent_name][rollout, time_step] = action
                if action in environment.get_all_optimal_actions():
                    results["optimal_actions"][agent_name][rollout, time_step] = 1
    return results