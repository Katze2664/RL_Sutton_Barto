import numpy as np

def run_simulation(max_episodes, max_time_steps, environment, agents):
    agent_names = []
    for agent in agents:
        assert agent.name not in agent_names, "Agent names must be unique"
        agent_names.append(agent.name)

    results = {}
    agent_categories = [("states_actual", (max_episodes, max_time_steps)),
                        ("rewards", (max_episodes, max_time_steps)),
                        ("actions", (max_episodes, max_time_steps)),
                        ("optimal_actions", (max_episodes, max_time_steps)),
                        ("action_values", (max_episodes, max_time_steps, environment.n, environment.k))]
    
    environment_categories = [("q_stars", (max_episodes, max_time_steps, environment.n, environment.k))]

    for category_name, category_shape in agent_categories:
        results[category_name] = {agent_name: np.zeros(category_shape) for agent_name in agent_names}
    
    for category_name, category_shape in environment_categories:
        results[category_name] = {"environment": np.zeros(category_shape)}

    for episode in range(max_episodes):
        environment.reset()
        for agent in agents:
            agent_name = agent.name
            agent.reset()

        for time_step in range(max_time_steps):
            environment.update_time_step(time_step)
            results["q_stars"]["environment"][episode, time_step] = environment.get_q_stars()

            for agent in agents:
                agent_name = agent.name
                state_observed, reward = environment.get_observation(agent_name)
                agent.receive_observation(state_observed, reward)

                state_actual = environment.get_state_actual(agent_name)
                results["states_actual"][agent_name][episode, time_step] = state_actual

                for state, action_values in agent.get_action_values().items():
                    results["action_values"][agent_name][episode, time_step, state] = action_values

                if time_step > 0:
                    results["rewards"][agent_name][episode, time_step] = reward
                else:
                    results["rewards"][agent_name][episode, time_step] = np.nan
                
                action = agent.get_action()
                environment.receive_action(agent_name, action)
                
                results["actions"][agent_name][episode, time_step] = action
                if action in environment.get_all_optimal_actions(state_actual):
                    results["optimal_actions"][agent_name][episode, time_step] = 1
    return results