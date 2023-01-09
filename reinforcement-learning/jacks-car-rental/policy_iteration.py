import numpy as np

from common import *

def evaluate_policy(
        env: Environment, 
        policy: Policy, 
        state_values: StateValues,
        cfg: Config) -> StateValues:
    '''
    Evaluate a policy given state values.
    '''
    state_values = state_values.copy()
    for _ in range(cfg.max_iterations):
        delta = 0
        for state in env.get_all_states():
            old_value = state_values[state]
            action = policy[state]
            new_value = 0
            for outcome in env.get_possible_outcomes(state, action):
                p = env.get_outcome_probability(state, action, outcome)
                new_value += p * (outcome.reward + cfg.discount_factor * state_values[outcome.new_state])
            state_values[state] = new_value
            delta = max(delta, abs(old_value - new_value))
        if delta < cfg.accuracy_threshold:
            break
    return state_values

def improve_policy(
    env: Environment,
    policy: Policy,
    state_values: StateValues,
    cfg: Config) -> Policy:
    '''
    Improve a policy given state values.
    '''
    for _ in range(cfg.max_iterations):
        improved = False
        for state in env.get_all_states():
            old_action = policy[state]
            best_action = None
            best_value = -np.inf
            for action in env.get_all_actions():
                value = 0
                for outcome in env.get_possible_outcomes(state, action):
                    p = env.get_outcome_probability(state, action, outcome)
                    value += p * (outcome.reward + cfg.discount_factor * state_values[outcome.new_state])
                if value > best_value:
                    best_value = value
                    best_action = action
            policy[state] = best_action
            if best_action != old_action:
                improved = True
        if not improved:
            break
    return policy


def policy_iteration(env: Environment, cfg: Config, epochs: int) -> Policy:
    '''
    Policy iteration algorithm.
    '''
    actions = env.get_all_actions()
    policy = {state: actions[0] for state in env.get_all_states()}
    state_values = {state: 0 for state in env.get_all_states()}
    for i in range(epochs):
        print("Epoch", i)
        state_values = evaluate_policy(env, policy, state_values, cfg)
        policy = improve_policy(env, policy, state_values, cfg)
    return policy, state_values