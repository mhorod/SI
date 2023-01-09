import numpy as np

from common import *

def improve_values(
    env: Environment,
    values: StateValues,
    cfg: Config) -> StateValues:
    '''
    Improve state values
    '''
    for _ in range(cfg.max_iterations):
        delta = 0
        for state in env.get_all_states():
            old_value = values[state]
            best_value = -np.inf
            for action in env.get_all_actions():
                value = 0
                for outcome in env.get_possible_outcomes(state, action):
                    p = env.get_outcome_probability(state, action, outcome)
                    value += p * (outcome.reward + cfg.discount_factor * values[outcome.new_state])
                if value > best_value:
                    best_value = value
            values[state] = best_value
            delta = max(delta, abs(old_value - best_value))
        if delta < cfg.accuracy_threshold:
            break
    return values

def greedy_policy(env: Environment, values: StateValues, cfg: Config):
    policy = {}
    for state in env.get_all_states():
        best_action = None
        best_value = -np.inf
        for action in env.get_all_actions():
            value = 0
            for outcome in env.get_possible_outcomes(state, action):
                p = env.get_outcome_probability(state, action, outcome)
                value += p * (outcome.reward + cfg.discount_factor * values[outcome.new_state])
            if value > best_value:
                best_value = value
                best_action = action
        policy[state] = best_action
    
    return policy



def value_iteration(
    env: Environment,
    cfg: Config,
    epochs: int) -> Policy:
    '''
    Value iteration algorithm
    '''
    values = {state: 0 for state in env.get_all_states()}
    for i in range(epochs):
        print("Epoch", i)
        values = improve_values(env, values, cfg)
    return greedy_policy(env, values, cfg), values