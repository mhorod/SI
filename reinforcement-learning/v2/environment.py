from enum import Enum, auto
from dataclasses import dataclass

import numpy as np

import random

Action = [i for i in range(-5, 6)]

class Location:
    def __init__(self, request_lambda, return_lambda):
        self.cars = 0
        self.request_lambda = request_lambda
        self.return_lambda = return_lambda

    def request_cars(self):
        return np.random.poisson(self.request_lambda)

    def return_cars(self):
        return np.random.poisson(self.return_lambda)

@dataclass
class Rewards:
    move_car: int = -2
    reward_per_car: int = 10
    game_over: int = 0
    invalid_move: int = 0
    
class Environment:
    def __init__(self, max_cars, first_location, second_location, rewards):
        self.max_cars = max_cars
        self.first_location = first_location
        self.second_location = second_location
        self.rewards = rewards
        self.reset()

        self.poissons = {}
        ls = [self.first_location.request_lambda, self.first_location.return_lambda, self.second_location.request_lambda, self.second_location.return_lambda]
        for l in ls:
            for k in range(self.max_cars + 1):
                self.poissons[(l, k)] = self.get_poisson_probability(l, k)

        self.all_states = []
        for i in range(max_cars + 1):
            for j in range(max_cars + 1):
                self.all_states.append((i, j))

        self.transition_probabilities = {}
        self.generate_transition_probability()

    def reset(self):
        self.first_location.cars = random.randint(0, self.max_cars)
        self.second_location.cars = random.randint(0, self.max_cars)

    def generate_transition_probability(self):
        for state in self.all_states:
            for action in Action:
                for next_state in self.all_states:
                    rps = self.get_rewards_and_probabilities(state, action, next_state)

                    if (state, action) not in self.transition_probabilities:
                        self.transition_probabilities[(state, action)] = {}

                    for (reward, probability) in rps:
                        if probability > 0:
                            self.transition_probabilities[(state, action)][(next_state, reward)] = probability


    def get_rewards_and_probabilities(self, state, action, next_state):
        '''
        Return the reward and probability of the next state given the current state and action
        '''
        if state[0] - action < 0 or state[1] + action < 0:
            return [(self.rewards.invalid_move, 0)]

        if state[0] - action > self.max_cars or state[1] + action > self.max_cars:
            return [(self.rewards.invalid_move, 0)]
        
        cost = self.rewards.move_car * abs(action)
        state = (state[0] - action, state[1] + action)
        state = (min(state[0], self.max_cars), min(state[1], self.max_cars))
            
        open_results = self.get_rewards_and_probabilities_open_store(state, next_state)
        open_results = [(cost + open_reward, probability) for (open_reward, probability) in open_results]
        return open_results

    def get_rewards_and_probabilities_open_store(self, state, next_state):
        # d1 = requested - returned
        # returned = requested - d1
        d1 = state[0] - next_state[0]
        d2 = state[1] - next_state[1]

        rewards = {}
        for first_requested in range(max(d1, 0), min(state[0], self.max_cars) + 1):
            for second_requested in range(max(d2, 0), min(state[1], self.max_cars) + 1):
                p11 = self.fast_poisson(self.first_location.request_lambda, first_requested)
                p12 = self.fast_poisson(self.first_location.return_lambda, first_requested - d1)

                p21 = self.fast_poisson(self.second_location.request_lambda, second_requested)
                p22 = self.fast_poisson(self.second_location.return_lambda, second_requested - d2)

                probability = p11 * p12 * p21 * p22
                reward = self.rewards.reward_per_car * (first_requested + second_requested)

                rewards[reward] = rewards.get(reward, 0) + probability

        return [(reward, probability) for reward, probability in rewards.items()]

    def fast_poisson(self, l, k):
        return self.poissons[(l, k)]

    def get_rewards_and_probabilities_open_store_game_over(self, state):
        # Calculate probability that net change is less or equal to the number of cars in state
        p1 = 0
        p2 = 0

        for i in range(0, state[0] + 1):
            p1 += self.fast_poisson(self.first_location.request_lambda, i)
        
        for i in range(0, state[1] + 1):
            p2 += self.fast_poisson(self.second_location.request_lambda, i)

        
        # p1 * p2 is the probability that game is not over so we invert it
        probability = 1 - p1 * p2
        reward = self.rewards.game_over

        return [(reward, probability)]

    def get_poisson_probability(self, l, k):
        if k < 0:
            return 0
        else:
            return (l ** k) * (np.exp(-l)) / np.math.factorial(k)

    def get_transition_probability(self, current_state, action, next_state, reward):
        if (current_state, action) not in self.transition_probabilities:
            return 0
        elif (next_state, reward) not in self.transition_probabilities[(current_state, action)]:
            return 0
        else:
            return self.transition_probabilities[(current_state, action)][(next_state, reward)]

    def get_possible_outcomes(self, state, action):
        return self.transition_probabilities[(state, action)].items()