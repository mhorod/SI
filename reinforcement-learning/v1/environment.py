from enum import Enum, auto
from dataclasses import dataclass

import numpy as np

import random

class Action(Enum):
    MoveToSecond = auto()
    MoveToFirst = auto()
    OpenStore = auto()


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
    invalid_move: int = -1000
    
class Environment:
    GAME_OVER_STATE = (-1, -1)
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

        self.all_states = [Environment.GAME_OVER_STATE]
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
        if action == Action.MoveToSecond:
            return self.get_rewards_and_probabilities_move_to_second(state, next_state)
        elif action == Action.MoveToFirst:
            return self.get_rewards_and_probabilities_move_to_first(state, next_state)
        elif action == Action.OpenStore:
            return self.get_rewards_and_probabilities_open_store(state, next_state)
        else:
            raise Exception("Invalid action")

    def get_rewards_and_probabilities_move_to_second(self, state, next_state):
        d1 = next_state[0] - state[0]
        d2 = next_state[1] - state[1]

        if d1 == -1 and d2 == 1:
            return [(self.rewards.move_car, 1)]
        else:
            return [(0, 0)]
        
    def get_rewards_and_probabilities_move_to_first(self, state, next_state):
        d1 = next_state[0] - state[0]
        d2 = next_state[1] - state[1]

        if d1 == 1 and d2 == -1:
            return [(self.rewards.move_car, 1)]
        else:
            return [(0, 0)]

    def get_rewards_and_probabilities_open_store(self, state, next_state):
        if next_state == Environment.GAME_OVER_STATE:
            return self.get_rewards_and_probabilities_open_store_game_over(state)
        else:
            d1 = state[0] - next_state[0]
            d2 = state[1] - next_state[1]

            rewards = {}
            for i in range(0, self.max_cars + 1):
                for j in range(0, self.max_cars + 1):
                    if i - d1 < 0 or j - d2 < 0:
                        continue
                    if i - d1 > self.max_cars or j - d2 > self.max_cars:
                        continue
                    p11 = self.fast_poisson(self.first_location.request_lambda, i)
                    p12 = self.fast_poisson(self.first_location.return_lambda, i - d1)

                    p21 = self.fast_poisson(self.second_location.request_lambda, j)
                    p22 = self.fast_poisson(self.second_location.return_lambda, j - d2)

                    probability = p11 * p12 * p21 * p22
                    reward = self.rewards.reward_per_car * (i + j)

                    rewards[reward] = rewards.get(reward, 0) + probability

            return [(reward, probability) for reward, probability in rewards.items()]

    def fast_poisson(self, l, k):
        return self.poissons[(l, k)]



    def get_rewards_and_probabilities_open_store_game_over(self, state):
        l1 = self.first_location.request_lambda + self.first_location.return_lambda
        l2 = self.second_location.request_lambda + self.second_location.return_lambda

        # Calculate probability that net change is less or equal to the number of cars in state
        p1 = 0
        p2 = 0

        for i in range(state[0] + 1):
            p1 += self.get_poisson_probability(l1, i)
        
        for i in range(state[1] + 1):
            p2 += self.get_poisson_probability(l2, i)


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