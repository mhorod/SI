import numpy as np

from common import *


@dataclass
class Location:
    request_lambda: int
    return_lambda: int

@dataclass
class Rewards:
    move_car: int = -2
    rent_car: int = 10
    store_car: int = -4

class Poisson:
    def __init__(self):
        self.cache = {}

    def __call__(self, l, k):
        if (l, k) not in self.cache:
            self.cache[(l, k)] = self.get_poisson_probability(l, k)
        return self.cache[(l, k)]
    
    def get_poisson_probability(self, l, k):
        return (l ** k) * np.exp(-l) / np.math.factorial(k)


class CarEnvironment(Environment):
    def __init__(self,
                    max_cars,
                    first_location,
                    second_location,
                    rewards):
        self.max_cars = max_cars
        self.first_location = first_location
        self.second_location = second_location
        self.rewards = rewards

        self.all_states = tuple([(f, s) for f in range(max_cars + 1) for s in range(max_cars + 1)])

        self.transition_probabilities = {}
        self.generate_transition_probabilities()

    def get_all_states(self) -> List[Any]:
        return self.all_states

    def get_possible_outcomes(self, state: Any, action: Any) -> List[Outcome]:
        return self.transition_probabilities[(state, action)].keys()

    def get_outcome_probability(self, state: Any, action: Any, outcome: Outcome) -> float:
        return self.transition_probabilities[(state, action)][outcome]

    @abstractmethod
    def get_all_actions(self) -> List[Any]:
        pass

    @abstractmethod
    def generate_transition_probabilities(self):
        pass