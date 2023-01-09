from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

@dataclass(frozen=True)
class Outcome:
    '''
    Outcome of making action in a state.
    '''
    new_state: Any
    reward: float


class Environment(ABC):
    @abstractmethod
    def get_possible_outcomes(self, state, action) -> List[Outcome]:
        '''
        Get outcomes you can get from a given state and action.
        '''
        pass

    def get_outcome_probability(self, state, action, outcome) -> float:
        '''
        Get probability of getting a given outcome from a given state and action.
        '''
        pass

    @abstractmethod
    def get_all_states(self) -> List[Any]:
        pass

    @abstractmethod
    def get_all_actions(self) -> List[Any]:
        pass


@dataclass
class Config:
    '''
    Configuration of the learning algorithm.
    '''
    discount_factor: float
    accuracy_threshold: float
    max_iterations: int

Policy = Dict[Any, Any]
StateValues = Dict[Any, float]