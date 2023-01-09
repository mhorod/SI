from environment import *

class NormalEnvironment(CarEnvironment):
    def __init__(self,
                    max_cars,
                    max_cars_moved,
                    first_location,
                    second_location,
                    rewards):
        self.poisson = Poisson()

        self.actions = tuple([0] + [i for i in range(1, max_cars_moved + 1)] + [i for i in range(-1, -max_cars_moved - 1, -1)])
        self.open_store_transitions = {}
        super().__init__(max_cars, first_location, second_location, rewards)
        
    def get_all_actions(self) -> List[Any]:
        return self.actions

    def generate_transition_probabilities(self):
        for state in self.all_states:
            for action in self.actions:
                self.transition_probabilities[(state, action)] = self.generate_for_state_and_action(state, action)

    def generate_for_state_and_action(self, state, action):
        f, s = state

        # Invalid action - transition to terminal state with reward 0
        if f - action < 0 or s + action < 0 or f - action > self.max_cars or s + action > self.max_cars:
            return {}

        # Valid action
        new_state = (f - action, s + action)
        cost = self.rewards.move_car * abs(action)
        return {
            Outcome(outcome.new_state, cost + outcome.reward) : p
            for outcome, p in self.get_open_store_transitions(new_state).items()
        }

    def get_open_store_transitions(self, state):
        if state in self.open_store_transitions:
            return self.open_store_transitions[state]
        else:
            transition = self.generate_open_store_transitions(state)
            self.open_store_transitions[state] = transition
            return transition

    def generate_open_store_transitions(self, state):
        transition = {}
        for requested_f in range(0, state[0] + 1):
            for requested_s in range(0, state[1] + 1):
                reward = self.rewards.rent_car * (requested_f + requested_s)
                requested_state = (state[0] - requested_f, state[1] - requested_s)
                request_p = self.poisson(self.first_location.request_lambda, requested_f) * self.poisson(self.second_location.request_lambda, requested_s)

                p_first_return_fits = 0
                p_second_return_fits = 0

                for returned_f in range(0, self.max_cars - requested_state[0] + 1):
                    p_first_return_fits += self.poisson(self.first_location.return_lambda, returned_f)
                
                for returned_s in range(0, self.max_cars - requested_state[1] + 1):
                    p_second_return_fits += self.poisson(self.second_location.return_lambda, returned_s)
                
                # Events where cars returned at both locations fit
                for returned_f in range(0, self.max_cars - requested_state[0] + 1):
                    for returned_s in range(0, self.max_cars - requested_state[1] + 1):
                        returned_state = (requested_state[0] + returned_f, requested_state[1] + returned_s)
                        p = request_p * self.poisson(self.first_location.return_lambda, returned_f) * self.poisson(self.second_location.return_lambda, returned_s)
                        outcome = Outcome(returned_state, reward)
                        transition[outcome] = transition.get(outcome, 0) + p
                
                # Events where only cars returned at first location fit
                for returned_f in range(0, self.max_cars - requested_state[0] + 1):
                    returned_state = (requested_state[0] + returned_f, self.max_cars)
                    p = request_p * self.poisson(self.first_location.return_lambda, returned_f) * (1 - p_second_return_fits)
                    outcome = Outcome(returned_state, reward)
                    transition[outcome] = transition.get(outcome, 0) + p
                
                # Events where only cars returned at second location fit
                for returned_s in range(0, self.max_cars - requested_state[1] + 1):
                    returned_state = (self.max_cars, requested_state[1] + returned_s)
                    p = request_p * self.poisson(self.second_location.return_lambda, returned_s) * (1 - p_first_return_fits)
                    outcome = Outcome(returned_state, reward)
                    transition[outcome] = transition.get(outcome, 0) + p

                # Events where no cars returned fit
                returned_state = (self.max_cars, self.max_cars)
                p = request_p * (1 - p_first_return_fits) * (1 - p_second_return_fits)
                outcome = Outcome(returned_state, reward)
                transition[outcome] = transition.get(outcome, 0) + p

        return transition 