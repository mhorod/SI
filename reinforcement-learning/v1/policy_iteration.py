# Policy iteration

from environment import *

import matplotlib.pyplot as plt

class Policy:
    def __init__(self):
        self.actions = {}
        self.state_values = {}

def random_policy(env):
    policy = Policy()
    for state in env.all_states:
        policy.actions[state] = Action.OpenStore # random.choice(list(Action))
        policy.state_values[state] = 0
    return policy


def evaluate(env, policy, discount_factor, accuracy_threshold):
    for _ in range(100):
        delta = 0
        for state in env.all_states:
            old_value = policy.state_values[state]
            action = policy.actions[state]
            new_value = 0
            for (next_state, reward), probability in env.get_possible_outcomes(state, action):
                new_value += probability * (reward + discount_factor * policy.state_values[next_state])
            policy.state_values[state] = new_value

            delta = max(delta, abs(old_value - new_value))
        if delta < accuracy_threshold:
            break

def improve(env, policy, discount_factor):
    for _ in range(100):
        improved = False
        for state in env.all_states:
            old_action = policy.actions[state]
            best_action = None
            best_value = -np.inf
            for action in Action:
                value = 0
                for (next_state, reward), probability in env.get_possible_outcomes(state, action):
                    value += probability * (reward + discount_factor * policy.state_values[next_state])
                if value > best_value:
                    best_value = value
                    best_action = action
            policy.actions[state] = best_action
            if best_action != old_action:
                improved = True

        if not improved:
            break


def policy_iteration(env, discount_factor, accuracy_threshold, max_iterations):
    policy = random_policy(env)
    for i in range(max_iterations):
        print("Iteration: ", i)
        evaluate(env, policy, discount_factor, accuracy_threshold)
        improve(env, policy, discount_factor)
    return policy


first_location = Location(3, 3)
second_location = Location(4, 2)
rewards = Rewards()

MAX_CARS = 20

env = Environment(MAX_CARS, first_location, second_location, rewards)
print("Environment created")
p = policy_iteration(env, 0.9, 0.01, 5)


actions = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
values = np.zeros((MAX_CARS + 1, MAX_CARS + 1))

for i in range(MAX_CARS + 1):
    for j in range(MAX_CARS + 1):
        action = p.actions[(MAX_CARS - i, j)]
        if action == Action.MoveToSecond:
            print("S", end=" ")
        elif action == Action.MoveToFirst:
            print("F", end=" ")
        elif action == Action.OpenStore:
            print("O", end=" ")

        actions[MAX_CARS - i, j] = action.value
        values[MAX_CARS - i, j] = p.state_values[(MAX_CARS - i, j)]
    print()

# draw heatmap of actions and values
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(actions, cmap='hot', interpolation='nearest')
ax1.set_title("Actions")
ax1.invert_yaxis()

ax2.imshow(values, cmap='hot', interpolation='nearest')
ax2.set_title("Values")

fig.colorbar(ax2.imshow(values, cmap='hot', interpolation='nearest'), ax=ax2)
ax2.invert_yaxis()

plt.show()

