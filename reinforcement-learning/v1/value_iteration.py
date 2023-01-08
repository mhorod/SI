
import matplotlib.pyplot as plt

from environment import *

def random_values(env):
    values = {}
    for state in env.all_states:
        values[state] = 0
    return values

def policy_from_values(env, values, discount_factor):
    policy = {}
    for state in env.all_states:
        best_action = None
        best_value = -np.inf
        for action in Action:
            value = 0
            for (next_state, reward), probability in env.get_possible_outcomes(state, action):
                value += probability * (reward + discount_factor * values[next_state])
            if value > best_value:
                best_value = value
                best_action = action
        policy[state] = best_action
    return policy


def improve(env, values, discount_factor):
    for _ in range(100):
        improved = False
        for state in env.all_states:
            old_value = values[state]
            best_value = -np.inf
            for action in Action:
                value = 0
                for (next_state, reward), probability in env.get_possible_outcomes(state, action):
                    value += probability * (reward + discount_factor * values[next_state])
                if value > best_value:
                    best_value = value
            values[state] = best_value
            if best_value != old_value:
                improved = True
        if not improved:
            break

def value_iteration(env, discount_factor, accuracy_threshold, max_iterations):
    values = random_values(env)
    for i in range(max_iterations):
        print("Iteration: ", i)
        improve(env, values, discount_factor)
    return values

first_location = Location(3, 3)
second_location = Location(4, 2)
rewards = Rewards()

MAX_CARS = 20

env = Environment(MAX_CARS, first_location, second_location, rewards)
print("Environment created")
values = value_iteration(env, 0.9, 0.01, 5)
actions = policy_from_values(env, values, 0.9)


actions_plt = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
values_plt = np.zeros((MAX_CARS + 1, MAX_CARS + 1))

for i in range(MAX_CARS + 1):
    for j in range(MAX_CARS + 1):
        action = actions[(MAX_CARS - i, j)]
        if action == Action.MoveToSecond:
            print("S", end=" ")
        elif action == Action.MoveToFirst:
            print("F", end=" ")
        elif action == Action.OpenStore:
            print("O", end=" ")

        actions_plt[MAX_CARS - i, j] = action.value
        values_plt[MAX_CARS - i, j] = values[(MAX_CARS - i, j)]
    print()

# draw heatmap of actions and values
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(actions_plt, cmap='hot', interpolation='nearest')
ax1.set_title("Actions")
ax1.invert_yaxis()

fig.colorbar(ax2.imshow(values_plt, cmap='hot', interpolation='nearest'), ax=ax2)
ax2.set_title("Values")
ax2.invert_yaxis()

plt.show()

