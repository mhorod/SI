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
        policy.actions[state] = 0
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

MAX_CARS = 10

env = Environment(MAX_CARS, first_location, second_location, rewards)

request_p = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
for i in range(MAX_CARS + 1):
    for j in range(MAX_CARS + 1):
        request_p[i, j] = env.fast_poisson(env.first_location.request_lambda, i) * env.fast_poisson(env.second_location.request_lambda, j)
        print (request_p[i, j], end=" ")
    print()

fig, ax = plt.subplots()
im = ax.imshow(request_p, interpolation='nearest')
ax.set_title("Probability of requests")
fig.colorbar(im)
ax.invert_yaxis()


game_over_p_with_no_move = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
for i in range(MAX_CARS + 1):
    for j in range(MAX_CARS + 1):
        p = 0
        for (next_state, reward), probability in env.get_possible_outcomes((i, j), 0):
            p += probability
        game_over_p_with_no_move[i, j] = 1 - p
        print(1 - p, end=" ")
    print()

fig, ax = plt.subplots()
im = ax.imshow(game_over_p_with_no_move, interpolation='nearest')
ax.set_title("Probability of game over with no move")
fig.colorbar(im)
ax.invert_yaxis()
plt.show()

exit()

print("Environment created")
p = policy_iteration(env, 0.9, 0.01, 1)


actions = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
values = np.zeros((MAX_CARS + 1, MAX_CARS + 1))

for i in range(MAX_CARS + 1):
    for j in range(MAX_CARS + 1):
        action = p.actions[i, j]
        print(action, end= " ")

        actions[i, j] = action
        values[i, j] = p.state_values[i, j]
    print()


for i in range(MAX_CARS + 1):
    for j in range(MAX_CARS + 1):
        print(p.state_values[i, j], end=" ")
    print()

# draw heatmap of actions and values
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(actions, interpolation='nearest')
ax1.set_title("Actions")
ax1.invert_yaxis()


fig.colorbar(ax2.imshow(values, interpolation='nearest'), ax=ax2)
ax2.invert_yaxis()
ax2.set_title("Values")

plt.show()

