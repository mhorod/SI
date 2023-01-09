import numpy as np
import matplotlib.pyplot as plt

from common import *
from policy_iteration import *
from value_iteration import *
from normal_environment import *


def draw_policy_heatmap(max_cars, policy: Policy, values: StateValues):
    action_map = np.zeros((max_cars + 1, max_cars + 1))
    value_map = np.zeros((max_cars + 1, max_cars + 1))

    for i in range(max_cars + 1):
        for j in range(max_cars + 1):
            action = policy[i, j]
            action_map[i, j] = action
            value_map[i, j] = values[i, j]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title("Actions")
    ax1.set_ylabel("Cars at first location")
    ax1.set_xlabel("Cars at second location")

    fig.colorbar(ax1.imshow(action_map, interpolation='nearest'), ax=ax1)
    ax1.invert_yaxis()

    ax2.imshow(value_map, interpolation='nearest')
    ax2.set_ylabel("Cars at first location")
    ax2.set_xlabel("Cars at second location")

    fig.colorbar(ax2.imshow(value_map, interpolation='nearest'), ax=ax2)
    ax2.invert_yaxis()

    plt.show()

def solve_environment(env: Environment, cfg: Config, epochs: int, method: callable):
    policy, values = method(env, cfg, epochs)
    draw_policy_heatmap(env.max_cars, policy, values)



first_location = Location(3, 3)
second_location = Location(4, 2)
rewards = Rewards()

MAX_CARS = 20
MAX_CARS_MOVED = 5
env = NormalEnvironment(MAX_CARS, MAX_CARS_MOVED, first_location, second_location, rewards)
method = policy_iteration
cfg = Config(0.9, 0.01, 100)

solve_environment(env, cfg, 3, method)
