import numpy as np
import matplotlib.pyplot as plt

from common import *
from policy_iteration import *
from value_iteration import *

from normal_environment import *
from advanced_environment import *


def draw_policy_heatmap(max_cars, policy: Policy, values: StateValues, title, show: bool = False):
    action_map = np.zeros((max_cars + 1, max_cars + 1))
    value_map = np.zeros((max_cars + 1, max_cars + 1))

    for i in range(max_cars + 1):
        for j in range(max_cars + 1):
            action = policy[i, j]
            action_map[i, j] = action
            value_map[i, j] = values[i, j]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(title)
    ax1.set_title("Actions")
    ax1.set_ylabel("Cars at first location")
    ax1.set_xlabel("Cars at second location")
    action_im = ax1.imshow(action_map, interpolation='nearest')
    fig.colorbar(action_im, ax=ax1)
    ax1.invert_yaxis()

    value_im = ax2.imshow(value_map, interpolation='nearest')
    ax2.set_title("Values")
    ax2.set_ylabel("Cars at first location")
    ax2.set_xlabel("Cars at second location")
    fig.colorbar(value_im, ax=ax2)
    ax2.invert_yaxis()

    if show:
        plt.show()

def solve_environment(env: Environment, cfg: Config, epochs: int, method: callable):
    policy, values = method(env, cfg, epochs)
    draw_policy_heatmap(env.max_cars, policy, values)



first_location = Location(3, 3)
second_location = Location(4, 2)
rewards = Rewards()

MAX_CARS = 20
MAX_CARS_MOVED = 5
cfg = Config(0.9, 0.01, 100)
EPOCHS = 3

print("Preparing environments...")
envs = [
        #("normal-env", NormalEnvironment(MAX_CARS, MAX_CARS_MOVED, first_location, second_location, rewards)),
        ("advanced-env", AdvancedEnvironment(MAX_CARS, MAX_CARS_MOVED, first_location, second_location, rewards))]
    
methods = [("policy-iteration", policy_iteration), ("value-iteration",value_iteration)]


print("Everything is prepared")
for env_name, env in envs:
    for method_name, method in methods:
        print(f"Running {method_name} on {env_name}")
        policy, value = method(env, cfg, EPOCHS)
        title = f"{env_name} {method_name}"
        draw_policy_heatmap(env.max_cars, policy, value, title, show=False)
        filename = f"{env_name}_{method_name}.png"
        plt.savefig(filename)
