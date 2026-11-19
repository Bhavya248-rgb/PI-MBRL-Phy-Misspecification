"""
task.py

The task: point-to-point navigation.

CHANGES IN THIS VERSION
  - Every function takes an optional `goal`, so the same code serves the
    training task and the frozen-evaluation tasks (new start AND goal).
  - Functions work on one state (3,) or a batch of states (N, 3).
  - TRAIN_START / TRAIN_GOAL and EVAL_TASKS are defined here, in one
    place, so every model, seed and error level is evaluated on exactly
    the same tasks.
"""

import numpy as np

GOAL = np.array([0.6, 0.6])          # training goal (unchanged)
TRAIN_START = (0.0, 0.0, 0.0)        # training start pose (unchanged)
TRAIN_GOAL = GOAL
SUCCESS_RADIUS = 0.1
MAX_STEPS = 120

# Sanity check: wheel_radius=0.03, omega_max=8 rad/s, slip 0.85 -> max
# linear speed ~0.20 m/s. Over dt=0.1 and MAX_STEPS=120 the robot can
# cover ~2.4 m, comfortably more than the longest task below (~0.85 m
# plus turning on the spot). Re-check if GOAL, MAX_STEPS, dt, radius or
# action bounds change.

# Frozen-evaluation tasks: (start_pose (x, y, theta), goal (x, y)).
# All lie inside the region the scattered warmup covers (about -0.1..0.7)
# and all differ from the training task in start position, start heading
# and goal. Task 5 starts facing directly away from its goal.
EVAL_TASKS = [
    ((0.6, 0.0,  np.pi / 2), (0.0, 0.6)),
    ((0.0, 0.6,  0.0),       (0.6, 0.0)),
    ((0.3, 0.3,  np.pi),     (0.7, 0.2)),
    ((0.7, 0.7, -np.pi / 2), (0.1, 0.1)),
    ((0.0, 0.0,  np.pi),     (0.5, -0.1)),
    ((-0.1, 0.3, 0.0),       (0.4, 0.7)),
]


def _dist(state, goal):
    state = np.asarray(state)
    return np.linalg.norm(state[..., :2] - np.asarray(goal), axis=-1)


def reward_fn(state, goal=GOAL):
    dist = _dist(state, goal)
    return -dist + 10.0 * (dist < SUCCESS_RADIUS)


def is_success(state, goal=GOAL):
    return bool(_dist(state, goal) < SUCCESS_RADIUS)


def distance_to_goal(state, goal=GOAL):
    return float(_dist(state, goal))
