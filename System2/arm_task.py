"""
arm_task.py

System 2 task: move the arm's TIP to a target point and stop there.
Same role as task.py in System 1 (drive to a goal point).

A task = (start pose (q1, q2) with the arm at rest, target (x, y)).
Success = tip within SUCCESS_RADIUS of the target AND the arm nearly
still (joint speeds below SETTLE_SPEED), so swinging through the target
does not count: the arm has to actually stop there.

Reward per step (used by the planner and for the return):
    - tip distance to target (m)
    - SPEED_COST * joint speed (rad/s)     small push to brake near the target
    + 10 if success condition holds        same bonus as System 1

Functions work on one state or a batch, and use only the first 4 numbers
of the state (q1, q2, dq1, dq2), so extra inputs (previous command,
torques, history) do not matter here.
"""

import numpy as np

from arm_sim import tip_position

SUCCESS_RADIUS = 0.05        # 5 cm
SETTLE_SPEED = 1.0           # rad/s, norm of (dq1, dq2)
SPEED_COST = 0.02
MAX_STEPS = 100              # 5 s at dt = 0.05

# Training task: arm hanging straight down -> target up and to the right.
TRAIN_START = (-np.pi / 2, 0.0)
TRAIN_GOAL = np.array([0.35, 0.35])

# Frozen-evaluation tasks: (start (q1, q2), target (x, y)). Each differs from
# the training task in start pose and target; all targets are reachable
# (0.46 to 0.55 m from the shoulder; the arm reaches 0.10 to 0.70 m).
# Task 3 starts balanced straight up (unstable), task 5 pointing left.
EVAL_TASKS = [
    ((-np.pi / 2, 0.0),       (-0.35, 0.30)),
    ((0.0, 0.0),              (0.20, -0.45)),
    ((np.pi / 2, 0.0),        (0.45, 0.10)),
    ((-np.pi / 2, np.pi / 2), (0.00, 0.55)),
    ((np.pi, 0.0),            (0.30, 0.40)),
    ((-0.5, 1.2),             (-0.25, -0.40)),
]


def _dist(state, goal):
    state = np.asarray(state, dtype=float)
    return np.linalg.norm(tip_position(state[..., :4]) - np.asarray(goal), axis=-1)


def _speed(state):
    state = np.asarray(state, dtype=float)
    return np.linalg.norm(state[..., 2:4], axis=-1)


def _settled(state, goal):
    return (_dist(state, goal) < SUCCESS_RADIUS) & (_speed(state) < SETTLE_SPEED)


def reward_fn(state, goal=TRAIN_GOAL):
    return -_dist(state, goal) - SPEED_COST * _speed(state) + 10.0 * _settled(state, goal)


def is_success(state, goal=TRAIN_GOAL):
    return bool(_settled(state, goal))


def distance_to_goal(state, goal=TRAIN_GOAL):
    return float(_dist(state, goal))
