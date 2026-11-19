"""
diffdrive_sim.py

Core differential-drive kinematics + simulator.

Two separate things live here:

1. DiffDriveSimulator  -> the "TRUE" robot (ground-truth reality). Uses
   the TRUE wheel radius and a slip/efficiency factor.

2. predict_next_state  -> the "PHYSICS MODEL" prediction used by the
   physics_only model and by the physics half of the hybrid model. It
   can be given a DELIBERATELY WRONG wheel radius or slip value -- that
   is how the misspecification sweep is created.

CHANGE IN THIS VERSION: predict_next_state is now vectorized. It accepts
either one state of shape (3,) or a batch of states of shape (N, 3)
(with matching wheel-speed arrays), and returns the same shape. The
equations are unchanged; batching only lets the CEM planner evaluate all
its candidate action sequences at once instead of one by one.
"""

import numpy as np


class DiffDriveSimulator:
    """
    Ground-truth differential-drive robot simulator.

    State: (x, y, theta)   position in metres, heading in radians
    Action: (omega_L, omega_R)   commanded wheel angular velocities, rad/s
    """

    def __init__(self, true_wheel_radius=0.03, wheelbase=0.15,
                 slip_factor=1.0, dt=0.1):
        self.r_true = true_wheel_radius
        self.L = wheelbase
        self.slip = slip_factor
        self.dt = dt
        self.reset()

    def reset(self, x=0.0, y=0.0, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        return self.get_state()

    def get_state(self):
        return np.array([self.x, self.y, self.theta], dtype=float)

    def step(self, omega_L, omega_R):
        v_L = self.slip * self.r_true * omega_L
        v_R = self.slip * self.r_true * omega_R

        v = (v_R + v_L) / 2.0
        omega = (v_R - v_L) / self.L

        self.x = self.x + v * np.cos(self.theta) * self.dt
        self.y = self.y + v * np.sin(self.theta) * self.dt
        self.theta = self.theta + omega * self.dt

        return self.get_state()

    def rollout(self, actions):
        states = [self.get_state()]
        for (oL, oR) in actions:
            states.append(self.step(oL, oR))
        return np.array(states)


def predict_next_state(state, omega_L, omega_R,
                        assumed_r, wheelbase, dt, assumed_slip=1.0):
    """
    PHYSICS MODEL prediction of the next state. Pure function.
    state: shape (3,) or (N, 3). omega_L, omega_R: scalars or shape (N,).
    assumed_r / assumed_slip are what the MODEL believes.
    """
    state = np.asarray(state, dtype=float)
    x, y, theta = state[..., 0], state[..., 1], state[..., 2]

    v_L = assumed_slip * assumed_r * np.asarray(omega_L, dtype=float)
    v_R = assumed_slip * assumed_r * np.asarray(omega_R, dtype=float)

    v = (v_R + v_L) / 2.0
    omega = (v_R - v_L) / wheelbase

    x_new = x + v * np.cos(theta) * dt
    y_new = y + v * np.sin(theta) * dt
    theta_new = theta + omega * dt

    return np.stack([x_new, y_new, theta_new], axis=-1)


def predict_rollout(start_state, actions, assumed_r, wheelbase, dt,
                     assumed_slip=1.0):
    states = [np.array(start_state, dtype=float)]
    s = states[0]
    for (oL, oR) in actions:
        s = predict_next_state(s, oL, oR, assumed_r, wheelbase, dt, assumed_slip)
        states.append(s)
    return np.array(states)
