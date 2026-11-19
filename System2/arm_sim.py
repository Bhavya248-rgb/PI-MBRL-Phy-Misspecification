"""
arm_sim.py

System 2: a 2-link robot arm moving in a VERTICAL plane (gravity acts on it),
with a motor at each joint. Same role as diffdrive_sim.py in System 1.

Two separate things live here:

1. ArmSimulator      -> the "TRUE" arm (ground-truth reality). Uses the TRUE
   link masses and can have MOTOR LAG on the joint torques (hidden state).

2. arm_dynamics_step -> the "PHYSICS MODEL" prediction used by physics_only
   and by the physics half of the hybrid. It can be given DELIBERATELY WRONG
   link masses (the physics error, like the wrong wheel radius in System 1).
   It always assumes the motors produce the commanded torque instantly
   (no lag), exactly as the System 1 prior assumed instant wheel speeds.

State:  (q1, q2, dq1, dq2)
        q1 = shoulder angle from the horizontal (rad)
        q2 = elbow angle relative to link 1 (rad, 0 = arm straight)
        dq1, dq2 = joint angular speeds (rad/s)
Action: (tau1, tau2) commanded motor torques at shoulder and elbow (N m)

Dynamics (each link a uniform rod; see the doc "2-Link Robot Arm:
Kinematics and Dynamics Explained"):
    M(q) ddq + C(q, dq) dq + G(q) + b dq = tau_actual

Motor lag (first-order, exact discretization, same as System 1):
    alpha = 1 - exp(-dt / motor_tau)
    tau_actual <- tau_actual + alpha * (tau_cmd - tau_actual)
motor_tau = 0 gives tau_actual = tau_cmd (no lag).

Integration: each control step dt is split into N_SUBSTEPS small steps of
classic Runge-Kutta 4 (RK4). The simulator and the
physics prior use exactly the same integrator, so with correct masses and
no lag the prior predicts the simulator EXACTLY (checked in __main__).
"""

import numpy as np

# ---------------------------------------------------------------------
# True arm parameters (the "real robot")
# ---------------------------------------------------------------------
L1, L2 = 0.40, 0.30          # link lengths (m)
M1, M2 = 1.00, 0.80          # link masses (kg)
DAMPING = 1.0               # joint friction / damping b (N m s / rad)
GRAVITY = 9.81
DT = 0.05                    # control step (s): 20 commands per second
N_SUBSTEPS = 5               # RK4 sub-steps per control step (0.01 s each)
TORQUE_LIMIT = np.array([8.0, 4.0])   # |tau1| <= 8, |tau2| <= 4 N m
SPEED_CAP = 50.0             # rad/s; safeguard for imagined rollouts only (see below)
# Holding the arm straight out horizontally needs about 6.3 N m at the
# shoulder and 1.2 N m at the elbow, so the limits leave room to move
# but not enough to fling the arm around instantly.


def arm_dynamics_step(state, torque, m1, m2, l1=L1, l2=L2, b=DAMPING,
                      g=GRAVITY, dt=DT, n_sub=N_SUBSTEPS):
    """
    Advances the arm by one control step with a CONSTANT torque.
    Pure function, vectorized: state (4,) or (N, 4); torque (2,) or (N, 2).
    m1, m2 are the masses the caller BELIEVES (true ones for the simulator,
    possibly wrong ones for the physics prior).
    """
    s = np.array(state, dtype=float, copy=True)
    tau = np.broadcast_to(np.asarray(torque, dtype=float), s[..., :2].shape)
    h_dt = dt / n_sub
    # Safeguard: a learned model can imagine absurd joint speeds inside the
    # planner, which would overflow these equations. Speeds are capped at
    # SPEED_CAP (50 rad/s). The real arm never gets close: with the torque
    # limits and damping its speed stays below about 15 rad/s, so the cap
    # never changes the simulator or a correct prior.
    s[..., 2:4] = np.clip(s[..., 2:4], -SPEED_CAP, SPEED_CAP)

    def deriv(x):
        """d/dt of (q1, q2, dq1, dq2) = (dq1, dq2, ddq1, ddq2)."""
        a = joint_accelerations(x, tau, m1, m2, l1, l2, b, g)
        return np.concatenate([x[..., 2:], a], axis=-1)

    for _ in range(n_sub):            # classic Runge-Kutta 4 (RK4)
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * h_dt * k1)
        k3 = deriv(s + 0.5 * h_dt * k2)
        k4 = deriv(s + h_dt * k3)
        s = s + h_dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        s[..., 2:4] = np.clip(s[..., 2:4], -SPEED_CAP, SPEED_CAP)
    return s


def joint_accelerations(state, tau, m1, m2, l1=L1, l2=L2, b=DAMPING, g=GRAVITY):
    """Solves M(q) ddq = tau - C(q,dq) dq - G(q) - b dq for ddq.
    Vectorized over the leading dimensions of state (..., 4) and tau (..., 2)."""
    q1, q2, d1, d2 = state[..., 0], state[..., 1], state[..., 2], state[..., 3]
    lc1, lc2 = l1 / 2.0, l2 / 2.0
    I1, I2 = m1 * l1 ** 2 / 12.0, m2 * l2 ** 2 / 12.0
    c2, s2 = np.cos(q2), np.sin(q2)
    # inertia matrix M(q)
    M11 = I1 + I2 + m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * c2)
    M12 = I2 + m2 * (lc2 ** 2 + l1 * lc2 * c2)
    M22 = I2 + m2 * lc2 ** 2
    # Coriolis / centrifugal terms
    h = m2 * l1 * lc2 * s2
    C1 = -h * (2 * d1 * d2 + d2 ** 2)
    C2 = h * d1 ** 2
    # gravity torques (q1 measured from the horizontal)
    G1 = g * ((m1 * lc1 + m2 * l1) * np.cos(q1) + m2 * lc2 * np.cos(q1 + q2))
    G2 = g * m2 * lc2 * np.cos(q1 + q2)
    # right-hand side, then solve the 2x2 system
    r1 = tau[..., 0] - C1 - G1 - b * d1
    r2 = tau[..., 1] - C2 - G2 - b * d2
    det = M11 * M22 - M12 * M12
    a1 = (M22 * r1 - M12 * r2) / det
    a2 = (M11 * r2 - M12 * r1) / det
    return np.stack([a1, a2], axis=-1)


def tip_position(state, l1=L1, l2=L2):
    """Forward kinematics: (x, y) of the arm's tip. Works on (4,) or (N, 4).
    Always uses the TRUE lengths (lengths are not part of the physics error)."""
    state = np.asarray(state, dtype=float)
    q1, q2 = state[..., 0], state[..., 1]
    x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
    return np.stack([x, y], axis=-1)


class ArmSimulator:
    """Ground-truth 2-link arm, optionally with motor lag on the torques."""

    def __init__(self, m1=M1, m2=M2, dt=DT, motor_tau=0.0):
        self.m1, self.m2, self.dt = m1, m2, dt
        self.motor_tau = motor_tau
        self.alpha = 1.0 if motor_tau <= 0 else 1.0 - np.exp(-dt / motor_tau)
        self.reset()

    def reset(self, q1=-np.pi / 2, q2=0.0):
        """Places the arm at rest in pose (q1, q2), motors producing zero
        torque (like the System 1 robot starting at rest)."""
        self.state = np.array([q1, q2, 0.0, 0.0], dtype=float)
        self.tau_actual = np.zeros(2)
        return self.get_state()

    def get_state(self):
        return self.state.copy()

    def get_torques(self):
        """ACTUAL motor torques (what a joint torque sensor / motor current
        would measure). Used only by the 'torque' input type."""
        return self.tau_actual.copy()

    def step(self, tau1, tau2):
        cmd = np.clip(np.array([tau1, tau2], dtype=float), -TORQUE_LIMIT, TORQUE_LIMIT)
        if self.alpha >= 1.0:
            self.tau_actual = cmd
        else:
            self.tau_actual = self.tau_actual + self.alpha * (cmd - self.tau_actual)
        self.state = arm_dynamics_step(self.state, self.tau_actual, self.m1, self.m2,
                                       dt=self.dt)
        return self.get_state()


if __name__ == '__main__':
    # Checks (python arm_sim.py):
    #  1. energy is conserved with no torque and no friction
    #  2. tau = G(q) holds the arm still
    #  3. with correct masses and no lag, the physics prior predicts the
    #     simulator exactly; with wrong masses it does not
    #  4. motor lag delays the torque as expected
    rng = np.random.default_rng(0)

    def energy(s, m1=M1, m2=M2, l1=L1, l2=L2, g=GRAVITY):
        q1, q2, d1, d2 = s
        lc1, lc2 = l1 / 2, l2 / 2
        I1, I2 = m1 * l1 ** 2 / 12, m2 * l2 ** 2 / 12
        c2 = np.cos(q2)
        M11 = I1 + I2 + m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * c2)
        M12 = I2 + m2 * (lc2 ** 2 + l1 * lc2 * c2)
        M22 = I2 + m2 * lc2 ** 2
        KE = 0.5 * (M11 * d1 ** 2 + 2 * M12 * d1 * d2 + M22 * d2 ** 2)
        PE = g * (m1 * lc1 * np.sin(q1) + m2 * (l1 * np.sin(q1) + lc2 * np.sin(q1 + q2)))
        return KE + PE

    s = np.array([0.3, 1.0, 0.0, 0.0])
    E0 = energy(s)
    for _ in range(200):                       # 10 s, fine integration
        s = arm_dynamics_step(s, np.zeros(2), M1, M2, b=0.0, n_sub=200)
    print(f"1. energy after 10 s free swing: {energy(s):.4f} vs start {E0:.4f} "
          f"(drift {abs(energy(s) - E0) / abs(E0) * 100:.3f}%)")

    q = np.array([0.4, 0.7])
    lc1, lc2 = L1 / 2, L2 / 2
    G = GRAVITY * np.array([(M1 * lc1 + M2 * L1) * np.cos(q[0]) + M2 * lc2 * np.cos(q.sum()),
                            M2 * lc2 * np.cos(q.sum())])
    s = np.array([q[0], q[1], 0.0, 0.0])
    for _ in range(100):
        s = arm_dynamics_step(s, G, M1, M2)
    print(f"2. tau = G(q) for 5 s: angles moved by {np.abs(s[:2] - q).max():.2e} rad "
          f"(gravity torque {G[0]:.2f}, {G[1]:.2f} N m)")

    S = np.column_stack([rng.uniform(-np.pi, np.pi, 500), rng.uniform(-np.pi, np.pi, 500),
                         rng.uniform(-3, 3, 500), rng.uniform(-3, 3, 500)])
    A = rng.uniform(-1, 1, (500, 2)) * TORQUE_LIMIT
    sim = ArmSimulator()
    sim_next = []
    for i in range(500):                       # true simulator, one sample at a time
        sim.state = S[i].copy()
        sim_next.append(sim.step(A[i, 0], A[i, 1]))
    sim_next = np.array(sim_next)
    prior_ok = arm_dynamics_step(S, A, M1, M2)
    prior_bad = arm_dynamics_step(S, A, M1 * 1.25, M2 * 1.25)
    print(f"3. prior vs simulator, correct masses: max diff {np.abs(prior_ok - sim_next).max():.2e}; "
          f"+25% masses: mean speed error {np.abs(prior_bad - sim_next)[:, 2:].mean():.3f} rad/s")

    sim = ArmSimulator(motor_tau=0.1)
    sim.reset(q1=0.0, q2=0.0)
    taus = []
    for _ in range(10):
        sim.step(8.0, 0.0)
        taus.append(sim.get_torques()[0])
    print("4. motor lag tau=0.1 s, step command 8 N m -> actual shoulder torque:",
          " ".join(f"{t:.1f}" for t in taus))
