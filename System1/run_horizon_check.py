"""
run_horizon_check.py

Checks ONE odd result from Study 1: physics_only does BETTER when it
assumes the wheel is LARGER than it really is (negative radius error)
than when its physics is exactly correct.

Hypothesis: this is a PLANNER effect, not a model effect. The CEM planner
is short-sighted (8-step horizon) and noisy (150 samples, 3 iterations).
A model that over-estimates speed might push the planner toward more
aggressive actions, which happens to help.

Test: run physics_only (no learning, so it is fast and almost noise-free)
at several radius errors with
   (a) the Study 1 planner        horizon 8,  150 samples, 3 iterations
   (b) a longer horizon           horizon 20, 150 samples, 3 iterations
   (c) a stronger optimizer       horizon 8,  500 samples, 6 iterations
If the "wrong model beats correct model" effect disappears in (b) or (c),
it is a planner limitation, and the paper says so in one sentence.

Also prints the mean commanded wheel speed, to see whether the
over-estimating model is simply driving faster.

Usage:
    python run_horizon_check.py
"""

import numpy as np

from diffdrive_sim import DiffDriveSimulator
from models import PhysicsOnlyModel
from model_based_rl import run_episode

TRUE_R, WHEELBASE, TRUE_SLIP, DT = 0.03, 0.15, 0.85, 0.1
ERRORS = [-50.0, -25.0, 0.0, 25.0]
SEEDS = [0, 1, 2, 3, 4]
EPISODES = 15
PLANNERS = {
    'study1   (h=8,  150 samples, 3 iters)': dict(horizon=8, n_samples=150, n_elite=15, n_iters=3),
    'longer   (h=20, 150 samples, 3 iters)': dict(horizon=20, n_samples=150, n_elite=15, n_iters=3),
    'stronger (h=8,  500 samples, 6 iters)': dict(horizon=8, n_samples=500, n_elite=50, n_iters=6),
}


def mean_ci(v):
    v = np.asarray(v, float)
    t = {5: 2.776, 10: 2.262}.get(len(v), 2.0)
    return v.mean(), t * v.std(ddof=1) / np.sqrt(len(v))


def main():
    for pname, pk in PLANNERS.items():
        print(f"\nPlanner: {pname}")
        print(f"  {'error':>6s}  {'mean return':>16s}  {'steps/episode':>14s}  {'mean |wheel speed|':>19s}")
        for err in ERRORS:
            model = PhysicsOnlyModel(TRUE_R * (1 - err / 100), WHEELBASE, DT, TRUE_SLIP)
            rets, steps, speeds = [], [], []
            for seed in SEEDS:
                np.random.seed(seed)
                sim = DiffDriveSimulator(TRUE_R, WHEELBASE, TRUE_SLIP, DT)
                r_seed, s_seed, sp_seed = [], [], []
                for _ in range(EPISODES):
                    out = run_episode(sim, model, **pk)
                    r_seed.append(out['return'])
                    s_seed.append(out['steps'])
                    sp_seed += [np.abs(t[1]).mean() for t in out['transitions']]
                rets.append(np.mean(r_seed))
                steps.append(np.mean(s_seed))
                speeds.append(np.mean(sp_seed))
            r, rc = mean_ci(rets)
            s, sc = mean_ci(steps)
            print(f"  {err:+5.0f}%  {r:7.2f} +- {rc:5.2f}  {s:6.1f} +- {sc:4.1f}  {np.mean(speeds):12.2f} rad/s")


if __name__ == '__main__':
    main()