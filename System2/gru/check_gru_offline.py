"""
check_gru_offline.py

Supporting check for the GRU experiment (System 2, arm). No planning, no
learning loop: just "how well can each model predict the lagged arm, given
N random-torque transitions?"

Models (all with correct masses, motor lag tau = 0.2 s):
    hybrid, state only | hybrid + history | hybrid + torque sensor | hybrid + GRU memory
Data: random torques from random rest poses, episodes of 40 steps;
N = 720 (about what the learning loop collects), 1500, 3000, 6000 transitions;
3 data seeds each. Measured on the SAME fixed test set as the studies:
1-step and 10-step tip error (mm), mean over the 3 data seeds.

Why: if the GRU only partly recovers in the learning loop, this shows
whether memory CAN infer the hidden torque given more data -- i.e. whether
the limit is the information in the data or the amount of data.

About 15 to 25 minutes on one core.
    python check_gru_offline.py
"""

import numpy as np
import torch

import arm_path  # noqa: F401  (makes the System2 code importable)
from arm_sim import ArmSimulator, M1, M2, TORQUE_LIMIT
from arm_mbrl import Observer, make_prediction_test_set, prediction_errors
from arm_models import HybridModel
from arm_gru import build_gru_hybrid, tip
from recurrent_residual import recurrent_prediction_errors

torch.set_num_threads(1)

TAU = 0.2
EP_LEN = 40
SIZES = [720, 1500, 3000, 6000]
DATA_SEEDS = [0, 1, 2]
MODES = ['state', 'history', 'torque']
NAMES = {'state': 'hybrid, state only', 'history': 'hybrid + history',
         'torque': 'hybrid + torque sensor', 'gru': 'hybrid + GRU memory'}


def make_episodes(n_transitions, data_seed):
    rng = np.random.default_rng(1000 + data_seed)
    eps = {m: [] for m in MODES}
    for _ in range(n_transitions // EP_LEN):
        q = rng.uniform(-np.pi, np.pi, 2)
        A = rng.uniform(-1, 1, (EP_LEN, 2)) * TORQUE_LIMIT
        for m in MODES:                       # same motion, seen through each input type
            sim = ArmSimulator(M1, M2, motor_tau=TAU)
            sim.reset(q[0], q[1])
            obs, ep = Observer(sim, m), []
            for a in A:
                s = obs.get().copy()
                obs.step(a)
                ep.append((s, a.copy(), obs.get().copy()))
            eps[m].append(ep)
    return eps


def main():
    tests = {m: make_prediction_test_set(M1, M2, motor_tau=TAU, obs_mode=m) for m in MODES}
    res = {(k, n): [] for k in MODES + ['gru'] for n in SIZES}
    for n in SIZES:
        for ds in DATA_SEEDS:
            eps = make_episodes(n, ds)
            for m in MODES:
                h = HybridModel(M1, M2, seed=ds, obs_mode=m)
                flat = [t for e in eps[m] for t in e]
                h.train_model(np.array([t[0] for t in flat]), np.array([t[1] for t in flat]),
                              np.array([t[2] for t in flat]))
                res[(m, n)].append(prediction_errors(h, *tests[m]))
            g = build_gru_hybrid(0.0, ds)
            g.train_model(eps['state'])
            res[('gru', n)].append(recurrent_prediction_errors(g, *tests['state'], tip))
            print(f"  done N={n} data seed {ds}", flush=True)

    for key, label in [('onestep_pos_err', '1-step tip error (mm)'),
                       ('kstep_pos_err', '10-step tip error (mm)')]:
        print(f"\n{label}, motor lag {TAU} s, mean over {len(DATA_SEEDS)} data seeds")
        print(f"  {'N transitions':>14s}" + "".join(f"{NAMES[k]:>26s}" for k in MODES + ['gru']))
        for n in SIZES:
            print(f"  {n:>14d}" + "".join(
                f"{1000 * np.mean([r[key] for r in res[(k, n)]]):>26.1f}" for k in MODES + ['gru']))


if __name__ == '__main__':
    main()
