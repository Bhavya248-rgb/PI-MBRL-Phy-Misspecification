"""
run_arm_gru.py

LAST EXPERIMENT (System 2, arm): the recurrent hybrid 'hybrid_gru'.
The residual network sees only the plain state (angles + speeds) but has a
memory (GRU). Question: can memory alone recover what the hidden motor lag
takes away -- without a torque sensor and without hand-made history?

Same robot, same correct masses and the same 6 motor-lag levels as Study 2
(0, 0.05, 0.1, 0.2, 0.3, 0.4 s), 10 seeds = 60 runs.
Physics only and the four feedforward hybrids are NOT rerun: analyze_arm_gru.py
takes them from results_arm_study2/ (same seeds, same settings).

Writes into results_arm_gru/ inside this gru folder (same CSV columns as every other study).
RESUMABLE; seed ranges work, e.g. in two terminals:
    python run_arm_gru.py 0-4
    python run_arm_gru.py 5-9
About 1.5 to 2 minutes per run (about 1 hour on two parts).
Then: python analyze_arm_gru.py
"""

import os
import sys
import time

import torch

import arm_path  # noqa: F401  (makes the System2 code importable)
from arm_runner import (SEEDS, MAX_EPISODES, SUMMARY_FIELDS, EPISODE_FIELDS, EVAL_FIELDS,
                        _append, _key, already_done)
from arm_gru import run_model_based_rl_gru

torch.set_num_threads(1)

MOTOR_TAUS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]
RESULTS_DIR = os.path.join(arm_path.HERE, 'results_arm_gru')


def planned_runs():
    return [('hybrid_gru', 'state', 0.0, tau, seed) for seed in SEEDS for tau in MOTOR_TAUS]


def save_gru(m, path, key):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tt = lambda a: torch.tensor(a, dtype=torch.float64)
    torch.save({**key, 'state_dict': m.net.state_dict(), 'n_in': m.n_in,
                'x_mean': tt(m.norm.x_mean), 'x_std': tt(m.norm.x_std),
                'y_mean': tt(m.norm.y_mean), 'y_std': tt(m.norm.y_std)}, path)


def run_one(model, mode, err, tau, seed, tag):
    res = run_model_based_rl_gru(mass_error_pct=err, motor_tau=tau, seed=seed,
                                 max_episodes=MAX_EPISODES, verbose=False)
    key = {'model': model, 'obs_mode': mode, 'mass_error_pct': err,
           'motor_tau': tau, 'seed': seed}
    _append(RESULTS_DIR, 'episodes.csv', EPISODE_FIELDS, [{**key, **e} for e in res['episodes']], tag)
    _append(RESULTS_DIR, 'eval.csv', EVAL_FIELDS, [{**key, **e} for e in res['eval_tasks']], tag)
    summary = dict(key)
    for f in SUMMARY_FIELDS:
        if f in res:
            summary[f] = res[f]
    summary['final_onestep_pos_err'] = res['final_model']['onestep_pos_err']
    summary['final_kstep_pos_err'] = res['final_model']['kstep_pos_err']
    save_gru(res['model_object'],
             os.path.join(RESULTS_DIR, 'models', f"{model}_{mode}_err{err:+.0f}_tau{tau:.2f}_seed{seed}.pt"),
             key)
    _append(RESULTS_DIR, 'summary.csv', SUMMARY_FIELDS, [summary], tag)   # written LAST
    return res


def main():
    runs, tag = planned_runs(), ''
    if len(sys.argv) > 1:
        lo, hi = (int(x) for x in sys.argv[1].split('-'))
        runs = [r for r in runs if lo <= r[4] <= hi]
        tag = f'_seeds{lo}-{hi}'
        print(f"Running seeds {lo} to {hi} only (files end in {tag})")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    done = already_done(RESULTS_DIR)
    todo = [r for r in runs if _key(*r) not in done]
    print(f"Planned runs: {len(runs)}   already done: {len(runs) - len(todo)}   "
          f"to run now: {len(todo)}\n")
    t0 = time.time()
    for i, (m, mode, err, tau, seed) in enumerate(todo, 1):
        res = run_one(m, mode, err, tau, seed, tag)
        eta = (time.time() - t0) / i * (len(todo) - i)
        print(f"[{i:3d}/{len(todo)}] seed={seed} {m} tau={tau:.2f}s  "
              f"mean_return={res['mean_return']:7.2f}  "
              f"eval_success={res['eval_success_rate']*100:3.0f}%  "
              f"({res['runtime_s']:.0f}s, ~{eta/60:.0f} min left)", flush=True)
    print(f"\nDONE. Results in {RESULTS_DIR}/  -> now run: python analyze_arm_gru.py")


if __name__ == '__main__':
    main()
