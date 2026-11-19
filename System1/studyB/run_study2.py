"""
run_study2.py

Study 2: STRUCTURAL misspecification (unmodelled motor lag).
The real robot's wheels take time to reach the commanded speed (time
constant motor_tau). The physics prior keeps the CORRECT wheel radius and
slip but assumes instant response, so the only error is the missing lag.
Runs every (seed, model, motor_tau) combination in the CONFIG block
and writes three CSV files into RESULTS_DIR:

  summary.csv   one row per run (model x motor_tau x seed): headline numbers
  episodes.csv  one row per training episode: steps, return, success,
                model prediction error, training epochs ...
  eval.csv      one row per frozen-evaluation task per run

RESUMABLE: before each run it checks summary.csv. If that
(model, motor_tau, seed) is already there, the run is skipped. So if the
script is interrupted (VS Code closed, laptop slept), just run it again
and it continues where it stopped. To redo everything, delete the
results folder.

Usage:
    python run_study2.py
"""

import csv
import os
import time

from model_based_rl import run_model_based_rl

# =====================================================================
# CONFIG
# =====================================================================
TRUE_WHEEL_RADIUS = 0.03
WHEELBASE = 0.15
TRUE_SLIP = 0.85
DT = 0.1

MAX_EPISODES = 15
HORIZON = 8

MODEL_TYPES = ['physics_only', 'hybrid', 'black_box']

# Motor time constant in seconds (0 = no lag = Study 1 with correct physics).
# With dt = 0.1 s the wheels reach this fraction of a new command in one step:
#   0.05 s -> 86%,  0.1 s -> 63%,  0.2 s -> 39%,  0.3 s -> 28%,  0.4 s -> 22%
MOTOR_TAUS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

RESULTS_DIR = 'results_study2'
# =====================================================================

SUMMARY_FIELDS = ['model', 'motor_tau', 'seed',
                  'warmup_steps', 'total_real_steps',
                  'steps_to_reliable_success', 'success_rate_all',
                  'success_rate_last5', 'mean_return', 'mean_return_last5',
                  'mean_epochs_per_retrain',
                  'final_onestep_pos_err', 'final_onestep_head_err',
                  'final_kstep_pos_err', 'final_kstep_head_err',
                  'eval_success_rate', 'eval_mean_steps',
                  'eval_mean_final_dist', 'eval_mean_return',
                  'eval_train_task_success',
                  'eval_onpolicy_onestep_pos_err', 'eval_onpolicy_kstep_pos_err',
                  'runtime_s']
EPISODE_FIELDS = ['model', 'motor_tau', 'seed', 'episode', 'steps',
                  'cumulative_steps', 'success', 'final_dist', 'return',
                  'model_epochs_run', 'model_best_epoch', 'model_val_loss',
                  'buffer_size_trained_on', 'onestep_pos_err',
                  'onestep_head_err', 'kstep_pos_err', 'kstep_head_err']
EVAL_FIELDS = ['model', 'motor_tau', 'seed', 'task_id', 'success',
               'steps', 'final_dist', 'return']


def path(name):
    return os.path.join(RESULTS_DIR, name)


def append_rows(filename, fields, rows):
    exists = os.path.isfile(path(filename))
    with open(path(filename), 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        if not exists:
            w.writeheader()
        w.writerows(rows)


def already_done():
    done = set()
    if os.path.isfile(path('summary.csv')):
        with open(path('summary.csv')) as f:
            for r in csv.DictReader(f):
                done.add((r['model'], float(r['motor_tau']), int(r['seed'])))
    return done


def planned_runs():
    """Unlike Study 1, black_box IS affected here (the real robot changes
    with motor_tau), so every model runs at every motor_tau."""
    return [(m, tau, seed) for seed in SEEDS for m in MODEL_TYPES
            for tau in MOTOR_TAUS]


def run_one(model, tau, seed):
    res = run_model_based_rl(model, TRUE_WHEEL_RADIUS, WHEELBASE, TRUE_SLIP, DT,
                             assumed_r=TRUE_WHEEL_RADIUS, assumed_slip=TRUE_SLIP,
                             max_episodes=MAX_EPISODES, seed=seed,
                             horizon=HORIZON, motor_tau=tau, verbose=False)

    key = {'model': model, 'motor_tau': tau, 'seed': seed}
    append_rows('episodes.csv', EPISODE_FIELDS,
                [{**key, **e} for e in res['episodes']])
    append_rows('eval.csv', EVAL_FIELDS,
                [{**key, **e} for e in res['eval_tasks']])

    summary = dict(key)
    for f in SUMMARY_FIELDS:
        if f in res:
            summary[f] = res[f]
    fm = res['final_model']
    summary['final_onestep_pos_err'] = fm['onestep_pos_err']
    summary['final_onestep_head_err'] = fm['onestep_head_err']
    summary['final_kstep_pos_err'] = fm['kstep_pos_err']
    summary['final_kstep_head_err'] = fm['kstep_head_err']
    append_rows('summary.csv', SUMMARY_FIELDS, [summary])   # written LAST
    return res


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    done = already_done()
    runs = planned_runs()
    todo = [r for r in runs if (r[0], r[1], r[2]) not in done]
    print(f"Planned runs: {len(runs)}   already done: {len(runs) - len(todo)}   "
          f"to run now: {len(todo)}\n")

    t0 = time.time()
    for i, (m, e, seed) in enumerate(todo, 1):
        label = f'{e:.2f}s'
        res = run_one(m, e, seed)
        srs = res['steps_to_reliable_success']
        elapsed = time.time() - t0
        eta = elapsed / i * (len(todo) - i)
        print(f"[{i:3d}/{len(todo)}] seed={seed} {m:12s} tau={label:>5s}  "
              f"steps_to_3_in_a_row={str(srs):>5s}  "
              f"mean_return={res['mean_return']:7.2f}  "
              f"eval_success={res['eval_success_rate']*100:3.0f}%  "
              f"({res['runtime_s']:.0f}s, ~{eta/60:.0f} min left)")

    print(f"\nDONE. Results in {RESULTS_DIR}/  -> now run: python analyze_study2.py")


if __name__ == '__main__':
    main()
