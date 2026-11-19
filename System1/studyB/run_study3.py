"""
run_study3.py

Study 3: BOTH errors at once (the realistic case).
The real robot has motor lag (motor_tau), AND the physics prior uses a
wrong wheel radius (radius_error_pct), while still assuming instant
wheel response. So the prior is wrong in two ways:
  - parametric: wrong radius (fixable from what the model sees)
  - structural: missing lag (needs the hidden wheel speeds)
Question: with the missing information (encoders), does the hybrid now
BEAT physics alone, because the prior is genuinely wrong?

Runs, for every (seed, radius error, motor_tau):
  physics_only  -- once (it ignores the extra inputs, so obs_mode does
                   not matter; checked in Study 2b)
  hybrid        -- with each obs_mode: 'pose', 'prev_action', 'wheel_speed'
black_box is NOT rerun: it does not use the physics, so the radius error
does not affect it. analyze_study3.py takes its runs (same tau, same
obs_mode, same seeds) from results_study2/ and results_study2b/.

RESUMABLE, like the other runners. Writes into results_study3/.

Usage:
    python run_study3.py
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

RADIUS_ERROR_LEVELS_PCT = [10.0, 25.0, 50.0]   # + = wheel assumed smaller
MOTOR_TAUS = [0.1, 0.2, 0.4]                   # seconds
OBS_MODES = ['pose', 'prev_action', 'wheel_speed']

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

RESULTS_DIR = 'results_study3'
# =====================================================================

SUMMARY_FIELDS = ['model', 'obs_mode', 'radius_error_pct', 'assumed_r',
                  'motor_tau', 'seed',
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
EPISODE_FIELDS = ['model', 'obs_mode', 'radius_error_pct', 'motor_tau', 'seed',
                  'episode', 'steps', 'cumulative_steps', 'success',
                  'final_dist', 'return', 'model_epochs_run',
                  'model_best_epoch', 'model_val_loss',
                  'buffer_size_trained_on', 'onestep_pos_err',
                  'onestep_head_err', 'kstep_pos_err', 'kstep_head_err']
EVAL_FIELDS = ['model', 'obs_mode', 'radius_error_pct', 'motor_tau', 'seed',
               'task_id', 'success', 'steps', 'final_dist', 'return']


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
                done.add((r['model'], r['obs_mode'], float(r['radius_error_pct']),
                          float(r['motor_tau']), int(r['seed'])))
    return done


def planned_runs():
    runs = []
    for seed in SEEDS:
        for err in RADIUS_ERROR_LEVELS_PCT:
            for tau in MOTOR_TAUS:
                runs.append(('physics_only', 'pose', err, tau, seed))
                for mode in OBS_MODES:
                    runs.append(('hybrid', mode, err, tau, seed))
    return runs


def run_one(model, mode, err, tau, seed):
    assumed_r = TRUE_WHEEL_RADIUS * (1.0 - err / 100.0)
    res = run_model_based_rl(model, TRUE_WHEEL_RADIUS, WHEELBASE, TRUE_SLIP, DT,
                             assumed_r=assumed_r, assumed_slip=TRUE_SLIP,
                             max_episodes=MAX_EPISODES, seed=seed,
                             horizon=HORIZON, motor_tau=tau, obs_mode=mode,
                             verbose=False)

    key = {'model': model, 'obs_mode': mode, 'radius_error_pct': err,
           'motor_tau': tau, 'seed': seed}
    append_rows('episodes.csv', EPISODE_FIELDS,
                [{**key, **e} for e in res['episodes']])
    append_rows('eval.csv', EVAL_FIELDS,
                [{**key, **e} for e in res['eval_tasks']])

    summary = {**key, 'assumed_r': assumed_r}
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
    todo = [r for r in runs if r not in done]
    print(f"Planned runs: {len(runs)}   already done: {len(runs) - len(todo)}   "
          f"to run now: {len(todo)}\n")

    t0 = time.time()
    for i, (m, mode, err, tau, seed) in enumerate(todo, 1):
        res = run_one(m, mode, err, tau, seed)
        elapsed = time.time() - t0
        eta = elapsed / i * (len(todo) - i)
        print(f"[{i:3d}/{len(todo)}] seed={seed} {m:12s} {mode:11s} "
              f"err={err:+3.0f}% tau={tau:.2f}s  "
              f"mean_return={res['mean_return']:7.2f}  "
              f"eval_success={res['eval_success_rate']*100:3.0f}%  "
              f"({res['runtime_s']:.0f}s, ~{eta/60:.0f} min left)")

    print(f"\nDONE. Results in {RESULTS_DIR}/  -> now run: python analyze_study3.py")


if __name__ == '__main__':
    main()
