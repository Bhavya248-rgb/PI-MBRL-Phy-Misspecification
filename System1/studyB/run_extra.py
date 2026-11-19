"""
run_extra.py

Two extra experiments on the Study 2 robot (motor lag, correct radius):

1. HISTORY (no extra sensor): the learned models also see their last
   measured motion (forward speed and turn rate computed from the change
   in pose over the last step) and their last two commands.
   Question: can a little memory replace wheel encoders?
   -> hybrid and black_box, obs_mode='history', all 6 lags (as Study 2b)

2. WARMUP CHECK: is part of the damage caused by the random warmup data?
   'drop'   : after episode 1, retrain WITHOUT the 45 warmup transitions
   'smooth' : warmup commands held for 5 steps instead of changing each step
   -> hybrid, pose only and with encoders, lags 0.1 / 0.2 / 0.4

Baselines (physics_only, default pose / prev_action / wheel_speed runs)
are NOT rerun: analyze_extra.py takes them from results_study2/ and
results_study2b/ (same seeds, same code; defaults reproduce them exactly).

RESUMABLE. Writes into results_extra/.

Usage:
    python run_extra.py
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
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

HISTORY_TAUS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]
WARMUP_TAUS = [0.1, 0.2, 0.4]
SMOOTH_HOLD_STEPS = 5

RESULTS_DIR = 'results_extra'
# =====================================================================

# variant -> settings passed to run_model_based_rl
VARIANTS = {
    'default': dict(),
    'drop': dict(drop_warmup=True),
    'smooth': dict(warmup_hold=SMOOTH_HOLD_STEPS),
}

SUMMARY_FIELDS = ['model', 'obs_mode', 'variant', 'motor_tau', 'seed',
                  'warmup_steps', 'total_real_steps',
                  'steps_to_reliable_success', 'success_rate_all',
                  'success_rate_last5', 'mean_return', 'mean_return_last5',
                  'mean_epochs_per_retrain',
                  'final_onestep_pos_err', 'final_kstep_pos_err',
                  'eval_success_rate', 'eval_mean_steps',
                  'eval_mean_final_dist', 'eval_mean_return',
                  'eval_onpolicy_onestep_pos_err', 'eval_onpolicy_kstep_pos_err',
                  'runtime_s']
EPISODE_FIELDS = ['model', 'obs_mode', 'variant', 'motor_tau', 'seed',
                  'episode', 'steps', 'cumulative_steps', 'success',
                  'final_dist', 'return', 'model_epochs_run',
                  'model_best_epoch', 'model_val_loss', 'onestep_pos_err',
                  'kstep_pos_err']
EVAL_FIELDS = ['model', 'obs_mode', 'variant', 'motor_tau', 'seed',
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
                done.add((r['model'], r['obs_mode'], r['variant'],
                          float(r['motor_tau']), int(r['seed'])))
    return done


def planned_runs():
    runs = []
    for seed in SEEDS:
        for tau in HISTORY_TAUS:                       # experiment 1
            for m in ['hybrid', 'black_box']:
                runs.append((m, 'history', 'default', tau, seed))
        for tau in WARMUP_TAUS:                        # experiment 2
            for mode in ['pose', 'wheel_speed']:
                for v in ['drop', 'smooth']:
                    runs.append(('hybrid', mode, v, tau, seed))
    return runs


def run_one(model, mode, variant, tau, seed):
    res = run_model_based_rl(model, TRUE_WHEEL_RADIUS, WHEELBASE, TRUE_SLIP, DT,
                             assumed_r=TRUE_WHEEL_RADIUS, assumed_slip=TRUE_SLIP,
                             max_episodes=MAX_EPISODES, seed=seed,
                             horizon=HORIZON, motor_tau=tau, obs_mode=mode,
                             verbose=False, **VARIANTS[variant])
    key = {'model': model, 'obs_mode': mode, 'variant': variant,
           'motor_tau': tau, 'seed': seed}
    append_rows('episodes.csv', EPISODE_FIELDS, [{**key, **e} for e in res['episodes']])
    append_rows('eval.csv', EVAL_FIELDS, [{**key, **e} for e in res['eval_tasks']])
    summary = dict(key)
    for f in SUMMARY_FIELDS:
        if f in res:
            summary[f] = res[f]
    summary['final_onestep_pos_err'] = res['final_model']['onestep_pos_err']
    summary['final_kstep_pos_err'] = res['final_model']['kstep_pos_err']
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
    for i, (m, mode, v, tau, seed) in enumerate(todo, 1):
        res = run_one(m, mode, v, tau, seed)
        eta = (time.time() - t0) / i * (len(todo) - i)
        print(f"[{i:3d}/{len(todo)}] seed={seed} {m:9s} {mode:11s} {v:7s} "
              f"tau={tau:.2f}s  mean_return={res['mean_return']:7.2f}  "
              f"eval_success={res['eval_success_rate']*100:3.0f}%  "
              f"({res['runtime_s']:.0f}s, ~{eta/60:.0f} min left)")
    print(f"\nDONE. Results in {RESULTS_DIR}/  -> now run: python analyze_extra.py")


if __name__ == '__main__':
    main()
