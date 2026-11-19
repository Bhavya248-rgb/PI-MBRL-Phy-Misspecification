"""
arm_runner.py

Shared code for the three System 2 study scripts (run_arm_study1/2/3.py):
runs one (model, input type, mass error, motor lag, seed) combination and
appends the results to three CSV files in the study's results folder:

  summary.csv   one row per run: headline numbers
  episodes.csv  one row per training episode
  eval.csv      one row per frozen-evaluation task

RESUMABLE: summary.csv is written LAST for each run, and a run already in
summary.csv is skipped. If a script stops (laptop sleeps, window closed),
run it again and it continues where it stopped.

PARALLEL (optional): give a seed range after the script name to run only
those seeds, e.g. in two terminals at the same time:
    python run_arm_study2.py 0-4
    python run_arm_study2.py 5-9
Each part writes its own files (summary_seeds0-4.csv, ...) in the same
results folder, so the parts never write to the same file. analyze_arm.py
reads all parts together. Without a range, all seeds run in one terminal.
Each script uses one CPU core (torch is limited to 1 thread), so run at
most as many parts as your CPU has cores, minus one.
"""

import csv
import glob
import os
import sys
import time

import torch

from arm_mbrl import run_model_based_rl

torch.set_num_threads(1)     # tiny networks: 1 thread is as fast, and lets parts run side by side

MAX_EPISODES = 15
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

SUMMARY_FIELDS = ['model', 'obs_mode', 'mass_error_pct', 'motor_tau', 'seed',
                  'warmup_steps', 'total_real_steps', 'steps_to_reliable_success',
                  'success_rate_all', 'success_rate_last5', 'mean_return',
                  'mean_return_last5', 'mean_epochs_per_retrain',
                  'final_onestep_pos_err', 'final_kstep_pos_err',
                  'eval_success_rate', 'eval_mean_steps', 'eval_mean_final_dist',
                  'eval_mean_return', 'eval_train_task_success',
                  'eval_onpolicy_onestep_pos_err', 'eval_onpolicy_kstep_pos_err',
                  'runtime_s']
EPISODE_FIELDS = ['model', 'obs_mode', 'mass_error_pct', 'motor_tau', 'seed',
                  'episode', 'steps', 'cumulative_steps', 'success', 'final_dist',
                  'return', 'model_epochs_run', 'model_best_epoch', 'model_val_loss',
                  'onestep_pos_err', 'kstep_pos_err']
EVAL_FIELDS = ['model', 'obs_mode', 'mass_error_pct', 'motor_tau', 'seed',
               'task_id', 'success', 'steps', 'final_dist', 'return']


def _append(results_dir, filename, fields, rows, tag=''):
    stem, ext = os.path.splitext(filename)
    p = os.path.join(results_dir, f"{stem}{tag}{ext}")
    exists = os.path.isfile(p)
    with open(p, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        if not exists:
            w.writeheader()
        w.writerows(rows)


def _key(model, mode, err, tau, seed):
    return (model, mode, round(float(err), 6), round(float(tau), 6), int(seed))


def already_done(results_dir):
    done = set()
    for p in glob.glob(os.path.join(results_dir, 'summary*.csv')):
        with open(p) as f:
            for r in csv.DictReader(f):
                done.add(_key(r['model'], r['obs_mode'], r['mass_error_pct'],
                              r['motor_tau'], r['seed']))
    return done


def run_one(results_dir, model, mode, err, tau, seed, tag=''):
    res = run_model_based_rl(model, mass_error_pct=err, motor_tau=tau, obs_mode=mode,
                             seed=seed, max_episodes=MAX_EPISODES, verbose=False)
    key = {'model': model, 'obs_mode': mode, 'mass_error_pct': err,
           'motor_tau': tau, 'seed': seed}
    _append(results_dir, 'episodes.csv', EPISODE_FIELDS, [{**key, **e} for e in res['episodes']], tag)
    _append(results_dir, 'eval.csv', EVAL_FIELDS, [{**key, **e} for e in res['eval_tasks']], tag)
    summary = dict(key)
    for f in SUMMARY_FIELDS:
        if f in res:
            summary[f] = res[f]
    summary['final_onestep_pos_err'] = res['final_model']['onestep_pos_err']
    summary['final_kstep_pos_err'] = res['final_model']['kstep_pos_err']
    _append(results_dir, 'summary.csv', SUMMARY_FIELDS, [summary], tag)   # written LAST
    return res


def run_all(results_dir, runs, next_step):
    """runs: list of (model, obs_mode, mass_error_pct, motor_tau, seed).
    An optional seed range on the command line (e.g. 0-4) runs only those seeds."""
    tag = ''
    if len(sys.argv) > 1:
        lo, hi = (int(x) for x in sys.argv[1].split('-'))
        runs = [r for r in runs if lo <= r[4] <= hi]
        tag = f'_seeds{lo}-{hi}'
        print(f"Running seeds {lo} to {hi} only (files end in {tag})")
    os.makedirs(results_dir, exist_ok=True)
    done = already_done(results_dir)
    todo = [r for r in runs if _key(*r) not in done]
    print(f"Planned runs: {len(runs)}   already done: {len(runs) - len(todo)}   "
          f"to run now: {len(todo)}\n")
    t0 = time.time()
    for i, (m, mode, err, tau, seed) in enumerate(todo, 1):
        res = run_one(results_dir, m, mode, err, tau, seed, tag)
        eta = (time.time() - t0) / i * (len(todo) - i)
        print(f"[{i:3d}/{len(todo)}] seed={seed} {m:12s} {mode:11s} mass_err={err:+4.0f}% "
              f"tau={tau:.2f}s  mean_return={res['mean_return']:7.2f}  "
              f"eval_success={res['eval_success_rate']*100:3.0f}%  "
              f"({res['runtime_s']:.0f}s, ~{eta/60:.0f} min left)", flush=True)
    print(f"\nDONE. Results in {results_dir}/  -> now run: {next_step}")
