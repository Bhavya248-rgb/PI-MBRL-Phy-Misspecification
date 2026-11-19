"""
run_gru.py

LAST EXPERIMENT (System 1, diff-drive robot): the recurrent hybrid
'hybrid_gru'. The residual network sees only the pose (x, y, theta) but has
a memory (GRU). Question: can memory alone recover what the hidden motor
lag takes away -- without wheel encoders and without hand-made history?

Same robot as Study 2 (correct radius and slip, motor lag), the same 6 lag
levels (0, 0.05, 0.1, 0.2, 0.3, 0.4 s), 10 seeds = 60 runs.
Physics only and the feedforward hybrids are NOT rerun: analyze_gru.py takes
them from results_study2/, results_study2b/ and results_extra/.

Keep the GRU files in their own folder NEXT TO studyB (see studyb_path.py);
the code is imported from studyB, nothing there changes.
Writes into results_gru/ inside this GRU folder. RESUMABLE; seed ranges work, e.g. two terminals:
    python run_gru.py 0-4
    python run_gru.py 5-9
About 1 minute per run. Then: python analyze_gru.py
"""

import csv
import glob
import os
import sys
import time

import torch

from gru_mbrl import run_model_based_rl_gru

torch.set_num_threads(1)

TRUE_WHEEL_RADIUS = 0.03
WHEELBASE = 0.15
TRUE_SLIP = 0.85
DT = 0.1
MAX_EPISODES = 15
HORIZON = 8
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
MOTOR_TAUS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_gru')

SUMMARY_FIELDS = ['model', 'obs_mode', 'motor_tau', 'seed',
                  'warmup_steps', 'total_real_steps', 'steps_to_reliable_success',
                  'success_rate_all', 'success_rate_last5', 'mean_return',
                  'mean_return_last5', 'mean_epochs_per_retrain',
                  'final_onestep_pos_err', 'final_kstep_pos_err',
                  'eval_success_rate', 'eval_mean_steps', 'eval_mean_final_dist',
                  'eval_mean_return', 'eval_train_task_success',
                  'eval_onpolicy_onestep_pos_err', 'eval_onpolicy_kstep_pos_err',
                  'runtime_s']
EPISODE_FIELDS = ['model', 'obs_mode', 'motor_tau', 'seed', 'episode', 'steps',
                  'cumulative_steps', 'success', 'final_dist', 'return',
                  'model_epochs_run', 'model_best_epoch', 'model_val_loss',
                  'onestep_pos_err', 'kstep_pos_err']
EVAL_FIELDS = ['model', 'obs_mode', 'motor_tau', 'seed', 'task_id', 'success',
               'steps', 'final_dist', 'return']


def append_rows(filename, fields, rows, tag):
    stem, ext = os.path.splitext(filename)
    p = os.path.join(RESULTS_DIR, f"{stem}{tag}{ext}")
    exists = os.path.isfile(p)
    with open(p, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        if not exists:
            w.writeheader()
        w.writerows(rows)


def already_done():
    done = set()
    for p in glob.glob(os.path.join(RESULTS_DIR, 'summary*.csv')):
        with open(p) as f:
            for r in csv.DictReader(f):
                done.add((round(float(r['motor_tau']), 6), int(r['seed'])))
    return done


def run_one(tau, seed, tag):
    res = run_model_based_rl_gru(TRUE_WHEEL_RADIUS, WHEELBASE, TRUE_SLIP, DT,
                                 assumed_r=TRUE_WHEEL_RADIUS, assumed_slip=TRUE_SLIP,
                                 max_episodes=MAX_EPISODES, seed=seed, horizon=HORIZON,
                                 motor_tau=tau, verbose=False)
    key = {'model': 'hybrid_gru', 'obs_mode': 'pose', 'motor_tau': tau, 'seed': seed}
    append_rows('episodes.csv', EPISODE_FIELDS, [{**key, **e} for e in res['episodes']], tag)
    append_rows('eval.csv', EVAL_FIELDS, [{**key, **e} for e in res['eval_tasks']], tag)
    summary = dict(key)
    for f in SUMMARY_FIELDS:
        if f in res:
            summary[f] = res[f]
    summary['final_onestep_pos_err'] = res['final_model']['onestep_pos_err']
    summary['final_kstep_pos_err'] = res['final_model']['kstep_pos_err']
    m = res['model_object']
    os.makedirs(os.path.join(RESULTS_DIR, 'models'), exist_ok=True)
    tt = lambda a: torch.tensor(a, dtype=torch.float64)
    torch.save({**key, 'state_dict': m.net.state_dict(), 'n_in': m.n_in,
                'x_mean': tt(m.norm.x_mean), 'x_std': tt(m.norm.x_std),
                'y_mean': tt(m.norm.y_mean), 'y_std': tt(m.norm.y_std)},
               os.path.join(RESULTS_DIR, 'models', f"hybrid_gru_pose_tau{tau:.2f}_seed{seed}.pt"))
    append_rows('summary.csv', SUMMARY_FIELDS, [summary], tag)   # written LAST
    return res


def main():
    runs = [(tau, seed) for seed in SEEDS for tau in MOTOR_TAUS]
    tag = ''
    if len(sys.argv) > 1:
        lo, hi = (int(x) for x in sys.argv[1].split('-'))
        runs = [r for r in runs if lo <= r[1] <= hi]
        tag = f'_seeds{lo}-{hi}'
        print(f"Running seeds {lo} to {hi} only (files end in {tag})")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    done = already_done()
    todo = [r for r in runs if (round(r[0], 6), r[1]) not in done]
    print(f"Planned runs: {len(runs)}   already done: {len(runs) - len(todo)}   "
          f"to run now: {len(todo)}\n")
    t0 = time.time()
    for i, (tau, seed) in enumerate(todo, 1):
        res = run_one(tau, seed, tag)
        eta = (time.time() - t0) / i * (len(todo) - i)
        print(f"[{i:3d}/{len(todo)}] seed={seed} hybrid_gru tau={tau:.2f}s  "
              f"mean_return={res['mean_return']:7.2f}  "
              f"eval_success={res['eval_success_rate']*100:3.0f}%  "
              f"({res['runtime_s']:.0f}s, ~{eta/60:.0f} min left)", flush=True)
    print(f"\nDONE. Results in {RESULTS_DIR}/  -> now run: python analyze_gru.py")


if __name__ == '__main__':
    main()
