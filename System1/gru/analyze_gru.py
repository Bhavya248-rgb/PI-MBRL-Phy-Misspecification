"""
analyze_gru.py

System 1 (diff-drive robot): the recurrent hybrid next to the Study 2 models.
Reads results_gru/ from this folder and results_study2/, results_study2b/
and results_extra/ from the studyB folder (see studyb_path.py).
    python analyze_gru.py
Prints tables, paired comparisons and the share of the loss won back by
memory; saves fig_gru.png.
"""

import csv
import glob
import os
from collections import defaultdict

from gru_analysis import report
from studyb_path import HERE, STUDY_B_DIR

METRICS = [
    ('mean_return', 'Training return (mean over 15 episodes)', 1, +1),
    ('eval_mean_return', 'Frozen-eval return on new tasks', 1, +1),
    ('eval_success_rate', 'Frozen-eval success on new tasks (%)', 100, +1),
    ('eval_onpolicy_kstep_pos_err', 'On-policy 8-step position error (mm, lower better)', 1000, -1),
]

CONDS = [  # (key, short, legend, colour, line style)
    (('physics_only', 'pose'), 'physics', 'Physics only', '#2a78d6', '-'),
    (('hybrid', 'pose'), 'hyb-pose', 'Hybrid, pose only', '#eb6834', ':'),
    (('hybrid', 'history'), 'hyb-hist', 'Hybrid + history', '#f2a541', '-.'),
    (('hybrid', 'wheel_speed'), 'hyb-enc', 'Hybrid + wheel encoders', '#b5361c', '--'),
    (('hybrid_gru', 'pose'), 'hyb-GRU', 'Hybrid + GRU memory (pose only)', '#8a3ffc', '-'),
]


def rows(pattern):
    out = []
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"WARNING: {pattern} not found")
    for p in files:
        with open(p) as f:
            out += list(csv.DictReader(f))
    return out


def load():
    by = defaultdict(dict)          # (model, obs_mode, tau) -> {seed: row}
    for r in rows(os.path.join(STUDY_B_DIR, 'results_study2', 'summary.csv')):
        by[(r['model'], 'pose', float(r['motor_tau']))][r['seed']] = r
    for r in rows(os.path.join(STUDY_B_DIR, 'results_study2b', 'summary.csv')):
        by[(r['model'], r['obs_mode'], float(r['motor_tau']))][r['seed']] = r
    for r in rows(os.path.join(STUDY_B_DIR, 'results_extra', 'summary.csv')):
        if r.get('variant', 'default') == 'default':
            by[(r['model'], r['obs_mode'], float(r['motor_tau']))][r['seed']] = r
    for r in rows(os.path.join(HERE, 'results_gru', 'summary*.csv')):
        by[(r['model'], r['obs_mode'], float(r['motor_tau']))][r['seed']] = r
    return by


if __name__ == '__main__':
    by = load()
    get = lambda c, t: by.get((c[0], c[1], t), {})
    taus = sorted({k[2] for k in by if k[0] == 'hybrid_gru'})
    report(get, CONDS, taus, METRICS,
           'System 1 (diff-drive): can memory replace the wheel encoders?', os.path.join(HERE, 'fig_gru.png'))
