"""
analyze_arm_gru.py

System 2 (arm): the recurrent hybrid next to the Study 2 models.
Reads results_arm_gru/ from this gru folder and results_arm_study2/ from the
System2 folder above it (see arm_path.py).
    python analyze_arm_gru.py
Prints tables, paired comparisons and the share of the loss won back by
memory; saves fig_arm_gru.png.
"""

import os

from arm_path import HERE, ARM_DIR
from analyze_arm import load
from gru_analysis import report

METRICS = [
    ('mean_return', 'Training return (mean over 15 episodes)', 1, +1),
    ('eval_mean_return', 'Frozen-eval return on new tasks', 1, +1),
    ('eval_success_rate', 'Frozen-eval success on new tasks (%)', 100, +1),
    ('eval_onpolicy_kstep_pos_err', 'On-policy 10-step tip error (mm, lower better)', 1000, -1),
]

CONDS = [  # (key, short, legend, colour, line style)
    (('physics_only', 'state'), 'physics', 'Physics only', '#2a78d6', '-'),
    (('hybrid', 'state'), 'hyb-state', 'Hybrid, state only', '#eb6834', ':'),
    (('hybrid', 'history'), 'hyb-hist', 'Hybrid + history', '#f2a541', '-.'),
    (('hybrid', 'torque'), 'hyb-sensor', 'Hybrid + torque sensor', '#b5361c', '--'),
    (('hybrid_gru', 'state'), 'hyb-GRU', 'Hybrid + GRU memory (state only)', '#8a3ffc', '-'),
]

if __name__ == '__main__':
    s2, gru = os.path.join(ARM_DIR, 'results_arm_study2'), os.path.join(HERE, 'results_arm_gru')
    for d in (s2, gru):
        if not os.path.isdir(d):
            print(f"WARNING: {d} not found")
    by = {**load(s2), **load(gru)}
    get = lambda c, t: by.get((c[0], c[1], 0.0, t), {})
    taus = sorted({k[3] for k in by if k[0] == 'hybrid_gru'})
    report(get, CONDS, taus, METRICS,
           'System 2 (arm): can memory replace the torque sensor?', os.path.join(HERE, 'fig_arm_gru.png'))
