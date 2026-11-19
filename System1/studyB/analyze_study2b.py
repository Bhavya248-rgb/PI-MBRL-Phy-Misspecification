"""
analyze_study2b.py

Does giving the learned models the missing information fix Study 2?

Combines
  results_study2/   (Study 2: models see the pose only  -> obs 'pose')
  results_study2b/  (Study 2b: 'prev_action' = fix (a), 'wheel_speed' = fix (b))
Both folders must be in the current folder (run this from your Study 2 folder).

Prints, for every motor_tau:
  1. a table per metric with all 7 conditions (mean +- 95% CI over seeds)
  2. paired comparisons with a verdict (significant only if the 95% CI excludes 0):
       FIX EFFECT   hybrid(fix)    vs hybrid(pose)     -- does the fix help the hybrid?
       VS PHYSICS   hybrid(fix)    vs physics_only     -- does it beat the prior alone again?
       Q1 PER MODE  hybrid(mode)   vs black_box(mode)  -- hybrid vs learning from scratch
       BLACK BOX    black_box(fix) vs black_box(pose)  -- does the fix help the black box too?
and saves fig2b_fix.png.

Usage:
    python analyze_study2b.py
"""

import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from analyze_study2 import mean_ci, to_num

DIR_POSE = 'results_study2'
DIR_FIX = 'results_study2b'
MODES = ['pose', 'prev_action', 'wheel_speed']
MODE_LABEL = {'pose': 'pose only', 'prev_action': '+ prev. command (a)',
              'wheel_speed': '+ wheel encoders (b)'}
CONDITIONS = [('physics_only', 'pose')] + \
             [('hybrid', m) for m in MODES] + [('black_box', m) for m in MODES]
SHORT = {('physics_only', 'pose'): 'physics',
         ('hybrid', 'pose'): 'hyb-pose', ('hybrid', 'prev_action'): 'hyb-prev(a)',
         ('hybrid', 'wheel_speed'): 'hyb-enc(b)',
         ('black_box', 'pose'): 'bb-pose', ('black_box', 'prev_action'): 'bb-prev(a)',
         ('black_box', 'wheel_speed'): 'bb-enc(b)'}
COLORS = {'physics_only': '#2a78d6', 'hybrid': '#eb6834', 'black_box': '#1baf7a'}
STYLES = {'pose': ':', 'prev_action': '--', 'wheel_speed': '-'}

METRICS = [
    ('mean_return', 'Training return (mean over 15 episodes)', 1, +1),
    ('eval_mean_return', 'Frozen-eval return on new tasks', 1, +1),
    ('eval_success_rate', 'Frozen-eval success on new tasks (%)', 100, +1),
    ('eval_onpolicy_kstep_pos_err', 'On-policy 8-step position error (mm) (lower is better)', 1000, -1),
]


def load():
    rows = []
    for d, default_mode in [(DIR_POSE, 'pose'), (DIR_FIX, None)]:
        p = os.path.join(d, 'summary.csv')
        if not os.path.isfile(p):
            print(f"WARNING: {p} not found")
            continue
        with open(p) as f:
            for r in csv.DictReader(f):
                r = dict(r)
                r['obs_mode'] = r.get('obs_mode') or default_mode
                rows.append(r)
    dedup = {(r['model'], r['obs_mode'], r['motor_tau'], r['seed']): r for r in rows}
    by = defaultdict(dict)        # (model, mode, tau) -> {seed: row}
    for r in dedup.values():
        by[(r['model'], r['obs_mode'], float(r['motor_tau']))][r['seed']] = r
    taus = sorted({float(r['motor_tau']) for r in dedup.values()})
    return by, taus


def verdict(mu, ci, n, a, b):
    if n < 2 or np.isnan(ci):
        return 'n/a'
    if mu - ci > 0:
        return f'{a} better'
    if mu + ci < 0:
        return f'{b} better'
    return 'no significant difference'


def paired(by, A, B, tau, key, scale, sign):
    ra, rb = by[(*A, tau)], by[(*B, tau)]
    diffs = []
    for s in ra:
        if s in rb:
            x, y = to_num(ra[s][key]) * scale, to_num(rb[s][key]) * scale
            if not (np.isnan(x) or np.isnan(y)):
                diffs.append(sign * (x - y))
    mu, ci, n = mean_ci(diffs)
    return mu, ci, n, sum(d > 0 for d in diffs)


def main():
    by, taus = load()

    # ------------------------------------------------------ 1. tables
    print("\nmean +- 95% CI over seeds.  physics = physics_only (from Study 2)\n")
    for key, title, scale, _ in METRICS:
        print(title)
        print(f"  {'tau':>6s} " + "".join(f"{SHORT[c]:>18s}" for c in CONDITIONS))
        for t in taus:
            cells = []
            for c in CONDITIONS:
                mu, ci, n = mean_ci([to_num(r[key]) * scale for r in by[(*c, t)].values()])
                cells.append(f"{mu:8.1f} +-{ci:5.1f}" if n else "      --         ")
            print(f"  {t:5.2f}s " + "".join(f"{c:>18s}" for c in cells))
        print()

    # ------------------------------------------------ 2. comparisons
    blocks = [
        ('FIX EFFECT: does the fix help the hybrid?  (fix minus pose-only, positive = fix better)',
         [(('hybrid', m), ('hybrid', 'pose')) for m in ('prev_action', 'wheel_speed')]),
        ('VS PHYSICS: hybrid vs the prior alone  (positive = hybrid better)',
         [(('hybrid', m), ('physics_only', 'pose')) for m in MODES]),
        ('Q1 PER MODE: hybrid vs black_box with the SAME inputs  (positive = hybrid better)',
         [(('hybrid', m), ('black_box', m)) for m in MODES]),
        ('BLACK BOX: does the fix help the black box too?  (positive = fix better)',
         [(('black_box', m), ('black_box', 'pose')) for m in ('prev_action', 'wheel_speed')]),
    ]
    print("=" * 86)
    print("PAIRED COMPARISONS (same seed, same tau). Significant only if the 95% CI excludes 0.")
    print("=" * 86)
    for title, pairs in blocks:
        print(f"\n{title}")
        for key, mtitle, scale, sign in METRICS:
            print(f"  {mtitle}")
            for A, B in pairs:
                print(f"    {SHORT[A]} vs {SHORT[B]}")
                for t in taus:
                    mu, ci, n, wins = paired(by, A, B, t, key, scale, sign)
                    print(f"      {t:5.2f}s  {mu:+8.2f} +- {ci:6.2f}  (n={n:2d}, "
                          f"{SHORT[A]} won {wins:2d}/{n:<2d})  "
                          f"{verdict(mu, ci, n, SHORT[A], SHORT[B])}")

    # ---------------------------------------------------- 3. figure
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, (key, title, scale, _) in zip(axes, METRICS):
        for c in CONDITIONS:
            model, mode = c
            stats = [mean_ci([to_num(r[key]) * scale for r in by[(*c, t)].values()])
                     for t in taus]
            mu = np.array([s[0] for s in stats])
            ci = np.nan_to_num(np.array([s[1] for s in stats]))
            if np.all(np.isnan(mu)):
                continue
            label = 'physics only' if model == 'physics_only' else \
                f"{'hybrid' if model == 'hybrid' else 'black box'}, {MODE_LABEL[mode]}"
            ax.errorbar(taus, mu, yerr=ci, color=COLORS[model], ls=STYLES[mode],
                        lw=2, marker='o', ms=4, capsize=3, label=label)
        ax.set_title(title.replace(' (lower is better)', '\n(lower is better)'), fontsize=9)
        ax.set_xlabel('motor time constant tau (s)')
        ax.grid(alpha=0.25, lw=0.5)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig('fig2b_fix.png', dpi=150)
    plt.close(fig)
    print("\nSaved: fig2b_fix.png")


if __name__ == '__main__':
    main()
