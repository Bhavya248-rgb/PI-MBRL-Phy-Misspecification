"""
analyze_study3.py

Study 3: wrong wheel radius AND unmodelled motor lag at the same time.

Combines
  results_study3/   physics_only and hybrid (all 3 obs_modes) per (radius error, tau)
  results_study2/   black_box, pose only        (same tau and seeds; the radius
  results_study2b/  black_box, prev_action and   error does not affect black_box,
                    wheel_speed                  so these are exactly the same runs)
All three folders must be in the current folder (your Study 2 folder).

Prints:
  1. tables per metric: rows = (radius error, tau), columns = 7 conditions
  2. paired comparisons (same seed), significant only if the 95% CI excludes 0:
       VS PHYSICS   hybrid(mode) vs physics_only    -- the main Study 3 question
       Q1 PER MODE  hybrid(mode) vs black_box(mode) -- hybrid vs learning from scratch
       FIX EFFECT   hybrid(fix)  vs hybrid(pose)
  3. a compact verdict table for "hybrid vs physics_only"
and saves fig3_combined.png.

Usage:
    python analyze_study3.py
"""

import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from analyze_study2 import mean_ci, to_num

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


def read(path):
    if not os.path.isfile(path):
        print(f"WARNING: {path} not found")
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def load():
    by = defaultdict(dict)       # (model, mode, err, tau) -> {seed: row}
    s3 = read(os.path.join('results_study3', 'summary.csv'))
    errs = sorted({float(r['radius_error_pct']) for r in s3})
    taus = sorted({float(r['motor_tau']) for r in s3})
    for r in s3:
        by[(r['model'], r['obs_mode'], float(r['radius_error_pct']),
            float(r['motor_tau']))][r['seed']] = r
    # black_box from Study 2 (pose) and 2b (prev_action, wheel_speed):
    # unaffected by the radius error, so the same run serves every error level
    bb = [dict(r, obs_mode='pose') for r in read(os.path.join('results_study2', 'summary.csv'))
          if r['model'] == 'black_box']
    bb += [r for r in read(os.path.join('results_study2b', 'summary.csv'))
           if r['model'] == 'black_box']
    for r in bb:
        t = float(r['motor_tau'])
        if t in taus:
            for e in errs:
                by[('black_box', r['obs_mode'], e, t)][r['seed']] = r
    return by, errs, taus


def verdict(mu, ci, n, a, b):
    if n < 2 or np.isnan(ci):
        return 'n/a'
    if mu - ci > 0:
        return f'{a} better'
    if mu + ci < 0:
        return f'{b} better'
    return 'no significant difference'


def paired(by, A, B, e, t, key, scale, sign):
    ra, rb = by[(*A, e, t)], by[(*B, e, t)]
    diffs = []
    for s in ra:
        if s in rb:
            x, y = to_num(ra[s][key]) * scale, to_num(rb[s][key]) * scale
            if not (np.isnan(x) or np.isnan(y)):
                diffs.append(sign * (x - y))
    mu, ci, n = mean_ci(diffs)
    return mu, ci, n, sum(d > 0 for d in diffs)


def main():
    by, errs, taus = load()
    cells = [(e, t) for e in errs for t in taus]

    # ------------------------------------------------------ 1. tables
    print("\nmean +- 95% CI over seeds.  rows = radius error / motor lag tau\n")
    for key, title, scale, _ in METRICS:
        print(title)
        print(f"  {'err':>4s} {'tau':>5s} " + "".join(f"{SHORT[c]:>17s}" for c in CONDITIONS))
        for e, t in cells:
            row = []
            for c in CONDITIONS:
                mu, ci, n = mean_ci([to_num(r[key]) * scale for r in by[(*c, e, t)].values()])
                row.append(f"{mu:7.1f} +-{ci:5.1f}" if n else "      --        ")
            print(f"  {e:3.0f}% {t:4.2f}s " + "".join(f"{x:>17s}" for x in row))
        print()

    # ------------------------------------------------ 2. comparisons
    blocks = [
        ('VS PHYSICS (main Study 3 question): hybrid vs physics_only  (positive = hybrid better)',
         [(('hybrid', m), ('physics_only', 'pose')) for m in MODES]),
        ('Q1 PER MODE: hybrid vs black_box with the SAME inputs  (positive = hybrid better)',
         [(('hybrid', m), ('black_box', m)) for m in MODES]),
        ('FIX EFFECT: hybrid with extra input vs hybrid pose-only  (positive = fix better)',
         [(('hybrid', m), ('hybrid', 'pose')) for m in ('prev_action', 'wheel_speed')]),
    ]
    print("=" * 90)
    print("PAIRED COMPARISONS (same seed). Significant only if the 95% CI excludes 0.")
    print("=" * 90)
    for title, pairs in blocks:
        print(f"\n{title}")
        for key, mtitle, scale, sign in METRICS:
            print(f"  {mtitle}")
            for A, B in pairs:
                print(f"    {SHORT[A]} vs {SHORT[B]}")
                for e, t in cells:
                    mu, ci, n, wins = paired(by, A, B, e, t, key, scale, sign)
                    print(f"      {e:3.0f}% {t:4.2f}s  {mu:+8.2f} +- {ci:6.2f}  "
                          f"(n={n:2d}, {SHORT[A]} won {wins:2d}/{n:<2d})  "
                          f"{verdict(mu, ci, n, SHORT[A], SHORT[B])}")

    # ---------------------------------- 3. compact verdict table
    print("\nVERDICT TABLE: hybrid vs physics_only, frozen-eval return on new tasks")
    print(f"  {'err':>4s} {'tau':>5s} " + "".join(f"{MODE_LABEL[m]:>30s}" for m in MODES))
    for e, t in cells:
        row = []
        for m in MODES:
            mu, ci, n, _ = paired(by, ('hybrid', m), ('physics_only', 'pose'), e, t,
                                  'eval_mean_return', 1, +1)
            v = verdict(mu, ci, n, 'HYBRID', 'PHYSICS')
            row.append(f"{mu:+6.1f} {v.replace(' better','').replace('no significant difference','tie')}")
        print(f"  {e:3.0f}% {t:4.2f}s " + "".join(f"{x:>30s}" for x in row))

    # ---------------------------------------------------- 4. figure
    show = [('eval_mean_return', 'Frozen-eval return on new tasks', 1),
            ('eval_success_rate', 'Frozen-eval success on new tasks (%)', 100)]
    fig, axes = plt.subplots(len(show), len(errs), figsize=(5 * len(errs), 3.6 * len(show)),
                             sharey='row', squeeze=False)
    for i, (key, title, scale) in enumerate(show):
        for j, e in enumerate(errs):
            ax = axes[i][j]
            for c in CONDITIONS:
                model, mode = c
                stats = [mean_ci([to_num(r[key]) * scale for r in by[(*c, e, t)].values()])
                         for t in taus]
                mu = np.array([s[0] for s in stats])
                ci = np.nan_to_num(np.array([s[1] for s in stats]))
                if np.all(np.isnan(mu)):
                    continue
                label = 'physics only' if model == 'physics_only' else \
                    f"{'hybrid' if model == 'hybrid' else 'black box'}, {MODE_LABEL[mode]}"
                ax.errorbar(taus, mu, yerr=ci, color=COLORS[model], ls=STYLES[mode],
                            lw=2, marker='o', ms=4, capsize=3, label=label)
            ax.set_title(f"radius error +{e:.0f}%", fontsize=10)
            if j == 0:
                ax.set_ylabel(title, fontsize=9)
            if i == len(show) - 1:
                ax.set_xlabel('motor time constant tau (s)')
            ax.grid(alpha=0.25, lw=0.5)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig('fig3_combined.png', dpi=150)
    plt.close(fig)
    print("\nSaved: fig3_combined.png")


if __name__ == '__main__':
    main()
