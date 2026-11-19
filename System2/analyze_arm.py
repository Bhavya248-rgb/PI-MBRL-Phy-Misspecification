"""
analyze_arm.py

Analysis for System 2 (the arm), one study at a time:
    python analyze_arm.py 1     Study 1  (wrong masses)        results_arm_study1/
    python analyze_arm.py 2     Study 2+2b (lag, input types)  results_arm_study2/
    python analyze_arm.py 3     Study 3  (masses x lag)        results_arm_study3/ + results_arm_study2/

Same statistics as System 1: mean +- 95% CI (t-distribution) over seeds,
and paired comparisons by seed; a difference is called significant only
if its 95% CI excludes 0. Prints tables and comparisons, saves a figure.
"""

import csv
import glob
import math
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

_T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]

MODES = ['state', 'prev_action', 'torque', 'history']
MODE_LABEL = {'state': 'state only', 'prev_action': '+ prev. command',
              'torque': '+ torque sensor', 'history': '+ history'}
SHORT = {'physics_only': 'physics', 'hybrid': 'hyb', 'black_box': 'bb'}
MODE_SHORT = {'state': 'state', 'prev_action': 'prev', 'torque': 'torq', 'history': 'hist'}
COLORS = {'physics_only': '#2a78d6', 'hybrid': '#eb6834', 'black_box': '#1baf7a'}
STYLES = {'state': ':', 'prev_action': '--', 'torque': '-', 'history': '-.'}

METRICS = [
    ('mean_return', 'Training return (mean over 15 episodes)', 1, +1),
    ('eval_mean_return', 'Frozen-eval return on new tasks', 1, +1),
    ('eval_success_rate', 'Frozen-eval success on new tasks (%)', 100, +1),
    ('eval_onpolicy_kstep_pos_err', 'On-policy 10-step tip error (mm) (lower is better)', 1000, -1),
]


def mean_ci(values):
    v = np.array([x for x in values if x is not None and not np.isnan(x)], float)
    n = len(v)
    if n == 0:
        return np.nan, np.nan, 0
    if n == 1:
        return v[0], np.nan, 1
    t = _T95[n - 2] if n - 1 <= 30 else 1.96
    return v.mean(), t * v.std(ddof=1) / math.sqrt(n), n


def to_num(x):
    if x in ('', 'None', 'nan', None):
        return np.nan
    if x in ('True', 'False'):
        return 1.0 if x == 'True' else 0.0
    return float(x)


def load(folder):
    """-> {(model, mode, err, tau): {seed: row}}"""
    by = defaultdict(dict)
    files = sorted(glob.glob(os.path.join(folder, 'summary*.csv')))   # all parallel parts
    if not files:
        print(f"WARNING: no summary*.csv in {folder}")
    for p in files:
        with open(p) as f:
            for r in csv.DictReader(f):
                by[(r['model'], r['obs_mode'], float(r['mass_error_pct']),
                    float(r['motor_tau']))][r['seed']] = r
    return by


def name(c):
    return SHORT[c[0]] if c[0] == 'physics_only' else f"{SHORT[c[0]]}-{MODE_SHORT[c[1]]}"


def verdict(mu, ci, n, a, b):
    if n < 2 or np.isnan(ci):
        return 'n/a'
    if mu - ci > 0:
        return f'{a} better'
    if mu + ci < 0:
        return f'{b} better'
    return 'no significant difference'


def paired(ra, rb, key, scale, sign):
    d = []
    for s in ra:
        if s in rb:
            x, y = to_num(ra[s][key]) * scale, to_num(rb[s][key]) * scale
            if not (np.isnan(x) or np.isnan(y)):
                d.append(sign * (x - y))
    mu, ci, n = mean_ci(d)
    return mu, ci, n, sum(v > 0 for v in d)


def print_tables(get, conds, cells, cell_label):
    for key, title, scale, _ in METRICS:
        print(title)
        print(f"  {'':>12s} " + "".join(f"{name(c):>16s}" for c in conds))
        for cell in cells:
            row = []
            for c in conds:
                mu, ci, n = mean_ci([to_num(r[key]) * scale for r in get(c, cell).values()])
                row.append(f"{mu:7.1f} +-{ci:5.1f}" if n > 1 else (f"{mu:7.1f}         " if n else "     --         "))
            print(f"  {cell_label(cell):>12s} " + "".join(f"{x:>16s}" for x in row))
        print()


def print_comparisons(get, blocks, cells, cell_label):
    print("=" * 90)
    print("PAIRED COMPARISONS (same seed). Significant only if the 95% CI excludes 0.")
    print("=" * 90)
    for title, pairs in blocks:
        print(f"\n{title}")
        for key, mtitle, scale, sign in METRICS:
            print(f"  {mtitle}")
            for A, B in pairs:
                print(f"    {name(A)} vs {name(B)}")
                for cell in cells:
                    mu, ci, n, wins = paired(get(A, cell), get(B, cell), key, scale, sign)
                    print(f"      {cell_label(cell):>12s}  {mu:+8.2f} +- {ci:6.2f}  "
                          f"(n={n:2d}, {name(A)} won {wins:2d}/{n:<2d})  "
                          f"{verdict(mu, ci, n, name(A), name(B))}")


def style_axes(ax):
    ax.grid(alpha=0.25, lw=0.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def line(ax, xs, get, c, cells, key, scale, label):
    st = [mean_ci([to_num(r[key]) * scale for r in get(c, cell).values()]) for cell in cells]
    mu = np.array([s[0] for s in st])
    ci = np.nan_to_num(np.array([s[1] for s in st]))
    if np.all(np.isnan(mu)):
        return
    ax.errorbar(xs, mu, yerr=ci, color=COLORS[c[0]], ls=STYLES[c[1]] if c[0] != 'physics_only' else '-',
                lw=2, marker='o', ms=4, capsize=3, label=label)


def label_of(c):
    if c[0] == 'physics_only':
        return 'physics only'
    return f"{'hybrid' if c[0] == 'hybrid' else 'black box'}, {MODE_LABEL[c[1]]}"


# ----------------------------------------------------------------- Study 1
def study1():
    by = load('results_arm_study1')
    errs = sorted({k[2] for k in by if k[0] != 'black_box'})
    get = lambda c, e: by[(c[0], c[1], 0.0 if c[0] == 'black_box' else e, 0.0)]
    P, H, B = ('physics_only', 'state'), ('hybrid', 'state'), ('black_box', 'state')
    lab = lambda e: f"{e:+.0f}%"
    print("\nSYSTEM 2, STUDY 1: wrong link masses, no lag.  mean +- 95% CI over seeds")
    print("(black box has no physics prior: the same runs appear in every row)\n")
    print_tables(get, [P, H, B], errs, lab)
    print_comparisons(get, [
        ('Q1: hybrid vs black box  (positive = hybrid better)', [(H, B)]),
        ('Q2: hybrid vs physics only  (positive = hybrid better)', [(H, P)]),
        ('Q3: physics only vs black box  (positive = physics better)', [(P, B)]),
    ], errs, lab)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, (key, title, scale, _) in zip(axes, METRICS):
        for c in (P, H, B):
            line(ax, errs, get, c, errs, key, scale, label_of(c).replace(', state only', ''))
        ax.set_title(title.replace(' (lower is better)', '\n(lower is better)'), fontsize=9)
        ax.set_xlabel('mass error in the physics prior (%)')
        style_axes(ax)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig('fig_arm_study1.png', dpi=150)
    print("\nSaved: fig_arm_study1.png")


# ----------------------------------------------------------------- Study 2
def study2():
    by = load('results_arm_study2')
    taus = sorted({k[3] for k in by})
    get = lambda c, t: by[(c[0], c[1], 0.0, t)]
    P = ('physics_only', 'state')
    conds = [P] + [('hybrid', m) for m in MODES] + [('black_box', m) for m in MODES]
    lab = lambda t: f"tau {t:.2f}s"
    print("\nSYSTEM 2, STUDY 2 + 2b: correct masses, motor lag, four input types.  mean +- 95% CI\n")
    print_tables(get, conds, taus, lab)
    print_comparisons(get, [
        ('FIX EFFECT: hybrid with extra input vs hybrid state only  (positive = fix better)',
         [(('hybrid', m), ('hybrid', 'state')) for m in MODES[1:]]),
        ('VS PHYSICS: hybrid vs physics only  (positive = hybrid better)',
         [(('hybrid', m), P) for m in MODES]),
        ('Q1 PER INPUT: hybrid vs black box with the SAME inputs  (positive = hybrid better)',
         [(('hybrid', m), ('black_box', m)) for m in MODES]),
        ('BLACK BOX: extra input vs state only  (positive = fix better)',
         [(('black_box', m), ('black_box', 'state')) for m in MODES[1:]]),
    ], taus, lab)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, (key, title, scale, _) in zip(axes, METRICS):
        for c in conds:
            line(ax, taus, get, c, taus, key, scale, label_of(c))
        ax.set_title(title.replace(' (lower is better)', '\n(lower is better)'), fontsize=9)
        ax.set_xlabel('motor time constant tau (s)')
        style_axes(ax)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=5, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig('fig_arm_study2.png', dpi=150)
    print("\nSaved: fig_arm_study2.png")


# ----------------------------------------------------------------- Study 3
def study3():
    s3 = load('results_arm_study3')
    s2 = load('results_arm_study2')
    errs = sorted({k[2] for k in s3})
    taus = sorted({k[3] for k in s3})
    cells = [(e, t) for e in errs for t in taus]

    def get(c, cell):
        e, t = cell
        if c[0] == 'black_box':           # unaffected by the mass error
            return s2[(c[0], c[1], 0.0, t)]
        return s3[(c[0], c[1], e, t)]

    P = ('physics_only', 'state')
    conds = [P] + [('hybrid', m) for m in MODES] + [('black_box', m) for m in MODES]
    lab = lambda cell: f"{cell[0]:.0f}% {cell[1]:.2f}s"
    print("\nSYSTEM 2, STUDY 3: wrong masses AND motor lag.  rows = mass error / tau.  mean +- 95% CI")
    print("(black box taken from Study 2: same lag, same seeds; the mass error does not affect it)\n")
    print_tables(get, conds, cells, lab)
    print_comparisons(get, [
        ('VS PHYSICS (main Study 3 question): hybrid vs physics only  (positive = hybrid better)',
         [(('hybrid', m), P) for m in MODES]),
        ('Q1 PER INPUT: hybrid vs black box with the SAME inputs  (positive = hybrid better)',
         [(('hybrid', m), ('black_box', m)) for m in MODES]),
        ('FIX EFFECT: hybrid with extra input vs hybrid state only  (positive = fix better)',
         [(('hybrid', m), ('hybrid', 'state')) for m in MODES[1:]]),
    ], cells, lab)

    print("\nVERDICT TABLE: hybrid vs physics only, frozen-eval return on new tasks")
    print(f"  {'':>12s} " + "".join(f"{MODE_LABEL[m]:>22s}" for m in MODES))
    for cell in cells:
        row = []
        for m in MODES:
            mu, ci, n, _ = paired(get(('hybrid', m), cell), get(P, cell), 'eval_mean_return', 1, +1)
            v = verdict(mu, ci, n, 'HYBRID', 'PHYSICS')
            row.append(f"{mu:+6.1f} {v.replace(' better', '').replace('no significant difference', 'tie')}")
        print(f"  {lab(cell):>12s} " + "".join(f"{x:>22s}" for x in row))

    show = [('eval_mean_return', 'Frozen-eval return on new tasks', 1),
            ('eval_success_rate', 'Frozen-eval success on new tasks (%)', 100)]
    fig, axes = plt.subplots(2, len(errs), figsize=(5 * len(errs), 7.2), sharey='row', squeeze=False)
    for i, (key, title, scale) in enumerate(show):
        for j, e in enumerate(errs):
            ax = axes[i][j]
            g = lambda c, t, e=e: get(c, (e, t))
            for c in conds:
                line(ax, taus, g, c, taus, key, scale, label_of(c))
            ax.set_title(f"mass error +{e:.0f}%", fontsize=10)
            if j == 0:
                ax.set_ylabel(title, fontsize=9)
            if i == 1:
                ax.set_xlabel('motor time constant tau (s)')
            style_axes(ax)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=5, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig('fig_arm_study3.png', dpi=150)
    print("\nSaved: fig_arm_study3.png")


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else '1'
    {'1': study1, '2': study2, '3': study3}[which]()
