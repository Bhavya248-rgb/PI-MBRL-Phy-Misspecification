"""
analyze_extra.py

Analyses the two extra experiments (run_extra.py) against the existing
Study 2 / 2b runs. Needs results_study2/, results_study2b/ and
results_extra/ in the current folder (your Study 2 folder).

Prints
  PART 1 HISTORY: hybrid and black box with the history inputs, next to
         physics only, pose only, previous command and encoders
  PART 2 WARMUP:  hybrid (pose only / encoders) with default, dropped and
         smooth warmup
with paired comparisons (same seed and lag; significant only if the 95%
CI excludes 0), and saves fig_extra.png.

Usage:
    python analyze_extra.py
"""

import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from analyze_study2 import mean_ci, to_num

COLORS = {'physics_only': '#2a78d6', 'hybrid': '#eb6834', 'black_box': '#1baf7a'}
METRICS = [
    ('mean_return', 'Training return (mean over 15 episodes)', 1, +1),
    ('eval_mean_return', 'Frozen-eval return on new tasks', 1, +1),
    ('eval_success_rate', 'Frozen-eval success on new tasks (%)', 100, +1),
    ('eval_onpolicy_kstep_pos_err', 'On-policy 8-step position error (mm) (lower is better)', 1000, -1),
]


def read(p):
    if not os.path.isfile(p):
        print(f"WARNING: {p} not found")
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def load():
    by = defaultdict(dict)       # (model, obs_mode, variant, tau) -> {seed: row}
    for r in read(os.path.join('results_study2', 'summary.csv')):
        by[(r['model'], 'pose', 'default', float(r['motor_tau']))][r['seed']] = r
    for r in read(os.path.join('results_study2b', 'summary.csv')):
        by[(r['model'], r['obs_mode'], 'default', float(r['motor_tau']))][r['seed']] = r
    for r in read(os.path.join('results_extra', 'summary.csv')):
        by[(r['model'], r['obs_mode'], r['variant'], float(r['motor_tau']))][r['seed']] = r
    return by


def verdict(mu, ci, n, a, b):
    if n < 2 or np.isnan(ci):
        return 'n/a'
    if mu - ci > 0:
        return f'{a} better'
    if mu + ci < 0:
        return f'{b} better'
    return 'no significant difference'


def paired(by, A, B, t, key, scale, sign):
    ra, rb = by[(*A, t)], by[(*B, t)]
    d = []
    for s in ra:
        if s in rb:
            x, y = to_num(ra[s][key]) * scale, to_num(rb[s][key]) * scale
            if not (np.isnan(x) or np.isnan(y)):
                d.append(sign * (x - y))
    mu, ci, n = mean_ci(d)
    return mu, ci, n, sum(v > 0 for v in d)


def table(by, conds, names, taus):
    for key, title, scale, _ in METRICS:
        print(title)
        print(f"  {'tau':>6s} " + "".join(f"{n:>17s}" for n in names))
        for t in taus:
            row = []
            for c in conds:
                mu, ci, n = mean_ci([to_num(r[key]) * scale for r in by[(*c, t)].values()])
                row.append(f"{mu:7.1f} +-{ci:5.1f}" if n else "      --        ")
            print(f"  {t:5.2f}s " + "".join(f"{x:>17s}" for x in row))
        print()


def comparisons(by, pairs, taus, short):
    for key, title, scale, sign in METRICS:
        print(f"  {title}")
        for A, B in pairs:
            print(f"    {short[A]} vs {short[B]}")
            for t in taus:
                mu, ci, n, wins = paired(by, A, B, t, key, scale, sign)
                print(f"      {t:5.2f}s  {mu:+8.2f} +- {ci:6.2f}  (n={n:2d}, "
                      f"{short[A]} won {wins:2d}/{n:<2d})  "
                      f"{verdict(mu, ci, n, short[A], short[B])}")


def main():
    by = load()
    hist_taus = sorted({k[3] for k in by if k[1] == 'history'})
    wu_taus = sorted({k[3] for k in by if k[2] in ('drop', 'smooth')})

    # ------------------------------------------------------- PART 1
    phys = ('physics_only', 'pose', 'default')
    hp = ('hybrid', 'pose', 'default')
    ha = ('hybrid', 'prev_action', 'default')
    hh = ('hybrid', 'history', 'default')
    he = ('hybrid', 'wheel_speed', 'default')
    bp = ('black_box', 'pose', 'default')
    bh = ('black_box', 'history', 'default')
    be = ('black_box', 'wheel_speed', 'default')
    short = {phys: 'physics', hp: 'hyb-pose', ha: 'hyb-prev', hh: 'hyb-hist',
             he: 'hyb-enc', bp: 'bb-pose', bh: 'bb-hist', be: 'bb-enc'}
    print("\n" + "=" * 86)
    print("PART 1  HISTORY INPUTS (last measured motion + last two commands, no extra sensor)")
    print("=" * 86 + "\nmean +- 95% CI over seeds\n")
    conds = [phys, hp, ha, hh, he, bp, bh, be]
    table(by, conds, [short[c] for c in conds], hist_taus)
    print("Paired comparisons (positive = first better):")
    comparisons(by, [(hh, hp), (hh, phys), (hh, he), (hh, ha), (hh, bh)], hist_taus, short)

    # ------------------------------------------------------- PART 2
    names2 = {}
    conds2 = []
    for mode, tag in [('pose', 'pose'), ('wheel_speed', 'enc')]:
        for v in ['default', 'drop', 'smooth']:
            c = ('hybrid', mode, v)
            conds2.append(c)
            names2[c] = f"hyb-{tag}-{v}"
    names2[phys] = 'physics'
    print("\n" + "=" * 86)
    print("PART 2  WARMUP CHECK: default vs drop (retrain without warmup) vs smooth (held commands)")
    print("=" * 86 + "\nmean +- 95% CI over seeds\n")
    table(by, [phys] + conds2, ['physics'] + [names2[c] for c in conds2], wu_taus)
    print("Paired comparisons (positive = first better):")
    pairs = []
    for mode in ['pose', 'wheel_speed']:
        for v in ['drop', 'smooth']:
            pairs.append((('hybrid', mode, v), ('hybrid', mode, 'default')))
            pairs.append((('hybrid', mode, v), phys))
    comparisons(by, pairs, wu_taus, names2)

    # ------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    ax = axes[0]
    for c, lab, ls in [(phys, 'physics only', '-'), (hp, 'hybrid, pose only', ':'),
                       (ha, 'hybrid, + prev. command', '--'),
                       (hh, 'hybrid, + history (no sensor)', '-.'),
                       (he, 'hybrid, + wheel encoders', '-'),
                       (bh, 'black box, + history', '-.')]:
        st = [mean_ci([to_num(r['eval_mean_return']) for r in by[(*c, t)].values()]) for t in hist_taus]
        mu = np.array([s[0] for s in st]); ci = np.nan_to_num(np.array([s[1] for s in st]))
        ax.errorbar(hist_taus, mu, yerr=ci, color=COLORS[c[0]], ls=ls, lw=2,
                    marker='o', ms=4, capsize=3, label=lab)
    ax.set_title('(a) History inputs: frozen-eval return on new tasks', fontsize=10)
    ax.set_xlabel('motor time constant tau (s)')
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    styles = {'default': '-', 'drop': '--', 'smooth': ':'}
    for mode, tag in [('pose', 'pose only'), ('wheel_speed', 'encoders')]:
        for v in ['default', 'drop', 'smooth']:
            c = ('hybrid', mode, v)
            st = [mean_ci([to_num(r['eval_mean_return']) for r in by[(*c, t)].values()]) for t in wu_taus]
            mu = np.array([s[0] for s in st]); ci = np.nan_to_num(np.array([s[1] for s in st]))
            ax.errorbar(wu_taus, mu, yerr=ci, color=COLORS['hybrid'] if mode == 'pose' else '#6250d6',
                        ls=styles[v], lw=2, marker='o', ms=4, capsize=3, label=f'hybrid, {tag}, {v} warmup')
    st = [mean_ci([to_num(r['eval_mean_return']) for r in by[(*phys, t)].values()]) for t in wu_taus]
    ax.plot(wu_taus, [s[0] for s in st], color=COLORS['physics_only'], lw=2, marker='o', ms=4, label='physics only')
    ax.set_title('(b) Warmup check: frozen-eval return on new tasks', fontsize=10)
    ax.set_xlabel('motor time constant tau (s)')
    ax.legend(frameon=False, fontsize=8)
    for a in axes:
        a.grid(alpha=0.25, lw=0.5)
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig('fig_extra.png', dpi=150)
    plt.close(fig)
    print("\nSaved: fig_extra.png")


if __name__ == '__main__':
    main()
