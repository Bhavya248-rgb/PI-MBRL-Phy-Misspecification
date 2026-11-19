"""
gru_analysis.py

Shared analysis code for the recurrent-hybrid experiment (used by
analyze_arm_gru.py for the arm and analyze_gru.py for the diff-drive robot;
the same file is in both folders).

For every motor-lag level it prints, side by side:
    physics only | hybrid, plain state | hybrid + history | hybrid + sensor | hybrid + GRU memory
then paired comparisons of the GRU hybrid against each of the others
(same seed and lag; significant only if the 95% CI excludes 0), and the
share of the hybrid's loss that the memory wins back:

    recovered = (GRU - plain hybrid) / (physics - plain hybrid)
    (100% = back to physics level, 0% = no better than the plain hybrid;
     shown only where the plain hybrid is clearly below physics)

and saves a 4-panel figure against the lag.
"""

import math

import numpy as np
import matplotlib.pyplot as plt

_T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


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


def report(get, conds, taus, metrics, title, fig_path):
    """get(cond, tau) -> {seed: row}. conds: list of (key, short name, label, color, style),
    in the order physics, plain hybrid, history, sensor, GRU."""
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)
    for key, mtitle, scale, _ in metrics:
        print(f"\n{mtitle}   (mean +- 95% CI over seeds)")
        print(f"  {'lag':>6s} " + "".join(f"{c[1]:>17s}" for c in conds))
        for t in taus:
            row = []
            for c in conds:
                mu, ci, n = mean_ci([to_num(r[key]) * scale for r in get(c[0], t).values()])
                row.append(f"{mu:7.1f} +-{ci:5.1f}" if n > 1 else "      --        ")
            print(f"  {t:5.2f}s " + "".join(f"{x:>17s}" for x in row))

    gru, others = conds[-1], conds[:-1]
    print("\nPAIRED: GRU hybrid minus each other model (positive = GRU better)")
    for key, mtitle, scale, sign in metrics:
        print(f"\n  {mtitle}")
        for o in others:
            print(f"    {gru[1]} vs {o[1]}")
            for t in taus:
                mu, ci, n, wins = paired(get(gru[0], t), get(o[0], t), key, scale, sign)
                print(f"      {t:5.2f}s  {mu:+8.2f} +- {ci:6.2f}  (n={n:2d}, GRU won {wins:2d}/{n:<2d})  "
                      f"{verdict(mu, ci, n, gru[1], o[1])}")

    phys, plain = conds[0], conds[1]
    print("\nSHARE OF THE HYBRID'S LOSS WON BACK BY MEMORY")
    print("  recovered = (GRU - plain hybrid) / (physics - plain hybrid)")
    for key, mtitle, scale, sign in metrics[:2]:
        print(f"  {mtitle}")
        for t in taus:
            mu_d, ci_d, n, _ = paired(get(phys[0], t), get(plain[0], t), key, scale, sign)
            mu_g, _, _, _ = paired(get(gru[0], t), get(plain[0], t), key, scale, sign)
            if n > 1 and mu_d - ci_d > 0:
                print(f"    {t:5.2f}s  physics - plain hybrid = {mu_d:6.2f};  GRU - plain hybrid = "
                      f"{mu_g:6.2f}  ->  recovered {100 * mu_g / mu_d:5.0f}%")
            else:
                print(f"    {t:5.2f}s  plain hybrid not clearly below physics here -> not computed")

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4.2))
    for ax, (key, mtitle, scale, _) in zip(axes, metrics):
        for c in conds:
            mus, cis, ts = [], [], []
            for t in taus:
                mu, ci, n = mean_ci([to_num(r[key]) * scale for r in get(c[0], t).values()])
                if n > 1:
                    mus.append(mu); cis.append(ci); ts.append(t)
            if not ts:
                continue
            mus, cis = np.array(mus), np.array(cis)
            ax.plot(ts, mus, c[4], color=c[3], marker='o', ms=4, label=c[2], lw=2 if c is gru else 1.5)
            ax.fill_between(ts, mus - cis, mus + cis, color=c[3], alpha=0.12)
        ax.set_xlabel('motor lag tau (s)')
        ax.set_title(mtitle, fontsize=9)
        if '(%)' in mtitle:
            ax.set_ylim(-5, 105)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    print(f"\nSaved {fig_path}")
