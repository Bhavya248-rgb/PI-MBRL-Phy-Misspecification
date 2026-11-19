"""
analyze_arm_checks.py

Analysis of the two robustness checks, each next to the matching
Study 2 runs (needs results_arm_study2/ in the same folder):

  1. NETWORK SIZE (results_arm_check_capacity/): 64-neuron hybrids vs the
     16-neuron hybrids of Study 2, and both vs physics only.
  2. DAMPING (results_arm_check_damping/): damping 0.5 vs Study 2's 1.0 --
     is the pattern (state only < physics, torque sensor = physics) the same?

Same statistics as everywhere: mean +- 95% CI over 10 seeds, paired by
seed, significant only if the CI excludes 0.
Usage:
    python analyze_arm_checks.py
"""

import os

from analyze_arm import load, mean_ci, paired, verdict, to_num

METRICS = [('eval_mean_return', 'Frozen-eval return on new tasks', 1, +1),
           ('eval_success_rate', 'Frozen-eval success on new tasks (%)', 100, +1),
           ('mean_return', 'Training return', 1, +1)]


def cell(rows, key, scale):
    mu, ci, n = mean_ci([to_num(r[key]) * scale for r in rows.values()])
    return f"{mu:7.1f} +-{ci:5.1f}" if n > 1 else "      --      "


def compare(title, A, B, nameA, nameB):
    print(f"    {title}")
    for key, mtitle, scale, sign in METRICS[:2]:
        mu, ci, n, wins = paired(A, B, key, scale, sign)
        print(f"      {mtitle:38s} {mu:+7.2f} +- {ci:5.2f}  (n={n:2d}, {nameA} won {wins:2d}/{n:<2d})  "
              f"{verdict(mu, ci, n, nameA, nameB)}")


def capacity(s2):
    cap = load('results_arm_check_capacity')
    if not cap:
        return
    print("\n" + "=" * 88)
    print("CHECK 1  NETWORK SIZE: 64 neurons per layer vs 16 (correct masses, motor lag)")
    print("=" * 88)
    for tau in sorted({k[3] for k in cap}):
        P = s2[('physics_only', 'state', 0.0, tau)]
        print(f"\n  tau = {tau:.2f} s")
        print(f"    {'':24s}" + "".join(f"{m[1]:>40s}" for m in METRICS))
        print(f"    {'physics only':24s}" + "".join(f"{cell(P, k, s):>40s}" for k, _, s, _ in METRICS))
        for mode in ['state', 'history', 'torque']:
            h16 = s2[('hybrid', mode, 0.0, tau)]
            h64 = cap[('hybrid', mode, 0.0, tau)]
            print(f"    {'hybrid ' + mode + ', 16':24s}" + "".join(f"{cell(h16, k, s):>40s}" for k, _, s, _ in METRICS))
            print(f"    {'hybrid ' + mode + ', 64':24s}" + "".join(f"{cell(h64, k, s):>40s}" for k, _, s, _ in METRICS))
        for mode in ['state', 'history', 'torque']:
            h16 = s2[('hybrid', mode, 0.0, tau)]
            h64 = cap[('hybrid', mode, 0.0, tau)]
            compare(f"hybrid {mode}: 64 vs 16 neurons", h64, h16, '64', '16')
            compare(f"hybrid {mode}, 64 neurons vs physics", h64, P, 'hyb64', 'physics')


def damping(s2):
    dmp = load('results_arm_check_damping')
    if not dmp:
        return
    print("\n" + "=" * 88)
    print("CHECK 2  DAMPING: 0.5 vs 1.0 (correct masses)")
    print("=" * 88)
    for tau in sorted({k[3] for k in dmp}):
        print(f"\n  tau = {tau:.2f} s          frozen-eval return (success %)")
        print(f"    {'':22s}{'damping 1.0 (Study 2)':>30s}{'damping 0.5':>30s}")
        conds = [('physics_only', 'state')] + [('hybrid', m) for m in ['state', 'torque', 'history']]
        for c in conds:
            a = s2[(c[0], c[1], 0.0, tau)]
            b = dmp[(c[0], c[1], 0.0, tau)]
            fmt = lambda r: f"{cell(r, 'eval_mean_return', 1).strip()} ({cell(r, 'eval_success_rate', 100).split('+-')[0].strip()}%)"
            name = 'physics only' if c[0] == 'physics_only' else f'hybrid {c[1]}'
            print(f"    {name:22s}{fmt(a):>30s}{fmt(b):>30s}")
        P = dmp[('physics_only', 'state', 0.0, tau)]
        for m in ['state', 'torque', 'history']:
            compare(f"damping 0.5: hybrid {m} vs physics", dmp[('hybrid', m, 0.0, tau)], P,
                    f'hyb-{m}', 'physics')


if __name__ == '__main__':
    if not os.path.isdir('results_arm_study2'):
        print("WARNING: results_arm_study2/ not found -- copy it into this folder first")
    s2 = load('results_arm_study2')
    capacity(s2)
    damping(s2)
