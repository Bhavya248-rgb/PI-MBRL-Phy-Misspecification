"""
run_all_models.py

Quick check: runs all three models at ONE radius error and ONE seed,
prints every episode and the frozen evaluation, and saves a plot.
For the real experiment use run_study1.py.

Usage:
    python run_all_models.py            # 0% radius error, seed 0
    python run_all_models.py 25         # 25% error, seed 0
    python run_all_models.py 25 3       # 25% error, seed 3
    python run_all_models.py -50 1      # wheel assumed 50% LARGER, seed 1
"""

import sys

import matplotlib.pyplot as plt

from model_based_rl import run_model_based_rl

TRUE_WHEEL_RADIUS = 0.03
WHEELBASE = 0.15
TRUE_SLIP = 0.85
DT = 0.1
MAX_EPISODES = 15
COLORS = {'physics_only': '#2a78d6', 'hybrid': '#eb6834', 'black_box': '#1baf7a'}


def main():
    err = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    assumed_r = TRUE_WHEEL_RADIUS * (1.0 - err / 100.0)
    print(f"radius error {err:+.0f}%  (assumed_r={assumed_r:.4f}, true 0.03)  seed={seed}\n")

    results = {}
    for m in ['physics_only', 'hybrid', 'black_box']:
        print(f"--- {m} ---")
        results[m] = run_model_based_rl(m, TRUE_WHEEL_RADIUS, WHEELBASE, TRUE_SLIP,
                                        DT, assumed_r=assumed_r,
                                        assumed_slip=TRUE_SLIP,
                                        max_episodes=MAX_EPISODES, seed=seed)
        print()

    print(f"{'model':13s} {'3-in-a-row':>11s} {'mean return':>12s} "
          f"{'eval success':>13s} {'8-step err (mm)':>16s}")
    for m, r in results.items():
        print(f"{m:13s} {str(r['steps_to_reliable_success']):>11s} "
              f"{r['mean_return']:12.2f} {r['eval_success_rate']*100:12.0f}% "
              f"{r['final_model']['kstep_pos_err']*1000:16.1f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    for m, r in results.items():
        ax.plot([e['cumulative_steps'] for e in r['episodes']],
                [e['return'] for e in r['episodes']],
                marker='o', ms=4, lw=2, color=COLORS[m], label=m)
    ax.set_xlabel('real environment steps used (incl. warmup)')
    ax.set_ylabel('episode return')
    ax.set_title(f'radius error {err:+.0f}%, seed {seed}')
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = f'single_run_{err:+.0f}pct_seed{seed}.png'
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")


if __name__ == '__main__':
    main()
