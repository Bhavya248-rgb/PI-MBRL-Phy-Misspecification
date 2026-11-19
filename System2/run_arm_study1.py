"""
run_arm_study1.py

System 2, Study 1: wrong physics only (wrong link masses), no motor lag.
Same question as System 1 Study 1: how does the hybrid's advantage change
as the physics prior gets worse?

Mass error levels: the prior believes masses = true x (1 + error/100).
Same 8 levels as the radius error in System 1, so the two can be plotted
side by side.
  physics_only and hybrid: every level x 10 seeds
  black_box: no physics prior, so once per seed (stored with error 0)

RESUMABLE (see arm_runner.py). About 1 to 1.5 hours.
Usage:
    python run_arm_study1.py
"""

from arm_runner import run_all, SEEDS

MASS_ERROR_LEVELS_PCT = [-50.0, -25.0, 0.0, 10.0, 25.0, 50.0, 75.0, 90.0]
RESULTS_DIR = 'results_arm_study1'


def planned_runs():
    runs = []
    for seed in SEEDS:
        for err in MASS_ERROR_LEVELS_PCT:
            runs.append(('physics_only', 'state', err, 0.0, seed))
            runs.append(('hybrid', 'state', err, 0.0, seed))
        runs.append(('black_box', 'state', 0.0, 0.0, seed))
    return runs


if __name__ == '__main__':
    run_all(RESULTS_DIR, planned_runs(), 'python analyze_arm.py 1')
