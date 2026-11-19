"""
run_arm_study3.py

System 2, Study 3: wrong masses AND motor lag at the same time, with the
four input types. Same grid as System 1 Study 3 (error 10/25/50% x lag
0.1/0.2/0.4 s).
  physics_only: 'state' only
  hybrid: all four input types
  black_box: NOT rerun -- it has no physics prior, so the mass error does
  not affect it; analyze_arm.py takes it from results_arm_study2/ (same
  lag, same seeds).

RESUMABLE (see arm_runner.py). About 4 to 5 hours.
Usage:
    python run_arm_study3.py
"""

from arm_runner import run_all, SEEDS

MASS_ERRORS = [10.0, 25.0, 50.0]
MOTOR_TAUS = [0.1, 0.2, 0.4]
MODES = ['state', 'prev_action', 'torque', 'history']
RESULTS_DIR = 'results_arm_study3'


def planned_runs():
    runs = []
    for seed in SEEDS:
        for err in MASS_ERRORS:
            for tau in MOTOR_TAUS:
                runs.append(('physics_only', 'state', err, tau, seed))
                for mode in MODES:
                    runs.append(('hybrid', mode, err, tau, seed))
    return runs


if __name__ == '__main__':
    run_all(RESULTS_DIR, planned_runs(), 'python analyze_arm.py 3')
