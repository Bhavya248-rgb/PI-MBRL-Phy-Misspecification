"""
run_arm_study2.py

System 2, Studies 2 and 2b together: correct masses, unmodelled MOTOR LAG
on the joint torques, and the four input types:
  'state'       angles + speeds only                  (Study 2)
  'prev_action' + previous command                    (fix a)
  'torque'      + measured actual torque (sensor)     (fix b, like encoders)
  'history'     + last measured acceleration + last two commands (no sensor)
Same 6 lag levels as System 1.
  physics_only: 'state' only (its prediction does not depend on the extras)
  hybrid and black_box: all four input types

RESUMABLE (see arm_runner.py). About 5 to 6 hours; can run at the same time
as run_arm_study1.py in a second terminal (separate results folders).
Usage:
    python run_arm_study2.py
"""

from arm_runner import run_all, SEEDS

MOTOR_TAUS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]
MODES = ['state', 'prev_action', 'torque', 'history']
RESULTS_DIR = 'results_arm_study2'


def planned_runs():
    runs = []
    for seed in SEEDS:
        for tau in MOTOR_TAUS:
            runs.append(('physics_only', 'state', 0.0, tau, seed))
            for m in ['hybrid', 'black_box']:
                for mode in MODES:
                    runs.append((m, mode, 0.0, tau, seed))
    return runs


if __name__ == '__main__':
    run_all(RESULTS_DIR, planned_runs(), 'python analyze_arm.py 2')
