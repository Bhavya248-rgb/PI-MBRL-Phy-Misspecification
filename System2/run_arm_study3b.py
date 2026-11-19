"""
run_arm_study3b.py

Extension of System 2 Study 3: the two LARGE mass errors (+75%, +90%),
where Study 1 showed physics alone starts to fail on the arm. The original
grid (+10/25/50%, copied from System 1) never reached that region, so it
could not test "with the hidden torque measured, the hybrid beats physics
once the masses are clearly wrong".

Same lag levels and input types as run_arm_study3.py. Writes into the SAME
folder (results_arm_study3/), so `python analyze_arm.py 3` then shows all
five error levels together. Runs already there are skipped.

RESUMABLE, and seed ranges work as before, e.g. in two terminals:
    python run_arm_study3b.py 0-4
    python run_arm_study3b.py 5-9
300 runs, about 3 to 4 hours on two terminals.
"""

from arm_runner import run_all, SEEDS

MASS_ERRORS = [75.0, 90.0]
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
