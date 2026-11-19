# Study 2 code: structural misspecification (unmodelled motor lag)

## Idea in one paragraph

On a real robot, a wheel cannot jump to a new speed: the motor and wheel have
inertia, so the actual speed moves gradually toward the commanded speed (time
constant `tau`). In Study 2 the REAL robot (simulator) has this lag. The physics
prior keeps the CORRECT wheel radius and slip but still assumes instant response.
So the only thing wrong with the physics is its FORM: a missing piece of dynamics,
not a wrong number. The actual wheel speeds are hidden: no model observes them,
all models still see only (x, y, theta) and the command.

| tau (s) | share of a new command reached in one 0.1 s step |
|---|---|
| 0    | 100% (no lag, identical to Study 1 at 0% error) |
| 0.05 | 86% |
| 0.1  | 63% |
| 0.2  | 39% |
| 0.3  | 28% |
| 0.4  | 22% |

## Folder setup

Make a NEW folder (e.g. `Capstone Study B`), unzip these files into it and reuse
the same virtual environment (activate `..\Capstone Study A\venv`, or create a
new one with the same three pip installs).

| File | Changed from Study 1? |
|---|---|
| `diffdrive_sim.py` | YES: motor lag added (`motor_tau`; 0 = old behaviour) |
| `model_based_rl.py` | small: passes `motor_tau` to the real robot and the test set |
| `run_study2.py` | NEW: sweeps `motor_tau` instead of radius error |
| `analyze_study2.py` | NEW: same Q1/Q2/Q3 tables, x-axis = tau |
| `task.py`, `cem_planner.py`, `models.py` | unchanged |

## Commands

```bash
python run_study2.py       # 180 runs: 3 models x 6 taus x 10 seeds (~30-45 min), resumable
python analyze_study2.py   # tables + fig2_learning_curves.png + fig2_vs_tau.png
```

## Built-in check

At tau = 0 the results must be IDENTICAL to your Study 1 results at 0% radius
error (same machine, same seeds): physics_only and hybrid mean return about
-20.2, black_box about -30.9. If they are, the lag code did not change anything else.

## Difference from Study 1

black_box is now run at EVERY tau (the real robot changes, so it is affected
too). In Study 1 it ran once per seed because only the prior changed.
