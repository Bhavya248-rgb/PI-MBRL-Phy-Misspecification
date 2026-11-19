# System 2: 2-link robot arm

A second, different robot for the paper. It uses the same method as
System 1 (the diff-drive robot): the same three models, the same CEM
planner, learning loop, frozen evaluation, metrics, 10 seeds and
statistics. Only the robot, the task and the input types change.

Put this folder anywhere, for example `C:\SEM-7\Capstone Study A\arm`, and
use the same venv as System 1 (numpy, torch, matplotlib). It does not need
any System 1 file.

## Files

| File | What it is |
| --- | --- |
| `arm_sim.py` | The arm: true simulator (with optional motor lag) and the physics prior (with possibly wrong masses). `python arm_sim.py` runs 4 checks |
| `arm_task.py` | Task: move the tip to a target and stop there; training task and 6 new evaluation tasks |
| `arm_models.py` | physics_only, black_box, hybrid; four input types |
| `arm_mbrl.py` | The learning loop and frozen evaluation (same as System 1) |
| `cem_planner.py` | Same planner as System 1 (unchanged file) |
| `arm_runner.py` | Shared code for the three run scripts (CSV writing, resume) |
| `run_arm_study1.py` | Study 1: wrong masses only |
| `run_arm_study2.py` | Studies 2 and 2b: motor lag x four input types |
| `run_arm_study3.py` | Study 3: wrong masses x motor lag x four input types |
| `analyze_arm.py` | Analysis: `python analyze_arm.py 1`, `2` or `3` |

## Order to run

1. `python arm_sim.py` (a few seconds). Expected: energy drift 0.000%,
   angles moved 0.00e+00, prior vs simulator max diff 0.00e+00, and the
   lagged torque rising 3.1, 5.1, 6.2 ... towards 8.
2. `python run_arm_study1.py`, then `python analyze_arm.py 1`
   (about 1 to 1.5 hours).
3. `python run_arm_study2.py`, then `python analyze_arm.py 2`
   (about 5 to 6 hours).
4. `python run_arm_study3.py`, then `python analyze_arm.py 3`
   (about 4 to 5 hours; needs results_arm_study2/ for the black box).

Every run script is resumable: if it stops, run it again and it continues.

To finish faster, split the seeds across terminals (each uses one CPU
core). For example, with 4 cores, in three terminals at the same time:

    python run_arm_study2.py 0-3
    python run_arm_study2.py 4-6
    python run_arm_study2.py 7-9

Each part writes its own files in the same results folder;
`analyze_arm.py` reads them all together. Run the analysis only after all
parts finish. A GPU does not help: the networks are tiny and the planner
and simulator run on the CPU.

## Settings (and why)

| Setting | Value | Why |
| --- | --- | --- |
| Links | 0.40 m and 0.30 m, 1.0 kg and 0.8 kg, uniform rods | A small arm; reach 0.10 to 0.70 m |
| Joint damping | 1.0 N m s/rad | Geared joint motors have high friction; with low damping the arm swung at up to 11 rad/s and each episode gave only about 13 data points |
| Torque limits | 8 N m shoulder, 4 N m elbow | Holding the arm straight out needs 6.3 and 1.2 N m |
| Control step | 0.05 s (20 commands per second) | |
| Integration | RK4, 5 sub-steps per control step | Simulator and prior use the same integrator, so with correct masses and no lag the prior is exact |
| Horizon | 10 steps (0.5 s) | The arm needs a few steps to speed up and slow down |
| Episode | up to 100 steps (5 s) | Perfect physics reaches the targets in about 1 s |
| Success | tip within 5 cm AND joint speeds below 1 rad/s | Swinging through the target does not count |
| Physics error | prior masses = true x (1 + error), same 8 levels as System 1 | Mass changes only the dynamics, not where the tip is |
| Motor lag | first-order on the joint torques, same 6 levels as System 1 | |

## Input types (obs_mode)

| Input | Model sees | System 1 counterpart |
| --- | --- | --- |
| `state` | angles + joint speeds | pose only |
| `prev_action` | + previous command | + previous command |
| `torque` | + actual motor torques (sensor) | + wheel encoders |
| `history` | + last measured joint acceleration + last two commands | + history |

## What the pilot runs (2 seeds) suggested

These are early signs only; the full runs decide.

- Perfect physics solves all 7 tasks.
- Physics alone degrades past about +-25 to 50% mass error.
- At +50% mass error, the hybrid restored 100% success on new tasks
  (physics alone 50%, black box 50%).
- With motor lag 0.1 to 0.2 s, the state-only hybrid did worse than physics
  alone, and the torque sensor brought it back.
- History and previous command helped much less than in System 1. A likely
  reason: on the robot, the last measured motion gives the wheel speeds
  almost directly; on the arm, getting the torque from the last
  acceleration needs the inverse of the inertia matrix, which is harder to
  learn from a few hundred data points. That would itself be a finding:
  history works when the hidden quantity follows simply from what was
  measured.
