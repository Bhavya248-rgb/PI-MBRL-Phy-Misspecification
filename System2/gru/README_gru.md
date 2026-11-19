# Last experiment: the recurrent (GRU) hybrid

## The question
The plain hybrid gets worse than physics alone when the motor lag is hidden.
Our explanation is that the problem is what the model can **observe**: from
the state alone it cannot tell what torque (or wheel speed) the motor is
producing. A sensor fixes it. The GRU hybrid tests the other way to fix it:
**give the network a memory** and still show it only the plain state. If
memory wins back most of the loss, that strongly supports the observability
explanation. It also answers the reviewer question "why not a recurrent
residual, like Golemo 2018 and Reuss 2022?"

## The model
    memory_t   = GRU( [state_t, command_t], memory_{t-1} )     (16 numbers)
    residual_t = the same 3 x 16 MLP as the hybrid, fed [memory_t, state_t, command_t]
    next state = physics(state_t, command_t) + residual_t
- The memory starts at 0 at every episode start (the robot starts at rest).
- After every real step the memory is updated with the real measured state.
- In the planner, every candidate plan starts from the current memory and the
  memory runs on with the imagined states.
- Training: same buffer, kept as whole episodes; same 10% validation split
  (by steps), same normalization, Adam, early stopping and warm start as the
  hybrid. One addition: gradient clipping at 1.0 (standard for GRUs).

## Files
| System | Files to ADD to the folder (nothing is replaced) |
| --- | --- |
| System 2 (a NEW folder `gru` inside `System2`) | `recurrent_residual.py`, `arm_gru.py`, `run_arm_gru.py`, `gru_analysis.py`, `analyze_arm_gru.py`, `check_gru_offline.py`, `arm_path.py` |
| System 1 (Study 2 folder, the one with `results_study2/`) | `recurrent_residual.py`, `gru_mbrl.py`, `run_gru.py`, `gru_analysis.py`, `analyze_gru.py` |

`recurrent_residual.py` and `gru_analysis.py` are the same file in both folders.

## Runs
| Script | Runs | Time | Where |
| --- | --- | --- | --- |
| `run_arm_gru.py` | 6 lags x 10 seeds = 60 | about 1.5 min each | Kaggle (where arm Study 2 ran) |
| `run_gru.py` | 6 lags x 10 seeds = 60 | about 1 min each | Laptop (where System 1 Study 2 ran) |
| `check_gru_offline.py` | prediction only | 15 to 25 min | Anywhere |

Both run scripts are resumable and take seed ranges (`0-4`, `5-9`).

## Analysis
- Arm: put `results_arm_gru/` inside `System2\gru\`, then `python analyze_arm_gru.py` there (reads `results_arm_study2` from `System2`, saves `fig_arm_gru.png` in `gru`).
- Diff-drive: `python analyze_gru.py` in the Study 2 folder (saves `fig_gru.png`).

Each prints, for every lag: physics | hybrid plain | hybrid + history |
hybrid + sensor | hybrid + GRU, then the paired comparisons, then
**"recovered"**: the share of the hybrid's loss that memory wins back
(100% = back to physics level).
