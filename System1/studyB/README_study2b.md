# Study 2b code: the fix (give the learned models the missing information)

## Where to put the files

Copy these 5 files into your EXISTING Study 2 folder (`Capstone Study A\studyB`),
replacing the old ones when asked:

| File | What changed |
|---|---|
| `models.py` | models accept 3 or 5 state numbers (`obs_mode`) |
| `model_based_rl.py` | builds what the models see from the simulator (`observe`) |
| `diffdrive_sim.py` | adds `get_wheel_speeds()` (what encoders would measure) |
| `run_study2b.py` | NEW: the fix sweep |
| `analyze_study2b.py` | NEW: compares pose-only (Study 2) with fixes (a) and (b) |

The changed files are backward compatible: with `obs_mode='pose'` (the default)
they reproduce Study 2 exactly (checked). Do NOT rerun Study 2.

## What the models see

| obs_mode | state the models see | network predicts |
|---|---|---|
| `pose` (Study 2) | x, y, theta | change in pose |
| `prev_action` (fix a) | x, y, theta, previous command L, R | change in pose (the previous command part is just copied) |
| `wheel_speed` (fix b) | x, y, theta, actual wheel speed L, R | change in pose AND in wheel speeds |

The physics prior is the same wrong "instant response" physics in every mode.
physics_only ignores the extra inputs, so it is not rerun (its Study 2 results are reused).

## Commands (run inside the Study 2 folder)

```bash
python run_study2b.py      # 240 runs: 2 fixes x 2 models x 6 taus x 10 seeds (~1-1.5 h), resumable
python analyze_study2b.py  # needs results_study2/ AND results_study2b/ in this folder
```
