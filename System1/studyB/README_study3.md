# Study 3 code: wrong wheel radius AND motor lag together

## Where to put the files
Copy `run_study3.py` and `analyze_study3.py` into your Study 2 folder
(`Capstone Study A\studyB`), next to the Study 2b files. Nothing else changes:
they use the same models.py / model_based_rl.py / diffdrive_sim.py you already have.

## What it runs
| Setting | Values |
|---|---|
| Radius error (physics prior) | +10%, +25%, +50% |
| Motor lag tau (real robot) | 0.1, 0.2, 0.4 s |
| Inputs the hybrid sees | pose only, + previous command (a), + wheel encoders (b) |
| Models | physics_only (once per setting), hybrid (all 3 input types) |
| Seeds | 0-9 |

360 runs (90 physics_only, fast + 270 hybrid), about 1-1.5 h, resumable.
black_box is NOT rerun: the radius error does not affect it, so the analysis
reuses its Study 2 / 2b runs at the same tau and seeds.

## Commands (inside the Study 2 folder)
```bash
python run_study3.py
python analyze_study3.py   # needs results_study2/, results_study2b/ AND results_study3/ here
```

## Output to look at first
The VERDICT TABLE at the end: for each (radius error, tau), does the hybrid with
each input type beat physics_only on new tasks? Plus fig3_combined.png.
