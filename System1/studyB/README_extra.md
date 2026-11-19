# Extra experiments: history inputs + warmup check

## Where to put the files
Copy these 4 files into your Study 2 folder (`Capstone Study A\studyB`), replacing
`models.py` and `model_based_rl.py` when asked. With default settings they
reproduce Studies 1, 2, 2b and 3 exactly (checked), so do NOT rerun those.

| File | What changed |
|---|---|
| `models.py` | new input version `history` |
| `model_based_rl.py` | builds the history inputs; warmup options `warmup_hold`, `drop_warmup` |
| `run_extra.py` | NEW: both experiments, 240 runs (~1.5 h), resumable |
| `analyze_extra.py` | NEW: compares with your Study 2 / 2b results, `fig_extra.png` |

## Experiment 1: history inputs (no extra sensor)
The hybrid (and black box) also see their **last measured motion** (forward speed
and turn rate, computed from the change in pose over the last step) and their
**last two commands**. Question: can a little memory replace wheel encoders?
Lags 0, 0.05, 0.1, 0.2, 0.3, 0.4 s, 10 seeds.

## Experiment 2: warmup check
Is part of the damage caused by the random warmup data?
- `drop`: after episode 1, retrain without the 45 warmup transitions
- `smooth`: each random warmup command held for 5 steps
Hybrid with pose only and with encoders, lags 0.1, 0.2, 0.4 s, 10 seeds.

## Commands (inside the Study 2 folder)
```bash
python run_extra.py
python analyze_extra.py   # needs results_study2/, results_study2b/, results_extra/
```
