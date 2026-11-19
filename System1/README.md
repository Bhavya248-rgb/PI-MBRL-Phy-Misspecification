# Study 1 code (parametric misspecification: wrong wheel radius)

## Files

Replace your old files with these 8:

| File | What it is |
|---|---|
| `diffdrive_sim.py` | True robot + physics equations (now also works on batches) |
| `task.py` | Reward, success check, training task, **6 frozen-evaluation tasks** |
| `cem_planner.py` | CEM planner (batched, much faster, same algorithm) |
| `models.py` | physics_only, black_box, hybrid; **validation-based early stopping** |
| `model_based_rl.py` | The MBRL loop + frozen evaluation + prediction-error measurement |
| `run_study1.py` | The full sweep (resumable) |
| `analyze_study1.py` | Tables with 95% CIs + figures |
| `run_all_models.py` | Quick check of one error level and one seed |

Delete `run_custom_sweep.py` and `run_horizon_test.py` (replaced). Keep your old
CSVs in a folder called `old_results_before_fixes/`. Don't mix them with the new ones.

## Commands

```bash
python models.py               # 10-second check that both learned models train
python run_all_models.py 50 0  # one condition, all 3 models, prints every episode
python run_study1.py           # full sweep: 10 seeds x 8 error levels (~25-40 min)
python analyze_study1.py       # tables + fig_learning_curves.png + fig_vs_error.png
```

If `run_study1.py` gets interrupted, just run it again. It skips everything
already in `results_study1/summary.csv`.

## What changed, and why

1. **Validation-based early stopping** (`models.py`). At every retrain the
   buffer is randomly split 90% train / 10% validation (at least 5 transitions).
   The network trains on the 90%. The weights with the lowest validation
   loss are kept. Training stops after 100 epochs without the validation loss
   improving by more than 1e-4. The normalization statistics come from the
   training part only.
   Patience is 100 (was 20): with a fresh Adam optimizer at lr 0.01, the first
   few epochs make the loss jump up. With patience 20, about 40% of retrains
   stopped before recovering and kept the old weights unchanged. Because the
   best-validation weights are always restored, a longer patience costs only
   time, not overfitting.
2. **Warmup counted.** The 45 random warmup steps are now included in the
   real-step totals of black_box and hybrid. physics_only uses none.
3. **Return per episode** recorded. This is the main learning-curve metric.
   `steps_to_reliable_success` (3 in a row) is still reported as a secondary
   metric.
4. **Frozen evaluation.** After the 15 training episodes, learning is switched
   off and the final model drives 6 new tasks (new start position, start heading
   and goal) plus the original training task as a reference.
5. **Prediction error** of the model the planner is about to use, before every
   episode, on a fixed held-out test set (40 random-action trajectories, the same
   for every model and seed):
   - one-step: predict the next state from the true current state.
   - 8-step: predict 8 steps ahead, feeding back its own predictions (what the
     planner does).
   Also measured "on-policy" on the trajectories driven in the frozen evaluation.
6. **Batched planning.** The planner rolls out all 150 candidate plans together.
   A run went from about 6 minutes to about 8 seconds, so 10 seeds are now practical.
7. **Wider radius sweep:** -50, -25, 0, 10, 25, 50, 75, 90%.
   Positive = wheel assumed smaller than it is (model under-predicts motion).
   Negative = wheel assumed larger (model over-predicts motion).

## Output files (`results_study1/`)

- `summary.csv`: one row per run (model x error x seed)
- `episodes.csv`: one row per training episode (return, steps, success, model error, epochs)
- `eval.csv`: one row per frozen-evaluation task
- `table_summary.csv`: mean and 95% CI per model and error (written by the analysis script)
