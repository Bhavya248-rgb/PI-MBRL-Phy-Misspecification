"""
cem_planner.py

Cross-Entropy Method (CEM) planner. Samples candidate action sequences,
scores them with the model's imagined rollout, keeps the best ("elite")
ones, refits the sampling distribution, repeats, and returns the first
action of the refined plan (MPC: execute one action, observe the REAL
result, replan).

CHANGE IN THIS VERSION: all n_samples candidate sequences are rolled out
together with model.predict_batch(), instead of one model.predict() call
per sample per step. Same algorithm, same random samples, same scores
(up to float rounding) -- it is just much faster.

reward_fn must accept a batch of states and return (N,).

SAFEGUARD (System 2): if a learned model's imagined rollout blows up
(non-finite score), that plan is scored as the WORST, never the best.
(np.argsort puts NaN last, i.e. would otherwise pick it as an elite.)
"""

import numpy as np


def cem_plan(model, state, reward_fn, horizon=8, n_samples=150,
             n_elite=15, n_iters=3, action_low=-8.0, action_high=8.0,
             action_dim=2):
    mean = np.zeros((horizon, action_dim))
    std = np.ones((horizon, action_dim)) * (action_high - action_low) / 4.0

    for _ in range(n_iters):
        samples = np.random.normal(loc=mean, scale=std,
                                    size=(n_samples, horizon, action_dim))
        samples = np.clip(samples, action_low, action_high)

        s = np.repeat(np.asarray(state, dtype=float)[None, :], n_samples, axis=0)
        scores = np.zeros(n_samples)
        for t in range(horizon):
            s = model.predict_batch(s, samples[:, t])
            scores += reward_fn(s)

        scores = np.where(np.isfinite(scores), scores, -np.inf)
        elite_idx = np.argsort(scores)[-n_elite:]
        elite_samples = samples[elite_idx]
        mean = elite_samples.mean(axis=0)
        std = elite_samples.std(axis=0) + 1e-6

    return mean[0]
