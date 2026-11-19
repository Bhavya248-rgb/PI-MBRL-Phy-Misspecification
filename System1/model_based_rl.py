"""
model_based_rl.py

The model-based RL loop (Dyna-style, as in Nagabandi et al. 2018):
  warmup (random actions) -> train model -> repeat for max_episodes:
      run one episode with CEM-MPC on the current model, add its data
      to the buffer, retrain on the whole buffer.
Then a FROZEN EVALUATION: learning switched off, the final model is used
on new start/goal tasks it never trained on.

Kept from earlier versions:
  - Every model always runs the full max_episodes (no early exit).
  - Scattered warmup (3 random starts x 15 random-action steps).

CHANGED / ADDED IN THIS VERSION
  1. Warmup steps are COUNTED as real environment steps. Learned models
     start their step counter at 45; physics_only (no warmup) at 0.
  2. Per-episode RETURN (sum of rewards) is recorded.
  3. Model quality is measured before every episode (i.e. for the exact
     model the planner is about to use) on a FIXED held-out test set of
     random-action trajectories, identical for every model and seed:
       - one-step error: position (m) and heading (rad)
       - k-step open-loop error (k = planning horizon = 8): the model
         predicts 8 steps ahead feeding back its own predictions, as the
         planner does.
  4. Training diagnostics per retrain: epochs run, epoch kept, val loss.
  5. Frozen evaluation on task.EVAL_TASKS (+ the training task as a
     reference), learning off. On-policy prediction errors are also
     measured on the trajectories the robot actually drove there.
  6. Planning uses batched model prediction (same algorithm, faster).
"""

import time

import numpy as np

from diffdrive_sim import DiffDriveSimulator
from cem_planner import cem_plan
from task import (reward_fn, is_success, distance_to_goal, MAX_STEPS,
                  TRAIN_START, TRAIN_GOAL, EVAL_TASKS)
from models import PhysicsOnlyModel, BlackBoxModel, HybridModel

WARMUP_EPISODES = 3
WARMUP_STEPS_PER_EPISODE = 15
TEST_SET_SEED = 12345          # fixed: same prediction test set everywhere
TEST_SET_TRAJECTORIES = 40


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------
def run_episode(true_sim, model, start_pose=TRAIN_START, goal=TRAIN_GOAL,
                max_steps=MAX_STEPS, horizon=8, n_samples=150, n_elite=15,
                n_iters=3):
    """
    Runs one episode with CEM-MPC. Returns a dict with transitions,
    success, final distance, return (sum of rewards) and step count.
    """
    true_sim.reset(x=start_pose[0], y=start_pose[1], theta=start_pose[2])
    goal = np.asarray(goal, dtype=float)
    rfn = lambda s: reward_fn(s, goal)

    transitions = []
    ep_return = 0.0
    succeeded = False

    for _ in range(max_steps):
        state = true_sim.get_state()
        if is_success(state, goal):
            succeeded = True
            break
        action = cem_plan(model, state, rfn, horizon=horizon,
                          n_samples=n_samples, n_elite=n_elite, n_iters=n_iters)
        next_state = true_sim.step(action[0], action[1])
        ep_return += float(reward_fn(next_state, goal))
        transitions.append((state.copy(), action.copy(), next_state.copy()))

    # the state after the final action must also be checked
    if not succeeded and is_success(true_sim.get_state(), goal):
        succeeded = True

    return {'transitions': transitions, 'success': succeeded,
            'final_dist': distance_to_goal(true_sim.get_state(), goal),
            'return': ep_return, 'steps': len(transitions)}


def collect_scattered_warmup(true_sim, rng, n_episodes=WARMUP_EPISODES,
                             steps_per_episode=WARMUP_STEPS_PER_EPISODE,
                             arena_half_size=0.7, action_low=-8.0,
                             action_high=8.0):
    """Random-action data from several random starting poses."""
    buffer = []
    for _ in range(n_episodes):
        true_sim.reset(x=rng.uniform(-0.1, arena_half_size),
                       y=rng.uniform(-0.1, arena_half_size),
                       theta=rng.uniform(-np.pi, np.pi))
        for _ in range(steps_per_episode):
            s = true_sim.get_state().copy()
            a = rng.uniform(action_low, action_high, size=2)
            buffer.append((s, a, true_sim.step(a[0], a[1]).copy()))
    return buffer


def train_model_from_buffer(model, buffer, lr=0.01, max_epochs=1000,
                            patience=100, min_delta=1e-4, verbose=False):
    S = np.array([t[0] for t in buffer])
    A = np.array([t[1] for t in buffer])
    S2 = np.array([t[2] for t in buffer])
    return model.train_model(S, A, S2, lr=lr, max_epochs=max_epochs,
                             patience=patience, min_delta=min_delta,
                             verbose=verbose)


# ---------------------------------------------------------------------
# Prediction-error measurement
# ---------------------------------------------------------------------
def make_prediction_test_set(true_wheel_radius, wheelbase, true_slip, dt,
                             horizon=8, n_traj=TEST_SET_TRAJECTORIES,
                             seed=TEST_SET_SEED):
    """
    Fixed held-out test set: n_traj random-action trajectories of length
    `horizon` from random poses in the arena. Uses its own fixed seed, so
    it is identical for every model, seed and error level, and never
    enters any training buffer.
    Returns states (n_traj, horizon+1, 3) and actions (n_traj, horizon, 2).
    """
    rng = np.random.default_rng(seed)
    sim = DiffDriveSimulator(true_wheel_radius, wheelbase, true_slip, dt)
    S = np.zeros((n_traj, horizon + 1, 3))
    A = rng.uniform(-8, 8, size=(n_traj, horizon, 2))
    for i in range(n_traj):
        S[i, 0] = sim.reset(x=rng.uniform(-0.1, 0.7), y=rng.uniform(-0.1, 0.7),
                            theta=rng.uniform(-np.pi, np.pi))
        for t in range(horizon):
            S[i, t + 1] = sim.step(A[i, t, 0], A[i, t, 1])
    return S, A


def prediction_errors(model, S, A):
    """
    S: (M, k+1, 3) true states, A: (M, k, 2) actions.
    one-step: predict each S[:, t+1] from the TRUE S[:, t] (all t).
    k-step:   start from S[:, 0] and feed back the model's own
              predictions for k steps; error at the final step.
    Returns mean position errors (m) and heading errors (rad).
    """
    M, k = A.shape[0], A.shape[1]
    pred1 = model.predict_batch(S[:, :-1].reshape(-1, 3), A.reshape(-1, 2))
    true1 = S[:, 1:].reshape(-1, 3)
    one_pos = np.linalg.norm(pred1[:, :2] - true1[:, :2], axis=1).mean()
    one_head = np.abs(wrap_angle(pred1[:, 2] - true1[:, 2])).mean()

    s = S[:, 0].copy()
    for t in range(k):
        s = model.predict_batch(s, A[:, t])
    k_pos = np.linalg.norm(s[:, :2] - S[:, -1, :2], axis=1).mean()
    k_head = np.abs(wrap_angle(s[:, 2] - S[:, -1, 2])).mean()
    return {'onestep_pos_err': float(one_pos), 'onestep_head_err': float(one_head),
            'kstep_pos_err': float(k_pos), 'kstep_head_err': float(k_head)}


def windows_from_transitions(transitions_list, k):
    """Cuts driven trajectories into all length-k windows (for on-policy error)."""
    Ss, As = [], []
    for trans in transitions_list:
        if len(trans) < k:
            continue
        states = np.array([t[0] for t in trans] + [trans[-1][2]])
        actions = np.array([t[1] for t in trans])
        for i in range(len(trans) - k + 1):
            Ss.append(states[i:i + k + 1])
            As.append(actions[i:i + k])
    if not Ss:
        return None, None
    return np.array(Ss), np.array(As)


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
def build_model(model_type, wheelbase, dt, assumed_r, assumed_slip, seed,
                val_fraction):
    if model_type == 'physics_only':
        return PhysicsOnlyModel(assumed_r, wheelbase, dt, assumed_slip)
    if model_type == 'black_box':
        return BlackBoxModel(seed=seed, val_fraction=val_fraction)
    if model_type == 'hybrid':
        return HybridModel(assumed_r, wheelbase, dt, assumed_slip, seed=seed,
                           val_fraction=val_fraction)
    raise ValueError(model_type)


def run_model_based_rl(model_type, true_wheel_radius, wheelbase, true_slip, dt,
                       assumed_r=None, assumed_slip=None,
                       max_episodes=15, success_streak_needed=3, seed=0,
                       train_lr=0.01, train_max_epochs=1000,
                       train_patience=100, train_min_delta=1e-4,
                       val_fraction=0.1, horizon=8, do_eval=True,
                       train_epochs=None, verbose=True, verbose_training=False):
    """
    Runs the full MBRL loop for one model type, then the frozen evaluation.

    Returns a dict:
      warmup_steps, total_real_steps, steps_to_reliable_success (incl.
      warmup; None if never 3-in-a-row), success_rate_last5,
      mean_return (mean over all episodes = normalized area under the
      return curve), episodes (list of per-episode dicts), final_model
      (training info + prediction errors of the model used for eval),
      eval_tasks (list of per-task dicts), eval summary fields,
      runtime_s, plus the old list-style keys for older scripts.
    """
    if train_epochs is not None:          # compatibility with old runners
        train_max_epochs = train_epochs
    t_start = time.time()

    np.random.seed(seed)                  # CEM sampling noise
    rng = np.random.default_rng(seed)     # warmup data
    true_sim = DiffDriveSimulator(true_wheel_radius, wheelbase, true_slip, dt)
    model = build_model(model_type, wheelbase, dt, assumed_r, assumed_slip,
                        seed, val_fraction)
    learned = model_type in ('black_box', 'hybrid')
    S_test, A_test = make_prediction_test_set(true_wheel_radius, wheelbase,
                                              true_slip, dt, horizon=horizon)
    train_kwargs = dict(lr=train_lr, max_epochs=train_max_epochs,
                        patience=train_patience, min_delta=train_min_delta,
                        verbose=verbose_training)

    no_train = {'epochs_run': 0, 'best_epoch': 0, 'train_loss': np.nan,
                'val_loss': np.nan, 'n_train': 0, 'n_val': 0}
    buffer, train_info = [], dict(no_train)
    if learned:
        buffer = collect_scattered_warmup(true_sim, rng)
        train_info = train_model_from_buffer(model, buffer, **train_kwargs)
    warmup_steps = len(buffer)

    episodes = []
    total_steps = warmup_steps
    streak = 0
    steps_to_reliable_success = None

    for ep in range(1, max_episodes + 1):
        err = prediction_errors(model, S_test, A_test)   # model about to be used
        out = run_episode(true_sim, model, horizon=horizon)
        buffer.extend(out['transitions'])
        total_steps += out['steps']

        streak = streak + 1 if out['success'] else 0
        if streak >= success_streak_needed and steps_to_reliable_success is None:
            steps_to_reliable_success = total_steps

        row = {'episode': ep, 'steps': out['steps'],
               'cumulative_steps': total_steps, 'success': out['success'],
               'final_dist': out['final_dist'], 'return': out['return'],
               'model_epochs_run': train_info['epochs_run'],
               'model_best_epoch': train_info['best_epoch'],
               'model_val_loss': train_info['val_loss'],
               'buffer_size_trained_on': train_info['n_train'] + train_info['n_val']}
        row.update(err)
        episodes.append(row)

        if verbose:
            print(f"    [{model_type}] ep {ep:2d}  steps={out['steps']:3d}  "
                  f"success={str(out['success']):5s}  return={out['return']:8.2f}  "
                  f"cum_steps={total_steps:4d}  1-step pos err={err['onestep_pos_err']*1000:.2f} mm  "
                  f"8-step={err['kstep_pos_err']*1000:.1f} mm"
                  + (f"  epochs={train_info['epochs_run']}" if learned else ""))

        if learned:
            train_info = train_model_from_buffer(model, buffer, **train_kwargs)

    final_err = prediction_errors(model, S_test, A_test)
    final_model = dict(train_info)
    final_model.update(final_err)

    # ---------------- frozen evaluation (learning OFF) ----------------
    eval_rows, eval_trans = [], []
    if do_eval:
        np.random.seed(seed + 10_000)     # reproducible CEM noise in eval
        tasks = [('train_task', TRAIN_START, TRAIN_GOAL)] + \
                [(f'eval_{i+1}', s, g) for i, (s, g) in enumerate(EVAL_TASKS)]
        for task_id, start, goal in tasks:
            out = run_episode(true_sim, model, start_pose=start, goal=goal,
                              horizon=horizon)
            eval_rows.append({'task_id': task_id, 'success': out['success'],
                              'steps': out['steps'], 'final_dist': out['final_dist'],
                              'return': out['return']})
            if task_id != 'train_task':
                eval_trans.append(out['transitions'])
            if verbose:
                print(f"    [{model_type}] FROZEN {task_id:10s} success={str(out['success']):5s}  "
                      f"steps={out['steps']:3d}  final_dist={out['final_dist']:.3f}")

    new_rows = [r for r in eval_rows if r['task_id'] != 'train_task']
    Sw, Aw = windows_from_transitions(eval_trans, horizon)
    onpol = prediction_errors(model, Sw, Aw) if Sw is not None else {}

    last5 = [e['success'] for e in episodes[-5:]]
    result = {
        'model': model_type,
        'warmup_steps': warmup_steps,
        'total_real_steps': total_steps,
        'steps_to_reliable_success': steps_to_reliable_success,
        'success_rate_all': float(np.mean([e['success'] for e in episodes])),
        'success_rate_last5': float(np.mean(last5)),
        'mean_return': float(np.mean([e['return'] for e in episodes])),
        'mean_return_last5': float(np.mean([e['return'] for e in episodes[-5:]])),
        'mean_epochs_per_retrain': (float(np.mean([e['model_epochs_run'] for e in episodes]))
                                    if learned else 0.0),
        'episodes': episodes,
        'final_model': final_model,
        'eval_tasks': eval_rows,
        'eval_success_rate': float(np.mean([r['success'] for r in new_rows])) if new_rows else np.nan,
        'eval_mean_steps': float(np.mean([r['steps'] for r in new_rows])) if new_rows else np.nan,
        'eval_mean_final_dist': float(np.mean([r['final_dist'] for r in new_rows])) if new_rows else np.nan,
        'eval_mean_return': float(np.mean([r['return'] for r in new_rows])) if new_rows else np.nan,
        'eval_train_task_success': eval_rows[0]['success'] if eval_rows else None,
        'eval_onpolicy_onestep_pos_err': onpol.get('onestep_pos_err', np.nan),
        'eval_onpolicy_kstep_pos_err': onpol.get('kstep_pos_err', np.nan),
        'runtime_s': time.time() - t_start,
    }
    # old list-style keys, so older scripts keep working
    result['real_steps_per_episode'] = [e['steps'] for e in episodes]
    result['cumulative_real_steps'] = [e['cumulative_steps'] for e in episodes]
    result['success_per_episode'] = [e['success'] for e in episodes]
    result['final_dist_per_episode'] = [e['final_dist'] for e in episodes]
    result['training_epochs_per_retrain'] = [e['model_epochs_run'] for e in episodes]
    return result
