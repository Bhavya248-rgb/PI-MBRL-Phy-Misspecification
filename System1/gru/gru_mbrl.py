"""
gru_mbrl.py

System 1 (diff-drive robot): the learning loop for the RECURRENT hybrid
'hybrid_gru' (see recurrent_residual.py). Identical to
model_based_rl.run_model_based_rl in every setting and measurement -- same
warmup (same random draws), same 15 episodes, same CEM (same random draws),
same frozen evaluation, same fixed test set and the same result fields --
except:
  - the buffer is kept as whole episodes (the memory needs step order);
  - the planner carries the model's memory through its imagined rollouts;
  - the memory is reset to 0 at every episode start and updated with the
    real measured pose after every real step.
The model sees only the pose (x, y, theta) -> obs_mode 'pose'.
Nothing in the existing files is changed. The robot, task, models and
learning-loop code are imported from the studyB folder (see studyb_path.py).
"""

import time

import numpy as np

import studyb_path  # noqa: F401  (makes the studyB code importable)
from diffdrive_sim import DiffDriveSimulator
from task import (reward_fn, is_success, distance_to_goal, MAX_STEPS,
                  TRAIN_START, TRAIN_GOAL, EVAL_TASKS)
from models import PhysicsOnlyModel, state_to_network_input
from model_based_rl import (Observer, collect_scattered_warmup, make_prediction_test_set,
                            WARMUP_STEPS_PER_EPISODE)
from recurrent_residual import (GRUHybridModel, cem_plan_recurrent,
                                recurrent_prediction_errors, onpolicy_errors)

OBS_MODE = 'pose'
pos = lambda s: s[..., :2]


def build_gru_hybrid(assumed_r, wheelbase, dt, assumed_slip, seed, val_fraction=0.1):
    physics = PhysicsOnlyModel(assumed_r, wheelbase, dt, assumed_slip, obs_mode=OBS_MODE)
    return GRUHybridModel(physics, state_to_network_input, n_out=3, seed=seed,
                          val_fraction=val_fraction)


def run_episode_gru(true_sim, model, start_pose=TRAIN_START, goal=TRAIN_GOAL,
                    max_steps=MAX_STEPS, horizon=8, n_samples=150, n_elite=15, n_iters=3):
    true_sim.reset(x=start_pose[0], y=start_pose[1], theta=start_pose[2])
    obs = Observer(true_sim, OBS_MODE, true_sim.dt)
    memory = model.init_memory(1)                  # robot at rest -> empty memory
    goal = np.asarray(goal, dtype=float)
    rfn = lambda s: reward_fn(s, goal)
    transitions, ep_return, succeeded = [], 0.0, False
    for _ in range(max_steps):
        state = obs.get()
        if is_success(state, goal):
            succeeded = True
            break
        action = cem_plan_recurrent(model, state, memory, rfn, horizon=horizon,
                                    n_samples=n_samples, n_elite=n_elite, n_iters=n_iters)
        _, memory = model.step(state[None], action[None], memory)   # update with what really happened
        obs.step(action)
        next_state = obs.get()
        ep_return += float(reward_fn(next_state, goal))
        transitions.append((state.copy(), action.copy(), next_state.copy()))
    if not succeeded and is_success(true_sim.get_state(), goal):
        succeeded = True
    return {'transitions': transitions, 'success': succeeded,
            'final_dist': distance_to_goal(true_sim.get_state(), goal),
            'return': ep_return, 'steps': len(transitions)}


def run_model_based_rl_gru(true_wheel_radius, wheelbase, true_slip, dt,
                           assumed_r=None, assumed_slip=None, max_episodes=15,
                           success_streak_needed=3, seed=0, train_lr=0.01,
                           train_max_epochs=1000, train_patience=100, train_min_delta=1e-4,
                           val_fraction=0.1, horizon=8, do_eval=True, motor_tau=0.0,
                           verbose=True):
    t_start = time.time()
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    true_sim = DiffDriveSimulator(true_wheel_radius, wheelbase, true_slip, dt,
                                  motor_tau=motor_tau)
    model = build_gru_hybrid(assumed_r, wheelbase, dt, assumed_slip, seed, val_fraction)
    S_test, A_test = make_prediction_test_set(true_wheel_radius, wheelbase, true_slip, dt,
                                              horizon=horizon, motor_tau=motor_tau,
                                              obs_mode=OBS_MODE)
    train_kw = dict(lr=train_lr, max_epochs=train_max_epochs, patience=train_patience,
                    min_delta=train_min_delta)

    flat = collect_scattered_warmup(true_sim, rng, obs_mode=OBS_MODE)   # same draws as the hybrid
    W = WARMUP_STEPS_PER_EPISODE
    episodes_data = [flat[i:i + W] for i in range(0, len(flat), W)]
    train_info = model.train_model(episodes_data, **train_kw)
    warmup_steps = len(flat)

    episodes, total_steps, streak, steps_to_reliable = [], warmup_steps, 0, None
    for ep in range(1, max_episodes + 1):
        err = recurrent_prediction_errors(model, S_test, A_test, pos)
        out = run_episode_gru(true_sim, model, horizon=horizon)
        episodes_data.append(out['transitions'])
        total_steps += out['steps']
        streak = streak + 1 if out['success'] else 0
        if streak >= success_streak_needed and steps_to_reliable is None:
            steps_to_reliable = total_steps
        row = {'episode': ep, 'steps': out['steps'], 'cumulative_steps': total_steps,
               'success': out['success'], 'final_dist': out['final_dist'],
               'return': out['return'], 'model_epochs_run': train_info['epochs_run'],
               'model_best_epoch': train_info['best_epoch'],
               'model_val_loss': train_info['val_loss']}
        row.update(err)
        episodes.append(row)
        if verbose:
            print(f"    [hybrid_gru] ep {ep:2d}  steps={out['steps']:3d}  "
                  f"success={str(out['success']):5s}  return={out['return']:8.2f}  "
                  f"cum_steps={total_steps:4d}  1-step pos err={err['onestep_pos_err']*1000:.2f} mm  "
                  f"8-step={err['kstep_pos_err']*1000:.1f} mm  epochs={train_info['epochs_run']}")
        train_info = model.train_model(episodes_data, **train_kw)

    final_model = dict(train_info)
    final_model.update(recurrent_prediction_errors(model, S_test, A_test, pos))

    eval_rows, eval_trans = [], []
    if do_eval:
        np.random.seed(seed + 10_000)
        tasks = [('train_task', TRAIN_START, TRAIN_GOAL)] + \
                [(f'eval_{i+1}', s, g) for i, (s, g) in enumerate(EVAL_TASKS)]
        for task_id, start, goal in tasks:
            out = run_episode_gru(true_sim, model, start_pose=start, goal=goal, horizon=horizon)
            eval_rows.append({'task_id': task_id, 'success': out['success'],
                              'steps': out['steps'], 'final_dist': out['final_dist'],
                              'return': out['return']})
            if task_id != 'train_task':
                eval_trans.append(out['transitions'])
            if verbose:
                print(f"    [hybrid_gru] FROZEN {task_id:10s} success={str(out['success']):5s}  "
                      f"steps={out['steps']:3d}  final_dist={out['final_dist']:.3f}")

    new_rows = [r for r in eval_rows if r['task_id'] != 'train_task']
    onpol = onpolicy_errors(model, eval_trans, horizon, pos)
    mean_of = lambda rows, k: float(np.mean([r[k] for r in rows])) if rows else np.nan
    return {
        'model': 'hybrid_gru', 'model_object': model,
        'warmup_steps': warmup_steps, 'total_real_steps': total_steps,
        'steps_to_reliable_success': steps_to_reliable,
        'success_rate_all': mean_of(episodes, 'success'),
        'success_rate_last5': mean_of(episodes[-5:], 'success'),
        'mean_return': mean_of(episodes, 'return'),
        'mean_return_last5': mean_of(episodes[-5:], 'return'),
        'mean_epochs_per_retrain': mean_of(episodes, 'model_epochs_run'),
        'episodes': episodes, 'final_model': final_model, 'eval_tasks': eval_rows,
        'eval_success_rate': mean_of(new_rows, 'success'),
        'eval_mean_steps': mean_of(new_rows, 'steps'),
        'eval_mean_final_dist': mean_of(new_rows, 'final_dist'),
        'eval_mean_return': mean_of(new_rows, 'return'),
        'eval_train_task_success': eval_rows[0]['success'] if eval_rows else None,
        'eval_onpolicy_onestep_pos_err': onpol.get('onestep_pos_err', np.nan),
        'eval_onpolicy_kstep_pos_err': onpol.get('kstep_pos_err', np.nan),
        'runtime_s': time.time() - t_start,
    }
