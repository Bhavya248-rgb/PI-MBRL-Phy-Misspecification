"""
arm_mbrl.py

System 2: the model-based RL loop for the arm. Same algorithm and the
same measurements as System 1 (model_based_rl.py); only the robot, the
task and the input types are the arm's:

  warmup (random torques, 3 random rest poses x 15 steps, counted as real
  steps) -> train -> for 15 episodes: one episode with CEM-MPC on the
  current model, add its data, retrain on the whole buffer
  -> FROZEN EVALUATION on the training task + 6 new tasks, learning off.

Measured (as in System 1):
  - per episode: return, success, steps, model training info
  - model quality before every episode on a FIXED random-torque test set
    (40 trajectories, seed 12345): 1-step and k-step TIP position error (m)
  - frozen eval: success, return, final distance per task, and on-policy
    1-step / k-step tip error on the trajectories driven there

Physics error: the prior's link masses = true masses x (1 + error).
Hidden effect: motor lag on the joint torques (motor_tau).
"""

import time

import numpy as np

from arm_sim import ArmSimulator, tip_position, M1, M2, DT, TORQUE_LIMIT
from cem_planner import cem_plan
from arm_task import (reward_fn, is_success, distance_to_goal, MAX_STEPS,
                      TRAIN_START, TRAIN_GOAL, EVAL_TASKS)
from arm_models import (PhysicsOnlyModel, BlackBoxModel, HybridModel, state_dim,
                        history_extras)

WARMUP_EPISODES = 3
WARMUP_STEPS_PER_EPISODE = 15
TEST_SET_SEED = 12345
TEST_SET_TRAJECTORIES = 40
HORIZON = 10                 # 0.5 s look-ahead at dt = 0.05


class Observer:
    """What the model is given as 'the state' (see arm_models.py, obs_mode).
    reset() at every episode start (arm at rest, zero torque, no history)."""

    def __init__(self, sim, obs_mode, dt=DT):
        self.sim, self.mode, self.dt = sim, obs_mode, dt
        self.reset()

    def reset(self):
        self.prev_core = self.sim.get_state()
        self.a1 = np.zeros(2)
        self.a2 = np.zeros(2)

    def get(self):
        core = self.sim.get_state()
        if self.mode == 'state':
            return core
        if self.mode == 'prev_action':
            return np.concatenate([core, self.a1])
        if self.mode == 'torque':
            return np.concatenate([core, self.sim.get_torques()])
        if self.mode == 'history':
            # previous state layout: core(4), acc(2), cmd t-1(2), cmd t-2(2);
            # its 'cmd t-1' is today's 'cmd t-2' (= self.a2)
            prev = np.concatenate([self.prev_core, np.zeros(2), self.a2, np.zeros(2)])
            return np.concatenate([core, history_extras(prev, self.a1, core, self.dt)])
        raise ValueError(self.mode)

    def step(self, action):
        self.prev_core = self.sim.get_state()
        self.sim.step(action[0], action[1])
        self.a2, self.a1 = self.a1, np.clip(np.array(action, dtype=float),
                                            -TORQUE_LIMIT, TORQUE_LIMIT)


def run_episode(true_sim, model, start=TRAIN_START, goal=TRAIN_GOAL,
                max_steps=MAX_STEPS, horizon=HORIZON, n_samples=150, n_elite=15,
                n_iters=3, obs_mode='state'):
    true_sim.reset(q1=start[0], q2=start[1])
    obs = Observer(true_sim, obs_mode)
    goal = np.asarray(goal, dtype=float)
    rfn = lambda s: reward_fn(s, goal)
    transitions, ep_return, succeeded = [], 0.0, False
    for _ in range(max_steps):
        state = obs.get()
        if is_success(state, goal):
            succeeded = True
            break
        action = cem_plan(model, state, rfn, horizon=horizon, n_samples=n_samples,
                          n_elite=n_elite, n_iters=n_iters,
                          action_low=-TORQUE_LIMIT, action_high=TORQUE_LIMIT)
        obs.step(action)
        next_state = obs.get()
        ep_return += float(reward_fn(next_state, goal))
        transitions.append((state.copy(), np.clip(action, -TORQUE_LIMIT, TORQUE_LIMIT),
                            next_state.copy()))
    if not succeeded and is_success(true_sim.get_state(), goal):
        succeeded = True
    return {'transitions': transitions, 'success': succeeded,
            'final_dist': distance_to_goal(true_sim.get_state(), goal),
            'return': ep_return, 'steps': len(transitions)}


def collect_warmup(true_sim, rng, obs_mode='state', n_episodes=WARMUP_EPISODES,
                   steps_per_episode=WARMUP_STEPS_PER_EPISODE):
    """Random torques from random rest poses."""
    buffer = []
    for _ in range(n_episodes):
        true_sim.reset(q1=rng.uniform(-np.pi, np.pi), q2=rng.uniform(-np.pi, np.pi))
        obs = Observer(true_sim, obs_mode)
        for _ in range(steps_per_episode):
            s = obs.get().copy()
            a = rng.uniform(-1, 1, size=2) * TORQUE_LIMIT
            obs.step(a)
            buffer.append((s, a.copy(), obs.get().copy()))
    return buffer


def train_model_from_buffer(model, buffer, **kw):
    S = np.array([t[0] for t in buffer])
    A = np.array([t[1] for t in buffer])
    S2 = np.array([t[2] for t in buffer])
    return model.train_model(S, A, S2, **kw)


def make_prediction_test_set(true_m1, true_m2, horizon=HORIZON, n_traj=TEST_SET_TRAJECTORIES,
                             seed=TEST_SET_SEED, motor_tau=0.0, obs_mode='state'):
    rng = np.random.default_rng(seed)
    sim = ArmSimulator(true_m1, true_m2, motor_tau=motor_tau)
    S = np.zeros((n_traj, horizon + 1, state_dim(obs_mode)))
    A = rng.uniform(-1, 1, size=(n_traj, horizon, 2)) * TORQUE_LIMIT
    for i in range(n_traj):
        sim.reset(q1=rng.uniform(-np.pi, np.pi), q2=rng.uniform(-np.pi, np.pi))
        obs = Observer(sim, obs_mode)
        S[i, 0] = obs.get()
        for t in range(horizon):
            obs.step(A[i, t])
            S[i, t + 1] = obs.get()
    return S, A


def prediction_errors(model, S, A):
    """Mean TIP position error (m): one-step from true states, and k-step
    open-loop (model fed its own predictions, as the planner does)."""
    k, d = A.shape[1], S.shape[-1]
    pred1 = model.predict_batch(S[:, :-1].reshape(-1, d), A.reshape(-1, 2))
    true1 = S[:, 1:].reshape(-1, d)
    one = np.linalg.norm(tip_position(pred1[:, :4]) - tip_position(true1[:, :4]), axis=1).mean()
    s = S[:, 0].copy()
    for t in range(k):
        s = model.predict_batch(s, A[:, t])
    kst = np.linalg.norm(tip_position(s[:, :4]) - tip_position(S[:, -1, :4]), axis=1).mean()
    return {'onestep_pos_err': float(one), 'kstep_pos_err': float(kst)}


def windows_from_transitions(transitions_list, k):
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


def build_model(model_type, assumed_m1, assumed_m2, seed, val_fraction, obs_mode):
    if model_type == 'physics_only':
        return PhysicsOnlyModel(assumed_m1, assumed_m2, obs_mode=obs_mode)
    if model_type == 'black_box':
        return BlackBoxModel(seed=seed, val_fraction=val_fraction, obs_mode=obs_mode)
    if model_type == 'hybrid':
        return HybridModel(assumed_m1, assumed_m2, seed=seed, val_fraction=val_fraction,
                           obs_mode=obs_mode)
    raise ValueError(model_type)


def run_model_based_rl(model_type, mass_error_pct=0.0, motor_tau=0.0, obs_mode='state',
                       seed=0, max_episodes=15, success_streak_needed=3,
                       horizon=HORIZON, n_samples=150, train_lr=0.01,
                       train_max_epochs=1000, train_patience=100, train_min_delta=1e-4,
                       val_fraction=0.1, do_eval=True, verbose=True):
    """Full MBRL loop for one model, then the frozen evaluation.
    mass_error_pct: the prior believes masses = true x (1 + pct/100)."""
    t_start = time.time()
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    scale = 1.0 + mass_error_pct / 100.0
    true_sim = ArmSimulator(M1, M2, motor_tau=motor_tau)
    model = build_model(model_type, M1 * scale, M2 * scale, seed, val_fraction, obs_mode)
    learned = model_type in ('black_box', 'hybrid')
    S_test, A_test = make_prediction_test_set(M1, M2, horizon=horizon,
                                              motor_tau=motor_tau, obs_mode=obs_mode)
    train_kw = dict(lr=train_lr, max_epochs=train_max_epochs, patience=train_patience,
                    min_delta=train_min_delta)
    plan_kw = dict(horizon=horizon, n_samples=n_samples, obs_mode=obs_mode)

    no_train = {'epochs_run': 0, 'best_epoch': 0, 'train_loss': np.nan,
                'val_loss': np.nan, 'n_train': 0, 'n_val': 0}
    buffer, train_info = [], dict(no_train)
    if learned:
        buffer = collect_warmup(true_sim, rng, obs_mode=obs_mode)
        train_info = train_model_from_buffer(model, buffer, **train_kw)
    warmup_steps = len(buffer)

    episodes, total_steps, streak, steps_to_reliable = [], warmup_steps, 0, None
    for ep in range(1, max_episodes + 1):
        err = prediction_errors(model, S_test, A_test)
        out = run_episode(true_sim, model, **plan_kw)
        buffer.extend(out['transitions'])
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
            print(f"    [{model_type}] ep {ep:2d}  steps={out['steps']:3d}  "
                  f"success={str(out['success']):5s}  return={out['return']:8.2f}  "
                  f"final_dist={out['final_dist']:.3f}  1-step tip err={err['onestep_pos_err']*1000:.1f} mm  "
                  f"{horizon}-step={err['kstep_pos_err']*1000:.0f} mm"
                  + (f"  epochs={train_info['epochs_run']}" if learned else ""))
        if learned:
            train_info = train_model_from_buffer(model, buffer, **train_kw)

    final_model = dict(train_info)
    final_model.update(prediction_errors(model, S_test, A_test))

    eval_rows, eval_trans = [], []
    if do_eval:
        np.random.seed(seed + 10_000)
        tasks = [('train_task', TRAIN_START, TRAIN_GOAL)] + \
                [(f'eval_{i+1}', s, g) for i, (s, g) in enumerate(EVAL_TASKS)]
        for task_id, start, goal in tasks:
            out = run_episode(true_sim, model, start=start, goal=goal, **plan_kw)
            eval_rows.append({'task_id': task_id, 'success': out['success'],
                              'steps': out['steps'], 'final_dist': out['final_dist'],
                              'return': out['return']})
            if task_id != 'train_task':
                eval_trans.append(out['transitions'])
            if verbose:
                print(f"    [{model_type}] FROZEN {task_id:10s} success={str(out['success']):5s}  "
                      f"steps={out['steps']:3d}  final_dist={out['final_dist']:.3f}  "
                      f"return={out['return']:.2f}")

    new_rows = [r for r in eval_rows if r['task_id'] != 'train_task']
    Sw, Aw = windows_from_transitions(eval_trans, horizon)
    onpol = prediction_errors(model, Sw, Aw) if Sw is not None else {}
    mean_of = lambda rows, k: float(np.mean([r[k] for r in rows])) if rows else np.nan
    return {
        'model': model_type, 'model_object': model,
        'warmup_steps': warmup_steps, 'total_real_steps': total_steps,
        'steps_to_reliable_success': steps_to_reliable,
        'success_rate_all': mean_of(episodes, 'success'),
        'success_rate_last5': mean_of(episodes[-5:], 'success'),
        'mean_return': mean_of(episodes, 'return'),
        'mean_return_last5': mean_of(episodes[-5:], 'return'),
        'mean_epochs_per_retrain': mean_of(episodes, 'model_epochs_run') if learned else 0.0,
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
