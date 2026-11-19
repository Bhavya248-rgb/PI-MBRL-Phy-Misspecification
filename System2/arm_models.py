"""
arm_models.py

System 2: the same three models as System 1 (models.py), for the arm.
All share one interface:
    model.predict_batch(states, actions) -> next_states

  1. PhysicsOnlyModel -- arm dynamics equation with the ASSUMED (possibly
     wrong) link masses and no motor lag. No learning.
  2. BlackBoxModel    -- MLP learns the state change from data, no physics.
  3. HybridModel      -- frozen physics prior + residual MLP:
         next_state = physics_prediction + residual_NN(state, action)

Everything about learning is IDENTICAL to System 1: same MLP (3 hidden
layers of 16, ReLU), predicts the change, mean/std normalization with a
1e-6 std floor, 10% validation split (min 5), Adam lr 0.01 recreated at
every retrain, up to 1000 epochs, patience 100, min_delta 1e-4, best
validation weights kept, warm start between retrains.

INPUT TYPES (obs_mode) -- what the models see as the state:
  'state'       (q1, q2, dq1, dq2)                      standard arm state
  'prev_action' + previous command (tau1, tau2)         partial hint of the
                hidden actual torque (copied, not learned)
  'torque'      + ACTUAL motor torques                  the arm's version of
                wheel encoders (joint torque sensor / motor current); the
                network also predicts the change in actual torque
  'history'     + last measured joint acceleration (change in joint speed
                over the last step / dt) + last two commands  -- no sensor;
                computed from the state, not learned
The physics prior is the same in every mode: it assumes actual torque =
command (instant motors). Only what the network can see changes.

Network input: sin and cos of both angles (no jump at +-pi), the joint
speeds, the extra inputs, and the command.
"""

import copy

import numpy as np
import torch
import torch.nn as nn

from arm_sim import arm_dynamics_step, DT

OBS_MODES = ('state', 'prev_action', 'torque', 'history')


def state_dim(obs_mode):
    return {'state': 4, 'prev_action': 6, 'torque': 6, 'history': 10}[obs_mode]


def history_extras(states, actions, next_core, dt):
    """History part of the NEXT state: (measured acceleration over this step,
    this command, previous command). states (..., 10), next_core (..., 4)."""
    acc = (next_core[..., 2:4] - states[..., 2:4]) / dt
    return np.concatenate([acc, actions, states[..., 6:8]], axis=-1)


def state_to_network_input(state):
    """(q1, q2, dq1, dq2, [extras]) -> (sin q1, cos q1, sin q2, cos q2, dq1, dq2, [extras])."""
    q1, q2 = state[..., 0], state[..., 1]
    ang = np.stack([np.sin(q1), np.cos(q1), np.sin(q2), np.cos(q2)], axis=-1)
    return np.concatenate([ang, state[..., 2:]], axis=-1)


class Normalizer:
    STD_FLOOR = 1e-6

    def fit(self, X, Y):
        self.x_mean = X.mean(axis=0)
        self.x_std = np.maximum(X.std(axis=0), self.STD_FLOOR)
        self.y_mean = Y.mean(axis=0)
        self.y_std = np.maximum(Y.std(axis=0), self.STD_FLOOR)

    def norm_x(self, X):
        return (X - self.x_mean) / self.x_std

    def norm_y(self, Y):
        return (Y - self.y_mean) / self.y_std

    def denorm_y(self, Y):
        return Y * self.y_std + self.y_mean


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=16, n_hidden=3):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_size), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers += [nn.Linear(hidden_size, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _train_with_val_early_stopping(net, X_tr, Y_tr, X_val, Y_val, lr,
                                   max_epochs, patience, min_delta):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        best_val = loss_fn(net(X_val), Y_val).item()
        best_train = loss_fn(net(X_tr), Y_tr).item()
    best_state = copy.deepcopy(net.state_dict())
    best_epoch, since_improve, epochs_run = 0, 0, 0
    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad()
        loss = loss_fn(net(X_tr), Y_tr)
        loss.backward()
        optimizer.step()
        epochs_run = epoch
        with torch.no_grad():
            val = loss_fn(net(X_val), Y_val).item()
        significant = val < best_val - min_delta
        if val < best_val:
            best_val, best_train = val, loss.item()
            best_state = copy.deepcopy(net.state_dict())
            best_epoch = epoch
        if significant:
            since_improve = 0
        else:
            since_improve += 1
            if since_improve >= patience:
                break
    net.load_state_dict(best_state)
    return {'epochs_run': epochs_run, 'best_epoch': best_epoch,
            'train_loss': best_train, 'val_loss': best_val}


class PhysicsOnlyModel:
    """Arm dynamics with assumed masses, instant motors. No learning."""

    def __init__(self, assumed_m1, assumed_m2, dt=DT, obs_mode='state'):
        self.m1, self.m2, self.dt, self.obs_mode = assumed_m1, assumed_m2, dt, obs_mode

    def predict_batch(self, states, actions):
        states = np.asarray(states, dtype=float)
        actions = np.asarray(actions, dtype=float)
        core = arm_dynamics_step(states[..., :4], actions, self.m1, self.m2, dt=self.dt)
        if self.obs_mode == 'state':
            return core
        if self.obs_mode == 'history':
            return np.concatenate([core, history_extras(states, actions, core, self.dt)], axis=-1)
        # 'prev_action': next previous-command = this command (exact)
        # 'torque': instant-motor assumption -> next actual torque = command
        return np.concatenate([core, actions], axis=-1)

    def predict(self, state, action):
        return self.predict_batch(np.asarray(state)[None], np.asarray(action)[None])[0]


class _LearnedModel:
    def __init__(self, seed=0, val_fraction=0.1, min_val=5, obs_mode='state', dt=DT):
        torch.manual_seed(seed)
        self.obs_mode, self.dt = obs_mode, dt
        self.n_in = state_dim(obs_mode) + 2 + 2       # sin/cos adds 2, plus command
        self.n_out = 6 if obs_mode == 'torque' else 4
        self.net = MLP(self.n_in, self.n_out)
        self.norm = Normalizer()
        self.val_fraction, self.min_val = val_fraction, min_val
        self.split_rng = np.random.default_rng(seed + 777)

    def _make_input(self, states, actions):
        return np.concatenate([state_to_network_input(states), actions], axis=-1)

    def _base_next(self, states, actions):
        raise NotImplementedError

    def train_model(self, states, actions, next_states, lr=0.01, max_epochs=1000,
                    patience=100, min_delta=1e-4, verbose=False):
        states = np.asarray(states, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        next_states = np.asarray(next_states, dtype=np.float64)
        X_np = self._make_input(states, actions)
        Y_np = (next_states - self._base_next(states, actions))[:, :self.n_out]
        n = len(X_np)
        n_val = max(self.min_val, int(round(self.val_fraction * n)))
        perm = self.split_rng.permutation(n)
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        self.norm.fit(X_np[tr_idx], Y_np[tr_idx])
        to_t = lambda a: torch.tensor(a, dtype=torch.float32)
        info = _train_with_val_early_stopping(
            self.net, to_t(self.norm.norm_x(X_np[tr_idx])), to_t(self.norm.norm_y(Y_np[tr_idx])),
            to_t(self.norm.norm_x(X_np[val_idx])), to_t(self.norm.norm_y(Y_np[val_idx])),
            lr, max_epochs, patience, min_delta)
        info['n_train'], info['n_val'] = len(tr_idx), len(val_idx)
        return info

    def predict_batch(self, states, actions):
        states = np.asarray(states, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        X = torch.tensor(self.norm.norm_x(self._make_input(states, actions)), dtype=torch.float32)
        with torch.no_grad():
            corr = self.norm.denorm_y(self.net(X).numpy().astype(np.float64))
        out = self._base_next(states, actions)
        out[..., :self.n_out] += corr
        if self.obs_mode == 'history':        # recompute from the corrected state
            out[..., 4:] = history_extras(states, actions, out[..., :4], self.dt)
        return out

    def predict(self, state, action):
        return self.predict_batch(np.asarray(state)[None], np.asarray(action)[None])[0]


class BlackBoxModel(_LearnedModel):
    """Learns the state change from data, no physics."""

    def _base_next(self, states, actions):
        out = np.array(states, dtype=np.float64, copy=True)
        if self.obs_mode == 'prev_action':
            out[..., 4:] = actions            # copied exactly, not learned
        return out                            # ('history' extras set in predict_batch)


class HybridModel(_LearnedModel):
    """Frozen (possibly wrong) physics prior + learned residual."""

    def __init__(self, assumed_m1, assumed_m2, dt=DT, seed=0, val_fraction=0.1,
                 min_val=5, obs_mode='state'):
        super().__init__(seed=seed, val_fraction=val_fraction, min_val=min_val,
                         obs_mode=obs_mode, dt=dt)
        self.physics = PhysicsOnlyModel(assumed_m1, assumed_m2, dt, obs_mode=obs_mode)

    def _base_next(self, states, actions):
        return self.physics.predict_batch(states, actions)
