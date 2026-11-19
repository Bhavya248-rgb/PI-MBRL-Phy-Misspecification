"""
models.py

Three dynamics models, all exposing the same interface:
    model.predict(state, action)            -> next_state        (3,)
    model.predict_batch(states, actions)    -> next_states       (N, 3)

  1. PhysicsOnlyModel -- kinematic equations only (possibly WRONG wheel
     radius / slip). No learning.
  2. BlackBoxModel    -- MLP learns the state change from data, with NO
     physics knowledge.
  3. HybridModel      -- physics equations (possibly wrong, kept FROZEN)
     PLUS a residual MLP that learns what the physics gets wrong:
         next_state = physics_prediction + residual_NN(state, action)

NORMALIZATION (unchanged): inputs and targets are mean/std normalized,
as in Nagabandi et al. (2018), with a std floor of 1e-6 so a near-zero
residual (correct physics) is not blown up to unit scale.

TRAINING -- CHANGED IN THIS VERSION: validation-based early stopping.
  At every retrain the buffer is randomly split into a training part
  (90%) and a held-out validation part (10%, at least 5 transitions).
  The network is trained on the training part only; after every epoch
  its loss on the validation part is measured. The weights with the
  LOWEST VALIDATION LOSS are kept (any improvement counts), and training
  stops once `patience` epochs pass without the validation loss
  improving by more than `min_delta`. If no epoch beats the starting
  weights, the previous model is kept unchanged (best_epoch = 0).
  This replaces the previous rule, which watched the TRAINING loss and
  therefore could not tell learning apart from memorizing.
  Normalization statistics are computed from the training part only, so
  the validation part is genuinely unseen.
  (For reference: MBPO, Janner et al. 2019, uses the same idea with a
  20% holdout. 10% is used here because the buffer is small early on;
  `val_fraction` can be changed.)

The optimizer is recreated fresh at every retrain (a persistent optimizer
was tested earlier and made results worse). The network weights carry
over between retrains (warm start), as before.
"""

import copy

import numpy as np
import torch
import torch.nn as nn

from diffdrive_sim import predict_next_state


def state_to_network_input(state):
    """(x, y, theta) -> (x, y, sin(theta), cos(theta)); avoids the +-pi jump."""
    x, y, theta = state[..., 0], state[..., 1], state[..., 2]
    return np.stack([x, y, np.sin(theta), np.cos(theta)], axis=-1)


class Normalizer:
    """Mean/std normalization of network inputs (X) and targets (Y)."""
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
    """Small MLP: 4 linear layers, 16 neurons per hidden layer."""
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
                                   max_epochs, patience, min_delta,
                                   verbose, log_every, tag):
    """
    Full-batch training with validation-based early stopping.
    Restores the weights from the epoch with the lowest validation loss.

    Returns a dict:
        epochs_run   -- epochs actually trained before stopping
        best_epoch   -- epoch whose weights were kept (1-based; 0 = initial)
        train_loss   -- training loss at best_epoch (normalized units)
        val_loss     -- validation loss at best_epoch (normalized units)
    """
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    with torch.no_grad():
        best_val = loss_fn(net(X_val), Y_val).item()
        best_train = loss_fn(net(X_tr), Y_tr).item()
    best_state = copy.deepcopy(net.state_dict())
    best_epoch = 0
    since_improve = 0
    epochs_run = 0

    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad()
        loss = loss_fn(net(X_tr), Y_tr)
        loss.backward()
        optimizer.step()
        epochs_run = epoch

        with torch.no_grad():
            val = loss_fn(net(X_val), Y_val).item()
        train_val = loss.item()

        if verbose and epoch % log_every == 0:
            print(f"  [{tag}] epoch {epoch:4d}  train={train_val:.5f}  "
                  f"val={val:.5f}  best_val={best_val:.5f}")

        # Any improvement in validation loss is kept (weights saved);
        # only an improvement larger than min_delta resets the patience
        # counter. So min_delta decides WHEN TO STOP, not which weights
        # are kept.
        significant = val < best_val - min_delta
        if val < best_val:
            best_val = val
            best_train = train_val
            best_state = copy.deepcopy(net.state_dict())
            best_epoch = epoch
        if significant:
            since_improve = 0
        else:
            since_improve += 1
            if since_improve >= patience:
                break

    net.load_state_dict(best_state)
    if verbose:
        print(f"  [{tag}] stopped after {epochs_run} epochs, kept epoch "
              f"{best_epoch} (val={best_val:.5f}, train={best_train:.5f})")
    return {'epochs_run': epochs_run, 'best_epoch': best_epoch,
            'train_loss': best_train, 'val_loss': best_val}


class PhysicsOnlyModel:
    """Kinematic equations only. No learning."""

    def __init__(self, assumed_r, wheelbase, dt, assumed_slip=1.0):
        self.assumed_r = assumed_r
        self.wheelbase = wheelbase
        self.dt = dt
        self.assumed_slip = assumed_slip

    def predict_batch(self, states, actions):
        actions = np.asarray(actions, dtype=float)
        return predict_next_state(states, actions[..., 0], actions[..., 1],
                                  assumed_r=self.assumed_r,
                                  wheelbase=self.wheelbase, dt=self.dt,
                                  assumed_slip=self.assumed_slip)

    def predict(self, state, action):
        return self.predict_batch(np.asarray(state)[None, :],
                                  np.asarray(action)[None, :])[0]


class _LearnedModel:
    """
    Shared code for BlackBoxModel and HybridModel. Subclasses define
    _base_next(states, actions): the prediction BEFORE the network's
    correction (the unchanged state for black_box, the physics
    prediction for hybrid). The network learns
        target = true_next_state - _base_next(state, action).
    """

    def __init__(self, seed=0, val_fraction=0.1, min_val=5):
        torch.manual_seed(seed)
        self.net = MLP(input_dim=6, output_dim=3)
        self.norm = Normalizer()
        self.val_fraction = val_fraction
        self.min_val = min_val
        self.split_rng = np.random.default_rng(seed + 777)

    def _make_input(self, states, actions):
        return np.concatenate([state_to_network_input(states), actions], axis=-1)

    def _base_next(self, states, actions):
        raise NotImplementedError

    def train_model(self, states, actions, next_states,
                    lr=0.01, max_epochs=1000, patience=100, min_delta=1e-4,
                    verbose=False, log_every=20):
        states = np.asarray(states, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        next_states = np.asarray(next_states, dtype=np.float64)

        X_np = self._make_input(states, actions)
        Y_np = next_states - self._base_next(states, actions)

        n = len(X_np)
        n_val = max(self.min_val, int(round(self.val_fraction * n)))
        perm = self.split_rng.permutation(n)
        val_idx, tr_idx = perm[:n_val], perm[n_val:]

        self.norm.fit(X_np[tr_idx], Y_np[tr_idx])   # training part only
        to_t = lambda a: torch.tensor(a, dtype=torch.float32)
        X_tr = to_t(self.norm.norm_x(X_np[tr_idx]))
        Y_tr = to_t(self.norm.norm_y(Y_np[tr_idx]))
        X_val = to_t(self.norm.norm_x(X_np[val_idx]))
        Y_val = to_t(self.norm.norm_y(Y_np[val_idx]))

        info = _train_with_val_early_stopping(
            self.net, X_tr, Y_tr, X_val, Y_val, lr, max_epochs, patience,
            min_delta, verbose, log_every, tag=type(self).__name__)
        info['n_train'] = len(tr_idx)
        info['n_val'] = len(val_idx)
        return info

    def predict_batch(self, states, actions):
        states = np.asarray(states, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        X = torch.tensor(self.norm.norm_x(self._make_input(states, actions)),
                         dtype=torch.float32)
        with torch.no_grad():
            correction = self.norm.denorm_y(self.net(X).numpy().astype(np.float64))
        return self._base_next(states, actions) + correction

    def predict(self, state, action):
        return self.predict_batch(np.asarray(state)[None, :],
                                  np.asarray(action)[None, :])[0]


class BlackBoxModel(_LearnedModel):
    """Pure black-box model: learns the state change, no physics."""

    def _base_next(self, states, actions):
        return states


class HybridModel(_LearnedModel):
    """Frozen (possibly wrong) physics prior + learned residual."""

    def __init__(self, assumed_r, wheelbase, dt, assumed_slip=1.0, seed=0,
                 val_fraction=0.1, min_val=5):
        super().__init__(seed=seed, val_fraction=val_fraction, min_val=min_val)
        self.physics = PhysicsOnlyModel(assumed_r, wheelbase, dt, assumed_slip)

    def _base_next(self, states, actions):
        return self.physics.predict_batch(states, actions)


if __name__ == '__main__':
    # Diagnostic (python models.py): trains both learned models on a
    # realistic buffer and reports held-out prediction error in metres.
    import time
    from diffdrive_sim import DiffDriveSimulator

    def make_data(seed, n_episodes=10, steps=60):
        rng = np.random.default_rng(seed)
        sim = DiffDriveSimulator(true_wheel_radius=0.03, wheelbase=0.15,
                                 slip_factor=0.85, dt=0.1)
        S, A, S2 = [], [], []
        for _ in range(n_episodes):
            sim.reset(x=rng.uniform(-0.1, 0.7), y=rng.uniform(-0.1, 0.7),
                      theta=rng.uniform(-np.pi, np.pi))
            for _ in range(steps):
                s = sim.get_state().copy()
                a = rng.uniform(-8, 8, size=2)
                S.append(s); A.append(a); S2.append(sim.step(a[0], a[1]).copy())
        return np.array(S), np.array(A), np.array(S2)

    S, A, S2 = make_data(0)
    St, At, S2t = make_data(999, n_episodes=4)
    print(f"Train buffer: {len(S)} transitions, test: {len(St)}\n")
    for name, model in [
        ("BlackBoxModel", BlackBoxModel(seed=0)),
        ("HybridModel (correct physics)",
         HybridModel(assumed_r=0.03, wheelbase=0.15, dt=0.1, assumed_slip=0.85)),
        ("HybridModel (50% radius error)",
         HybridModel(assumed_r=0.015, wheelbase=0.15, dt=0.1, assumed_slip=0.85)),
    ]:
        t0 = time.time()
        info = model.train_model(S, A, S2)
        pred = model.predict_batch(St, At)
        pos = np.linalg.norm(pred[:, :2] - S2t[:, :2], axis=1).mean()
        step = np.linalg.norm(S2t[:, :2] - St[:, :2], axis=1).mean()
        print(f"{name}: {info['epochs_run']} epochs (kept {info['best_epoch']}), "
              f"val={info['val_loss']:.5f}, {time.time()-t0:.1f}s")
        print(f"  test position error {pos:.5f} m vs mean step {step:.5f} m "
              f"({100*pos/step:.1f}%)\n")
