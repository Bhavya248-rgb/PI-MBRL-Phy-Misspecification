"""
recurrent_residual.py

The RECURRENT hybrid ('hybrid_gru'): the same frozen physics prior, but the
residual network has a memory (a GRU) that it carries from step to step.
The SAME file is used by both robots (System 1 and System 2).

    h_t        = GRU( x_t , h_{t-1} )          x_t = [state_t features, command_t]
    residual_t = MLP( [h_t , x_t] )            same 3 x 16 ReLU MLP as the hybrid
    next_state = physics(state_t, command_t) + residual_t

What the network SEES is only the plain state (pose / angles + speeds) and
the command -- no sensor, no hand-made history. Anything it needs about the
hidden motor state it must work out from the past, through its memory h.
If the MLP ignored h it would be exactly the feedforward hybrid, so this
model is "the same hybrid, plus a memory".

MEMORY RULES
  - h = 0 at every episode start. Every episode starts with the robot at
    rest and zero motor torque, so "no memory" matches reality there.
  - During a real episode, after each real step the memory is updated with
    the REAL measured state and the command sent (a filter).
  - In the planner, every candidate plan starts from the current real
    memory, and the memory then runs forward on the model's OWN imagined
    states, exactly like the state does.

TRAINING (kept as close to the hybrid's as possible)
  - Data: the same buffer, but kept as whole episodes (warmup episodes and
    driving episodes), because the memory needs the order of the steps.
  - The GRU runs over each whole episode from h = 0 (teacher forcing: it is
    fed the real states). Loss = MSE of the residual on every step.
  - Validation: a random 10% of the STEPS (at least 5) are left out of the
    loss and used for early stopping -- same split rule as the hybrid.
  - Same normalization (mean/std, std floor 1e-6, training steps only),
    Adam lr 0.01 recreated at every retrain, up to 1000 epochs, patience
    100, min_delta 1e-4, best validation weights kept, warm start.
  - One addition that is standard for recurrent networks: gradient norm
    clipping at 1.0 (prevents rare exploding-gradient steps).
"""

import copy

import numpy as np
import torch
import torch.nn as nn

GRU_HIDDEN = 16            # memory size (same width as the MLP layers)
GRAD_CLIP = 1.0


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


class GRUResidualNet(nn.Module):
    def __init__(self, n_in, n_out, memory=GRU_HIDDEN, hidden_size=16, n_hidden=3):
        super().__init__()
        self.memory = memory
        self.gru = nn.GRU(n_in, memory, batch_first=True)
        layers = [nn.Linear(memory + n_in, hidden_size), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers += [nn.Linear(hidden_size, n_out)]
        self.head = nn.Sequential(*layers)

    def forward(self, x, h0=None):
        """x (B, T, n_in), h0 (1, B, memory) or None -> residuals (B, T, n_out), h (1, B, memory)."""
        hs, hT = self.gru(x, h0)
        return self.head(torch.cat([hs, x], dim=-1)), hT


def _masked_mse(pred, Y, mask):
    se = ((pred - Y) ** 2).mean(dim=-1)            # mean over outputs, as nn.MSELoss
    return (se * mask).sum() / mask.sum()


class GRUHybridModel:
    """physics: a physics-only model with predict_batch(states, actions) -> next states.
    features: state -> network features (sin/cos of angles etc.).
    n_out: how many state entries the residual corrects (the plain state)."""

    is_recurrent = True

    def __init__(self, physics, features, n_out, seed=0, val_fraction=0.1, min_val=5):
        torch.manual_seed(seed)
        self.physics, self.features, self.n_out = physics, features, n_out
        self.n_in = None
        self.net = None
        self.norm = Normalizer()
        self.val_fraction, self.min_val = val_fraction, min_val
        self.split_rng = np.random.default_rng(seed + 777)
        self._seed = seed

    # ------------------------------------------------------------------
    def _input(self, states, actions):
        return np.concatenate([self.features(states), actions], axis=-1)

    def _ensure_net(self, n_in):
        if self.net is None:
            torch.manual_seed(self._seed)
            self.n_in = n_in
            self.net = GRUResidualNet(n_in, self.n_out)

    def init_memory(self, n=1):
        return torch.zeros(1, n, self.net.memory)

    # ------------------------------------------------------------------
    def train_model(self, episodes, lr=0.01, max_epochs=1000, patience=100,
                    min_delta=1e-4, verbose=False):
        """episodes: list of episodes, each a list of (state, action, next_state)."""
        episodes = [e for e in episodes if len(e) > 0]
        B, T = len(episodes), max(len(e) for e in episodes)
        S = [np.array([t[0] for t in e], dtype=np.float64) for e in episodes]
        A = [np.array([t[1] for t in e], dtype=np.float64) for e in episodes]
        S2 = [np.array([t[2] for t in e], dtype=np.float64) for e in episodes]
        X_list = [self._input(s, a) for s, a in zip(S, A)]
        Y_list = [(s2 - self.physics.predict_batch(s, a))[:, :self.n_out]
                  for s, a, s2 in zip(S, A, S2)]
        self._ensure_net(X_list[0].shape[1])

        # same split rule as the hybrid: random 10% of the steps (min 5)
        n = sum(len(e) for e in episodes)
        n_val = max(self.min_val, int(round(self.val_fraction * n)))
        is_val = np.zeros(n, dtype=bool)
        is_val[self.split_rng.permutation(n)[:n_val]] = True

        X_all, Y_all = np.concatenate(X_list), np.concatenate(Y_list)
        self.norm.fit(X_all[~is_val], Y_all[~is_val])          # training steps only

        X = np.zeros((B, T, self.n_in)); Y = np.zeros((B, T, self.n_out))
        m_tr = np.zeros((B, T)); m_val = np.zeros((B, T))
        k = 0
        for i, (x, y) in enumerate(zip(X_list, Y_list)):
            L = len(x)
            X[i, :L] = self.norm.norm_x(x)
            Y[i, :L] = self.norm.norm_y(y)
            v = is_val[k:k + L]
            m_tr[i, :L] = ~v
            m_val[i, :L] = v
            k += L
        f = lambda a: torch.tensor(a, dtype=torch.float32)
        X, Y, m_tr, m_val = f(X), f(Y), f(m_tr), f(m_val)

        net = self.net
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        with torch.no_grad():
            p, _ = net(X)
            best_val = _masked_mse(p, Y, m_val).item()
            best_train = _masked_mse(p, Y, m_tr).item()
        best_state = copy.deepcopy(net.state_dict())
        best_epoch, since, epochs_run = 0, 0, 0
        for epoch in range(1, max_epochs + 1):
            opt.zero_grad()
            p, _ = net(X)
            loss = _masked_mse(p, Y, m_tr)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            opt.step()
            epochs_run = epoch
            with torch.no_grad():
                p, _ = net(X)
                val = _masked_mse(p, Y, m_val).item()
            significant = val < best_val - min_delta
            if val < best_val:
                best_val, best_train = val, loss.item()
                best_state = copy.deepcopy(net.state_dict())
                best_epoch = epoch
            if significant:
                since = 0
            else:
                since += 1
                if since >= patience:
                    break
        net.load_state_dict(best_state)
        return {'epochs_run': epochs_run, 'best_epoch': best_epoch,
                'train_loss': best_train, 'val_loss': best_val,
                'n_train': int(n - n_val), 'n_val': int(n_val)}

    # ------------------------------------------------------------------
    def step(self, states, actions, memory):
        """One step for a batch. states (N, d), actions (N, 2), memory (1, N, H).
        Returns (next_states (N, d), new memory)."""
        states = np.asarray(states, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        x = torch.tensor(self.norm.norm_x(self._input(states, actions)),
                         dtype=torch.float32)[:, None, :]
        with torch.no_grad():
            r, memory = self.net(x, memory)
        out = self.physics.predict_batch(states, actions)
        out[..., :self.n_out] += self.norm.denorm_y(r[:, 0].numpy().astype(np.float64))
        return out, memory

    def teacher_forced(self, states, actions):
        """Runs the memory along REAL trajectories from h = 0.
        states (M, k, d), actions (M, k, 2) -> one-step predictions (M, k, d)
        and the memory BEFORE each step (M, k, H)."""
        M, k = actions.shape[:2]
        mem = self.init_memory(M)
        preds, mems = [], []
        for t in range(k):
            mems.append(mem[0].clone())
            p, mem = self.step(states[:, t], actions[:, t], mem)
            preds.append(p)
        return np.stack(preds, axis=1), torch.stack(mems, dim=1)


# ----------------------------------------------------------------------
# Planner: CEM exactly as cem_planner.cem_plan (same random draws), but the
# imagined rollouts also carry the memory.
# ----------------------------------------------------------------------
def cem_plan_recurrent(model, state, memory, reward_fn, horizon=8, n_samples=150,
                       n_elite=15, n_iters=3, action_low=-8.0, action_high=8.0,
                       action_dim=2):
    mean = np.zeros((horizon, action_dim))
    std = np.ones((horizon, action_dim)) * (action_high - action_low) / 4.0
    for _ in range(n_iters):
        samples = np.random.normal(loc=mean, scale=std, size=(n_samples, horizon, action_dim))
        samples = np.clip(samples, action_low, action_high)
        s = np.repeat(np.asarray(state, dtype=float)[None, :], n_samples, axis=0)
        h = memory.repeat(1, n_samples, 1)
        scores = np.zeros(n_samples)
        for t in range(horizon):
            s, h = model.step(s, samples[:, t], h)
            scores += reward_fn(s)
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        elite = samples[np.argsort(scores)[-n_elite:]]
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + 1e-6
    return mean[0]


# ----------------------------------------------------------------------
# Prediction errors with memory (same definitions as prediction_errors)
# ----------------------------------------------------------------------
def recurrent_prediction_errors(model, S, A, pos_fn):
    """Fixed test set. S (M, k+1, d) true states, A (M, k, 2); every test
    trajectory starts at rest, so the memory starts at 0.
    pos_fn(states) -> positions (..., 2).
    one-step: from each TRUE state, with the memory built from the true past.
    k-step: open loop from S[:, 0], feeding back its own predictions.
    Returns mean position errors (m)."""
    M, k = A.shape[:2]
    pred1, _ = model.teacher_forced(S[:, :-1], A)
    one = np.linalg.norm(pos_fn(pred1) - pos_fn(S[:, 1:]), axis=-1).mean()
    s, h = S[:, 0].copy(), model.init_memory(M)
    for t in range(k):
        s, h = model.step(s, A[:, t], h)
    kst = np.linalg.norm(pos_fn(s) - pos_fn(S[:, -1]), axis=-1).mean()
    return {'onestep_pos_err': float(one), 'kstep_pos_err': float(kst)}


def onpolicy_errors(model, transitions_list, k, pos_fn):
    """Same windows as windows_from_transitions() (every length-k window of
    every evaluation episode), but each window starts from the memory the
    model had at that point of the real episode."""
    one_errs, k_errs = [], []
    for trans in transitions_list:
        if len(trans) < k:
            continue
        states = np.array([t[0] for t in trans] + [trans[-1][2]])
        actions = np.array([t[1] for t in trans])
        pred1, mems = model.teacher_forced(states[None, :-1], actions[None])
        pred1, mems = pred1[0], mems[0]                          # (L, d), (L, H)
        e1 = np.linalg.norm(pos_fn(pred1) - pos_fn(states[1:]), axis=-1)
        n_win = len(trans) - k + 1
        for i in range(n_win):
            one_errs.append(e1[i:i + k])
        s, h = states[:n_win].copy(), mems[:n_win][None].contiguous()
        for t in range(k):
            s, h = model.step(s, actions[np.arange(n_win) + t], h)
        k_errs.append(np.linalg.norm(pos_fn(s) - pos_fn(states[np.arange(n_win) + k]), axis=-1))
    if not k_errs:
        return {}
    return {'onestep_pos_err': float(np.concatenate(one_errs).mean()),
            'kstep_pos_err': float(np.concatenate(k_errs).mean())}
