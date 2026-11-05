"""V6 feature contract: robot-centred (egocentric) inputs + time-based joint history.

Why V6 exists
-------------
Up to V5.4 every network saw *absolute world coordinates* (base x/y, goal x/y, target, EE).
The demonstrations only spawn the base within +-0.25 m of the world origin, so a base that
starts 1.5 m away produces input combinations the networks never saw, and the (never DAgger-
trained) wheel network fails. V6 removes that dependence:

* all positions are expressed in the base frame (rotation by -yaw, translation by base x/y),
  so a goal 1 m ahead looks identical wherever the robot is on the floor;
* angles enter as sin/cos, so there is no +-pi wrap discontinuity in the inputs;
* the arm history lags are defined in *seconds* (10 ms, 30 ms). Every dataset is resampled to
  a 10 ms row spacing before training, and at deployment (1 ms steps) the lags are 10/30 steps,
  so the history has the same physical meaning in training and deployment.

The V6 inputs are computed from the 38 raw columns of dagger_common.FEATURES, which every
demonstration and DAgger rollout CSV already contains, so all earlier data stay usable.
"""
from collections import deque
import json
from pathlib import Path

import numpy as np
import torch

from dagger_common import FEATURES as RAW_FEATURES, OUTPUTS, MLP

R = {name: i for i, name in enumerate(RAW_FEATURES)}
ARM_STATE = [f'q_mes_{i}' for i in range(4, 11)] + [f'qd_mes_{i}' for i in range(4, 11)]

EGO_FEATURES = (
    ['goal_fwd', 'goal_left',                                   # goal position in base frame
     'target_fwd', 'target_left', 'target_up',                  # target in base frame
     'ee_fwd', 'ee_left', 'ee_up',                              # end-effector in base frame
     'target_minus_ee_fwd', 'target_minus_ee_left', 'target_minus_ee_up',
     'goal_distance', 'sin_heading_error', 'cos_heading_error',
     'sin_orientation_error', 'cos_orientation_error']
    + [f'q_mes_{i}' for i in range(4, 11)]
    + [f'qd_mes_{i}' for i in range(4, 11)]
    + [f'q_init_{i}' for i in range(4, 11)]
    + ['navigation_complete'])

DT_ROW = 0.010            # training row spacing (s)
LAGS_S = (0.010, 0.030)   # arm history lags (s)
LAG_ROWS = tuple(int(round(l / DT_ROW)) for l in LAGS_S)       # (1, 3) at 10 ms rows
ARM_EGO_FEATURES = EGO_FEATURES + [f'{c}_lag{int(l*1000)}ms' for l in LAGS_S for c in ARM_STATE]
MODEL_VERSION = 'v6_egocentric'
WHEEL_OUTPUTS = OUTPUTS['wheel_mlp']
ARM_OUTPUTS = OUTPUTS['torque_mlp']


def ego_from_raw(raw):
    """(N, 38) raw dagger_common.FEATURES -> (N, 38) egocentric V6 features (float32)."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[None]
    bx, by, bz = raw[:, R['base_x']], raw[:, R['base_y']], raw[:, R['base_z']]
    yaw = raw[:, R['base_bearing']]
    c, s = np.cos(yaw), np.sin(yaw)

    def to_base(px, py):
        dx, dy = px - bx, py - by
        return c * dx + s * dy, -s * dx + c * dy

    gf, gl = to_base(raw[:, R['goal_x']], raw[:, R['goal_y']])
    tf, tl = to_base(raw[:, R['target_x']], raw[:, R['target_y']])
    ef, el = to_base(raw[:, R['ee_x']], raw[:, R['ee_y']])
    tu, eu = raw[:, R['target_z']] - bz, raw[:, R['ee_z']] - bz
    head, ori = raw[:, R['base_heading_error']], raw[:, R['base_orientation_error']]
    joints = raw[:, R['q_mes_4']:R['q_init_10'] + 1]                     # q, qd, q_init (21)
    out = np.column_stack([gf, gl, tf, tl, tu, ef, el, eu, tf - ef, tl - el, tu - eu,
                           raw[:, R['base_goal_distance']], np.sin(head), np.cos(head),
                           np.sin(ori), np.cos(ori), joints, raw[:, R['navigation_complete']]])
    return out.astype(np.float32)


def arm_matrix(ego, raw):
    """Arm inputs for an episode resampled to 10 ms rows: ego features + lagged q/qd."""
    n = len(ego)
    idx = [R[c] for c in ARM_STATE]
    lagged = [raw[np.maximum(np.arange(n) - k, 0)][:, idx] for k in LAG_ROWS]
    return np.concatenate([ego] + lagged, axis=1).astype(np.float32)


def resample_10ms(df, time_col='time'):
    """Keep one row per 10 ms. Demos are already 10 ms; 1 ms rollouts keep every 10th step."""
    t = df[time_col].to_numpy(dtype=float)
    slot = np.round(t / DT_ROW).astype(np.int64)
    on_grid = np.flatnonzero(np.abs(t - slot * DT_ROW) < 2.5e-4)
    # first on-grid row of every 10 ms slot (compare with the previous *on-grid* row only)
    first = on_grid[np.r_[True, slot[on_grid][1:] != slot[on_grid][:-1]]]
    return df.iloc[first].reset_index(drop=True)


def is_v6_model(model_dir):
    p = Path(model_dir) / 'v6_metadata.json'
    return p.is_file()


class V6Policy:
    """Wheel net + phase-gated arm nets on egocentric features. Input: raw 38-vector per 1 ms step."""

    def __init__(self, model_dir, device='cpu', sim_dt=0.001):
        self.device = torch.device(device)
        folder = Path(model_dir)
        meta = json.loads((folder / 'v6_metadata.json').read_text())
        if meta['model_version'] != MODEL_VERSION or meta['wheel_inputs'] != EGO_FEATURES \
                or meta['arm_inputs'] != ARM_EGO_FEATURES:
            raise ValueError('Incompatible V6 feature contract; retrain with train_v6.py')
        self.models = {}
        for name, n_in, n_out in [('wheel', len(EGO_FEATURES), 4),
                                  ('arm_hold', len(ARM_EGO_FEATURES), 7),
                                  ('arm_reach', len(ARM_EGO_FEATURES), 7)]:
            m = MLP(n_in, n_out, meta['hidden_dims'], meta['dropout'])
            m.load_state_dict(torch.load(folder / f'{name}.pth', map_location='cpu', weights_only=True))
            self.models[name] = m.to(self.device).eval()
        self.lag_steps = tuple(int(round(l / sim_dt)) for l in LAGS_S)
        self.history = deque(maxlen=max(self.lag_steps) + 1)
        self.latched = False
        self.nav_idx = R['navigation_complete']
        self.state_idx = [R[c] for c in ARM_STATE]

    @torch.no_grad()
    def predict(self, raw_x):
        raw = np.asarray(raw_x, dtype=np.float64).copy()
        if raw[self.nav_idx] >= 0.5:
            self.latched = True
        raw[self.nav_idx] = float(self.latched)
        self.history.append(raw[self.state_idx].astype(np.float32))
        n = len(self.history)
        lagged = [self.history[max(0, n - 1 - k)] for k in self.lag_steps]
        ego = ego_from_raw(raw)[0]
        arm_x = np.concatenate([ego] + lagged).astype(np.float32)
        t = lambda v: torch.as_tensor(v, device=self.device).reshape(1, -1)
        w = self.models['wheel'](t(ego)).cpu().numpy()[0]
        tau = self.models['arm_reach' if self.latched else 'arm_hold'](t(arm_x)).cpu().numpy()[0]
        if not (np.isfinite(w).all() and np.isfinite(tau).all()):
            raise FloatingPointError('Nonfinite model output')
        return w, tau
