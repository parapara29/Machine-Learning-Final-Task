"""Train the V6 policy: wheel MLP + phase-gated arm MLPs on egocentric features.

Data = original demonstrations (training split only) + any number of DAgger rollout folders.
* Every source is resampled to 10 ms rows and converted to V6 features (v6_features.py).
* Fixed-target rollouts (target equals targets.csv for that episode ID) are used only for
  training-split IDs; random-target rollouts are always usable. 5 % of the DAgger rollouts
  (chosen by a hash of the file name) are held out for validation alongside the demonstration
  validation split. Early stopping uses the mean of the two validation MSEs.
* Parsed rollouts are cached as .npz under --cache-dir, so later DAgger rounds only parse
  new files.

Pipeline of this script:
  load_data()      demonstrations + DAgger rollouts -> per-episode arrays (raw features, labels)
  wheel_arrays()   egocentric inputs + wheel labels, at most --max-rows-per-episode rows per episode
  arm_arrays()     the same for one arm phase (HOLD or REACH) + recovery weights + previous rows
  fit()            AdamW training with progress bars, LR reduction, early stopping, best checkpoint
  main()           trains wheel -> arm_hold -> arm_reach and writes v6_metadata.json

Example:
  python train_v6.py --dagger-dirs arm_dagger_v6_r0 arm_dagger_v6_r1 \
      --output-dir trained_model_dagger_v6_r0
"""
import argparse
import json
import random
import shutil
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from dagger_common import FEATURES as RAW, MLP
from v6_features import (ARM_EGO_FEATURES, ARM_OUTPUTS, EGO_FEATURES, LAGS_S, MODEL_VERSION,
                         WHEEL_OUTPUTS, R, arm_matrix, ego_from_raw, resample_10ms)

HIDDEN = [256, 256, 128]     # hidden layer sizes of all three networks
DROPOUT = 0.1
# The demonstration CSVs lack the navigation signals; they are recomputed by _nav_features().
NAV_COLS = ['base_goal_distance', 'base_heading_error', 'base_orientation_error', 'navigation_complete']


# ----------------------------------------------------------------------------- loading
def _nav_features(df, timeout=25.0):
    """Recompute distance / heading error / orientation error / navigation flag for a demo episode."""
    dx, dy = df.goal_x - df.base_x, df.goal_y - df.base_y
    wrap = lambda a: np.arctan2(np.sin(a), np.cos(a))
    d = np.hypot(dx, dy)
    head = wrap(np.arctan2(dy, dx) - df.base_bearing)
    ori = wrap(df.goal_bearing - df.base_bearing)
    done = ((d < 0.08) & (np.abs(ori) < 0.05)) | (df.time > timeout)
    return np.column_stack([d, head, ori, done.astype(float)])


def _episode_arrays(df):
    """Episode DataFrame -> dict of float32 arrays. The navigation flag is latched (running max),
    exactly as the deployed policy latches it."""
    raw = df[RAW].to_numpy(np.float64, copy=True)
    raw[:, R['navigation_complete']] = np.maximum.accumulate(raw[:, R['navigation_complete']] >= 0.5)
    ee_err = (df['ee_error'].to_numpy(np.float64) if 'ee_error' in df.columns else
              np.linalg.norm(df[['target_x', 'target_y', 'target_z']].to_numpy(float)
                             - df[['ee_x', 'ee_y', 'ee_z']].to_numpy(float), axis=1))
    return {'raw': raw.astype(np.float32),
            'wheel': df[WHEEL_OUTPUTS].to_numpy(np.float32),
            'arm': df[ARM_OUTPUTS].to_numpy(np.float32),
            'ee_err': ee_err.astype(np.float32)}


def _cached(path, cache_dir, parse):
    """Parse a CSV once and store the result as .npz; the cache is reused while the CSV's size and
    modification time are unchanged. Unusable files are cached as empty arrays (returns None)."""
    cache = cache_dir / path.parent.name / (path.name + '.npz')
    stamp = f'{path.stat().st_size}-{int(path.stat().st_mtime)}'
    if cache.is_file():
        with np.load(cache) as z:
            if str(z['stamp']) == stamp:
                return {k: z[k] for k in ('raw', 'wheel', 'arm', 'ee_err')} if z['raw'].size else None
    ep = parse(path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if ep is None:
        np.savez(cache, stamp=stamp, raw=np.zeros(0), wheel=np.zeros(0), arm=np.zeros(0), ee_err=np.zeros(0))
    else:
        np.savez(cache, stamp=stamp, **ep)
    return ep


def _parse_demo(path, targets):
    """One demonstration CSV: add its target (from targets.csv), the navigation signals, and drop
    rows with missing values. Demos are already at 10 ms spacing."""
    df = pd.read_csv(path)
    eid = int(path.stem.split('_')[1])
    for k in ('target_x', 'target_y', 'target_z'):
        df[k] = float(targets.loc[eid, k])
    df[NAV_COLS] = _nav_features(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=RAW + WHEEL_OUTPUTS + ARM_OUTPUTS)
    df = resample_10ms(df.reset_index(drop=True))
    return _episode_arrays(df) if len(df) > 4 else None


def _parse_rollout(path):
    """One DAgger rollout CSV: read only the needed columns and resample to 10 ms rows."""
    wanted = set(RAW + WHEEL_OUTPUTS + ARM_OUTPUTS + ['time', 'ee_error'])
    df = pd.read_csv(path, usecols=lambda c: c in wanted, dtype=np.float64)
    missing = wanted - {'ee_error'} - set(df.columns)
    if missing:
        raise ValueError(f'{path.name} lacks {sorted(missing)}')
    df = resample_10ms(df)
    vals = df[RAW + WHEEL_OUTPUTS + ARM_OUTPUTS].to_numpy()
    if len(df) <= 4 or not np.isfinite(vals).all():
        return None
    return _episode_arrays(df)


def load_data(args):
    """Return (train, val_demo, val_dagger, manifest).

    train: demo training episodes + 95 % of the DAgger rollouts; val_demo: the 300 validation demos;
    val_dagger: 5 % of the rollouts, chosen by a CRC32 hash of the file name (order-independent).
    Rollouts whose target equals a validation/test demo target are excluded from training.
    """
    targets = pd.read_csv(Path(args.data_dir) / 'targets.csv').set_index('episode')
    split = json.loads(Path(args.split).read_text())
    train_ids, val_ids = set(split['train']), set(split['validation'])
    cache_dir = Path(args.cache_dir)
    train, val_demo, val_dagger, manifest = [], [], [], {}

    demo_files = [p for p in sorted(Path(args.data_dir).glob('episode_*.csv'))
                  if int(p.stem.split('_')[1]) in train_ids | val_ids]
    for p in tqdm(demo_files, desc='demos', unit='file', dynamic_ncols=True):
        ep = _cached(p, cache_dir, lambda q: _parse_demo(q, targets))
        if ep is None:
            continue
        (train if int(p.stem.split('_')[1]) in train_ids else val_demo).append(ep)
    manifest['demonstrations'] = {'train': len(train), 'validation': len(val_demo)}

    for folder in args.dagger_dirs:
        files = sorted(Path(folder).glob('dagger_*.csv'))
        used = held_out_split = held_out_dagger_val = skipped = 0
        for p in tqdm(files, desc=Path(folder).name, unit='file', dynamic_ncols=True):
            eid = int(p.name.split('_')[1])
            ep = _cached(p, cache_dir, _parse_rollout)
            if ep is None:
                skipped += 1; continue
            target = ep['raw'][0, [R['target_x'], R['target_y'], R['target_z']]]
            fixed = eid in targets.index and np.allclose(
                target, targets.loc[eid, ['target_x', 'target_y', 'target_z']].to_numpy(float), atol=1e-4)
            if fixed and eid not in train_ids:
                held_out_split += 1; continue          # never train on validation/test targets
            if zlib.crc32(p.name.encode()) % 20 == 0:
                val_dagger.append(ep); held_out_dagger_val += 1
            else:
                train.append(ep); used += 1
        manifest[folder] = {'train': used, 'dagger_validation': held_out_dagger_val,
                            'excluded_heldout_targets': held_out_split, 'skipped_unusable': skipped}
        tqdm.write(f'{folder}: {manifest[folder]}')
    return train, val_demo, val_dagger, manifest


# ----------------------------------------------------------------------------- datasets
def wheel_arrays(episodes, cap, rng):
    """Inputs/labels for the wheel net; cap = max random rows per episode (0 = all rows)."""
    xs, ys = [], []
    for ep in episodes:
        idx = np.arange(len(ep['raw']))
        if cap and len(idx) > cap:
            idx = np.sort(rng.choice(idx, cap, replace=False))
        xs.append(ego_from_raw(ep['raw'][idx])); ys.append(ep['wheel'][idx])
    return np.concatenate(xs), np.concatenate(ys)


def arm_arrays(episodes, phase, cap, rng, args):
    """Inputs/labels for one arm net. Returns (x, y, w, x_prev):
    w[:, 0] = recovery weight (1 + recovery_weight near the target in REACH),
    w[:, 1] = 1 if the previous row exists and is in the same phase (smoothness term mask),
    x_prev = input of the previous row (10 ms earlier) for the smoothness penalty."""
    xs, ys, ws, xp = [], [], [], []
    for ep in episodes:
        flag = ep['raw'][:, R['navigation_complete']] >= 0.5
        idx = np.flatnonzero(flag if phase == 'reach' else ~flag)
        if cap and len(idx) > cap:
            idx = np.sort(rng.choice(idx, cap, replace=False))
        if not len(idx):
            continue
        full = arm_matrix(ego_from_raw(ep['raw']), ep['raw'])
        prev = np.maximum(idx - 1, 0)
        w = np.where((ep['ee_err'][idx] < args.recovery_radius) & (phase == 'reach'),
                     1.0 + args.recovery_weight, 1.0)
        same = (idx > 0) & (flag[prev] == flag[idx])
        xs.append(full[idx]); xp.append(full[prev]); ys.append(ep['arm'][idx])
        ws.append(np.column_stack([w, same]).astype(np.float32))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws), np.concatenate(xp)


# ----------------------------------------------------------------------------- training
def mse(model, x, y, device, bs=8192):
    """Unweighted mean squared error over all outputs (evaluation mode, no dropout)."""
    model.eval(); tot = 0.0
    with torch.no_grad():
        for i in range(0, len(x), bs):
            d = model(torch.from_numpy(x[i:i + bs]).to(device)) - torch.from_numpy(y[i:i + bs]).to(device)
            tot += torch.sum(d ** 2).item()
    return tot / max(y.size, 1)


def fit(name, train, vals, args, device, dest, init=None):
    """train = (x, y[, w, xprev]); vals = {label: (x, y)}; early stopping on mean val MSE."""
    x, y = train[0], train[1]
    w = train[2] if len(train) > 2 else np.ones((len(x), 2), np.float32)
    xp = train[3] if len(train) > 3 else x
    smooth = args.smoothness if len(train) > 3 else 0.0
    model = MLP(x.shape[1], y.shape[1], HIDDEN, DROPOUT).to(device)
    if init is not None and init.is_file():
        model.load_state_dict(torch.load(init, map_location=device, weights_only=True))
        tqdm.write(f'{name}: warm start from {init}')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    loader = DataLoader(TensorDataset(*(torch.from_numpy(a) for a in (x, y, w, xp))),
                        batch_size=args.batch_size, shuffle=True, pin_memory=device.type == 'cuda')
    best, wait, history = float('inf'), 0, []
    epochs = tqdm(range(1, args.epochs + 1), desc=f'{name}', unit='epoch', dynamic_ncols=True)
    for epoch in epochs:
        model.train(); total = n = 0
        for xb, yb, wb, xpb in tqdm(loader, desc=f'  epoch {epoch}', leave=False, unit='batch', dynamic_ncols=True):
            xb, yb, wb, xpb = (t.to(device, non_blocking=True) for t in (xb, yb, wb, xpb))
            pred = model(xb)
            # weighted MSE (recovery weight) + lambda * |f(x_t) - f(x_{t-1})|^2 (smooth torques)
            loss = (((pred - yb) ** 2).mean(1) * wb[:, 0]).mean()
            if smooth > 0:
                loss = loss + smooth * (((pred - model(xpb)) ** 2).mean(1) * wb[:, 1]).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(xb); n += len(xb)
        # validation: demo hold-out and DAgger hold-out; model selection on their mean
        scores = {k: mse(model, vx, vy, device) for k, (vx, vy) in vals.items() if len(vx)}
        val = float(np.mean(list(scores.values())))
        sched.step(val)
        history.append({'epoch': epoch, 'train_objective': total / n, 'val_mean_mse': val,
                        **{f'val_{k}_mse': v for k, v in scores.items()}, 'lr': opt.param_groups[0]['lr']})
        if val < best - 1e-7:
            best, wait = val, 0
            torch.save(model.state_dict(), dest / f'{name}.pth')
        else:
            wait += 1
        epochs.set_postfix(train=f'{total / n:.4f}', val=f'{val:.4f}', best=f'{best:.4f}', wait=wait)
        if wait >= args.patience:
            tqdm.write(f'{name}: early stop at epoch {epoch} (best val {best:.5f})')
            break
    pd.DataFrame(history).to_csv(dest / f'{name}_training_history.csv', index=False)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', default='../data_randomized')
    ap.add_argument('--split', default='episode_split.json')
    ap.add_argument('--dagger-dirs', nargs='*', default=[])
    ap.add_argument('--output-dir', default='trained_model_dagger_v6')
    ap.add_argument('--init-from', default='', help='Warm start from an earlier V6 checkpoint folder')
    ap.add_argument('--cache-dir', default='cache_v6')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--max-rows-per-episode', type=int, default=1500,
                    help='Random rows per episode (per phase for the arm nets), at 10 ms spacing')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--recovery-radius', type=float, default=0.35)
    ap.add_argument('--recovery-weight', type=float, default=4.0)
    ap.add_argument('--smoothness', type=float, default=0.01)
    ap.add_argument('--models', nargs='*', default=['wheel', 'arm_hold', 'arm_reach'],
                    choices=['wheel', 'arm_hold', 'arm_reach'])
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dest = Path(args.output_dir); dest.mkdir(parents=True, exist_ok=True)
    init = Path(args.init_from) if args.init_from else None
    print(f'Device: {device} | output: {dest}')

    train, val_demo, val_dagger, manifest = load_data(args)
    # Train the three networks one after the other (each on its own input/label arrays)
    print(f'Training episodes: {len(train)} | validation: {len(val_demo)} demos + {len(val_dagger)} DAgger rollouts')
    rng = np.random.default_rng(args.seed)
    best = {}
    if 'wheel' in args.models:
        tr = wheel_arrays(train, args.max_rows_per_episode, rng)
        vals = {'demo': wheel_arrays(val_demo, 0, rng)}
        if val_dagger: vals['dagger'] = wheel_arrays(val_dagger, 0, rng)
        print(f'wheel: {len(tr[0]):,} training rows')
        best['wheel'] = fit('wheel', tr, vals, args, device, dest, init and init / 'wheel.pth')
        del tr
    for phase in ('hold', 'reach'):
        name = f'arm_{phase}'
        if name not in args.models:
            continue
        tr = arm_arrays(train, phase, args.max_rows_per_episode, rng, args)
        vals = {'demo': arm_arrays(val_demo, phase, 0, rng, args)[:2]}
        if val_dagger: vals['dagger'] = arm_arrays(val_dagger, phase, 0, rng, args)[:2]
        print(f'{name}: {len(tr[0]):,} training rows')
        best[name] = fit(name, tr, vals, args, device, dest, init and init / f'{name}.pth')
        del tr

    if init is not None:   # models not retrained this run are carried over unchanged
        for name in {'wheel', 'arm_hold', 'arm_reach'} - set(args.models):
            shutil.copy2(init / f'{name}.pth', dest / f'{name}.pth')
    shutil.copy2(args.split, dest / 'episode_split.json')
    meta = {'model_version': MODEL_VERSION, 'wheel_inputs': EGO_FEATURES, 'arm_inputs': ARM_EGO_FEATURES,
            'wheel_outputs': WHEEL_OUTPUTS, 'arm_outputs': ARM_OUTPUTS, 'hidden_dims': HIDDEN,
            'dropout': DROPOUT, 'history_lags_s': list(LAGS_S), 'row_spacing_s': 0.01,
            'normalization': 'none', 'best_validation_mse': best, 'data': manifest,
            'args': vars(args)}
    (dest / 'v6_metadata.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps({'best_validation_mse': best}, indent=2))
    print('Saved V6 policy to', dest)


if __name__ == '__main__':
    main()
