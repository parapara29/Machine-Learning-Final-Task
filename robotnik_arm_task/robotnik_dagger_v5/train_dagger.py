"""Train independent state-only wheel and arm imitation policies without scaling.

This is the core imitation-learning pipeline: load full episodes, create an episode-level
train/validation/test split, train separate MLPs for wheel commands and arm torque, and
save metadata so the runtime loader can validate that a checkpoint matches the expected
feature contract. The split is by episode to avoid leakage between nearby time steps from
the same trajectory.

Usage: python train_models_v2.py --data-dir data_randomized --output-dir trained_model_v2
The split is by episode, and test metrics are evaluated only after training.
"""
import argparse
import glob
import html
import json
import os
import random
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dagger_common import FEATURES as SHARED_FEATURES

from dagger_common import navigation_features
COMMON_FEATURES = SHARED_FEATURES

POLICIES = {
    'wheel_mlp': [f'wheel_cmds_{i}' for i in range(4)],
    'torque_mlp': [f'torque_arm_{i}' for i in range(7)],
}
HIDDEN = [256, 256, 128]
DROPOUT = 0.1

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=HIDDEN, dropout=DROPOUT):
        super().__init__()
        layers = []
        prev = input_dim
        for width in hidden_dims:
            layers += [nn.Linear(prev, width), nn.ReLU(), nn.Dropout(dropout)]
            prev = width
        layers.append(nn.Linear(prev, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def add_phase_features(df):
    """Derive deployment-observable navigation features from measured base and goal."""
    if all(c in df.columns for c in COMMON_FEATURES):
        return df
    required=['base_x','base_y','base_bearing','goal_x','goal_y','goal_bearing']
    if not set(required).issubset(df.columns):
        raise ValueError('Missing base/goal columns required for phase features')
    values=[]
    for idx, row in df.iterrows():
        elapsed=float(row['time']) if 'time' in df.columns else float(idx)*.001
        values.append(navigation_features([row.base_x,row.base_y],row.base_bearing,
                        [row.goal_x,row.goal_y],row.goal_bearing,elapsed))
    for j,col in enumerate(COMMON_FEATURES[-4:]):
        df[col]=np.asarray(values)[:,j]
    return df

def read_episodes(folder):
    # A single CSV episode contains the trajectory of one target-reaching task. We keep each
    # full episode together and never split rows from the same episode across train/val/test.
    folder = Path(folder)
    target_path = folder / 'targets.csv'
    if not target_path.is_file():
        raise FileNotFoundError(f'Missing {target_path}')
    targets = pd.read_csv(target_path)
    required_targets = {'episode', 'target_x', 'target_y', 'target_z'}
    if not required_targets.issubset(targets.columns):
        raise ValueError(f'{target_path} lacks {sorted(required_targets - set(targets.columns))}')
    if targets['episode'].duplicated().any():
        raise ValueError('Duplicate episode IDs in targets.csv')
    targets = targets.set_index('episode')
    files = sorted(folder.glob('episode_*.csv'))
    if not files:
        raise FileNotFoundError(f'No episode_*.csv files in {folder}')
    episodes = []
    required = set(COMMON_FEATURES).union(*map(set, POLICIES.values()))
    for path in files:
        match = re.fullmatch(r'episode_(\d+)\.csv', path.name)
        if not match:
            continue
        eid = int(match.group(1))
        if eid not in targets.index:
            raise ValueError(f'No target entry for {path.name}')
        df = pd.read_csv(path)
        df.columns = [re.sub(r'<[^>]+>', '', html.unescape(str(c))).replace('*','').strip() for c in df.columns]
        for key in ('target_x','target_y','target_z'):
            df[key] = float(targets.loc[eid, key])
        df=add_phase_features(df)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f'{path.name} lacks columns {sorted(missing)}')
        df[list(required)] = df[list(required)].apply(pd.to_numeric, errors='coerce')
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required)).reset_index(drop=True)
        if len(df) >= 2:
            episodes.append((eid, df))
    if len(episodes) < 10:
        raise ValueError(f'Only {len(episodes)} usable episodes; at least 10 recommended')
    print(f'Loaded {len(episodes)} episodes')
    return episodes

def add_dagger_episodes(train_eps, folder, train_ids, max_rows_per_episode=0):
    """Append expert-labelled learner states ONLY for original training IDs.

    This preserves the original validation/test episode holdouts. A DAgger CSV
    must use the same 38 raw input columns and 11 expert label columns.
    """
    if not folder:
        return train_eps
    directory=Path(folder)
    paths=sorted(directory.glob('dagger_*.csv'))
    required=set(COMMON_FEATURES).union(*map(set,POLICIES.values()))
    extra=[]
    skipped=0
    for path in paths:
        match=re.fullmatch(r'dagger_(\d+)(?:_seed\d+_beta\d+)?\.csv',path.name)
        if not match: continue
        eid=int(match.group(1))
        if eid not in train_ids:
            skipped+=1
            continue
        df=add_phase_features(pd.read_csv(path))
        if not required.issubset(df.columns):
            raise ValueError(f'{path}: missing {sorted(required-set(df.columns))}')
        df[list(required)]=df[list(required)].apply(pd.to_numeric,errors='coerce')
        df=df.replace([np.inf,-np.inf],np.nan).dropna(subset=list(required))
        if max_rows_per_episode>0 and len(df)>max_rows_per_episode:
            df=df.iloc[np.linspace(0,len(df)-1,max_rows_per_episode,dtype=int)].copy()
        if len(df)>=2: extra.append((eid,df.reset_index(drop=True)))
    print(f'Loaded {len(extra)} DAgger rollouts for training; excluded {skipped} holdout rollouts')
    return train_eps+extra

def raw_arrays(episodes, outputs):
    x = np.concatenate([df[COMMON_FEATURES].to_numpy(np.float32) for _,df in episodes])
    y = np.concatenate([df[outputs].to_numpy(np.float32) for _,df in episodes])
    return x,y

def metrics(y_true, y_pred, labels):
    result = {}
    for i, name in enumerate(labels):
        a, b = y_true[:,i], y_pred[:,i]
        result[name] = {
            'mae': float(mean_absolute_error(a,b)),
            'rmse': float(np.sqrt(mean_squared_error(a,b))),
            'r2': float(r2_score(a,b)),
        }
    return result

def plot_episode(episode, model, outputs, device, dest):
    eid, df = episode
    x = df[COMMON_FEATURES].to_numpy(np.float32)
    with torch.no_grad():
        pred = model(torch.from_numpy(x).to(device)).cpu().numpy()
    true = df[outputs].to_numpy(np.float32)
    fig, axes = plt.subplots(len(outputs), 1, figsize=(12, 2.2*len(outputs)), sharex=True, squeeze=False)
    for j, name in enumerate(outputs):
        ax = axes[j,0]
        ax.plot(true[:,j], label='Recorded expert', lw=1)
        ax.plot(pred[:,j], label='Predicted', lw=1, alpha=.85)
        ax.set_ylabel(name)
        ax.grid(alpha=.25)
        if j == 0: ax.legend()
    axes[-1,0].set_xlabel('Timestep')
    fig.suptitle(f'Unseen episode {eid}: teacher-forced command predictions')
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)

def train_one(name, outputs, train_eps, val_eps, test_eps, args, device, output_dir):
    # Each independent model is trained on the same raw feature vector but predicts a different
    # actuator output set. Early stopping is based on validation error, while test metrics are
    # reported only after the model has been selected from the validation set.
    print(f'\nTraining {name}: {len(COMMON_FEATURES)} inputs -> {len(outputs)} outputs')
    x_train, y_train = raw_arrays(train_eps, outputs)
    x_val, y_val = raw_arrays(val_eps, outputs)
    x_test, y_test = raw_arrays(test_eps, outputs)
    # Temporal pairs stay within episodes. They are sampled together in a single loader.
    if args.smoothness_weight > 0:
        pair_x1 = np.concatenate([df[COMMON_FEATURES].to_numpy(np.float32)[:-1] for _,df in train_eps])
        pair_x2 = np.concatenate([df[COMMON_FEATURES].to_numpy(np.float32)[1:] for _,df in train_eps])
        pair_loader = DataLoader(TensorDataset(
            torch.from_numpy(pair_x1),
            torch.from_numpy(pair_x2)),
            batch_size=args.batch_size, shuffle=True)
    else:
        pair_loader = None
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train),torch.from_numpy(y_train)),
                        batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val),torch.from_numpy(y_val)),
                            batch_size=args.batch_size, shuffle=False)
    model = MLP(len(COMMON_FEATURES),len(outputs)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=5)
    best, waiting, history = float('inf'), 0, []
    checkpoint = output_dir / f'{name}.pth'
    for epoch in range(args.epochs):
        model.train()
        train_total, train_count = 0., 0
        pair_it = iter(pair_loader) if pair_loader is not None else None
        for xb,yb in loader:
            xb,yb = xb.to(device),yb.to(device)
            optimizer.zero_grad()
            loss = torch.mean((model(xb)-yb)**2)
            if pair_it is not None:
                try: x1,x2 = next(pair_it)
                except StopIteration:
                    pair_it = iter(pair_loader)
                    x1,x2 = next(pair_it)
                x1,x2 = x1.to(device),x2.to(device)
                loss = loss + args.smoothness_weight * torch.mean((model(x2)-model(x1))**2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            train_total += loss.item()*len(xb)
            train_count += len(xb)
        model.eval()
        val_total, val_count = 0., 0
        with torch.no_grad():
            for xb,yb in val_loader:
                xb,yb = xb.to(device),yb.to(device)
                val_total += torch.sum((model(xb)-yb)**2).item()
                val_count += yb.numel()
        val_mse = val_total / max(val_count,1)
        scheduler.step(val_mse)
        history.append({'epoch':epoch+1, 'train_objective':train_total/max(train_count,1), 'val_mse_raw':val_mse})
        if val_mse < best - 1e-8:
            best, waiting = val_mse, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            waiting += 1
        if epoch == 0 or (epoch+1)%5 == 0:
            print(f'epoch {epoch+1:3d} train objective={train_total/max(train_count,1):.5f} val MSE={val_mse:.5f}')
        if waiting >= args.patience:
            print('Early stopping')
            break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.from_numpy(x_test).to(device)).cpu().numpy()
    y_true = y_test
    report = {'model':name,'test_episodes':len(test_eps), 'test_samples':len(y_true),
              'test_metrics_physical_units':metrics(y_true,y_pred,outputs),
              'best_validation_mse_raw':best}
    (output_dir/f'{name}_test_metrics.json').write_text(json.dumps(report,indent=2))
    metadata = {'input_features':COMMON_FEATURES,'output_features':outputs,
                'input_dim':len(COMMON_FEATURES),'output_dim':len(outputs),
                'hidden_dims':HIDDEN,'dropout':DROPOUT,
                'best_validation_loss':best,'model_version':'phase_observable_raw_v4','normalization':'none'}
    (output_dir/f'{name}_metadata.json').write_text(json.dumps(metadata,indent=2))
    pd.DataFrame(history).to_csv(output_dir/f'{name}_training_history.csv',index=False)
    plt.figure(figsize=(8,4))
    plt.plot([h['epoch'] for h in history],[h['val_mse_raw'] for h in history],label='Validation MSE (raw units)')
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
    plt.savefig(output_dir/f'{name}_validation_loss.png',dpi=150); plt.close()
    plot_episode(test_eps[0],model,outputs,device,output_dir/f'{name}_heldout_episode.png')
    print(json.dumps(report,indent=2))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir',default='data_randomized')
    p.add_argument('--dagger-dir',default='',help='Directory of dagger_*.csv labelled rollouts')
    p.add_argument('--max-dagger-rows-per-episode',type=int,default=0)
    p.add_argument('--output-dir',default='trained_model_dagger')
    p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--batch-size',type=int,default=1024)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--patience',type=int,default=20)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--smoothness-weight',type=float,default=0.0,
                   help='Optional temporal penalty. Default 0 to prioritise accurate imitation.')
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:',device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    episodes = read_episodes(args.data_dir)
    train_eps,temp_eps = train_test_split(episodes,test_size=.30,random_state=args.seed)
    val_eps,test_eps = train_test_split(temp_eps,test_size=.50,random_state=args.seed)
    split = {'train':[e for e,_ in train_eps], 'validation':[e for e,_ in val_eps],
             'test':[e for e,_ in test_eps]}
    split_path=output_dir/'episode_split.json'
    if split_path.exists():
        old=json.loads(split_path.read_text())
        if old != split:
            raise ValueError('Existing episode split differs; use original seed or a new output directory')
    else:
        split_path.write_text(json.dumps(split,indent=2))
    print('Episode split:',{k:len(v) for k,v in split.items()})
    train_eps=add_dagger_episodes(train_eps,args.dagger_dir,set(split['train']),args.max_dagger_rows_per_episode)
    for name,outputs in POLICIES.items():
        train_one(name,outputs,train_eps,val_eps,test_eps,args,device,output_dir)
    print('\nSaved both independent models to',output_dir)

if __name__=='__main__':
    main()
