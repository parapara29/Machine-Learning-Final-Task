"""Train phase-specific temporal direct torque MLPs; preserve the validated V2 wheel model.
No scaling or normalization. Never train on held-out episode IDs.
"""
import argparse
import json
import random
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from dagger_common import OUTPUTS, MLP
from arm_v4 import ARM_FEATURES, arm_history_matrix
from train_dagger import read_episodes
import re

ARM_OUTPUTS=OUTPUTS['torque_mlp']

def arrays(episodes, phase, cap, seed, args):
    rng=np.random.default_rng(seed)
    xs=[]; ys=[]; ws=[]; xp=[]; counts=[]
    for _,df in episodes:
        if len(df)<2: continue
        df = df.copy()

        raw_nav = (
            df["navigation_complete"].to_numpy(dtype=float) >= 0.5
        )

        phase_flag = np.maximum.accumulate(raw_nav)

        df["navigation_complete"] = phase_flag.astype(np.float32)

        mask = phase_flag if phase == "reach" else ~phase_flag
        indices=np.flatnonzero(mask)
        if cap>0 and len(indices)>cap: indices=np.sort(rng.choice(indices,cap,replace=False))
        if not len(indices): continue
        full_x=arm_history_matrix(df)
        x=full_x[indices]
        xp_i=full_x[np.maximum(indices-1,0)]
        y=df[ARM_OUTPUTS].to_numpy(np.float32)[indices]
        if 'ee_error' in df.columns:
            errors=df['ee_error'].to_numpy(dtype=float)[indices]
        else:
            errors=np.linalg.norm(df[['target_x','target_y','target_z']].to_numpy(dtype=float)[indices] - df[['ee_x','ee_y','ee_z']].to_numpy(dtype=float)[indices],axis=1)
        w=np.where((errors<args.recovery_radius)&(phase=='reach'),1.+args.recovery_weight,1.).astype(np.float32)
        # Smoothness only for consecutive rows within this episode and phase.
        valid_previous=(indices>0)&(phase_flag[np.maximum(indices-1,0)]==phase_flag[indices])
        w_prev=valid_previous.astype(np.float32)
        finite=np.isfinite(x).all(axis=1)&np.isfinite(y).all(axis=1)
        if finite.any():
            xs.append(x[finite]); ys.append(y[finite]); ws.append(np.column_stack([w[finite],w_prev[finite]])); xp.append(xp_i[finite]); counts.append(int(finite.sum()))
    if not xs: raise ValueError(f'No {phase} samples: check navigation_complete phase and episodes')
    return np.concatenate(xs),np.concatenate(ys),np.concatenate(ws),np.concatenate(xp),counts

def train_phase(phase, train_eps, val_eps, args, dest, device):
    x,y,weights,xprev,_=arrays(train_eps,phase,args.max_rows_per_episode,args.seed,args)
    xv,yv,_,_,_=arrays(val_eps,phase,args.max_rows_per_episode,args.seed+1,args)
    print(f'{phase}: train {len(x):,}, validation {len(xv):,}, features {x.shape[1]}')
    model=MLP(len(ARM_FEATURES),7,hidden_dims=[256,256,128],dropout=.1).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-5)
    loader=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(weights),torch.from_numpy(xprev)),batch_size=args.batch_size,shuffle=True)
    val_loader=DataLoader(TensorDataset(torch.from_numpy(xv),torch.from_numpy(yv)),batch_size=args.batch_size)
    best=float('inf'); wait=0; history=[]
    name=f'arm_{phase}_v4'  # Retain V4 names for drop-in V4Policy compatibility
    for epoch in range(args.epochs):
        model.train(); total=0.; n=0
        for xb,yb,wb,xpb in loader:
            xb=xb.to(device); yb=yb.to(device); wb=wb.to(device); xpb=xpb.to(device)
            pred=model(xb)
            imitation=((pred-yb)**2).mean(dim=1)
            loss=(imitation*wb[:,0]).mean()
            if args.smoothness>0:
                prev_pred=model(xpb)
                loss=loss+args.smoothness*((pred-prev_pred)**2).mean(dim=1).mul(wb[:,1]).mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            total+=loss.item()*len(xb); n+=len(xb)
        model.eval(); vsum=0.; vn=0
        with torch.no_grad():
            for xb,yb in val_loader:
                pred=model(xb.to(device)); delta=pred-yb.to(device)
                vsum+=torch.sum(delta**2).item(); vn+=delta.numel()
        vmse=vsum/vn
        history.append({'epoch':epoch+1,'train_mse':total/n,'validation_mse':vmse})
        if vmse<best-1e-7:
            best=vmse; wait=0; torch.save(model.state_dict(),dest/f'{name}.pth')
        else: wait+=1
        if epoch==0 or (epoch+1)%5==0: print(f'  {name} epoch {epoch+1}: train={total/n:.5f}, val={vmse:.5f}')
        if wait>=args.patience: break
    meta={'input_features':ARM_FEATURES,'output_features':ARM_OUTPUTS,'hidden_dims':[256,256,128],
          'dropout':.1,'normalization':'none','phase':phase,'history_lags_steps':[10,30],
          'best_validation_mse':best,'model_version':'arm_temporal_phase_v5_dagger'}
    (dest/f'{name}_metadata.json').write_text(json.dumps(meta,indent=2))
    pd.DataFrame(history).to_csv(dest/f'{name}_training_history.csv',index=False)
    return best

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',default='../data_randomized')
    ap.add_argument('--dagger-dir',default='')
    ap.add_argument('--wheel-model-dir',default='trained_model_dagger_v2')
    ap.add_argument('--output-dir',default='trained_model_dagger_v4')
    ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--patience',type=int,default=15)
    ap.add_argument('--batch-size',type=int,default=1024)
    ap.add_argument('--lr',type=float,default=5e-4)
    ap.add_argument('--max-rows-per-episode',type=int,default=3000)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--recovery-radius',type=float,default=.35)
    ap.add_argument('--recovery-weight',type=float,default=4.)
    ap.add_argument('--smoothness',type=float,default=.01)
    args=ap.parse_args()
    if args.recovery_radius<=0 or args.recovery_weight<0 or args.smoothness<0: ap.error('Invalid recovery/smoothness settings')
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    src=Path(args.wheel_model_dir); dest=Path(args.output_dir)
    dest.mkdir(parents=True,exist_ok=True)
    for filename in ('wheel_mlp.pth','wheel_mlp_metadata.json','episode_split.json'):
        if not (src/filename).is_file(): raise FileNotFoundError(src/filename)
    episodes=read_episodes(args.data_dir)
    split=json.loads((src/'episode_split.json').read_text())
    train_ids=set(split['train']); val_ids=set(split['validation']); test_ids=set(split['test'])
    if train_ids&val_ids or train_ids&test_ids or val_ids&test_ids: raise ValueError('Overlapping split IDs')
    train_eps=[(eid,df) for eid,df in episodes if eid in train_ids]
    val_eps=[(eid,df) for eid,df in episodes if eid in val_ids]
    if not train_eps or not val_eps: raise ValueError('Missing training or validation episodes')
    if args.dagger_dir:
        extras=[]; excluded=0
        for path in sorted(Path(args.dagger_dir).glob('dagger_*.csv')):
            match=re.match(r'dagger_(\d+)(?:_|\.)',path.name)
            if not match: continue
            eid=int(match.group(1))
            if eid not in train_ids:
                excluded+=1; continue
            df=pd.read_csv(path)
            required=set(ARM_FEATURES[:38]+ARM_OUTPUTS)
            if not required.issubset(df.columns):
                raise ValueError(f'{path} missing {sorted(required-set(df.columns))}')
            # Preserve temporal row spacing; never drop individual rows before computing lags.
            df=df.replace([np.inf,-np.inf],np.nan)
            if not np.isfinite(df[list(required)].to_numpy(dtype=float)).all():
                print(f'Skipping nonfinite DAgger trajectory: {path.name}'); continue
            df=df.reset_index(drop=True)
            if len(df)>1: extras.append((eid,df))
        print(f'Loaded {len(extras)} arm DAgger rollouts; excluded {excluded} held-out rollouts')
        train_eps+=extras
    for filename in ('wheel_mlp.pth','wheel_mlp_metadata.json','episode_split.json'):
        if (src/filename).resolve()!=(dest/filename).resolve(): shutil.copy2(src/filename,dest/filename)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:',device,'Train episode IDs:',len(train_ids),'Validation:',len(val_ids),'Test held out:',len(test_ids))
    results={phase:train_phase(phase,train_eps,val_eps,args,dest,device) for phase in ('hold','reach')}
    (dest/'arm_v4_validation.json').write_text(json.dumps(results,indent=2))
    print('Saved V5-trained arm models and unchanged V2 wheel weights:',dest)

if __name__=='__main__': main()
