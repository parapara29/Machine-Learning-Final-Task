"""Shared raw 34-feature contract and independent PyTorch policies."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

FEATURES = ['base_x','base_y','base_z','base_bearing','target_x','target_y','target_z',
            'ee_x','ee_y','ee_z','goal_x','goal_y','goal_bearing',
            *[f'q_mes_{i}' for i in range(4,11)],
            *[f'qd_mes_{i}' for i in range(4,11)],
            *[f'q_init_{i}' for i in range(4,11)],
            'base_goal_distance','base_heading_error','base_orientation_error','navigation_complete']
OUTPUTS = {'wheel_mlp':[f'wheel_cmds_{i}' for i in range(4)],
           'torque_mlp':[f'torque_arm_{i}' for i in range(7)]}

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(256,256,128), dropout=.1):
        super().__init__()
        layers=[]; n=input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(n,h),nn.ReLU(),nn.Dropout(dropout)]); n=h
        layers.append(nn.Linear(n,output_dim)); self.network=nn.Sequential(*layers)
    def forward(self,x): return self.network(x)

def wrap(x):
    return float(np.arctan2(np.sin(x), np.cos(x)))

def navigation_features(base, bearing, goal, goal_bearing, elapsed=0.):
    delta=np.asarray(goal)[:2]-np.asarray(base)[:2]
    distance=float(np.linalg.norm(delta))
    heading=wrap(float(np.arctan2(delta[1],delta[0]))-bearing)
    orientation=wrap(goal_bearing-bearing)
    # This is an observable, deployable navigation-stage estimate, NOT expert internal state.
    complete=float((distance < 0.08 and abs(orientation)<0.05) or elapsed>25.)
    return [distance, heading, orientation, complete]

def features(base, bearing, target, ee, goal, goal_bearing, q, qd, qinit, elapsed=0.):
    vals=[*base[:3],bearing,*target[:3],*ee[:3],*goal[:2],goal_bearing,
          *q[4:11],*qd[4:11],*qinit[4:11],
          *navigation_features(base,bearing,goal,goal_bearing,elapsed)]
    arr=np.asarray(vals,dtype=np.float32)
    if arr.shape!=(len(FEATURES),) or not np.isfinite(arr).all():
        raise ValueError(f'Invalid features: shape={arr.shape} finite={np.isfinite(arr).all()}')
    return arr

class Policy:
    def __init__(self, model_dir, device='cpu'):
        self.device=torch.device(device)
        self.models={}
        for name,outputs in OUTPUTS.items():
            folder=Path(model_dir)
            meta=json.loads((folder/f'{name}_metadata.json').read_text())
            if meta['input_features']!=FEATURES or meta['output_features']!=outputs:
                raise ValueError(f'{name}: incompatible feature contract; retrain with train_dagger.py')
            if meta.get('normalization') not in (None,'none'):
                raise ValueError(f'{name}: scaled model supplied, but this pipeline uses raw inputs')
            model=MLP(len(FEATURES),len(outputs),meta.get('hidden_dims',[256,256,128]),meta.get('dropout',.1))
            weights=torch.load(folder/f'{name}.pth',map_location='cpu',weights_only=True)
            model.load_state_dict(weights); model.to(self.device).eval()
            self.models[name]=model
    @torch.no_grad()
    def predict(self,x):
        t=torch.from_numpy(np.asarray(x,dtype=np.float32).reshape(1,len(FEATURES))).to(self.device)
        results={name:model(t).cpu().numpy()[0] for name,model in self.models.items()}
        if not all(np.isfinite(v).all() for v in results.values()):
            raise FloatingPointError('MLP generated nonfinite commands')
        return results['wheel_mlp'],results['torque_mlp']
