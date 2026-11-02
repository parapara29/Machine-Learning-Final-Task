"""Phase-gated, temporal, direct-torque imitation policy. Raw features only."""
from collections import deque
from pathlib import Path
import json
import numpy as np
import torch
from dagger_common import FEATURES, OUTPUTS, MLP

LAGS=(10,30)  # history in simulator timesteps (1 ms each)
STATE_COLS=[f'q_mes_{i}' for i in range(4,11)]+[f'qd_mes_{i}' for i in range(4,11)]
ARM_FEATURES=FEATURES+[f'{c}_lag{lag}' for lag in LAGS for c in STATE_COLS]

class ArmHistory:
    def __init__(self): self.states=deque(maxlen=max(LAGS)+1)
    def push(self, x):
        state=np.asarray(x,dtype=np.float32)
        if state.shape!=(len(FEATURES),): raise ValueError('Expected raw base features')
        self.states.append(state.copy())
    def vector(self):
        if not self.states: raise RuntimeError('Call push before vector')
        now=self.states[-1]
        return np.concatenate([now]+[self.states[max(0,len(self.states)-1-lag)][[FEATURES.index(c) for c in STATE_COLS]] for lag in LAGS]).astype(np.float32)

def arm_history_matrix(df):
    x=df[FEATURES].to_numpy(dtype=np.float32)
    idx=[FEATURES.index(c) for c in STATE_COLS]
    n=len(x)
    return np.concatenate([x]+[x[np.maximum(np.arange(n)-lag,0)][:,idx] for lag in LAGS],axis=1)

class V4Policy:
    def __init__(self, model_dir, device='cpu'):
        self.device=torch.device(device)
        folder=Path(model_dir)
        self.models={}
        for name,ins,outs in [('wheel_mlp',FEATURES,OUTPUTS['wheel_mlp']),
                              ('arm_hold_v4',ARM_FEATURES,OUTPUTS['torque_mlp']),
                              ('arm_reach_v4',ARM_FEATURES,OUTPUTS['torque_mlp'])]:
            meta=json.loads((folder/f'{name}_metadata.json').read_text())
            if meta['input_features']!=ins or meta['output_features']!=outs or meta.get('normalization')!='none':
                raise ValueError(f'{name}: incompatible feature contract or scaling')
            model=MLP(len(ins),len(outs),meta.get('hidden_dims',[256,256,128]),meta.get('dropout',0.1))
            model.load_state_dict(torch.load(folder/f'{name}.pth',map_location='cpu',weights_only=True))
            model.to(self.device).eval()
            self.models[name]=model
        self.history=ArmHistory()
        self.navigation_latched = False
    @torch.no_grad()
    def predict(self,x):
        x = np.asarray(x, dtype=np.float32).copy()

        nav_idx = FEATURES.index('navigation_complete')

        if x[nav_idx] >= 0.5:
            self.navigation_latched = True

        x[nav_idx] = float(self.navigation_latched)

        self.history.push(x)
        arm_x = self.history.vector()

        phase = (
            'arm_reach_v4'
            if self.navigation_latched
            else 'arm_hold_v4'
        )
        w=self.models['wheel_mlp'](torch.as_tensor(x,dtype=torch.float32,device=self.device).reshape(1,-1)).cpu().numpy()[0]
        tau=self.models[phase](torch.as_tensor(arm_x,dtype=torch.float32,device=self.device).reshape(1,-1)).cpu().numpy()[0]
        if not (np.isfinite(w).all() and np.isfinite(tau).all()): raise FloatingPointError('Nonfinite model output')
        return w,tau
