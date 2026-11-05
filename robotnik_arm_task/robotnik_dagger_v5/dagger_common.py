"""Shared raw feature contract (38 inputs, 11 outputs) and the MLP used by every network.

This module is the contract boundary between the data pipeline and the learned policies:
all CSV trajectories, training scripts, and deployment code are expected to agree on the
same 38 input features and the same 11 output command names. Changing this schema without
updating the models and metadata files would silently break training and inference.
"""
import numpy as np
import torch
from torch import nn

# The policy sees only the current state and the deployment-observable navigation phase.
# No hidden expert state is provided. This keeps the learned controller portable and
# means the recorded features must be sufficient for both driving and arm control.
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

def navigation_features(base, bearing, goal, goal_bearing, elapsed=0., timeout=25.):
    delta=np.asarray(goal)[:2]-np.asarray(base)[:2]
    distance=float(np.linalg.norm(delta))
    heading=wrap(float(np.arctan2(delta[1],delta[0]))-bearing)
    orientation=wrap(goal_bearing-bearing)
    # This is an observable, deployable navigation-stage estimate, NOT expert internal state.
    complete=float((distance < 0.08 and abs(orientation)<0.05) or elapsed>timeout)
    return [distance, heading, orientation, complete]

def features(base, bearing, target, ee, goal, goal_bearing, q, qd, qinit, elapsed=0., timeout=25.):
    vals=[*base[:3],bearing,*target[:3],*ee[:3],*goal[:2],goal_bearing,
          *q[4:11],*qd[4:11],*qinit[4:11],
          *navigation_features(base,bearing,goal,goal_bearing,elapsed,timeout)]
    arr=np.asarray(vals,dtype=np.float32)
    if arr.shape!=(len(FEATURES),) or not np.isfinite(arr).all():
        raise ValueError(f'Invalid features: shape={arr.shape} finite={np.isfinite(arr).all()}')
    return arr
