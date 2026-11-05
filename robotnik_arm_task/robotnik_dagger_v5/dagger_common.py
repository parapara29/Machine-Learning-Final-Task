"""Shared raw feature contract (38 inputs, 11 outputs) and the MLP used by every network.

This module is the contract boundary between the data pipeline and the learned policies:
all CSV trajectories, training scripts, and deployment code are expected to agree on the
same 38 input features and the same 11 output command names. Changing this schema without
updating the models and metadata files would silently break training and inference.

Who uses what:
  FEATURES / features()  -> robotnik_dagger_collect.py logs these 38 raw values every step;
                            v6_features.py turns them into the egocentric network inputs.
  OUTPUTS                -> column names of the expert labels (wheel speeds, arm torques).
  MLP                    -> the network class of all three V6 networks (wheel, arm HOLD, arm REACH).
"""
import numpy as np
import torch
from torch import nn

# The policy sees only the current state and the deployment-observable navigation phase.
# No hidden expert state is provided. This keeps the learned controller portable and
# means the recorded features must be sufficient for both driving and arm control.
# Order matters: it is the column order of every CSV and of the vector returned by features().
FEATURES = ['base_x','base_y','base_z','base_bearing','target_x','target_y','target_z',   # base pose (yaw = bearing), target
            'ee_x','ee_y','ee_z','goal_x','goal_y','goal_bearing',                       # end-effector, base goal pose
            *[f'q_mes_{i}' for i in range(4,11)],                                        # 7 arm joint angles (rad)
            *[f'qd_mes_{i}' for i in range(4,11)],                                       # 7 arm joint velocities (rad/s)
            *[f'q_init_{i}' for i in range(4,11)],                                       # arm angles at episode start
            'base_goal_distance','base_heading_error','base_orientation_error','navigation_complete']
# Network outputs = expert labels: 4 wheel angular velocities (rad/s) and 7 arm joint torques (N*m).
OUTPUTS = {'wheel_mlp':[f'wheel_cmds_{i}' for i in range(4)],
           'torque_mlp':[f'torque_arm_{i}' for i in range(7)]}

class MLP(nn.Module):
    """Fully connected network: [Linear -> ReLU -> Dropout] per hidden layer, then a linear output.

    Default 256 -> 256 -> 128 hidden units with 10 % dropout (active only in training mode).
    No output activation: the outputs are unbounded regression targets (speeds / torques).
    """
    def __init__(self, input_dim, output_dim, hidden_dims=(256,256,128), dropout=.1):
        super().__init__()
        layers=[]; n=input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(n,h),nn.ReLU(),nn.Dropout(dropout)]); n=h
        layers.append(nn.Linear(n,output_dim)); self.network=nn.Sequential(*layers)
    def forward(self,x): return self.network(x)

def wrap(x):
    """Wrap an angle to (-pi, pi] so that the shortest rotation is used (179 deg vs -179 deg = 2 deg)."""
    return float(np.arctan2(np.sin(x), np.cos(x)))

def navigation_features(base, bearing, goal, goal_bearing, elapsed=0., timeout=25.):
    """Distance to the base goal, heading error toward it, final orientation error, and the
    navigation-complete flag (within 8 cm and 0.05 rad of the goal pose, or past the timeout)."""
    delta=np.asarray(goal)[:2]-np.asarray(base)[:2]
    distance=float(np.linalg.norm(delta))
    heading=wrap(float(np.arctan2(delta[1],delta[0]))-bearing)   # turn needed to face the goal point
    orientation=wrap(goal_bearing-bearing)                        # turn needed to reach the final heading
    # This is an observable, deployable navigation-stage estimate, NOT expert internal state.
    complete=float((distance < 0.08 and abs(orientation)<0.05) or elapsed>timeout)
    return [distance, heading, orientation, complete]

def features(base, bearing, target, ee, goal, goal_bearing, q, qd, qinit, elapsed=0., timeout=25.):
    """Assemble the 38 raw features (order = FEATURES) for one simulation step.

    q, qd, qinit are full motor vectors (wheels 0..3, arm 4..10); only the arm part is used.
    Raises if the vector has the wrong length or contains NaN/inf (a broken simulation state).
    """
    vals=[*base[:3],bearing,*target[:3],*ee[:3],*goal[:2],goal_bearing,
          *q[4:11],*qd[4:11],*qinit[4:11],
          *navigation_features(base,bearing,goal,goal_bearing,elapsed,timeout)]
    arr=np.asarray(vals,dtype=np.float32)
    if arr.shape!=(len(FEATURES),) or not np.isfinite(arr).all():
        raise ValueError(f'Invalid features: shape={arr.shape} finite={np.isfinite(arr).all()}')
    return arr
