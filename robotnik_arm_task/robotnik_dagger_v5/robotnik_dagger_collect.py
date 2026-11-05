"""Collect DAgger expert labels on learner-visited states. Expert state is advanced every step.

This is the data-collection loop used to generate imitation labels for states that the learner
actually encounters during deployment. Instead of replaying an ideal expert trajectory, the
simulation runs the current learner policy and then asks the expert controller for the action
that would be appropriate for that observed state. This is the standard DAgger setup used to
reduce compounding error.
"""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import pybullet as p
import pinocchio as pin
from simulation_and_control import pb, MotorCommands, PinWrapper
from robotnik_arm_control import (ArmControllerParams, BaseControllerParams, BaseNavState,
    arm_controller,base_controller,configure_gripper_fingers,quaternion2bearing)
from dagger_common import FEATURES, OUTPUTS, Policy, features
from arm_v4 import V4Policy

FINGER_ANGLES=np.full(6,.1)
STANDOFF_Z=[.1,.5,1.1,1.25]
STANDOFF=[.8,.65,.65,.35]


def sample_random_target(rng, x_range=(0.4, 1.2), y_range=(-1.0, 1.0), z_range=(0.2, 1.2)):
    """Sample a world-frame end-effector target inside the requested ranges."""
    return np.asarray([
        float(rng.uniform(*x_range)),
        float(rng.uniform(*y_range)),
        float(rng.uniform(*z_range)),
    ], dtype=np.float64)


def sample_random_spawn(rng, x_range=(-1.5, 1.5), y_range=(-1.5, 1.5), yaw_range=(-np.pi, np.pi)):
    """Sample a base spawn position and yaw in world coordinates."""
    return np.asarray([
        float(rng.uniform(*x_range)),
        float(rng.uniform(*y_range)),
        float(rng.uniform(*yaw_range)),
    ], dtype=np.float64)


def run(args, target, episode_id, seed, initial_base_position=None):
    # The simulation loop is the central state machine: it tracks base navigation and arm reach
    # while alternating between expert and learner commands depending on the requested mode.
    np.random.seed(seed)
    rng=np.random.default_rng(seed)
    execute_expert=False
    interval_expert=False
    here=Path(__file__).resolve().parent
    sim=pb.SimInterface(args.config,conf_file_path_ext=str(here),use_gui=args.gui)
    sim.SetFloorFriction(1.0)
    client=sim.pybullet_client; rid=sim.bot[0].bot_pybullet
    if initial_base_position is not None:
        spawn=np.asarray(initial_base_position,dtype=np.float64)
        if spawn.size not in (2,3):
            raise ValueError(f'Initial base pose must contain x/y or x/y/yaw; got {spawn.tolist()}')
        base_xy = spawn[:2]
        yaw = float(spawn[2]) if spawn.size == 3 else 0.0
        client.resetBasePositionAndOrientation(rid, [float(base_xy[0]), float(base_xy[1]), 0.0],
                                               client.getQuaternionFromEuler([0.0, 0.0, yaw]))
        client.resetBaseVelocity(rid, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    for j in range(client.getNumJoints(rid)):
        if 'wheel' in client.getJointInfo(rid,j)[1].decode():
            client.changeDynamics(rid,j,contactStiffness=30000,contactDamping=1000,
                                  anisotropicFriction=[1.,.01,1.])
    sim.bot[0].servo_motor_model.set_motor_gains(400,30)
    configure_gripper_fingers(sim,client,rid,FINGER_ANGLES)
    if args.gui:
        shape=client.createVisualShape(p.GEOM_SPHERE,radius=.03,rgbaColor=[1,0,0,.85])
        client.createMultiBody(baseMass=0,baseVisualShapeIndex=shape,basePosition=target.tolist())
    names=np.expand_dims(np.array(sim.getNameActiveJoints()),axis=0)
    dyn=PinWrapper(args.config,'pybullet',names,['pybullet'],False,0,str(here))
    ee_id=dyn.pin_model.getFrameId('robot_j2s7s300_end_effector')
    shoulder_id=dyn.pin_model.getFrameId('robot_j2s7s300_link_2')
    if ee_id>=len(dyn.pin_model.frames) or shoulder_id>=len(dyn.pin_model.frames):
        raise ValueError('Missing expected end-effector/shoulder frames')
    init_q=np.asarray(sim.GetMotorAngles(0))
    lower,upper=sim.GetBotJointsLimit()
    arm_lower=np.asarray(lower[4:11]); arm_upper=np.asarray(upper[4:11])
    q_des=init_q[4:11].copy()
    base_state=BaseNavState(); base_params=BaseControllerParams(nav_timeout=25.)
    arm_params=ArmControllerParams(reach_limit=.85)
    standoff=float(np.interp(target[2],STANDOFF_Z,STANDOFF))
    bearing=float(np.arctan2(target[1],target[0])) if np.linalg.norm(target[:2])>1e-4 else 0.
    goal=target[:2]-standoff*np.array([np.cos(bearing),np.sin(bearing)])
    need_learner = args.mode in ('collect','learner','deploy','diagnose')
    policy=(V4Policy(args.model_dir,device=args.device) if (Path(args.model_dir)/'arm_hold_v4.pth').exists() else Policy(args.model_dir,device=args.device)) if need_learner else None
    dt=sim.GetTimeStep(); cmd=MotorCommands(); records=[]
    best=float('inf'); stable=0; stop_reason='step_cap'; elapsed=0.
    expert_steps=0; guard_joint=None; guard_velocity=None
    velocity_exceed_steps=0; peak_velocity=0.0; first_exceed_time=None
    contacts_at_anomaly=None
    anomaly=None; previous_row=None
    wheel_source = args.wheel_source or ('expert' if args.mode=='expert' else 'learner' if args.mode in ('deploy','learner') else 'mix')
    arm_source = args.arm_source or ('expert' if args.mode=='expert' else 'learner' if args.mode in ('deploy','learner') else 'mix')
    if args.mode=='collect' and (wheel_source=='mix' or arm_source=='mix') and not 0 <= args.beta <= 1:
        raise ValueError('Invalid DAgger beta')
    if args.mode=='deploy' and (wheel_source!='learner' or arm_source!='learner'):
        raise ValueError('deploy mode is strictly learner-only; use --mode diagnose for hybrid experiments')
    if args.mode=='expert' and (wheel_source!='expert' or arm_source!='expert'):
        raise ValueError('expert mode must use expert for both actuators; use diagnose for hybrid experiments')
    for step in range(args.max_steps):
        base=np.asarray(sim.GetBasePosition()); ori=np.asarray(sim.GetBaseOrientation())
        yaw=quaternion2bearing(ori[3],ori[0],ori[1],ori[2])
        q=np.asarray(sim.GetMotorAngles(0)); qd=np.asarray(sim.GetMotorVelocities(0))
        qpin=dyn.ReoderJoints2PinVec(np.concatenate((base,ori,q)),'pos')
        pin.forwardKinematics(dyn.pin_model,dyn.pin_data,qpin)
        pin.updateFramePlacements(dyn.pin_model,dyn.pin_data)
        ee=dyn.pin_data.oMf[ee_id].translation.copy()
        error=float(np.linalg.norm(target-ee)); best=min(best,error)
        x=features(base,yaw,target,ee,goal,bearing,q,qd,init_q,elapsed)
        # IMPORTANT: advance the stateful expert on *every* observed learner state.
        if args.mode=='deploy':
            # No expert controller calls during deployment.
            expert_w=np.full(4,np.nan)
            expert_tau=np.full(7,np.nan)
            nav_done=(np.linalg.norm(base[:2]-goal)<base_params.pos_tol and
                      abs(np.arctan2(np.sin(bearing-yaw),np.cos(bearing-yaw)))<base_params.ang_tol)
        else:
            expert_w=base_controller(base_state,base,yaw,goal,bearing,elapsed,base_params)
            expert_tau,q_des,qd_des=arm_controller(dyn,qpin,q[4:11],qd[4:11],q_des,
                 init_q[4:11],target,ee,ee_id,shoulder_id,arm_lower,arm_upper,
                 dt,arm_params,reach_active=base_state.done)
            nav_done=base_state.done
        if policy is None:
            learned_w,learned_tau=expert_w,expert_tau
        else:
            learned_w,learned_tau=policy.predict(x)
        if args.mode=='collect' and step % args.mix_interval_steps == 0:
            interval_expert=bool(rng.random()<args.beta)
        def choose(source, expert_cmd, learned_cmd):
            if source=='expert': return expert_cmd
            if source=='learner': return learned_cmd
            if source=='mix': return expert_cmd if interval_expert else learned_cmd
            raise ValueError(f'Unknown source: {source}')
        w=choose(wheel_source,expert_w,learned_w)
        tau=choose(arm_source,expert_tau,learned_tau)
        wheel_expert = wheel_source=='expert' or (wheel_source=='mix' and interval_expert)
        arm_expert = arm_source=='expert' or (arm_source=='mix' and interval_expert)
        execute_expert = wheel_expert and arm_expert
        w = np.clip(w, -3.18, 3.18)

        # Original learned/expert torque command
        tau_raw = np.asarray(tau, dtype=np.float64).copy()

        # Measured arm joint velocities
        arm_qd = np.asarray(qd[4:11], dtype=np.float64)

        # Velocity-dependent damping, applied only when enabled
        if args.arm_damping > 0.0 and arm_source == "learner":
            tau = tau_raw - args.arm_damping * arm_qd
        else:
            tau = tau_raw

        # Preserve the original actuator torque limits
        tau = np.clip(tau, -30.0, 30.0)
        row=dict(zip(FEATURES,x.tolist()))
        # Diagnostic logging of the expert's internal reference.
# These values are NOT supplied to the learned policy.

        if args.mode != "deploy":

            for j in range(7):
                row[f"expert_q_des_{j}"] = float(q_des[j])
                row[f"expert_qd_des_{j}"] = float(qd_des[j])

                row[f"expert_position_error_{j}"] = float(
                    q_des[j] - q[4 + j]
                )

                row[f"expert_position_torque_{j}"] = float(
                    arm_params.kp * (q_des[j] - q[4 + j])
                )

                row[f"expert_velocity_torque_{j}"] = float(
                    arm_params.kd * (qd_des[j] - qd[4 + j])
                )
        row.update({key:float(expert_w[i]) for i,key in enumerate(OUTPUTS['wheel_mlp'])})
        row.update({key:float(expert_tau[i]) for i,key in enumerate(OUTPUTS['torque_mlp'])})
        row.update({f'pred_wheel_{i}':float(learned_w[i]) for i in range(4)})
        row.update({f'pred_torque_{i}':float(learned_tau[i]) for i in range(7)})
        row.update({f'executed_wheel_{i}':float(w[i]) for i in range(4)})
        row.update({f'executed_torque_{i}':float(tau[i]) for i in range(7)})
        row.update({
            f'raw_torque_{i}': float(tau_raw[i])
            for i in range(7)
        })

        row["arm_damping"] = float(args.arm_damping)
        row.update({'max_abs_arm_velocity':float(np.max(np.abs(qd[4:11]))),
                    'max_abs_arm_angle':float(np.max(np.abs(q[4:11]))),
                    'learner_nav_complete':float(x[-1])})
        row.update({'episode':episode_id,'time':elapsed,'ee_error':error,
                    'executed_expert':int(execute_expert),
                    'wheel_expert':int(wheel_expert),'arm_expert':int(arm_expert),
                    'expert_nav_done':int(nav_done)})
        row['base_goal_distance_logged']=float(np.linalg.norm(base[:2]-goal))
        row['arm_recovery_priority']=float(1.0 + args.recovery_weight * (error < args.recovery_radius) * (nav_done or x[-1] >= .5))
        row['label_phase_complete']=float(x[-1])
        row['contact_count']=int(len(client.getContactPoints(bodyA=rid))) if args.log_contacts else -1
        # Detect discontinuities in observed velocity before sending this step's commands.
        # The observed state was produced by the PREVIOUS simulation step.
        if previous_row is not None:
            prev_vel=np.array([previous_row[f'qd_mes_{j}'] for j in range(4,11)])
            dv=qd[4:11]-prev_vel
            if np.max(np.abs(dv)) > args.velocity_jump_threshold and anomaly is None:
                j=int(np.argmax(np.abs(dv)))
                contacts_at_anomaly=[{'link_a':int(c[3]),'link_b':int(c[4]),
                                     'normal_force':float(c[9]),'other_body':int(c[2])}
                                     for c in client.getContactPoints(bodyA=rid)] if args.log_contacts else None
                anomaly={
                    'time_s':float(elapsed), 'joint':j+1,
                    'velocity_before':float(prev_vel[j]),
                    'velocity_after':float(qd[4+j]),
                    'velocity_jump':float(dv[j]),
                    'previous_executed_torque':float(previous_row[f'executed_torque_{j}']),
                    'current_predicted_torque':float(learned_tau[j]),
                    'previous_executed_expert':int(previous_row['executed_expert']),
                    'previous_wheel_expert':int(previous_row['wheel_expert']),
                    'previous_arm_expert':int(previous_row['arm_expert']),
                    'contacts':contacts_at_anomaly,
                    'base_goal_distance':float(x[FEATURES.index('base_goal_distance')])
                        if 'base_goal_distance' in FEATURES else None,
                }
        records.append(row)
        previous_row=row
        cmd.SetControlCmd(np.concatenate((w,tau,np.zeros(6))),
                          ['velocity']*4+['torque']*13)
        sim.Step(cmd,'torque')
        elapsed+=dt
        stable=stable+1 if error<args.tolerance and nav_done else 0
        if stable>=args.hold_steps:
            stop_reason='reached'; break
        if args.max_abs_joint>0 and np.any(np.abs(q[4:11])>args.max_abs_joint):
            guard_joint=int(np.argmax(np.abs(q[4:11])))+1
            stop_reason='joint_guard'; break
        max_vel=float(np.max(np.abs(qd[4:11])))
        peak_velocity=max(peak_velocity,max_vel)
        if args.max_joint_velocity>0 and max_vel>args.max_joint_velocity:
            velocity_exceed_steps+=1
            if first_exceed_time is None: first_exceed_time=float(elapsed)
            if velocity_exceed_steps>=args.velocity_guard_consecutive_steps:
                guard_velocity=int(np.argmax(np.abs(qd[4:11])))+1
                stop_reason='velocity_guard'; break
        else:
            velocity_exceed_steps=0
            first_exceed_time=None
        if args.gui: time.sleep(.001)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    if args.mode=='collect':
        pd.DataFrame.from_records(records).to_csv(out/f'dagger_{episode_id:06d}_seed{seed}_beta{int(args.beta*100):03d}_{wheel_source}W_{arm_source}A.csv',index=False)
    elif args.save_trajectory:
        suffix=f'_{wheel_source}W_{arm_source}A' if args.mode=='diagnose' else ''
        pd.DataFrame.from_records(records).to_csv(out/f'{args.mode}_{episode_id:06d}{suffix}.csv',index=False)
    if anomaly is not None:
        (out/f'anomaly_{args.mode}_{episode_id:06d}_{wheel_source}W_{arm_source}A.json').write_text(
            json.dumps(anomaly,indent=2))
    if args.gui:
        print("\nSimulation finished.")
        print("Press Ctrl+C in the terminal to close PyBullet.")

        try:
            while client.isConnected():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nClosing simulation.")
    return {'episode':episode_id,'target_x':float(target[0]),'target_y':float(target[1]),
            'target_z':float(target[2]),'seed':seed,'mode':args.mode,'steps':len(records),
            'time_s':elapsed,'final_error_m':error,'min_error_m':best,
            'success':stop_reason=='reached','stop_reason':stop_reason,
            'executed_expert_fraction':float(np.mean([r['executed_expert'] for r in records])),
            'guard_joint':guard_joint,'guard_velocity_joint':guard_velocity,
            'max_arm_velocity_observed':float(max(r['max_abs_arm_velocity'] for r in records)),
             'velocity_jump_anomaly':anomaly,
             'wheel_source':wheel_source,'arm_source':arm_source,
             'peak_arm_velocity':peak_velocity,
             'velocity_guard_consecutive_steps':args.velocity_guard_consecutive_steps}

def _csv_ranges(csv_path, columns):
    path=Path(csv_path)
    if not path.exists():
        return None
    table=pd.read_csv(path)
    missing=[c for c in columns if c not in table.columns]
    if missing:
        return None
    return [(float(table[c].min()), float(table[c].max())) for c in columns]


def _resolve_random_ranges(args):
    default_target_x=( -2.0, 2.0)
    default_target_y=( -2.0, 2.0)
    default_target_z=( 0.2, 1.2)
    default_spawn_x=( -2.0, 2.0)
    default_spawn_y=( -2.0, 2.0)
    default_spawn_yaw=( -np.pi, np.pi)

    csv_path = args.targets_csv if args.targets_csv else '../data_randomized/targets.csv'
    csv_ranges = _csv_ranges(csv_path, ['target_x','target_y','target_z'])
    if args.random_target:
        if args.target_x_range is None:
            args.target_x_range = list(csv_ranges[0]) if csv_ranges is not None else list(default_target_x)
        if args.target_y_range is None:
            args.target_y_range = list(csv_ranges[1]) if csv_ranges is not None else list(default_target_y)
        if args.target_z_range is None:
            args.target_z_range = list(csv_ranges[2]) if csv_ranges is not None else list(default_target_z)

    spawn_ranges = _csv_ranges(csv_path, ['base_offset_x','base_offset_y','base_yaw'])
    if args.random_spawn:
        if args.spawn_x_range is None:
            args.spawn_x_range = list(spawn_ranges[0]) if spawn_ranges is not None else list(default_spawn_x)
        if args.spawn_y_range is None:
            args.spawn_y_range = list(spawn_ranges[1]) if spawn_ranges is not None else list(default_spawn_y)
        if args.spawn_yaw_range is None:
            args.spawn_yaw_range = list(spawn_ranges[2]) if spawn_ranges is not None else list(default_spawn_yaw)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['collect','learner','expert','deploy','diagnose'],default='collect')
    ap.add_argument('--wheel-source',choices=['expert','learner','mix'],default=None,help='Override wheel authority; hybrid tests use diagnose mode')
    ap.add_argument('--arm-source',choices=['expert','learner','mix'],default=None,help='Override arm authority; hybrid tests use diagnose mode')
    ap.add_argument('--recovery-radius',type=float,default=0.35,help='Near-target radius in metres for training priority annotation')
    ap.add_argument('--recovery-weight',type=float,default=4.0,help='Additional near-target sample priority')
    ap.add_argument('--log-contacts',action='store_true',help='Record contact counts and contacts at first velocity anomaly')
    ap.add_argument('--velocity-guard-consecutive-steps',type=int,default=1,help='Consecutive sampled steps over velocity limit before stop; default preserves original guard')
    ap.add_argument('--model-dir',default='trained_model_dagger')
    ap.add_argument('--config',default='robotnik_arm.json')
    ap.add_argument('--targets-csv',default='../data_randomized/targets.csv')
    ap.add_argument('--episode-ids',nargs='*',type=int)
    ap.add_argument('--start-episode',type=int,default=0)
    ap.add_argument('--num-episodes',type=int,default=10)
    ap.add_argument('--output-dir',default='dagger_rollouts')
    ap.add_argument('--beta',type=float,default=.8,help='Probability expert controls each mixing interval')
    ap.add_argument('--mix-interval-steps',type=int,default=100,help='Hold expert/learner choice for this many simulation steps')
    ap.add_argument('--max-steps',type=int,default=20000)
    ap.add_argument('--tolerance',type=float,default=.05)
    ap.add_argument('--hold-steps',type=int,default=30)
    ap.add_argument('--max-abs-joint',type=float,default=15.,help='Safety guard on measured arm angles; 0 disables')
    ap.add_argument('--velocity-jump-threshold',type=float,default=2.0,help='Log first per-step arm velocity jump larger than this rad/s')
    ap.add_argument('--max-joint-velocity',type=float,default=8.,help='Safety guard in rad/s; 0 disables')
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--gui',action='store_true')
    ap.add_argument('--save-trajectory',action='store_true')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--random-target',action='store_true',help='Sample a random target for each episode instead of reading from targets.csv')
    ap.add_argument('--random-spawn',action='store_true',help='Sample a random base start pose for each episode')
    ap.add_argument('--spawn-position',nargs=3,type=float,metavar=('X','Y','YAW'),help='Exact robot start pose in world space (x y yaw)')
    ap.add_argument('--target-x-range',nargs=2,type=float,default=None,metavar=('MIN','MAX'))
    ap.add_argument('--target-y-range',nargs=2,type=float,default=None,metavar=('MIN','MAX'))
    ap.add_argument('--target-z-range',nargs=2,type=float,default=None,metavar=('MIN','MAX'))
    ap.add_argument('--spawn-x-range',nargs=2,type=float,default=None,metavar=('MIN','MAX'))
    ap.add_argument('--spawn-y-range',nargs=2,type=float,default=None,metavar=('MIN','MAX'))
    ap.add_argument('--spawn-yaw-range',nargs=2,type=float,default=None,metavar=('MIN','MAX'))
    ap.add_argument(
    "--arm-damping",
    type=float,
    default=0.0,
    help="Velocity-dependent damping coefficient for learned arm torques"
    )

    args=ap.parse_args()
    if args.arm_damping < 0:
            ap.error("--arm-damping must be non-negative")
    if not 0<=args.beta<=1: ap.error('--beta must be between 0 and 1')
    if args.recovery_radius<=0 or args.recovery_weight<0: ap.error('Invalid recovery radius/weight')
    if args.mix_interval_steps<1: ap.error('--mix-interval-steps must be >= 1')
    if args.velocity_guard_consecutive_steps<1: ap.error('--velocity-guard-consecutive-steps must be >= 1')
    if args.mode=='diagnose' and (args.wheel_source is None or args.arm_source is None):
        ap.error('diagnose mode requires --wheel-source and --arm-source')
    if args.spawn_position is not None and len(args.spawn_position)!=3:
        ap.error('--spawn-position must contain x y yaw')
    _resolve_random_ranges(args)
    rng=np.random.default_rng(args.seed)
    # Collect mode permits independent sources for arm-focused DAgger.
    # Labels always come from the expert, never from executed learner commands.
    target_table=None
    if not args.random_target:
        target_table=pd.read_csv(args.targets_csv).set_index('episode')
    ids=args.episode_ids if args.episode_ids else list(range(args.start_episode,args.start_episode+args.num_episodes))
    reports=[]
    for eid in ids:
        if args.random_target:
            target=sample_random_target(rng,
                x_range=tuple(args.target_x_range),
                y_range=tuple(args.target_y_range),
                z_range=tuple(args.target_z_range))
        else:
            if eid not in target_table.index: raise ValueError(f'No target for episode {eid}')
            target=target_table.loc[eid,['target_x','target_y','target_z']].to_numpy(dtype=float)
        initial_base_position = None
        if args.random_spawn:
            initial_base_position = sample_random_spawn(rng,
                x_range=tuple(args.spawn_x_range),
                y_range=tuple(args.spawn_y_range),
                yaw_range=tuple(args.spawn_yaw_range))
        elif args.spawn_position is not None:
            initial_base_position = np.asarray(args.spawn_position, dtype=np.float64)
        if args.random_target or args.random_spawn:
            print(f'episode {eid}: target={target.tolist()} spawn={None if initial_base_position is None else initial_base_position.tolist()}', flush=True)
        report=run(args,target,int(eid),args.seed+int(eid), initial_base_position=initial_base_position)
        if initial_base_position is not None:
            report['initial_spawn_x'] = float(initial_base_position[0])
            report['initial_spawn_y'] = float(initial_base_position[1])
            report['initial_spawn_yaw'] = float(initial_base_position[2])
        reports.append(report); print(json.dumps(report),flush=True)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    summary=out/f'{args.mode}_summary.csv'
    pd.DataFrame(reports).to_csv(summary,index=False) if not summary.exists() else pd.concat(
        [pd.read_csv(summary),pd.DataFrame(reports)],ignore_index=True).to_csv(summary,index=False)
    print('Success rate:',np.mean([r['success'] for r in reports]))

if __name__=='__main__': main()
