#!/usr/bin/env python3
"""Evaluate a Robotnik V5.3 deployment trajectory, optionally running a rollout.

This script is a post-hoc analysis tool. It checks whether a saved trajectory reached the
requested target, computes sustained-reach metrics, and optionally profiles CPU/GPU usage while
running a deployment command. It is useful for benchmarking the final model quality and for
reproducing the published evaluation numbers.

Examples:
  python evaluate_robotnik_v53.py --trajectory custom_target_demo/deploy_000000.csv \
      --model-dir trained_model_dagger_v5_3_A --target 0.5 1.2 0.6
  python evaluate_robotnik_v53.py --trajectory custom_target_demo/deploy_000000.csv \
      --model-dir trained_model_dagger_v5_3_A --target 0.5 1.2 0.6 \
      --run-command pixi run -e ml python run_robotnik_v53.py --target 0.5 1.2 0.6 \
          --model-dir trained_model_dagger_v5_3_A --save-trajectory

The optional run command must write the trajectory specified by --trajectory.
CPU utilization is sampled for the process tree if psutil is installed. GPU utilization
requires nvidia-smi. These are utilization estimates, NOT energy or FLOP measurements.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trajectory', required=True, type=Path, help='Deployment trajectory CSV')
    ap.add_argument('--model-dir', type=Path, default=Path('trained_model_dagger_v5_3_A'))
    ap.add_argument('--target', nargs=3, type=float, metavar=('X','Y','Z'), help='World-frame EE target in metres')
    ap.add_argument('--output-dir', type=Path, default=Path('evaluation_metrics'))
    ap.add_argument('--tolerance', type=float, default=0.05, help='Euclidean EE tolerance in metres')
    ap.add_argument('--hold-steps', type=int, default=30, help='Consecutive steps inside tolerance and nav_done')
    ap.add_argument('--sample-period', type=float, default=0.2, help='Wall-clock sampling period for resources')
    ap.add_argument('--run-command', nargs=argparse.REMAINDER, help='Optional deployment command, must generate --trajectory')
    a = ap.parse_args()
    if a.tolerance <= 0 or a.hold_steps < 1 or a.sample_period <= 0:
        ap.error('tolerance, hold-steps, and sample-period must be positive')
    return a


def run_and_profile(command, sample_period):
    if command and command[0] == '--': command = command[1:]
    if not command: raise ValueError('--run-command needs an executable command')
    try:
        import psutil
    except ImportError:
        psutil = None
    has_gpu = shutil.which('nvidia-smi') is not None
    stats = []
    start = time.perf_counter()
    proc = subprocess.Popen(command)
    try:
        root = psutil.Process(proc.pid) if psutil else None
        while proc.poll() is None:
            item = {'wall_time_s': time.perf_counter()-start}
            if root:
                try:
                    group = [root] + root.children(recursive=True)
                    item['rss_mb'] = sum(p.memory_info().rss for p in group if p.is_running()) / 2**20
                    item['cpu_percent_one_core_100'] = sum(p.cpu_percent(interval=None) for p in group if p.is_running())
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            if has_gpu:
                try:
                    r = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=1)
                    if r.returncode == 0:
                        pairs = [line.split(',') for line in r.stdout.strip().splitlines() if line.strip()]
                        item['gpu_util_percent_system'] = max(float(p[0]) for p in pairs)
                        item['gpu_memory_mb_system'] = sum(float(p[1]) for p in pairs)
                except (OSError, subprocess.TimeoutExpired, ValueError):
                    pass
            stats.append(item)
            time.sleep(sample_period)
        rc = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try: proc.wait(timeout=5)
        except subprocess.TimeoutExpired: proc.kill()
        raise
    return {'exit_code': rc, 'wall_time_s': time.perf_counter()-start, 'samples': stats,
            'resource_notes': 'CPU is process-tree utilization; GPU values are device-wide, not process-specific.'}


def checkpoint_complexity(model_dir):
    try:
        import torch
    except ImportError as exc:
        return {'warning': f'PyTorch unavailable: {exc}'}
    if not model_dir.is_dir():
        return {'warning': f'Model directory not found: {model_dir}'}
    results = {}
    for file in sorted(list(model_dir.glob('*.pth')) + list(model_dir.glob('*.pt'))):
        try:
            try: data = torch.load(file, map_location='cpu', weights_only=True)
            except TypeError: data = torch.load(file, map_location='cpu')
            # Handle common checkpoint wrappers; avoid double-counting nested tensors.
            state = data
            if isinstance(data, dict):
                for key in ('state_dict', 'model_state_dict', 'model', 'network', 'policy_state_dict'):
                    if key in data and isinstance(data[key], dict):
                        state = data[key]; break
            if not isinstance(state, dict):
                results[file.name] = {'warning':'Checkpoint is not a state dictionary'}; continue
            tensors = {str(k): v for k,v in state.items() if isinstance(v, torch.Tensor)}
            if not tensors:
                results[file.name] = {'warning':'No direct tensor weights found; nested format may need inspection'}; continue
            params = sum(v.numel() for v in tensors.values())
            bytes_total = sum(v.numel()*v.element_size() for v in tensors.values())
            linear_layers = []
            macs = 0
            for key, weight in tensors.items():
                if key.endswith('weight') and weight.ndim == 2:
                    out_dim, in_dim = weight.shape
                    linear_layers.append({'name':key, 'in':int(in_dim), 'out':int(out_dim)})
                    macs += int(in_dim*out_dim)
            results[file.name] = {'tensor_elements':int(params), 'tensor_bytes':int(bytes_total),
                'file_bytes':file.stat().st_size, 'linear_layers':linear_layers,
                'dense_linear_macs_per_forward_estimate':macs if linear_layers else None,
                'approx_dense_linear_flops_per_forward':2*macs if linear_layers else None,
                'note':'MACs/FLOPs cover identified dense linear layers only; preprocessing, activation, normalization, routing, and simulation excluded.'}
        except Exception as exc:
            results[file.name] = {'warning':f'Could not inspect: {type(exc).__name__}: {exc}'}
    if not results:
        return {'warning':f'No .pth or .pt checkpoints found in {model_dir}'}
    return results


def first_existing(df, names):
    for name in names:
        if name in df.columns: return name
    return None


def extract_matrix(df, patterns):
    for names in patterns:
        if all(n in df.columns for n in names):
            return df[list(names)].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float), list(names)
    return None, None


def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches='tight')
    plt.close()


def main():
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    resource = None
    if a.run_command:
        resource = run_and_profile(a.run_command, a.sample_period)
        (a.output_dir/'resource_samples.csv').write_text(pd.DataFrame(resource['samples']).to_csv(index=False))
        if resource['exit_code'] != 0:
            raise SystemExit(f"Deployment exited {resource['exit_code']}; see its console output")
    if not a.trajectory.is_file():
        raise SystemExit(f'Trajectory not found: {a.trajectory}. Run deployment with --save-trajectory first.')
    df = pd.read_csv(a.trajectory)
    if df.empty: raise SystemExit('Trajectory CSV has no rows')
    if 'time' not in df: raise SystemExit('Trajectory must have a time column')
    t = pd.to_numeric(df['time'], errors='coerce').to_numpy(dtype=float)
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) < 0):
        raise SystemExit('Invalid or nonmonotonic trajectory times')
    target = np.array(a.target, dtype=float) if a.target else None
    if target is None:
        cols = ['target_x','target_y','target_z']
        if all(c in df for c in cols): target = df[cols].iloc[0].to_numpy(dtype=float)
    ee, ee_names = extract_matrix(df, [
        ('ee_x','ee_y','ee_z'), ('ee_pos_x','ee_pos_y','ee_pos_z'),
        ('ee_world_x','ee_world_y','ee_world_z'), ('end_effector_x','end_effector_y','end_effector_z')])
    err_name = first_existing(df, ['ee_error','ee_dist_error','ee_distance_error','cartesian_error'])
    if ee is not None and target is not None:
        err = np.linalg.norm(ee-target[None,:],axis=1)
        err_source = 'computed from EE xyz and target'
    elif err_name:
        err = pd.to_numeric(df[err_name], errors='coerce').to_numpy(dtype=float)
        err_source = err_name
    else:
        err = None; err_source = None
    if err is not None and not np.all(np.isfinite(err)):
        raise SystemExit('EE error contains non-finite values')
    dt = float(np.median(np.diff(t))) if len(t)>1 else None
    duration = float(t[-1] - t[0]) if len(t)>1 else 0.
    report = {'trajectory':str(a.trajectory), 'samples':len(df), 'time_start_s':float(t[0]),
        'time_last_observation_s':float(t[-1]), 'observed_duration_s':duration,
        'median_timestep_s':dt, 'target_xyz_m':target.tolist() if target is not None else None,
        'ee_coordinate_columns':ee_names, 'error_source':err_source,
        'reach_tolerance_m':a.tolerance, 'required_consecutive_steps':a.hold_steps,
        'compute_profile':resource, 'checkpoint_complexity':checkpoint_complexity(a.model_dir)}
    nav_col = first_existing(df, ['expert_nav_done','learner_nav_complete','nav_done'])
    nav = df[nav_col].fillna(0).to_numpy(dtype=float)>=0.5 if nav_col else None
    if err is not None:
        within = err < a.tolerance
        report.update({'initial_error_m':float(err[0]),'minimum_error_m':float(np.min(err)),
            'final_observed_error_m':float(err[-1]),
            'final_endpoint_within_tolerance':bool(within[-1]),
            'fraction_samples_within_tolerance':float(np.mean(within)),
            'first_time_inside_tolerance_s':float(t[np.flatnonzero(within)[0]]) if within.any() else None,
            'first_time_inside_tolerance_and_nav_done_s':float(t[np.flatnonzero(within & nav)[0]]) if nav is not None and (within & nav).any() else None})
        # Strict benchmark success needs both EE threshold and nav_done, continuously for hold_steps.
        valid = within & nav if nav is not None else within
        counter=0; reach_idx=None
        for i, ok in enumerate(valid):
            counter = counter+1 if ok else 0
            if counter >= a.hold_steps:
                reach_idx=i;break
        report['time_to_sustained_reach_s'] = float(t[reach_idx]) if reach_idx is not None else None
        report['sustained_reach_achieved'] = reach_idx is not None
        report['sustained_reach_uses_nav_done'] = nav is not None
        if nav is None: report['warning_nav'] = 'No nav_done column; sustained reach is EE-only, not benchmark-equivalent.'
        plt.figure(figsize=(10,4.8))
        plt.plot(t,err,label='Euclidean EE error')
        plt.axhline(a.tolerance,ls='--',label=f'Tolerance {a.tolerance:g} m')
        if reach_idx is not None: plt.axvline(t[reach_idx],ls=':',label='Sustained reach')
        plt.xlabel('Simulation time (s)');plt.ylabel('Error (m)');plt.title('End-effector Cartesian distance error')
        plt.grid(alpha=.3);plt.legend();save_plot(a.output_dir/'02_ee_error.png')
        plt.figure(figsize=(10,4))
        plt.plot(t,within.astype(int),label='EE inside tolerance')
        if nav is not None:plt.plot(t,nav.astype(int),label='Navigation complete',alpha=.7)
        plt.ylim(-.1,1.1);plt.xlabel('Simulation time (s)');plt.ylabel('Boolean condition')
        plt.title('Reach and navigation conditions');plt.grid(alpha=.3);plt.legend()
        save_plot(a.output_dir/'05_reach_conditions.png')
    if ee is not None and target is not None:
        fig, axes = plt.subplots(3,1,figsize=(10,8),sharex=True)
        for i,(ax,label) in enumerate(zip(axes,'XYZ')):
            ax.plot(t,ee[:,i],label=f'Actual EE {label}')
            ax.axhline(target[i],ls='--',label=f'Target {label}')
            ax.set_ylabel(f'{label} (m)');ax.legend();ax.grid(alpha=.3)
        axes[-1].set_xlabel('Simulation time (s)')
        fig.suptitle('End-effector Cartesian position tracking')
        save_plot(a.output_dir/'01_ee_xyz.png')
    else:
        report['warning_ee_xyz'] = ('EE xyz tracking plot unavailable: CSV lacks world-frame ee_x, ee_y, ee_z. '
            'Add row.update({"ee_x": float(ee[0]), "ee_y": float(ee[1]), "ee_z": float(ee[2])}) '
            'to robotnik_dagger_collect.py after ee is computed, then rerun deployment.')
    q, q_names = extract_matrix(df, [tuple(f'q_mes_{i}' for i in range(4,11)),
        tuple(f'q_arm_{i}' for i in range(7))])
    if q is not None:
        fig,axes=plt.subplots(4,2,figsize=(12,10),sharex=True)
        for i,ax in enumerate(axes.flat):
            if i<7:
                ax.plot(t,q[:,i]);ax.set_title(f'Arm joint {i+1}');ax.set_ylabel('Position (rad)');ax.grid(alpha=.3)
            else:ax.axis('off')
        for ax in axes[-1]:ax.set_xlabel('Simulation time (s)')
        fig.suptitle('Measured arm joint positions')
        save_plot(a.output_dir/'03_arm_joint_positions.png')
    else:report['warning_joint_positions']='Missing q_mes_4..q_mes_10 in CSV.'
    qd, _ = extract_matrix(df,[tuple(f'qd_mes_{i}' for i in range(4,11))])
    if qd is not None:
        report['peak_abs_arm_joint_velocity_rad_s'] = float(np.max(np.abs(qd)))
        fig,ax=plt.subplots(figsize=(10,5))
        for i in range(7):ax.plot(t,qd[:,i],label=f'J{i+1}',lw=.9)
        ax.set_xlabel('Simulation time (s)');ax.set_ylabel('Velocity (rad/s)')
        ax.set_title('Measured arm joint velocities');ax.legend(ncol=4);ax.grid(alpha=.3)
        save_plot(a.output_dir/'04_arm_joint_velocities.png')
    base, _ = extract_matrix(df,[('base_x','base_y'),('base_pos_x','base_pos_y')])
    if base is not None:
        plt.figure(figsize=(7,6));plt.plot(base[:,0],base[:,1],label='Base trajectory')
        plt.scatter(base[0,0],base[0,1],marker='o',label='Start')
        plt.scatter(base[-1,0],base[-1,1],marker='x',label='End')
        plt.xlabel('Base X (m)');plt.ylabel('Base Y (m)');plt.axis('equal');plt.grid(alpha=.3)
        plt.title('Mobile base XY path');plt.legend();save_plot(a.output_dir/'06_base_path.png')
    if resource and resource['samples']:
        samples=pd.DataFrame(resource['samples'])
        cols=[c for c in ('cpu_percent_one_core_100','rss_mb','gpu_util_percent_system','gpu_memory_mb_system') if c in samples]
        if cols:
            fig,axes=plt.subplots(len(cols),1,figsize=(10,2.5*len(cols)),sharex=True)
            axes=np.atleast_1d(axes)
            for ax,c in zip(axes,cols):ax.plot(samples['wall_time_s'],samples[c]);ax.set_ylabel(c);ax.grid(alpha=.3)
            axes[-1].set_xlabel('Wall time (s)');fig.suptitle('Measured process/device resource usage')
            save_plot(a.output_dir/'07_resource_usage.png')
            report['resource_summary']={c:{'mean':float(samples[c].mean()),'peak':float(samples[c].max())} for c in cols}
    with (a.output_dir/'metrics.json').open('w') as f:json.dump(report,f,indent=2,allow_nan=False)
    with (a.output_dir/'metrics.txt').open('w') as f:
        for k,v in report.items():
            if k not in ('checkpoint_complexity','compute_profile'):f.write(f'{k}: {v}\n')
        f.write('\nCheckpoint complexity:\n'+json.dumps(report['checkpoint_complexity'],indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('checkpoint_complexity','compute_profile')},indent=2))
    print(f'\nSaved plots and metrics in {a.output_dir.resolve()}')

if __name__=='__main__':main()
