"""Run the trained V6 DAgger policy in the PyBullet GUI on one (random or given) target and spawn.

This is a small deployment wrapper around the main simulation loop in
robotnik_dagger_collect.py. It resolves either a fixed target or randomized target/spawn,
loads the trained checkpoint, and then calls the same expert/learner control loop used by the
main DAgger collector.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from robotnik_dagger_collect import (DEFAULT_SPAWN_XY, DEFAULT_TARGET_RADIUS, DEFAULT_TARGET_Z,
                                     sample_random_spawn, sample_random_target)


def resolve_target(args, parser, rng):
    # The target is interpreted in the world frame, not the robot base frame. This is why the
    # later controller computes a base standoff pose relative to the target before driving.
    if args.random_target:
        target = sample_random_target(rng, x_range=args.target_x_range, y_range=args.target_y_range,
                                      z_range=tuple(args.target_z_range), radius_range=tuple(args.target_radius_range))
        print('Randomized end-effector target (world frame, m):', target.tolist(), flush=True)
        return target
    if args.target is None:
        parser.error('Provide --target X Y Z or enable --random-target.')
    target = np.asarray(args.target, dtype=np.float64)
    if not np.all(np.isfinite(target)):
        parser.error('All target coordinates must be finite.')
    return target


def resolve_spawn(args, parser, rng):
    if args.spawn_position is not None:
        spawn = np.asarray(args.spawn_position, dtype=np.float64)
        if spawn.size not in (2, 3):
            parser.error('--spawn-position must contain x y or x y yaw.')
        if not np.all(np.isfinite(spawn)):
            parser.error('All spawn coordinates must be finite.')
        return spawn
    if args.random_spawn:
        spawn = sample_random_spawn(rng, x_range=tuple(args.spawn_x_range), y_range=tuple(args.spawn_y_range), yaw_range=tuple(args.spawn_yaw_range))
        print('Randomized base spawn pose (x, y, yaw):', spawn.tolist(), flush=True)
        return spawn
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                        help='End-effector target in world coordinates (metres)')
    parser.add_argument('--random-target', action='store_true',
                        help='Sample a random end-effector target from the configured ranges.')
    parser.add_argument('--target-radius-range', nargs=2, type=float, default=DEFAULT_TARGET_RADIUS, metavar=('MIN', 'MAX'),
                        help='Horizontal distance of the target from the world origin (default 0.4 1.8)')
    parser.add_argument('--target-x-range', nargs=2, type=float, default=None, metavar=('MIN', 'MAX'),
                        help='Box sampling instead of radius sampling (give together with --target-y-range)')
    parser.add_argument('--target-y-range', nargs=2, type=float, default=None, metavar=('MIN', 'MAX'))
    parser.add_argument('--target-z-range', nargs=2, type=float, default=DEFAULT_TARGET_Z, metavar=('MIN', 'MAX'))
    parser.add_argument('--spawn-position', nargs=3, type=float, metavar=('X', 'Y', 'YAW'),
                        help='Initial robot base pose in world coordinates (metres + radians).')
    parser.add_argument('--random-spawn', action='store_true',
                        help='Randomize the robot base spawn pose in the configured x/y/yaw ranges.')
    parser.add_argument('--spawn-x-range', nargs=2, type=float, default=DEFAULT_SPAWN_XY, metavar=('MIN', 'MAX'))
    parser.add_argument('--spawn-y-range', nargs=2, type=float, default=DEFAULT_SPAWN_XY, metavar=('MIN', 'MAX'))
    parser.add_argument('--spawn-yaw-range', nargs=2, type=float, default=(-np.pi, np.pi), metavar=('MIN', 'MAX'))
    parser.add_argument('--model-dir', default='trained_model_dagger_v6')
    parser.add_argument('--config', default='robotnik_arm.json')
    parser.add_argument('--output-dir', default='custom_target_demo')
    parser.add_argument('--episode-id', type=int, default=0,
                        help='Identifier for output filenames only')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--gui', dest='gui', action='store_true', default=True)
    parser.add_argument('--headless', dest='gui', action='store_false')
    parser.add_argument('--save-trajectory', action='store_true')
    parser.add_argument('--max-steps', type=int, default=60000)
    parser.add_argument('--nav-timeout', type=float, default=45.0)
    parser.add_argument('--tolerance', type=float, default=0.05)
    parser.add_argument('--hold-steps', type=int, default=30)
    parser.add_argument('--max-abs-joint', type=float, default=15.0)
    parser.add_argument('--max-joint-velocity', type=float, default=8.0)
    parser.add_argument('--velocity-guard-consecutive-steps', type=int, default=1)
    parser.add_argument('--velocity-jump-threshold', type=float, default=2.0)
    parser.add_argument('--arm-damping', type=float, default=0.0)
    parser.add_argument('--log-contacts', action='store_true')
    args = parser.parse_args()
    if (args.target_x_range is None) != (args.target_y_range is None):
        parser.error('--target-x-range and --target-y-range must be given together')

    rng = np.random.default_rng(args.seed)
    target = resolve_target(args, parser, rng)
    spawn_pose = resolve_spawn(args, parser, rng)
    if not np.all(np.isfinite(target)):
        parser.error('All target coordinates must be finite.')
    if args.max_steps < 1 or args.hold_steps < 1:
        parser.error('--max-steps and --hold-steps must be positive.')
    if args.arm_damping < 0:
        parser.error('--arm-damping must be nonnegative.')
    if args.velocity_guard_consecutive_steps < 1:
        parser.error('--velocity-guard-consecutive-steps must be positive.')
    if args.tolerance <= 0:
        parser.error('--tolerance must be positive.')

    # Parameters accessed by robotnik_dagger_collect.run(). The existing
    # controller remains the single source of truth for inference and physics.
    args.mode = 'deploy'
    args.wheel_source = 'learner'
    args.arm_source = 'learner'
    args.beta = 0.0
    args.mix_interval_steps = 100
    args.recovery_radius = 0.35
    args.recovery_weight = 4.0
    args.log_every = 1
    args.stop_rule = 'hold'

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        parser.error(f'Model directory does not exist: {model_dir}')
    if not (model_dir / 'v6_metadata.json').exists():
        parser.error(f'No V6 checkpoint (v6_metadata.json) in {model_dir}')

    if spawn_pose is not None:
        print('Initial robot base pose (x, y, yaw):', spawn_pose.tolist(), flush=True)
    print('Custom end-effector target (world frame, m):', target.tolist(), flush=True)
    print('Using trained model:', model_dir, flush=True)
    from robotnik_dagger_collect import run
    report = run(args, target, args.episode_id, args.seed, initial_base_position=spawn_pose)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / f'custom_target_{args.episode_id:06d}_summary.json'
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print('\n--- Simulation result ---')
    print(json.dumps(report, indent=2, default=str))
    print('Summary saved to:', report_path)
    print('SUCCESS' if report['success'] else 'NOT REACHED', '|', report['stop_reason'])


if __name__ == '__main__':
    main()
