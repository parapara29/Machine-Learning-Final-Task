"""Run the existing Robotnik V5.3-A policy on a custom world-frame EE target.

Place alongside robotnik_dagger_collect.py and run from that project directory.
"""
import argparse
import json
from pathlib import Path
import numpy as np

from robotnik_dagger_collect import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', nargs=3, type=float, required=True,
                        metavar=('X', 'Y', 'Z'),
                        help='End-effector target in world coordinates (metres)')
    parser.add_argument('--model-dir', default='trained_model_dagger_v5_3_A')
    parser.add_argument('--config', default='robotnik_arm.json')
    parser.add_argument('--output-dir', default='custom_target_demo')
    parser.add_argument('--episode-id', type=int, default=0,
                        help='Identifier for output filenames only')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--gui', dest='gui', action='store_true', default=True)
    parser.add_argument('--headless', dest='gui', action='store_false')
    parser.add_argument('--save-trajectory', action='store_true')
    parser.add_argument('--max-steps', type=int, default=30000)
    parser.add_argument('--tolerance', type=float, default=0.05)
    parser.add_argument('--hold-steps', type=int, default=30)
    parser.add_argument('--max-abs-joint', type=float, default=15.0)
    parser.add_argument('--max-joint-velocity', type=float, default=8.0)
    parser.add_argument('--velocity-guard-consecutive-steps', type=int, default=1)
    parser.add_argument('--velocity-jump-threshold', type=float, default=2.0)
    parser.add_argument('--arm-damping', type=float, default=0.0)
    parser.add_argument('--log-contacts', action='store_true')
    args = parser.parse_args()

    target = np.asarray(args.target, dtype=np.float64)
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

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        parser.error(f'Model directory does not exist: {model_dir}')
    if not (model_dir / 'arm_hold_v4.pth').exists():
        parser.error(f'Expected V4/V5.3 arm checkpoint missing: {model_dir / "arm_hold_v4.pth"}')

    print('Custom end-effector target (world frame, m):', target.tolist(), flush=True)
    print('Using trained model:', model_dir, flush=True)
    report = run(args, target, args.episode_id, args.seed)
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
