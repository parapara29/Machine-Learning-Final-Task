"""Collect the expert demonstrations used to train the DAgger policy (-> data_randomized/).

Each episode runs the expert simulation in robotnik_arm_task.py (headless, in a fresh process) with
  * a random target: uniform direction, horizontal radius 0.9-1.8 m from the origin, height 0.3-0.9 m;
  * a random base start pose: x, y offset within +-0.25 m of the origin, any yaw;
  * random arm start angles: each joint offset within +-0.1 rad (kept 0.05 rad inside its limits).
The expert's commands and the robot state are recorded every 10th simulation step (100 Hz).

Outputs in --output-dir:
  targets.csv          one row per episode: target, base offset/yaw, arm offsets, seed
  episode_XXXX.csv     time, the 34 raw state columns, 7 arm torques, 4 wheel commands, ee_error

Usage (from robotnik_arm_task/):
  python collect_data.py                                   # 2000 episodes -> data_randomized/
  python collect_data.py --num-episodes 5 --output-dir /tmp/demo_check
The DAgger pipeline (robotnik_dagger_v5/) reads ../data_randomized and splits it 1400/300/300.
"""
import argparse
import csv
import os
from multiprocessing import Process
from pathlib import Path

import numpy as np

BASE_XY_RANGE = 0.25    # m, base start offset in x and y
ARM_ANGLE_RANGE = 0.1   # rad, start offset per arm joint
RECORD_EVERY = 10       # the simulation runs at 1 kHz; record every 10th step (100 Hz)

# Column layout of every episode CSV (identical to robotnik_dagger_v5/dagger_common.FEATURES[:34]
# followed by the expert outputs). Joints 0..3 are the wheels, 4..10 the arm, 11..16 the fingers.
STATE_COLUMNS = (
    ["base_x", "base_y", "base_z", "base_bearing", "target_x", "target_y", "target_z",
     "ee_x", "ee_y", "ee_z", "goal_x", "goal_y", "goal_bearing"]
    + [f"q_mes_{i}" for i in range(4, 11)]
    + [f"qd_mes_{i}" for i in range(4, 11)]
    + [f"q_init_{i}" for i in range(4, 11)]
)
OUTPUT_COLUMNS = [f"torque_arm_{i}" for i in range(7)] + [f"wheel_cmds_{i}" for i in range(4)]


def record(target, output, base_pose, arm_offsets):
    """Run one expert episode and write its CSV.

    The simulation module is imported inside the worker process, and its controllers and
    initial-state hook are wrapped, so that every command the expert sends is also logged.
    """
    os.environ["ROBOENV_HEADLESS"] = "1"
    import robotnik_arm_task as simulation

    simulation.TARGET_EE_POS = target
    base_controller = simulation.base_controller
    arm_controller = simulation.arm_controller
    configure_fingers = simulation.configure_gripper_fingers
    base_data = []
    wheel_cmds = []
    step = 0

    with output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time"] + STATE_COLUMNS + OUTPUT_COLUMNS + ["ee_error"])

        def set_initial_state(sim, client, rid, fixed_angles):
            # Replaces configure_gripper_fingers (called once at start-up): applies the random base
            # pose and arm offsets, then configures the fingers as usual.
            robot = sim.bot[0]
            position = np.array(robot.link_floating_base_pos, dtype=float)
            position[:2] += base_pose[:2]
            robot.link_floating_base_pos = position.tolist()
            robot.link_floating_base_ori = client.getQuaternionFromEuler([0, 0, base_pose[2]])
            angles = np.array(sim.GetInitMotorAngles(), dtype=float)
            arm = angles[4:11]
            lower, upper = (np.asarray(limits)[4:11] for limits in sim.GetBotJointsLimit())
            limited = lower < upper
            # Start from a valid nominal pose, then perturb within the limits.
            arm[limited] = np.clip(arm[limited], lower[limited] + 0.05, upper[limited] - 0.05)
            arm += arm_offsets
            arm[limited] = np.clip(arm[limited], lower[limited] + 0.05, upper[limited] - 0.05)
            robot.init_joint_angles = angles.tolist()
            sim.ResetPose()
            return configure_fingers(sim, client, rid, fixed_angles)

        def record_base(state, base_pos, base_bearing, goal_pos, goal_bearing, current_time, params):
            # Wraps the expert base controller: remember its wheel command and the base state.
            nonlocal base_data, wheel_cmds
            wheel_cmds = base_controller(state, base_pos, base_bearing, goal_pos, goal_bearing, current_time, params)
            base_data = (current_time, base_pos, base_bearing, goal_pos, goal_bearing)
            return wheel_cmds

        def record_arm(model, q_pin, q_arm, qd_arm, q_des, q_hold, target_pos, ee_pos, *args, **kwargs):
            # Wraps the expert arm controller (called once per step, after the base controller):
            # writes one CSV row every RECORD_EVERY steps with the state and both expert commands.
            nonlocal step
            torque, q_des, qd_des = arm_controller(model, q_pin, q_arm, qd_arm, q_des, q_hold, target_pos, ee_pos, *args, **kwargs)
            if step % RECORD_EVERY == 0:
                time, base_pos, bearing, goal_pos, goal_bearing = base_data
                writer.writerow([
                    time, *base_pos, bearing, *target, *ee_pos, *goal_pos, goal_bearing,
                    *q_arm, *qd_arm, *q_hold, *torque, *wheel_cmds, np.linalg.norm(target_pos - ee_pos),
                ])
            step += 1
            return torque, q_des, qd_des

        simulation.configure_gripper_fingers = set_initial_state
        simulation.base_controller = record_base
        simulation.arm_controller = record_arm
        simulation.main()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-episodes", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default=str(Path(__file__).parent / "data_randomized"))
    args = ap.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)   # never overwrite an existing dataset
    rng = np.random.default_rng(args.seed)
    with (output_dir / "targets.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["episode", "target_x", "target_y", "target_z", "base_offset_x", "base_offset_y", "base_yaw"]
            + [f"arm_offset_{i}" for i in range(4, 11)] + ["seed"]
        )
        for episode in range(args.num_episodes):
            # Draw order matters for reproducibility: target, base pose, arm offsets.
            angle = rng.uniform(-np.pi, np.pi)
            radius = rng.uniform(0.9, 1.8)
            target = np.array([radius * np.cos(angle), radius * np.sin(angle), rng.uniform(0.3, 0.9)])
            base_pose = np.array([*rng.uniform(-BASE_XY_RANGE, BASE_XY_RANGE, 2), rng.uniform(-np.pi, np.pi)])
            arm_offsets = rng.uniform(-ARM_ANGLE_RANGE, ARM_ANGLE_RANGE, 7)
            writer.writerow([episode, *target, *base_pose, *arm_offsets, args.seed])
            output = output_dir / f"episode_{episode:04d}.csv"
            print(f"Episode {episode + 1}/{args.num_episodes}: target={target}", flush=True)

            # One simulation per fresh process: PyBullet leaks memory when reused across episodes.
            process = Process(target=record, args=(target, output, base_pose, arm_offsets))
            process.start()
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"Episode {episode} failed")
    print(f"Saved data to {output_dir}")


if __name__ == "__main__":
    main()
