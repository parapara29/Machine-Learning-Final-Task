import csv
import os
from multiprocessing import Process
from pathlib import Path

import numpy as np

from model_features import STATE_COLUMNS, OUTPUT_COLUMNS

NUM_TARGETS = 2000
SEED = 42
OUTPUT_DIR = Path(__file__).parent / "data_randomized"
BASE_XY_RANGE = 0.25
ARM_ANGLE_RANGE = 0.1
RECORD_EVERY = 10


def record(target, output, base_pose, arm_offsets):
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
            nonlocal base_data, wheel_cmds
            wheel_cmds = base_controller(state, base_pos, base_bearing, goal_pos, goal_bearing, current_time, params)
            base_data = (current_time, base_pos, base_bearing, goal_pos, goal_bearing)
            return wheel_cmds

        def record_arm(model, q_pin, q_arm, qd_arm, q_des, q_hold, target_pos, ee_pos, *args, **kwargs):
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(SEED)
    with (OUTPUT_DIR / "targets.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["episode", "target_x", "target_y", "target_z", "base_offset_x", "base_offset_y", "base_yaw"]
            + [f"arm_offset_{i}" for i in range(4, 11)] + ["seed"]
        )
        for episode in range(NUM_TARGETS):
            angle = rng.uniform(-np.pi, np.pi)
            radius = rng.uniform(0.9, 1.8)
            target = np.array([radius * np.cos(angle), radius * np.sin(angle), rng.uniform(0.3, 0.9)])
            base_pose = np.array([*rng.uniform(-BASE_XY_RANGE, BASE_XY_RANGE, 2), rng.uniform(-np.pi, np.pi)])
            arm_offsets = rng.uniform(-ARM_ANGLE_RANGE, ARM_ANGLE_RANGE, 7)
            writer.writerow([episode, *target, *base_pose, *arm_offsets, SEED])
            output = OUTPUT_DIR / f"episode_{episode:04d}.csv"
            print(f"Episode {episode + 1}/{NUM_TARGETS}: target={target}", flush=True)

            # Run one simulation at a time, each in a fresh process.
            process = Process(target=record, args=(target, output, base_pose, arm_offsets))
            process.start()
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"Episode {episode} failed")
    print(f"Saved data to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
