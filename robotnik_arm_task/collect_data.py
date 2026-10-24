import csv
import os
from multiprocessing import Process
from pathlib import Path

import numpy as np

NUM_TARGETS = 200
SEED = 42
OUTPUT_DIR = Path(__file__).parent / "data"


def record(target, output):
    os.environ["ROBOENV_HEADLESS"] = "1"
    import robotnik_arm_task as simulation

    simulation.TARGET_EE_POS = target
    base_controller = simulation.base_controller
    arm_controller = simulation.arm_controller
    base_data = []
    wheel_cmds = []

    with output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["time", "target_x", "target_y", "target_z", "base_x", "base_y", "base_z", "base_bearing"]
            + [f"q_mes_{i}" for i in range(4, 11)]
            + [f"qd_mes_{i}" for i in range(4, 11)]
            + [f"wheel_cmds_{i}" for i in range(4)]
            + [f"torque_arm_{i}" for i in range(7)]
            + ["ee_error"]
        )

        # Capture the base measurements and the wheel commands computed from them.
        def record_base(state, base_pos, base_bearing, goal_pos, goal_bearing, current_time, params):
            nonlocal base_data, wheel_cmds
            wheel_cmds = base_controller(state, base_pos, base_bearing, goal_pos, goal_bearing, current_time, params)
            base_data = [current_time, *target, *base_pos, base_bearing]
            return wheel_cmds

        # The arm controller runs next, so all values for this step are now available.
        def record_arm(model, q_pin, q_arm, qd_arm, q_des, q_hold, target_pos, ee_pos, *args, **kwargs):
            result = arm_controller(model, q_pin, q_arm, qd_arm, q_des, q_hold, target_pos, ee_pos, *args, **kwargs)
            writer.writerow([
                *base_data, *q_arm, *qd_arm, *wheel_cmds, *result[0], np.linalg.norm(target_pos - ee_pos),
            ])
            return result

        simulation.base_controller = record_base
        simulation.arm_controller = record_arm
        simulation.main()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(SEED)
    with (OUTPUT_DIR / "targets.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "target_x", "target_y", "target_z", "seed"])
        for episode in range(NUM_TARGETS):
            angle = rng.uniform(-np.pi, np.pi)
            radius = rng.uniform(0.9, 1.8)
            target = np.array([radius * np.cos(angle), radius * np.sin(angle), rng.uniform(0.3, 0.9)])
            writer.writerow([episode, *target, SEED])
            output = OUTPUT_DIR / f"episode_{episode:04d}.csv"
            print(f"Episode {episode + 1}/{NUM_TARGETS}: target={target}", flush=True)

            # Run one simulation at a time, each in a fresh process.
            process = Process(target=record, args=(target, output))
            process.start()
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"Episode {episode} failed")
    print(f"Saved data to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
