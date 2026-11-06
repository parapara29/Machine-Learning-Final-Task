import os
os.environ["ROBOENV_HEADLESS"] = "1"

import csv
import multiprocessing as mp
import sys
import time
from pathlib import Path
import numpy as np
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pybullet as p
import pinocchio as pin
import rf_sim   # adds robotnik_arm_task to sys.path, so it has to come before robotnik_arm_control
from simulation_and_control import pb, MotorCommands, PinWrapper, wrap_angle
from robotnik_arm_control import StopCriteria, check_stop, configure_gripper_fingers, quaternion2bearing
from features import wheel_features

# Shivansh's random-spawn test (test_combined_random.py, branch shivansh/linear_regression) with the
# linear models replaced by the E4 controller: RF wheels (extra features) + DAgger V5.3-A arm.
# The test cases, the simulation setup, the stop rule and the success rule are unchanged.

NUM_TESTS = 1000  # number of test cases
REACH_TOLERANCE = 0.01    # Stop simulation once EE error <= 0.01 m (same as robotnik_arm_task.py)
SUCCESS_TOLERANCE = 0.05  # Count episode as a success if final EE error <= 0.05 m
RADIUS_MIN = 0.9          # Min target distance from robot start position (m)
RADIUS_MAX = 3.5          # Max target distance from robot start position (m)
SEED = 42  # same seed as Shivansh, so the same cases
NUM_WORKERS = max(1, (os.cpu_count() or 4) // 2)  # parallel processes
max_steps = int(os.environ.get("ROBOENV_MAX_STEPS", "40000"))  # step cap per episode (40 s)
# ==============================================================================

FEATURE_SET = "extra"  # wheel forest of E4
OUT_DIR = rf_sim.RUN_DIR / "eval_random_spawn"  # results of this test
FIELDS = ["episode", "start_x", "start_y", "start_yaw", "target_x", "target_y", "target_z",  # CSV columns
          "final_error", "time", "nav_done", "nav_time", "success", "wheel_ms", "arm_ms", "stop"]

# Global worker variables (initialized once per CPU worker process)
sim = None
dyn_model = None
ee_frame_id = None
default_base_z = 0.0
rf_wheel = dagger_arm = dagger_features = None


def init_worker():
    """Setup PyBullet, Pinocchio, and trained models once in each parallel worker."""
    global sim, dyn_model, ee_frame_id, default_base_z
    global rf_wheel, dagger_arm, dagger_features

    import torch
    torch.set_num_threads(1)
    conf_file_name = "robotnik_arm.json"
    cur_dir = str(rf_sim.ARM_TASK_DIR)

    rf_wheel = joblib.load(rf_sim.RUN_DIR / FEATURE_SET / "rf_wheel.joblib")
    rf_wheel.n_jobs = 1
    dagger_arm, dagger_features = rf_sim.load_dagger_arm()

    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=False)
    sim.SetFloorFriction(1.0)
    client = sim.pybullet_client
    rid = sim.bot[0].bot_pybullet
    default_base_z = float(sim.bot[0].link_floating_base_pos[2])

    wheel_joint_ids = [
        j for j in range(client.getNumJoints(rid))
        if "wheel" in client.getJointInfo(rid, j)[1].decode()
    ]
    for j in wheel_joint_ids:
        client.changeDynamics(
            rid, j,
            contactStiffness=30000,
            contactDamping=1000,
            anisotropicFriction=[1.0, 0.01, 1.0],
        )
    sim.bot[0].servo_motor_model.set_motor_gains(400, 30)

    # Configure gripper fingers once per worker so the torque hook is not stacked repeatedly
    configure_gripper_fingers(sim, client, rid, rf_sim.FIXED_FINGER_ANGLES)

    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, ["pybullet"], False, 0, cur_dir)
    ee_frame_id = dyn_model.pin_model.getFrameId("robot_j2s7s300_end_effector")


def test_one_point(case):
    """Run one test episode from a random base start pose to a random 3D target."""
    idx, start_x, start_y, start_yaw, target_pos = case

    # Reset robot to the randomized initial base position and orientation
    sim.bot[0].link_floating_base_pos = [start_x, start_y, default_base_z]
    sim.bot[0].link_floating_base_ori = list(p.getQuaternionFromEuler([0.0, 0.0, start_yaw]))
    sim.ResetPose()
    sim.ReceiveObservation()
    q_init = np.array(sim.GetMotorAngles(0))

    # The DAgger arm keeps a joint history and a latched navigation flag: clear them between episodes.
    dagger_arm.history.states.clear()
    dagger_arm.navigation_latched = False

    # Our own base goal (same standoff as rf_sim.py), facing the target from the randomized start
    delta_xy = target_pos[:2] - np.array([start_x, start_y])
    target_bearing = np.arctan2(delta_xy[1], delta_xy[0]) if np.linalg.norm(delta_xy) > 1e-4 else start_yaw
    standoff = float(np.interp(target_pos[2], rf_sim.STANDOFF_Z_POINTS, rf_sim.STANDOFF_POINTS))
    base_goal_pos = target_pos[:2] - standoff * np.array([np.cos(target_bearing), np.sin(target_bearing)])
    base_goal_bearing = target_bearing

    cmd = MotorCommands()
    control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6
    stop_criteria = StopCriteria(
        reach_tolerance=REACH_TOLERANCE,
        max_steps=max_steps,
        min_time=1.0,
        stall_window=1.5,
    )
    time_step = sim.GetTimeStep()
    current_time = 0.0
    step = 0
    nav_done = False
    nav_time = np.nan
    ee_dist_error = 0.0
    wheel_time = arm_time = 0.0
    wheel_calls = arm_calls = 0

    while True:
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        q_mes = np.array(sim.GetMotorAngles(0))
        qd_mes = np.array(sim.GetMotorVelocities(0))

        # Forward kinematics for end-effector position
        q_pin = dyn_model.ReoderJoints2PinVec(np.concatenate((base_pos, base_ori, q_mes)), "pos")
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
        ee_dist_error = float(np.linalg.norm(target_pos - ee_pos))

        # Supervisor (same rule as rf_sim.py)
        dist = np.linalg.norm(base_pos[:2] - base_goal_pos)
        ang = abs(wrap_angle(base_goal_bearing - base_bearing))
        nav_done = nav_done or (dist < 0.15 and ang < 0.05) or current_time > rf_sim.BASE_NAV_TIMEOUT
        if nav_done and np.isnan(nav_time):
            nav_time = current_time

        # Wheels: random forest, predicted every PREDICT_EVERY steps and held in between
        if step % rf_sim.PREDICT_EVERY == 0:
            s = rf_sim.sim_state(base_pos, base_bearing, q_mes, qd_mes, target_pos,
                                 base_goal_pos, base_goal_bearing, ee_pos, nav_done, q_init)
            t0 = time.perf_counter()
            wheel_cmds = rf_wheel.predict(wheel_features(s, FEATURE_SET == "extra"))[0]
            wheel_time += time.perf_counter() - t0
            wheel_calls += 1
            wheel_cmds = np.clip(wheel_cmds, -rf_sim.MAX_WHEEL_SPEED, rf_sim.MAX_WHEEL_SPEED)

        # Arm: DAgger MLP every step, with the supervisor's flag as its navigation input
        x = dagger_features(base_pos, base_bearing, target_pos, ee_pos, base_goal_pos, base_goal_bearing,
                            q_mes, qd_mes, q_init, current_time)
        x[-1] = float(nav_done)
        t0 = time.perf_counter()
        _, torque_arm = dagger_arm.predict(x)
        arm_time += time.perf_counter() - t0
        arm_calls += 1
        torque_arm = np.clip(torque_arm, -rf_sim.MAX_ARM_TORQUE, rf_sim.MAX_ARM_TORQUE)

        if nav_done:
            wheel_cmds = np.zeros(4)

        cmd_all = np.concatenate((wheel_cmds, torque_arm, np.zeros(6)))
        cmd.SetControlCmd(cmd_all, control_modes)
        sim.Step(cmd, "torque")

        stop_message = check_stop(stop_criteria, current_time, time_step, ee_dist_error, nav_done)
        if stop_message:
            break

        current_time += time_step
        step += 1

    is_success = ee_dist_error <= SUCCESS_TOLERANCE
    return {
        "episode": idx, "start_x": start_x, "start_y": start_y, "start_yaw": start_yaw,
        "target_x": target_pos[0], "target_y": target_pos[1], "target_z": target_pos[2],
        "final_error": ee_dist_error, "time": current_time, "nav_done": int(nav_done), "nav_time": nav_time,
        "success": int(is_success), "wheel_ms": 1000 * wheel_time / wheel_calls,
        "arm_ms": 1000 * arm_time / arm_calls, "stop": stop_message,
    }


def read_rows(path):
    # Rows of the result CSV, empty if it does not exist yet.
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def wilson(k, n, z=1.96):
    # 95 % Wilson interval of k successes in n trials, in percent.
    centre = (k + z**2 / 2) / (n + z**2)
    half = z * np.sqrt(k * (n - k) / n + z**2 / 4) / (n + z**2)
    return 100 * (centre - half), 100 * (centre + half)


def main():
    # python test_random_spawn.py      -> all 1000 cases
    # python test_random_spawn.py 10   -> only the first 10 (same cases as in the full run)
    num_tests = int(sys.argv[1]) if len(sys.argv) > 1 else NUM_TESTS

    # Generate 1,000 random initial base poses and 3D targets across [RADIUS_MIN, RADIUS_MAX]
    rng = np.random.default_rng(SEED)
    test_cases = []
    for i in range(NUM_TESTS):
        start_x = float(np.round(rng.uniform(-2.0, 2.0), 3))
        start_y = float(np.round(rng.uniform(-2.0, 2.0), 3))
        start_yaw = float(np.round(rng.uniform(-np.pi, np.pi), 3))

        angle = rng.uniform(-np.pi, np.pi)
        radius = rng.uniform(RADIUS_MIN, RADIUS_MAX)
        tx = float(np.round(start_x + radius * np.cos(angle), 3))
        ty = float(np.round(start_y + radius * np.sin(angle), 3))
        tz = float(np.round(rng.uniform(0.3, 0.9), 3))

        test_cases.append((i, start_x, start_y, start_yaw, np.array([tx, ty, tz])))
    test_cases = test_cases[:num_tests]

    # Resume: skip the cases that are already in the CSV
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "E4.csv"
    done = {int(row["episode"]) for row in read_rows(path)}
    jobs = [case for case in test_cases if case[0] not in done]

    print("==================================================================")
    print(f"Testing RF wheels ({FEATURE_SET}) + DAgger V5.3-A arm on {num_tests} Random Base Starts & Targets")
    print(f"Target Distance Range: [{RADIUS_MIN}, {RADIUS_MAX}] m | Parallel Workers: {NUM_WORKERS}")
    print(f"Stop Tolerance: {REACH_TOLERANCE} m | Success Criterion: <= {SUCCESS_TOLERANCE} m")
    print(f"{len(jobs)} to run, {len(done)} already done")
    print("==================================================================")

    start_wall_time = time.time()
    if jobs:
        new_file = not path.exists()
        ctx = mp.get_context("spawn")
        with open(path, "a", newline="") as f, ctx.Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                writer.writeheader()
            for count, res in enumerate(pool.imap_unordered(test_one_point, jobs, chunksize=5), 1):
                writer.writerow(res)
                f.flush()
                status = "SUCCESS" if res["success"] else "FAIL"
                print(
                    f"[{count:4d}/{len(jobs)}] Start: ({res['start_x']:5.2f}, {res['start_y']:5.2f}, "
                    f"yaw={res['start_yaw']:5.2f}) | EE Error: {res['final_error']:.4f} m | "
                    f"Time: {res['time']:5.2f}s | {status}"
                )

    # Summary over the requested cases
    wanted = {case[0] for case in test_cases}
    rows = sorted((row for row in read_rows(path) if int(row["episode"]) in wanted), key=lambda r: int(r["episode"]))
    r = {key: np.array([float(row[key]) for row in rows]) for key in FIELDS[:-1]}
    final_errors = r["final_error"]
    success_list = r["success"].astype(bool)
    n = len(rows)
    num_success = int(success_list.sum())
    low, high = wilson(num_success, n)

    print("\n==================================================================")
    print("FINAL TEST RESULTS SUMMARY")
    print("==================================================================")
    print(f"Total Points Tested          : {n}")
    print(f"Successful Reaches           : {num_success}")
    print(f"Success Rate                 : {100 * num_success / n:.2f}%  (95% Wilson [{low:.1f}, {high:.1f}]%)")
    print(f"Base Reached Its Goal        : {100 * r['nav_done'].mean():.1f}% (incl. {rf_sim.BASE_NAV_TIMEOUT:.0f} s timeout)")
    print(f"Within 1 cm                  : {100 * np.mean(final_errors <= 0.01):.1f}%")
    print("------------------------------------------------------------------")
    print(f"Mean EE Error (m)            : {np.mean(final_errors):.4f}")
    print(f"Median EE Error (m)          : {np.median(final_errors):.4f}")
    print(f"Std EE Error (m)             : {np.std(final_errors):.4f}")
    print(f"Max EE Error (m)             : {np.max(final_errors):.4f}")
    print(f"Average Episode Time (s)     : {np.mean(r['time']):.2f}")
    print(f"Wheel RF per prediction (ms) : {np.mean(r['wheel_ms']):.2f} (every {rf_sim.PREDICT_EVERY} steps, "
          f"{np.mean(r['wheel_ms']) / rf_sim.PREDICT_EVERY:.2f} per step)")
    print(f"DAgger arm per step (ms)     : {np.mean(r['arm_ms']):.2f}")
    print(f"Wall Time This Run (s)       : {time.time() - start_wall_time:.1f}")
    print("==================================================================")

    # Save a simple summary plot
    rel_targets = np.stack([r["target_x"] - r["start_x"], r["target_y"] - r["start_y"]], axis=1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(final_errors, bins=30, color="steelblue", edgecolor="black")
    plt.axvline(SUCCESS_TOLERANCE, color="r", linestyle="--", label=f"Success <= {SUCCESS_TOLERANCE}m")
    plt.xlabel("Final EE Error (m)")
    plt.ylabel("Count")
    plt.title(f"EE Error Histogram (Success Rate: {100 * num_success / n:.1f}%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(rel_targets[success_list, 0], rel_targets[success_list, 1], c="g", s=15, label="Success")
    if num_success < n:
        plt.scatter(rel_targets[~success_list, 0], rel_targets[~success_list, 1], c="r", s=25, marker="x", label="Fail")
    plt.xlabel("Target Offset dX from Start (m)")
    plt.ylabel("Target Offset dY from Start (m)")
    plt.title(f"Targets Relative to Random Base Start ({RADIUS_MIN}-{RADIUS_MAX}m)")
    plt.axis("equal")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = OUT_DIR / "test_summary.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved results to {path} and summary plot to {plot_path}")


if __name__ == "__main__":
    main()
