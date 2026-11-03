import os
import time
import numpy as np

headless = os.environ.get("ROBOENV_HEADLESS", "1").strip().lower() in ("1", "true", "yes")
max_steps = int(os.environ.get("ROBOENV_MAX_STEPS", "40000"))

import matplotlib
if headless:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pybullet as p
import pinocchio as pin
from simulation_and_control import pb, MotorCommands, PinWrapper
from robotnik_arm_control import StopCriteria, check_stop, configure_gripper_fingers, quaternion2bearing
from train_base_model import INPUT_COLUMNS as BASE_COLUMNS, make_base_features, wrap_angle
from train_arm_model import INPUT_COLUMNS as ARM_COLUMNS, make_arm_features
from robotnik_combined_model import (
    FIXED_FINGER_ANGLES,
    BASE_NAV_TIMEOUT,
    MAX_WHEEL_SPEED,
    MAX_ARM_TORQUE,
    compute_base_standoff,
    is_target_in_arm_reach,
)

# ==============================================================================
# TEST SETTINGS
# ==============================================================================
NUM_TESTS = 1000
REACH_TOLERANCE = 0.01    # Stop simulation early once EE error <= 0.01 m (same as robotnik_arm_task.py)
SUCCESS_TOLERANCE = 0.05  # Count episode as a success if final EE error <= 0.05 m
RADIUS_MIN = 0.9          # Min target distance from robot start position (m)
RADIUS_MAX = 3.5          # Max target distance from robot start position (m)
SEED = 42
# ==============================================================================


def main():
    # --------------------------------------------------------------------------
    # 1. Load Models and Setup Simulation
    # --------------------------------------------------------------------------
    conf_file_name = "robotnik_arm.json"
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(cur_dir, "..", "trained_model")

    with np.load(os.path.join(model_dir, "base_linear_regression.npz")) as base_model:
        base_weights = base_model["weights"]
        base_mean = base_model["input_mean"]
        base_scale = base_model["input_scale"]
        if base_model["input_columns"].tolist() != BASE_COLUMNS:
            raise ValueError("Base model features do not match train_base_model.py")

    with np.load(os.path.join(model_dir, "arm_linear_regression.npz")) as arm_model:
        arm_weights = arm_model["weights"]
        arm_mean = arm_model["input_mean"]
        arm_scale = arm_model["input_scale"]
        if arm_model["input_columns"].tolist() != ARM_COLUMNS:
            raise ValueError("Arm model features do not match train_arm_model.py")

    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=not headless)
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

    # Configure gripper fingers once so the torque hook is not stacked repeatedly
    configure_gripper_fingers(sim, client, rid, FIXED_FINGER_ANGLES)
    init_joint_angles = np.array(sim.GetMotorAngles(0))
    locked_arm_angles = init_joint_angles[4:11].copy()

    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, ["pybullet"], False, 0, cur_dir)
    ee_frame_id = dyn_model.pin_model.getFrameId("robot_j2s7s300_end_effector")

    cmd = MotorCommands()
    control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6
    time_step = sim.GetTimeStep()

    # --------------------------------------------------------------------------
    # 2. Generate 1,000 Random Initial Base Poses and Random 3D Target Points
    # --------------------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    test_cases = []
    for _ in range(NUM_TESTS):
        # Random initial robot base position (X, Y) and yaw angle
        start_x = np.round(rng.uniform(-2.0, 2.0), 3)
        start_y = np.round(rng.uniform(-2.0, 2.0), 3)
        start_yaw = np.round(rng.uniform(-np.pi, np.pi), 3)

        # Random 3D target around the robot's starting position across [RADIUS_MIN, RADIUS_MAX]
        angle = rng.uniform(-np.pi, np.pi)
        radius = rng.uniform(RADIUS_MIN, RADIUS_MAX)
        tx = np.round(start_x + radius * np.cos(angle), 3)
        ty = np.round(start_y + radius * np.sin(angle), 3)
        tz = np.round(rng.uniform(0.3, 0.9), 3)

        test_cases.append((
            np.array([start_x, start_y, default_base_z]),
            float(start_yaw),
            np.array([tx, ty, tz]),
        ))

    print("==================================================================")
    print(f"Testing Combined Model on {NUM_TESTS} Random Base Starts & Targets")
    print(f"Target Distance Range: [{RADIUS_MIN}, {RADIUS_MAX}] m")
    print(f"Stop Tolerance: {REACH_TOLERANCE} m | Success Criterion: <= {SUCCESS_TOLERANCE} m")
    print("==================================================================")

    final_errors = []
    success_list = []
    sim_times = []
    rel_targets = []
    start_wall_time = time.time()

    # --------------------------------------------------------------------------
    # 3. Test Loop Over All 1,000 Points
    # --------------------------------------------------------------------------
    for i, (start_base_pos, start_yaw, target_pos) in enumerate(test_cases):
        # Set randomized initial base position and yaw, then reset robot state
        sim.bot[0].link_floating_base_pos = start_base_pos.tolist()
        sim.bot[0].link_floating_base_ori = list(p.getQuaternionFromEuler([0.0, 0.0, start_yaw]))
        sim.ResetPose()
        sim.ReceiveObservation()

        # Compute base standoff goal facing the target from the randomized start position
        delta_xy = target_pos[:2] - start_base_pos[:2]
        target_bearing = np.arctan2(delta_xy[1], delta_xy[0]) if np.linalg.norm(delta_xy) > 1e-4 else start_yaw
        standoff = compute_base_standoff(target_pos[2])
        base_goal_pos = target_pos[:2] - standoff * np.array([np.cos(target_bearing), np.sin(target_bearing)])

        stop_criteria = StopCriteria(
            reach_tolerance=REACH_TOLERANCE,
            max_steps=max_steps,
            min_time=1.0,
            stall_window=1.5,
        )
        current_time = 0.0
        nav_done = False
        ee_dist_error = 0.0

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

            dist_to_goal = float(np.linalg.norm(base_pos[:2] - base_goal_pos))
            cur_target_bearing = float(np.arctan2(target_pos[1] - base_pos[1], target_pos[0] - base_pos[0]))
            in_reach, tf, tl, _ = is_target_in_arm_reach(base_pos, base_bearing, target_pos, expected_tf=standoff)

            # Base linear regression prediction
            base_state = {
                "base_x": base_pos[0],
                "base_y": base_pos[1],
                "base_bearing": base_bearing,
                "goal_x": base_goal_pos[0],
                "goal_y": base_goal_pos[1],
                "goal_bearing": cur_target_bearing,
            }
            base_features = (make_base_features(base_state)[0] - base_mean) / base_scale
            wheel_cmds = np.concatenate(([1.0], base_features)) @ base_weights
            max_speed = float(np.max(np.abs(wheel_cmds)))
            if max_speed > MAX_WHEEL_SPEED:
                wheel_cmds *= MAX_WHEEL_SPEED / max_speed

            # Stop wheels and switch to arm once target is in the arm's sweet spot
            if not nav_done and (
                in_reach
                or (dist_to_goal < 0.09 and abs(tl) <= 0.05 and max_speed < 0.35)
                or current_time >= BASE_NAV_TIMEOUT
            ):
                nav_done = True

            if not nav_done:
                # Phase 1: Drive wheels and hold arm locked at initial angles
                tau_g = pin.computeGeneralizedGravity(dyn_model.pin_model, dyn_model.pin_data, q_pin)
                torque_arm = np.clip(
                    tau_g[6:13] + 100.0 * (locked_arm_angles - q_mes[4:11]) - 30.0 * qd_mes[4:11],
                    -MAX_ARM_TORQUE,
                    MAX_ARM_TORQUE,
                )
            else:
                # Phase 2: Keep wheels stopped and move arm using arm linear regression
                wheel_cmds = np.zeros(4)
                tf_clamped = np.clip(tf, 0.55, 0.85)
                tl_clamped = np.clip(tl, -0.15, 0.15)
                th_clamped = np.clip(target_pos[2] - base_pos[2], 0.30, 1.15)
                c_yaw, s_yaw = np.cos(base_bearing), np.sin(base_bearing)
                arm_state = {
                    "base_x": base_pos[0],
                    "base_y": base_pos[1],
                    "base_z": base_pos[2],
                    "base_bearing": base_bearing,
                    "target_x": base_pos[0] + c_yaw * tf_clamped - s_yaw * tl_clamped,
                    "target_y": base_pos[1] + s_yaw * tf_clamped + c_yaw * tl_clamped,
                    "target_z": base_pos[2] + th_clamped,
                }
                for j in range(4, 11):
                    arm_state[f"q_mes_{j}"] = q_mes[j]
                    arm_state[f"qd_mes_{j}"] = qd_mes[j]
                arm_features = (make_arm_features(arm_state)[0] - arm_mean) / arm_scale
                torque_arm = np.clip(
                    np.concatenate(([1.0], arm_features)) @ arm_weights,
                    -MAX_ARM_TORQUE,
                    MAX_ARM_TORQUE,
                )

            cmd_all = np.concatenate((wheel_cmds, torque_arm, np.zeros(6)))
            cmd.SetControlCmd(cmd_all, control_modes)
            sim.Step(cmd, "torque")

            stop_message = check_stop(stop_criteria, current_time, time_step, ee_dist_error, nav_done)
            if stop_message:
                break

            current_time += time_step

        is_success = ee_dist_error <= SUCCESS_TOLERANCE
        final_errors.append(ee_dist_error)
        success_list.append(is_success)
        sim_times.append(current_time)
        rel_targets.append(delta_xy)

        status = "SUCCESS" if is_success else "FAIL"
        print(
            f"[{i + 1:4d}/{NUM_TESTS}] Start: ({start_base_pos[0]:5.2f}, {start_base_pos[1]:5.2f}, yaw={start_yaw:5.2f}) | "
            f"Target: {target_pos} | EE Error: {ee_dist_error:.4f} m | Time: {current_time:5.2f}s | {status}"
        )

    # --------------------------------------------------------------------------
    # 4. Final Summary Metrics
    # --------------------------------------------------------------------------
    final_errors = np.array(final_errors)
    success_list = np.array(success_list)
    sim_times = np.array(sim_times)
    rel_targets = np.array(rel_targets)

    num_success = int(np.sum(success_list))
    num_fail = NUM_TESTS - num_success
    success_rate = 100.0 * num_success / NUM_TESTS

    print("\n==================================================================")
    print("FINAL TEST RESULTS SUMMARY")
    print("==================================================================")
    print(f"Total Points Tested          : {NUM_TESTS}")
    print(f"Target Distance Range (m)    : [{RADIUS_MIN}, {RADIUS_MAX}]")
    print(f"Success Criterion            : EE Error <= {SUCCESS_TOLERANCE} m")
    print(f"Stop Tolerance               : EE Error <= {REACH_TOLERANCE} m")
    print(f"Successful Reaches           : {num_success}")
    print(f"Failed Reaches               : {num_fail}")
    print(f"Success Rate                 : {success_rate:.2f}%")
    print("------------------------------------------------------------------")
    print(f"Mean EE Error (m)            : {np.mean(final_errors):.4f}")
    print(f"Median EE Error (m)          : {np.median(final_errors):.4f}")
    print(f"Std EE Error (m)             : {np.std(final_errors):.4f}")
    print(f"Min EE Error (m)             : {np.min(final_errors):.4f}")
    print(f"Max EE Error (m)             : {np.max(final_errors):.4f}")
    print(f"Average Episode Time (s)     : {np.mean(sim_times):.2f}")
    print(f"Total Wall Time (s)          : {time.time() - start_wall_time:.1f}")
    print("==================================================================")

    # Save a simple summary plot
    out_dir = os.path.join(cur_dir, "combined_test_results")
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(final_errors, bins=30, color="steelblue", edgecolor="black")
    plt.axvline(SUCCESS_TOLERANCE, color="r", linestyle="--", label=f"Success <= {SUCCESS_TOLERANCE}m")
    plt.xlabel("Final EE Error (m)")
    plt.ylabel("Count")
    plt.title(f"EE Error Histogram (Success Rate: {success_rate:.1f}%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(rel_targets[success_list, 0], rel_targets[success_list, 1], c="g", s=15, label="Success")
    if num_fail > 0:
        plt.scatter(rel_targets[~success_list, 0], rel_targets[~success_list, 1], c="r", s=25, marker="x", label="Fail")
    plt.xlabel("Target Offset dX from Start (m)")
    plt.ylabel("Target Offset dY from Start (m)")
    plt.title(f"Targets Relative to Random Base Start ({RADIUS_MIN}-{RADIUS_MAX}m)")
    plt.axis("equal")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "test_summary.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved summary plot to {plot_path}")


if __name__ == "__main__":
    main()
