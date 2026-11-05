"""
Decoupled Mobile Manipulation — Combined Runtime Simulation
=============================================================
This script executes a full mobile manipulation episode in PyBullet using our two
trained linear regression models:

- Phase 1 (Base Navigation): The 15-feature base model drives the 4 wheels toward the
  standoff parking spot (~70cm in front of the target) while holding the arm locked.
- Phase 2 (Arm Reaching): Once the base parks within the arm's sweet spot, wheels stop (0 rad/s)
  and the 25-feature arm model commands joint torques to reach the 3D target.

How to Change the Target:
  Edit `TARGET_EE_POS = np.array([X, Y, Z])` around line 45.

Run:
  uv run robotnik_combined_model.py              # With PyBullet 3D GUI
  ROBOENV_HEADLESS=1 uv run robotnik_combined_model.py  # Headless (terminal only)
"""

import os
import time
import numpy as np

headless = os.environ.get("ROBOENV_HEADLESS", "").strip().lower() in ("1", "true", "yes")
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
from train_arm_model import (
    INPUT_COLUMNS as ARM_COLUMNS,
    SHOULDER_OFFSET,
    make_arm_features,
    target_in_base_frame,
)

# ==============================================================================
# TARGET SPECIFICATION
# Fixed 3D Cartesian coordinates for the target in the world frame: [X, Y, Z]
TARGET_EE_POS = np.array([-0.067, -0.263, 0.395])
FIXED_FINGER_ANGLES = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

# Base standoff distance curve (height-adjusted so base stops in the optimal arm reach window)
BASE_STANDOFF = 0.65

# Arm reachable workspace window in the robot base frame for stopping the wheels
ARM_FORWARD_MIN = 0.60
ARM_FORWARD_MAX = 0.85
ARM_LEFT_MAX = 0.045

# Navigation and arm control limits
POS_TOL = 0.08
ANG_TOL = 0.05
BASE_NAV_TIMEOUT = 35.0
REACH_TOLERANCE = 0.01
MAX_WHEEL_SPEED = 3.18
MAX_ARM_TORQUE = 30.0
# ==============================================================================


def compute_base_standoff(target_z):
    """Compute height-dependent standoff distance so the base stops in the arm's sweet spot."""
    return float(np.interp(target_z, [0.1, 0.5, 1.1, 1.25], [0.80, BASE_STANDOFF, BASE_STANDOFF, 0.35]))


def is_target_in_arm_reach(base_pos, base_bearing, target_pos, expected_tf=None):
    """Check if the target is aligned within the arm's sweet-spot window in the robot base frame."""
    tf, tl, th = target_in_base_frame({
        "base_x": base_pos[0],
        "base_y": base_pos[1],
        "base_z": base_pos[2],
        "base_bearing": base_bearing,
        "target_x": target_pos[0],
        "target_y": target_pos[1],
        "target_z": target_pos[2],
    })
    shoulder_dist = np.linalg.norm([tf - SHOULDER_OFFSET[0], tl - SHOULDER_OFFSET[1], th - SHOULDER_OFFSET[2]])
    ideal_tf = (BASE_STANDOFF if expected_tf is None else expected_tf) + 0.055
    in_reach = (ideal_tf - 0.04 <= tf <= ideal_tf + 0.03) and (abs(tl) <= ARM_LEFT_MAX)
    return in_reach, float(tf), float(tl), float(shoulder_dist)


def main():
    conf_file_name = "robotnik_arm.json"
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(cur_dir, "..", "trained_model")

    with np.load(os.path.join(model_dir, "base_linear_regression.npz")) as base_model:
        base_weights = base_model["weights"]
        base_mean = base_model["input_mean"]
        base_scale = base_model["input_scale"]
        if base_model["input_columns"].tolist() != BASE_COLUMNS:
            raise ValueError("Base model features do not match train_base_model.py; retrain the base model.")

    with np.load(os.path.join(model_dir, "arm_linear_regression.npz")) as arm_model:
        arm_weights = arm_model["weights"]
        arm_mean = arm_model["input_mean"]
        arm_scale = arm_model["input_scale"]
        forward_min = float(arm_model["forward_min"])
        forward_max = float(arm_model["forward_max"])
        left_max = float(arm_model["left_max"])
        if arm_model["input_columns"].tolist() != ARM_COLUMNS:
            raise ValueError("Arm model features do not match train_arm_model.py; retrain the arm model.")

    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=not headless)
    sim.SetFloorFriction(1.0)
    client = sim.pybullet_client
    rid = sim.bot[0].bot_pybullet

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

    # Keep gripper fingers fixed; arm joints remain in pure torque control throughout
    target_finger_angles = configure_gripper_fingers(sim, client, rid, FIXED_FINGER_ANGLES)
    init_joint_angles = np.array(sim.GetMotorAngles(0))
    locked_arm_angles = init_joint_angles[4:11].copy()

    # Visualize target as a red sphere in PyBullet
    target_pos_list = TARGET_EE_POS.tolist()
    visual_shape_id = client.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1.0, 0.0, 0.0, 0.85])
    client.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=target_pos_list)

    # Pinocchio model for forward kinematics and Phase-1 gravity holding
    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, ["pybullet"], False, 0, cur_dir)
    ee_frame_id = dyn_model.pin_model.getFrameId("robot_j2s7s300_end_effector")

    # Compute the base standoff goal pose facing the target from the initial base position
    init_base_pos = np.array(sim.GetBasePosition())
    delta_xy = TARGET_EE_POS[:2] - init_base_pos[:2]
    target_bearing = np.arctan2(delta_xy[1], delta_xy[0]) if np.linalg.norm(delta_xy) > 1e-4 else 0.0
    standoff = compute_base_standoff(TARGET_EE_POS[2])
    base_goal_pos = TARGET_EE_POS[:2] - standoff * np.array([np.cos(target_bearing), np.sin(target_bearing)])
    base_goal_bearing = target_bearing

    print("==================================================================")
    print(f"Target Cartesian EE Position : {TARGET_EE_POS}")
    print(f"Base Standoff Goal (X, Y)    : {np.round(base_goal_pos, 4)} (yaw={base_goal_bearing:.3f} rad)")
    print(f"Arm Workspace Stop Window    : forward=[{ARM_FORWARD_MIN:.2f}, {ARM_FORWARD_MAX:.2f}]m, |left|<={ARM_LEFT_MAX:.2f}m")
    print("==================================================================")

    cmd = MotorCommands()
    stop_criteria = StopCriteria(reach_tolerance=REACH_TOLERANCE, max_steps=max_steps)
    time_step = sim.GetTimeStep()
    current_time = 0.0
    nav_done = False

    time_all = []
    ee_pos_all = []
    ee_error_all = []
    base_pos_all = []
    q_mes_all = []

    while True:
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        q_mes = np.array(sim.GetMotorAngles(0))
        qd_mes = np.array(sim.GetMotorVelocities(0))

        # Forward kinematics for current end-effector position
        q_pin = dyn_model.ReoderJoints2PinVec(np.concatenate((base_pos, base_ori, q_mes)), "pos")
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
        ee_error = TARGET_EE_POS - ee_pos
        ee_dist_error = float(np.linalg.norm(ee_error))

        dist_to_goal = float(np.linalg.norm(base_pos[:2] - base_goal_pos))
        cur_target_bearing = float(np.arctan2(TARGET_EE_POS[1] - base_pos[1], TARGET_EE_POS[0] - base_pos[0]))
        bearing_error = float(wrap_angle(cur_target_bearing - base_bearing))
        in_reach, tf, tl, shoulder_dist = is_target_in_arm_reach(base_pos, base_bearing, TARGET_EE_POS, expected_tf=standoff)

        # Predict wheel commands for Phase 1
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

        # Switch from base navigation to arm reaching once the target is inside the arm's sweet spot
        if not nav_done and (
            in_reach
            or (dist_to_goal < 0.09 and abs(tl) <= 0.05 and max_speed < 0.35)
            or current_time >= BASE_NAV_TIMEOUT
        ):
            nav_done = True
            print(
                f"Wheels stopped at t={current_time:.2f}s | Target in base frame: "
                f"forward={tf:.3f}m, left={tl:.3f}m, shoulder_dist={shoulder_dist:.3f}m -> Starting arm model"
            )

        if not nav_done:
            # Phase 1: Base linear regression drives the wheels; hold arm locked at initial angles via torque control
            tau_g = pin.computeGeneralizedGravity(dyn_model.pin_model, dyn_model.pin_data, q_pin)
            torque_arm = np.clip(
                tau_g[6:13] + 100.0 * (locked_arm_angles - q_mes[4:11]) - 30.0 * qd_mes[4:11],
                -MAX_ARM_TORQUE,
                MAX_ARM_TORQUE,
            )
        else:
            # Phase 2: Wheels remain stopped while the arm linear regression reaches for the target
            wheel_cmds = np.zeros(4)
            tf_clamped = np.clip(tf, 0.55, 0.85)
            tl_clamped = np.clip(tl, -0.15, 0.15)
            th_clamped = np.clip(TARGET_EE_POS[2] - base_pos[2], 0.30, 1.15)
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
            for i in range(4, 11):
                arm_state[f"q_mes_{i}"] = q_mes[i]
                arm_state[f"qd_mes_{i}"] = qd_mes[i]
            arm_features = (make_arm_features(arm_state)[0] - arm_mean) / arm_scale
            torque_arm = np.clip(np.concatenate(([1.0], arm_features)) @ arm_weights, -MAX_ARM_TORQUE, MAX_ARM_TORQUE)

        cmd_all = np.concatenate((wheel_cmds, torque_arm, np.zeros(6)))
        control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6
        cmd.SetControlCmd(cmd_all, control_modes)
        sim.Step(cmd, "torque")

        time_all.append(current_time)
        ee_pos_all.append(ee_pos)
        ee_error_all.append(ee_error)
        base_pos_all.append(base_pos[:2].copy())
        q_mes_all.append(q_mes.copy())

        if int(current_time / time_step) % 500 == 0:
            mode_str = "ARM_REACH" if nav_done else "BASE_NAV"
            print(
                f"t={current_time:5.2f}s [{mode_str:9s}] | Base: {np.round(base_pos[:2], 3)} | "
                f"tf={tf:.2f}m, tl={tl:.2f}m | EE: {np.round(ee_pos, 3)} | EE err: {ee_dist_error:.4f}m"
            )

        stop_message = check_stop(stop_criteria, current_time, time_step, ee_dist_error, nav_done)
        if stop_message:
            print(stop_message)
            break

        keys = sim.GetPyBulletClient().getKeyboardEvents()
        q_key = ord("q")
        if q_key in keys and keys[q_key] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        if not headless:
            time.sleep(0.001)
        current_time += time_step

    print("==================================================================")
    print(f"Final Base Position          : {np.round(base_pos[:2], 4)}")
    print(f"Final End-Effector Position  : {np.round(ee_pos, 4)}")
    print(f"Target End-Effector Position : {TARGET_EE_POS}")
    print(f"Final Cartesian Error (m)    : {ee_dist_error:.4f}")
    print(f"Max Finger Deviation (rad)   : {np.max(np.abs(q_mes[11:17] - target_finger_angles)):.6f}")
    print("==================================================================")

    if headless:
        return

    time_all = np.array(time_all)
    ee_pos_all = np.array(ee_pos_all)
    ee_error_all = np.array(ee_error_all)

    # 1. Cartesian end-effector coordinates tracking
    fig1 = plt.figure(figsize=(10, 8))
    coords = ["X", "Y", "Z"]
    for idx in range(3):
        plt.subplot(3, 1, idx + 1)
        plt.plot(time_all, ee_pos_all[:, idx], label=f"Actual EE {coords[idx]}", color="b")
        plt.axhline(y=TARGET_EE_POS[idx], color="r", linestyle="--", label=f"Target EE {coords[idx]}")
        plt.ylabel(f"{coords[idx]} Position (m)")
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.xlabel("Time (s)")
    plt.suptitle("End-Effector Cartesian Position Reaching")
    plt.tight_layout()
    fig1.savefig(os.path.join(cur_dir, "combined_run_ee_tracking.png"), dpi=150)

    # 2. Cartesian distance error over time
    fig2 = plt.figure(figsize=(8, 4))
    ee_dist_norm = np.linalg.norm(ee_error_all, axis=1)
    plt.plot(time_all, ee_dist_norm, label="EE Euclidean Error (m)", color="g")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (m)")
    plt.title("Cartesian End-Effector Reaching Error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig2.savefig(os.path.join(cur_dir, "combined_run_ee_error.png"), dpi=150)

    # 3. Measured arm joint positions
    fig3 = plt.figure(figsize=(10, 8))
    for i in range(7):
        plt.subplot(4, 2, i + 1)
        plt.plot(time_all, [q[4 + i] for q in q_mes_all], label=f"Measured J{i + 1}")
        plt.ylabel(f"J{i + 1} (rad)")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
    plt.xlabel("Time (s)")
    plt.suptitle("Arm Joint Position Profiles")
    plt.tight_layout()
    fig3.savefig(os.path.join(cur_dir, "combined_run_joints.png"), dpi=150)

    print(f"Saved run plots to {cur_dir}/combined_run_*.png")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
