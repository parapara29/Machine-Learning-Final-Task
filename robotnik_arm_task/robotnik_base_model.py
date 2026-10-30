import os
import time
import numpy as np

headless = os.environ.get("ROBOENV_HEADLESS", "").strip().lower() in ("1", "true", "yes")
max_steps = int(os.environ.get("ROBOENV_MAX_STEPS", "25000"))

import matplotlib
if headless:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pybullet as p
from simulation_and_control import pb, MotorCommands
from robotnik_arm_control import configure_gripper_fingers, quaternion2bearing
from train_base_model import INPUT_COLUMNS, make_base_features, wrap_angle

# ==============================================================================
# TARGET SPECIFICATION
# Fixed 3D Cartesian coordinates for the target in the world frame: [X, Y, Z]
TARGET_EE_POS = np.array([2.0, 1.1, 1.0])
FIXED_FINGER_ANGLES = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

# Standoff distance from target based on height, matching robotnik_arm_task.py
STANDOFF_Z_POINTS = [0.1, 0.5, 1.1, 1.25]
STANDOFF_POINTS = [0.80, 0.65, 0.65, 0.35]

# Navigation arrival tolerances and limits
POS_TOL = 0.08
ANG_TOL = 0.05
BASE_NAV_TIMEOUT = 25.0
MAX_WHEEL_SPEED = 3.18
# ==============================================================================


def lock_arm_and_fingers(sim, client, rid, arm_angles, finger_angles):
    """Lock both the 7 arm joints and 6 gripper fingers at fixed angles."""
    target_finger_angles = configure_gripper_fingers(sim, client, rid, finger_angles)
    orig_set_torque = sim._SetMotorTorqueByIds

    def custom_set_torque(motor_ids, commands, index=0):
        # Only pass the 4 wheel commands through; hold arm (4..10) and fingers (11..16) fixed.
        orig_set_torque(motor_ids[:4], commands[:4], index=index)
        sim.pybullet_client.setJointMotorControlArray(
            bodyIndex=sim.bot[index].bot_pybullet,
            jointIndices=motor_ids[4:11],
            controlMode=sim.pybullet_client.POSITION_CONTROL,
            targetPositions=np.asarray(arm_angles).tolist(),
            forces=[80.0] * 7,
        )
        sim.pybullet_client.setJointMotorControlArray(
            bodyIndex=sim.bot[index].bot_pybullet,
            jointIndices=motor_ids[11:17],
            controlMode=sim.pybullet_client.POSITION_CONTROL,
            targetPositions=np.asarray(target_finger_angles).tolist(),
            forces=[5.0] * 6,
        )

    sim._SetMotorTorqueByIds = custom_set_torque
    return target_finger_angles


def main():
    conf_file_name = "robotnik_arm.json"
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(cur_dir, "..", "trained_model", "base_linear_regression.npz")

    with np.load(model_path) as model:
        weights = model["weights"]
        input_mean = model["input_mean"]
        input_scale = model["input_scale"]
        if model["input_columns"].tolist() != INPUT_COLUMNS:
            raise ValueError("Model features do not match train_base_model.py; retrain the base model.")

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

    # Lock arm at its initial configuration and lock gripper fingers.
    init_joint_angles = np.array(sim.GetMotorAngles(0))
    locked_arm_angles = init_joint_angles[4:11].copy()
    lock_arm_and_fingers(sim, client, rid, locked_arm_angles, FIXED_FINGER_ANGLES)

    # Visualize target as a red sphere in PyBullet
    target_pos_list = TARGET_EE_POS.tolist()
    visual_shape_id = client.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1.0, 0.0, 0.0, 0.85])
    client.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=target_pos_list)

    # Compute the base standoff goal pose facing the target
    standoff = float(np.interp(TARGET_EE_POS[2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
    target_xy = TARGET_EE_POS[:2]
    target_bearing = np.arctan2(target_xy[1], target_xy[0]) if np.linalg.norm(target_xy) > 1e-4 else 0.0
    base_goal_pos = target_xy - standoff * np.array([np.cos(target_bearing), np.sin(target_bearing)])
    base_goal_bearing = target_bearing

    print("==================================================================")
    print(f"Target Position              : {TARGET_EE_POS}")
    print(f"Base Standoff Goal (X, Y)    : {np.round(base_goal_pos, 4)}")
    print(f"Base Standoff Goal Bearing   : {base_goal_bearing:.4f} rad")
    print("==================================================================")

    cmd = MotorCommands()
    time_step = sim.GetTimeStep()
    current_time = 0.0
    nav_done = False

    time_all = []
    base_pos_all = []
    base_yaw_all = []
    goal_dist_all = []

    while True:
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        q_mes = np.array(sim.GetMotorAngles(0))

        dist_to_goal = np.linalg.norm(base_pos[:2] - base_goal_pos)
        bearing_error = wrap_angle(base_goal_bearing - base_bearing)

        if (dist_to_goal < POS_TOL and abs(bearing_error) < ANG_TOL) or current_time >= BASE_NAV_TIMEOUT:
            nav_done = True

        if nav_done:
            wheel_cmds = np.zeros(4)
        else:
            state = {
                "base_x": base_pos[0],
                "base_y": base_pos[1],
                "base_bearing": base_bearing,
                "goal_x": base_goal_pos[0],
                "goal_y": base_goal_pos[1],
                "goal_bearing": base_goal_bearing,
            }
            features = (make_base_features(state)[0] - input_mean) / input_scale
            wheel_cmds = np.concatenate(([1.0], features)) @ weights
            max_speed = np.max(np.abs(wheel_cmds))
            if max_speed > MAX_WHEEL_SPEED:
                wheel_cmds *= MAX_WHEEL_SPEED / max_speed

        # Send commands (arm and fingers are held locked by lock_arm_and_fingers)
        cmd_all = np.concatenate((wheel_cmds, np.zeros(7), np.zeros(6)))
        control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6
        cmd.SetControlCmd(cmd_all, control_modes)
        sim.Step(cmd, "torque")

        time_all.append(current_time)
        base_pos_all.append(base_pos[:2].copy())
        base_yaw_all.append(base_bearing)
        goal_dist_all.append(dist_to_goal)

        if int(current_time / time_step) % 500 == 0:
            print(
                f"t={current_time:5.2f}s | Base: {np.round(base_pos[:2], 3)} | "
                f"Goal dist: {dist_to_goal:.3f}m | Yaw err: {bearing_error:.3f}rad | "
                f"Wheels: {np.round(wheel_cmds, 2)}"
            )

        if nav_done:
            print(f"Base reached standoff goal at t={current_time:.2f}s (dist={dist_to_goal:.4f}m, yaw_err={bearing_error:.4f}rad)")
            break

        if max_steps and current_time / time_step >= max_steps:
            print(f"Reached step cap {max_steps} at t={current_time:.2f}s")
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
    print(f"Standoff Goal Position       : {np.round(base_goal_pos, 4)}")
    print(f"Final Distance to Goal (m)   : {dist_to_goal:.4f}")
    print(f"Final Distance to Target (m) : {np.linalg.norm(TARGET_EE_POS[:2] - base_pos[:2]):.4f} (standoff={standoff:.2f}m)")
    print(f"Final Yaw Error (rad)        : {bearing_error:.4f}")
    print(f"Max Arm Deviation (rad)      : {np.max(np.abs(q_mes[4:11] - locked_arm_angles)):.6f}")
    print("==================================================================")

    if headless:
        return

    time_all = np.array(time_all)
    base_pos_all = np.array(base_pos_all)
    base_yaw_all = np.array(base_yaw_all)
    goal_dist_all = np.array(goal_dist_all)

    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    axes[0].plot(time_all, base_pos_all[:, 0], label="Base X")
    axes[0].axhline(base_goal_pos[0], color="r", linestyle="--", label="Goal X")
    axes[0].set_ylabel("X (m)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_all, base_pos_all[:, 1], label="Base Y")
    axes[1].axhline(base_goal_pos[1], color="r", linestyle="--", label="Goal Y")
    axes[1].set_ylabel("Y (m)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_all, base_yaw_all, label="Base Yaw")
    axes[2].axhline(base_goal_bearing, color="r", linestyle="--", label="Goal Yaw")
    axes[2].set_ylabel("Yaw (rad)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(time_all, goal_dist_all, color="g", label="Distance to Standoff Goal")
    axes[3].axhline(POS_TOL, color="r", linestyle="--", label=f"Tolerance ({POS_TOL}m)")
    axes[3].set_ylabel("Goal Dist (m)")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    fig.suptitle("Mobile Base Linear Regression Navigation (Arm Locked)")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

