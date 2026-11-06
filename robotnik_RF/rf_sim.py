# Runs one episode in PyBullet: RF wheels with an RF, DAgger or expert arm.
import numpy as np
import time
import os
import sys
from pathlib import Path
import joblib
from features import wheel_features, arm_features, ARM_JOINTS

ARM_TASK_DIR = Path(__file__).parent.parent / "robotnik_arm_task"  # robot configs and expert controller
sys.path.insert(0, str(ARM_TASK_DIR))
RUN_DIR = Path(__file__).parent / "results" / "rf_run_4"  # trained forests
DAGGER_DIR = ARM_TASK_DIR / "robotnik_dagger_v5"  # Ricktho's DAgger arm
FEATURE_SET = "extra"       #we can use extra or raw
PREDICT_EVERY = 10  # wheel forest runs every N steps
ARM_MODEL = "dagger"  # "rf", "dagger" or "expert"
ARM_REACH_LIMIT = None  # reach limit passed to the expert arm controller

# Headless mode and step cap for CI / direct runs
headless = os.environ.get("ROBOENV_HEADLESS", "").strip().lower() in ("1", "true", "yes")
max_steps = int(os.environ.get("ROBOENV_MAX_STEPS", "40000"))

import matplotlib
if headless:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pybullet as p
import pinocchio as pin
from simulation_and_control import pb, MotorCommands, PinWrapper, wrap_angle
from robotnik_arm_control import (
    ArmControllerParams, BaseControllerParams, StopCriteria,
    arm_controller, check_stop, configure_gripper_fingers, quaternion2bearing,
)

def load_dagger_arm():
    # The friend's DAgger MLP arm (E3/E4) needs torch, which only the ml environment has.
    sys.path.insert(0, str(DAGGER_DIR))
    from dagger_common import features
    from arm_v4 import V4Policy
    return V4Policy(DAGGER_DIR / "trained_model_dagger_v5_3_A"), features


def load_models(feature_set):
    # Wheel and arm forests of one feature set.
    model_dir = RUN_DIR / feature_set
    rf_wheel = joblib.load(model_dir / "rf_wheel.joblib")
    rf_arm = joblib.load(model_dir / "rf_arm.joblib")
    rf_wheel.n_jobs = 1
    rf_arm.n_jobs = 1
    return rf_wheel, rf_arm, model_dir

def sim_state(base_pos, base_bearing, q_mes, qd_mes, target, goal_pos, goal_bearing, ee_pos, nav_done, q_init):
    # Live robot state as named columns, the same format as the training table.
    s = {
        "base_x": base_pos[0], "base_y": base_pos[1], "base_z": base_pos[2],
        "base_bearing": base_bearing,
        "target_x": target[0], "target_y": target[1], "target_z": target[2],
        "goal_x": goal_pos[0], "goal_y": goal_pos[1], "goal_bearing": goal_bearing,
        "ee_x" : ee_pos[0],"ee_y" : ee_pos[1],"ee_z" : ee_pos[2], 
        "nav_done": nav_done,
    }
    for i in ARM_JOINTS:
        s[f"q_mes_{i}"] = q_mes[i]
        s[f"qd_mes_{i}"] = qd_mes[i]
        s[f"q_init_{i}"] = q_init[i]

    # Every value becomes a length-1 array, like one row of the training table.
    return {name: np.array([value]) for name, value in s.items()}


# ==============================================================================
# TARGET SPECIFICATION
TARGET_EE_POS = np.array([1.5539, -0.6355, 0.7925])   # val episode 1389
# TARGET_EE_POS = None
# Fixed 3D Cartesian coordinates for the Kinova end-effector in the world frame:
# [X (forward/meters), Y (lateral/meters), Z (height/meters)]
# Fixed gripper finger configuration (angles in radians, held fixed during simulation):
# [finger_1, finger_tip_1, finger_2, finger_tip_2, finger_3, finger_tip_3]
# Set to None to use initial configuration from config/URDF, or an array of 6 angles.
FIXED_FINGER_ANGLES = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

# Max navigation time (s) used by the stopping criteria.
BASE_NAV_TIMEOUT = 25.0
# Same arm torque limit as the original controller (N m).
MAX_ARM_TORQUE = 30.0
# Expert wheel commands stay below 3.18 rad/s.
MAX_WHEEL_SPEED = 3.18
# Base-to-target horizontal standoff (m) as a function of target height: larger for low targets,
# default for mid heights, smaller for high targets (far above the shoulder, which sits ~0.49 m high).
STANDOFF_Z_POINTS = [0.1, 0.5, 1.1, 1.25]
STANDOFF_POINTS = [0.80, 0.65, 0.65, 0.35]
# Stop once the EE is this close to the target (m); max iterations set via ROBOENV_MAX_STEPS
REACH_TOLERANCE = 0.01
# The EE must stay within the tolerance this many steps in a row (same rule as the DAgger evaluation).
HOLD_STEPS = 30
# ==============================================================================


def main(target=TARGET_EE_POS, arm_model=ARM_MODEL, feature_set=FEATURE_SET, verbose=True,
         reach_tolerance=REACH_TOLERANCE, max_steps=max_steps, stop_on_stall=True):
    # verbose = False (evaluate.py) skips the prints and plots and only returns the result.
    # --------------------------------------------------------------------------
    # Configuration and Simulation Setup
    # --------------------------------------------------------------------------
    conf_file_name = "robotnik_arm.json"
    cur_dir = str(ARM_TASK_DIR)
    if arm_model == "dagger":
        dagger_arm, dagger_features = load_dagger_arm()
    rf_wheel, rf_arm, run_dir = load_models(feature_set)
    run_dir = run_dir / f"sim_arm_{arm_model}"
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=not headless)

    # Configure realistic ground contact and skid-steer friction dynamics
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
            anisotropicFriction=[1.0, 0.01, 1.0]
        )
    sim.bot[0].servo_motor_model.set_motor_gains(400, 30)

    # --------------------------------------------------------------------------
    # Gripper Fingers: Fix gripper finger configuration
    # --------------------------------------------------------------------------
    target_finger_angles = configure_gripper_fingers(sim, client, rid, FIXED_FINGER_ANGLES)

    # Visualize the target reach reference in the PyBullet 3D environment as a red dot
    target_pos_list = target.tolist()
    visual_shape_id = client.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.03,
        rgbaColor=[1.0, 0.0, 0.0, 0.85]
    )
    client.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=target_pos_list
    )
    client.addUserDebugPoints(
        pointPositions=[target_pos_list],
        pointColorsRGB=[[1.0, 0.0, 0.0]],
        pointSize=10
    )

    # Active joints from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names_arr = np.expand_dims(np.array(ext_names), axis=0)
    source_names = ["pybullet"]

    # Pinocchio dynamic & kinematic model
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names_arr, source_names, False, 0, cur_dir)

    controlled_frame_name = "robot_j2s7s300_end_effector"
    ee_frame_id = dyn_model.pin_model.getFrameId(controlled_frame_name)

    init_joint_angles = np.array(sim.GetMotorAngles(0))
    init_base_pos = np.array(sim.GetBasePosition())
    init_base_ori = np.array(sim.GetBaseOrientation())
    q_init = init_joint_angles.copy()
        # Expert arm controller, only used when ARM_MODEL == "expert".
    shoulder_frame_id = dyn_model.pin_model.getFrameId("robot_j2s7s300_link_2")
    lower_limits, upper_limits = sim.GetBotJointsLimit()
    arm_lower = np.array(lower_limits[4:11])
    arm_upper = np.array(upper_limits[4:11])
    arm_params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)
    q_des_arm = q_init[4:11].copy()
    qd_des_arm = np.zeros(7)


    # Forward kinematics for initial end-effector position
    init_q_all = np.concatenate((init_base_pos, init_base_ori, init_joint_angles))
    init_q_pin = dyn_model.ReoderJoints2PinVec(init_q_all, "pos")
    pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, init_q_pin)
    pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
    init_ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()

    if verbose:
        print("==================================================================")
        print(f"Target Cartesian EE Position : {target}")
        print(f"Initial Cartesian EE Position: {np.round(init_ee_pos, 4)}")
        print(f"Initial Cartesian Error (m)  : {np.linalg.norm(target - init_ee_pos):.4f}")
        print("==================================================================")

    time_step = sim.GetTimeStep()
    current_time = 0
    step = 0
    stable = 0            # steps in a row within the reach tolerance
    best_error = np.inf
    nav_time = np.nan     # time at which the supervisor switched to reaching

    # --------------------------------------------------------------------------
    # Coordinated Motion Planning: Base Navigation & Arm Reach
    # --------------------------------------------------------------------------
    # Standoff distance between robot base center and target, depending on target height.
    standoff = float(np.interp(target[2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
    target_xy = target[:2]
    target_bearing = np.arctan2(target_xy[1], target_xy[0]) if np.linalg.norm(target_xy) > 1e-4 else 0.0

    # Mobile base goal location facing the target
    base_goal_pos = target_xy - standoff * np.array([np.cos(target_bearing), np.sin(target_bearing)])
    base_goal_bearing = target_bearing

    # Track arrival at the base goal for the existing stopping criteria.
    base_navigation_done = False
    base_params = BaseControllerParams(nav_timeout=BASE_NAV_TIMEOUT)

    # Stop conditions (step cap, target reached, unreachable-target stall), see StopCriteria
    stop_criteria = StopCriteria(reach_tolerance=reach_tolerance, max_steps=max_steps)
    if not stop_on_stall:
        stop_criteria.stall_window = np.inf   # run until reached or the step cap, like the DAgger evaluation

    # Command interface
    cmd = MotorCommands()

    # Data logging for plotting
    time_all = []
    ee_pos_all = []
    ee_error_all = []
    base_pos_all = []
    q_mes_all = []
    qd_mes_all = []

    # --------------------------------------------------------------------------
    # Main Simulation Loop
    # --------------------------------------------------------------------------
    while True:
        # Measure current states
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        q_mes = np.array(sim.GetMotorAngles(0))
        qd_mes = np.array(sim.GetMotorVelocities(0))

        # Full kinematic state vector in Pinocchio format
        q_all = np.concatenate((base_pos, base_ori, q_mes))
        q_pin = dyn_model.ReoderJoints2PinVec(q_all, "pos")

        # Compute end-effector forward kinematics
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
        ee_error = target - ee_pos
        ee_dist_error = np.linalg.norm(ee_error)

        # 1. Supervisor first: the arm model needs nav_done as an input.
        dist = np.linalg.norm(base_pos[:2] - base_goal_pos)
        ang = abs(wrap_angle(base_goal_bearing - base_bearing))
        base_navigation_done = base_navigation_done or (
            dist < 0.15 and ang < base_params.ang_tol
        ) or current_time > base_params.nav_timeout
        if base_navigation_done and np.isnan(nav_time):
            nav_time = current_time
        best_error = min(best_error, ee_dist_error)

        # 2. Predict every PREDICT_EVERY steps, hold the commands in between.
        if step % PREDICT_EVERY == 0:
            s = sim_state(base_pos, base_bearing, q_mes, qd_mes, target,
                          base_goal_pos, base_goal_bearing, ee_pos, base_navigation_done, q_init)
            extra = feature_set == "extra"
            wheel_cmds = np.clip(rf_wheel.predict(wheel_features(s, extra))[0], -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
            if arm_model == "rf":
                torque_arm = np.clip(rf_arm.predict(arm_features(s, extra))[0], -MAX_ARM_TORQUE, MAX_ARM_TORQUE)
        if arm_model == "expert":
            torque_arm, q_des_arm, qd_des_arm = arm_controller(
                dyn_model, q_pin, q_mes[4:11], qd_mes[4:11], q_des_arm, q_init[4:11], target, ee_pos,
                ee_frame_id, shoulder_frame_id, arm_lower, arm_upper, time_step, arm_params,
                reach_active=base_navigation_done,
            )
        if arm_model == "dagger":
            x = dagger_features(base_pos, base_bearing, target, ee_pos, base_goal_pos, base_goal_bearing, q_mes, qd_mes, q_init, current_time)
            x[-1] = float(base_navigation_done) 
            _, torque_arm = dagger_arm.predict(x)
            torque_arm = np.clip(torque_arm, -MAX_ARM_TORQUE, MAX_ARM_TORQUE)


        # 3. Hardcoded stop: the wheels stay still once navigation is done.
        if base_navigation_done:
            wheel_cmds = np.zeros(4)
        # ----------------------------------------------------------------------
        # 3. Gripper / Finger Control (Fingers locked at fixed configuration)
        # ----------------------------------------------------------------------
        torque_fingers = np.zeros(6)

        # ----------------------------------------------------------------------
        # Send Multi-Interface Control Commands
        # ----------------------------------------------------------------------
        # 4 wheels (velocity) + 7 arm joints (torque) + 6 fingers (torque)
        cmd_all = np.concatenate((wheel_cmds, torque_arm, torque_fingers))
        control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6
        cmd.SetControlCmd(cmd_all, control_modes)
        sim.Step(cmd, "torque")

        # Visualization in Pinocchio GUI if active
        if dyn_model.visualizer:
            for index in range(len(sim.bot)):
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)

        # Exit logic with 'q' key in PyBullet GUI window
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            stop_message = "Stopped with the q key"
            break

        # Stop criteria: within tolerance for HOLD_STEPS steps in a row, maximum steps, or unreachable target
        stable = stable + 1 if base_navigation_done and ee_dist_error < reach_tolerance else 0
        if stable >= HOLD_STEPS:
            stop_message = f"Target reached within {reach_tolerance*1000:.0f}mm tolerance at t={current_time:.2f}s!"
        else:
            stop_message = check_stop(stop_criteria, current_time, time_step, ee_dist_error, base_navigation_done)
        if stop_message:
            if verbose:
                print(stop_message)
            break

        # Data logging for plotting
        time_all.append(current_time)
        ee_pos_all.append(ee_pos)
        ee_error_all.append(ee_error)
        base_pos_all.append(base_pos)
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)

        if not headless:
            time.sleep(0.001)

        current_time += time_step
        step += 1
        if verbose and int(current_time / time_step) % 500 == 0:
            print(f"t={current_time:.2f}s | Base pos: {np.round(base_pos[:2], 3)} | EE pos: {np.round(ee_pos, 3)} | EE error: {ee_dist_error:.4f}m | yaw {base_bearing:.2f} nav_done {base_navigation_done}")

    result = {
        "success": int(stop_message.startswith("Target reached")),
        "final_error": float(ee_dist_error),
        "best_error": float(best_error),
        "time": float(current_time),
        "nav_time": float(nav_time),
        "stop": stop_message,
    }
    if not verbose:
        return result

    # Final summary report
    print("==================================================================")
    print(f"Final Base Position          : {np.round(base_pos[:2], 4)}")
    print(f"Final End-Effector Position  : {np.round(ee_pos, 4)}")
    print(f"Target End-Effector Position : {target}")
    print(f"Final Cartesian Error (m)    : {np.linalg.norm(target - ee_pos):.4f}")
    print(f"Fixed Finger Angles          : {np.round(target_finger_angles, 4)}")
    print(f"Final Finger Angles          : {np.round(q_mes[11:17], 4)}")
    print(f"Max Finger Deviation (rad)   : {np.max(np.abs(q_mes[11:17] - target_finger_angles)):.6f}")
    print("==================================================================")

    if headless:
        print(f"Headless run finished at t={current_time:.2f}s")

    # --------------------------------------------------------------------------
    # Plotting Performance
    # --------------------------------------------------------------------------
    run_dir.mkdir(exist_ok=True)
    ee_pos_all = np.array(ee_pos_all)
    ee_error_all = np.array(ee_error_all)
    time_all = np.array(time_all)

    # 1. Cartesian end-effector coordinates tracking
    plt.figure(figsize=(10, 8))
    coords = ['X', 'Y', 'Z']
    for idx in range(3):
        plt.subplot(3, 1, idx + 1)
        plt.plot(time_all, ee_pos_all[:, idx], label=f'Actual EE {coords[idx]}', color='b')
        plt.axhline(y=target[idx], color='r', linestyle='--', label=f'Target EE {coords[idx]}')
        plt.ylabel(f'{coords[idx]} Position (m)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.xlabel('Time (s)')
    plt.suptitle('End-Effector Cartesian Position Reaching')
    plt.tight_layout()
    plt.savefig(run_dir / "sim_ee_position.png", dpi=150)
    plt.show()

    # 2. Cartesian distance error over time
    plt.figure(figsize=(8, 4))
    ee_dist_norm = np.linalg.norm(ee_error_all, axis=1)
    plt.plot(time_all, ee_dist_norm, label='EE Euclidean Error (m)', color='g')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.title('Cartesian End-Effector Reaching Error')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "sim_ee_error.png", dpi=150)
    plt.show()

    # 3. Measured arm joint positions (the model predicts torques, not joint references)
    plt.figure(figsize=(10, 8))
    for i in range(7):
        plt.subplot(4, 2, i + 1)
        plt.plot(time_all, [q[4 + i] for q in q_mes_all], label=f'Measured J{i+1}')
        plt.ylabel(f'J{i+1} (rad)')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
    plt.xlabel('Time (s)')
    plt.suptitle('Arm Joint Position Profiles')
    plt.tight_layout()
    plt.savefig(run_dir / "sim_joint_angles.png", dpi=150)
    plt.show()
    return result


if __name__ == '__main__':
    main()
