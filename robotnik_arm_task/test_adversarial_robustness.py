"""
Adversarial & Perturbation Robustness Evaluation
=================================================
This script benchmarks the decoupled linear regression system under 4 realistic adversarial stress tests:
  1. Shove / Force Impulse: Sudden 30 N external push on the arm gripper mid-reach.
  2. Target Jump Midway: The 3D target relocates by ~0.6m to 1.0m while the robot is en route.
  3. Arm Payload: An unmodeled 0.5 kg gravitational load attached to the end-effector.
  4. Base Displacement: The mobile base is shoved 0.4m backwards/sideways after arrival.

Total: 100 randomized tests (25 per condition by default) run in parallel across CPU cores.

Outputs:
  - Live console progress and per-condition breakdown table.
  - CSV report: `adversarial_results/adversarial_results.csv`
  - Publication-quality summary figure: `adversarial_results/adversarial_summary.png`

Run:
  uv run test_adversarial_robustness.py
"""

import argparse
import csv
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

import matplotlib
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

# Global worker handles (initialized once per CPU process)
sim = None
dyn_model = None
ee_frame_id = None
ee_link_idx = None
locked_arm_angles = None
default_base_z = 0.0
base_weights = base_mean = base_scale = None
arm_weights = arm_mean = arm_scale = None


def init_worker():
    """Setup PyBullet, Pinocchio, and trained models in each worker."""
    global sim, dyn_model, ee_frame_id, ee_link_idx, locked_arm_angles, default_base_z
    global base_weights, base_mean, base_scale
    global arm_weights, arm_mean, arm_scale

    conf_file_name = "robotnik_arm.json"
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(cur_dir, "..", "trained_model")

    with np.load(os.path.join(model_dir, "base_linear_regression.npz")) as base_model:
        base_weights = base_model["weights"]
        base_mean = base_model["input_mean"]
        base_scale = base_model["input_scale"]

    with np.load(os.path.join(model_dir, "arm_linear_regression.npz")) as arm_model:
        arm_weights = arm_model["weights"]
        arm_mean = arm_model["input_mean"]
        arm_scale = arm_model["input_scale"]

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

    configure_gripper_fingers(sim, client, rid, FIXED_FINGER_ANGLES)
    init_joint_angles = np.array(sim.GetMotorAngles(0))
    locked_arm_angles = init_joint_angles[4:11].copy()

    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, ["pybullet"], False, 0, cur_dir)
    ee_frame_id = dyn_model.pin_model.getFrameId("robot_j2s7s300_end_effector")

    # Find PyBullet link index for the end-effector
    ee_link_idx = None
    for j in range(client.getNumJoints(rid)):
        link_name = client.getJointInfo(rid, j)[12].decode()
        if "end_effector" in link_name:
            ee_link_idx = j
            break
    if ee_link_idx is None:
        for j in range(client.getNumJoints(rid)):
            link_name = client.getJointInfo(rid, j)[12].decode()
            if "link_7" in link_name:
                ee_link_idx = j
                break
    if ee_link_idx is None:
        ee_link_idx = client.getNumJoints(rid) - 1


def run_adversarial_test(case):
    """
    Simulates one episode under a specific adversarial perturbation.
    case: (idx, test_type, start_x, start_y, start_yaw, target_pos, param_dict)
    """
    idx, test_type, start_x, start_y, start_yaw, target_pos, params = case
    client = sim.pybullet_client
    rid = sim.bot[0].bot_pybullet

    # Reset robot to randomized initial pose
    sim.bot[0].link_floating_base_pos = [start_x, start_y, default_base_z]
    sim.bot[0].link_floating_base_ori = list(p.getQuaternionFromEuler([0.0, 0.0, start_yaw]))
    sim.ResetPose()
    sim.ReceiveObservation()

    # Initial target and standoff
    cur_target = target_pos.copy()
    delta_xy = cur_target[:2] - np.array([start_x, start_y])
    target_bearing = np.arctan2(delta_xy[1], delta_xy[0]) if np.linalg.norm(delta_xy) > 1e-4 else start_yaw
    standoff = compute_base_standoff(cur_target[2])
    base_goal_pos = cur_target[:2] - standoff * np.array([np.cos(target_bearing), np.sin(target_bearing)])

    cmd = MotorCommands()
    control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6
    stop_criteria = StopCriteria(
        reach_tolerance=0.01,
        max_steps=params.get("max_steps", 40000),
        min_time=1.5,
        stall_window=2.0,
    )
    time_step = sim.GetTimeStep()
    current_time = 0.0
    nav_done = False
    arm_phase_time = 0.0
    ee_dist_error = 0.0

    # Perturbation tracking flags
    perturbation_applied = False
    max_deflection = 0.0

    while True:
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        q_mes = np.array(sim.GetMotorAngles(0))
        qd_mes = np.array(sim.GetMotorVelocities(0))

        # Forward kinematics for end-effector
        q_pin = dyn_model.ReoderJoints2PinVec(np.concatenate((base_pos, base_ori, q_mes)), "pos")
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
        ee_dist_error = float(np.linalg.norm(cur_target - ee_pos))

        dist_to_goal = float(np.linalg.norm(base_pos[:2] - base_goal_pos))
        cur_target_bearing = float(np.arctan2(cur_target[1] - base_pos[1], cur_target[0] - base_pos[0]))
        in_reach, tf, tl, _ = is_target_in_arm_reach(base_pos, base_bearing, cur_target, expected_tf=standoff)

        # ----------------------------------------------------------------------
        # ADVERSARIAL STRESS LOGIC
        # ----------------------------------------------------------------------
        if test_type == "shove":
            # Apply a 30 N lateral/vertical force pulse on the gripper for 250 ms mid-reach
            if nav_done:
                arm_phase_time += time_step
                if 0.5 <= arm_phase_time < 0.75:
                    shove_force = params.get("shove_force", [25.0, 25.0, -20.0])
                    client.applyExternalForce(
                        objectUniqueId=rid,
                        linkIndex=ee_link_idx,
                        forceObj=shove_force,
                        posObj=[0.0, 0.0, 0.0],
                        flags=p.LINK_FRAME,
                    )
                    perturbation_applied = True
                    max_deflection = max(max_deflection, ee_dist_error)

        elif test_type == "target_jump":
            # Jump the target location while cruising (at t = 2.0s)
            if not perturbation_applied and current_time >= 2.0:
                jump_offset = params.get("jump_offset", np.array([0.6, -0.6, 0.1]))
                cur_target += jump_offset
                standoff = compute_base_standoff(cur_target[2])
                new_delta = cur_target[:2] - base_pos[:2]
                new_bearing = np.arctan2(new_delta[1], new_delta[0])
                base_goal_pos = cur_target[:2] - standoff * np.array([np.cos(new_bearing), np.sin(new_bearing)])
                nav_done = False  # Ensure base steers to new target
                perturbation_applied = True

        elif test_type == "payload":
            # Continuous downward gravity load from an unmodeled payload mass
            payload_mass = params.get("payload_mass", 0.5)  # kg
            payload_force = [0.0, 0.0, -payload_mass * 9.81]
            client.applyExternalForce(
                objectUniqueId=rid,
                linkIndex=ee_link_idx,
                forceObj=payload_force,
                posObj=[0.0, 0.0, 0.0],
                flags=p.WORLD_FRAME,
            )
            perturbation_applied = True

        elif test_type == "base_displace":
            # Push base 0.4m sideways/backwards right as reaching starts
            if nav_done and not perturbation_applied and arm_phase_time >= 0.2:
                displace = params.get("displace_offset", np.array([0.35, -0.25]))
                cur_pos = list(sim.GetBasePosition())
                cur_pos[0] += displace[0]
                cur_pos[1] += displace[1]
                sim.bot[0].link_floating_base_pos = cur_pos
                sim.ResetPose()
                perturbation_applied = True

            if nav_done:
                arm_phase_time += time_step
                # Closed-Loop Recovery: if kicked outside the sweet spot, resume base driving!
                if not in_reach and dist_to_goal > 0.20:
                    nav_done = False
                    arm_phase_time = 0.0

        # ----------------------------------------------------------------------
        # BASE CONTROLLER PREDICTION
        # ----------------------------------------------------------------------
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

        # Transition to arm when in sweet spot
        if not nav_done and (
            in_reach
            or (dist_to_goal < 0.09 and abs(tl) <= 0.05 and max_speed < 0.35)
            or current_time >= BASE_NAV_TIMEOUT
        ):
            nav_done = True

        if not nav_done:
            # Phase 1: Base driving, arm held locked
            tau_g = pin.computeGeneralizedGravity(dyn_model.pin_model, dyn_model.pin_data, q_pin)
            torque_arm = np.clip(
                tau_g[6:13] + 100.0 * (locked_arm_angles - q_mes[4:11]) - 30.0 * qd_mes[4:11],
                -MAX_ARM_TORQUE,
                MAX_ARM_TORQUE,
            )
        else:
            # Phase 2: Arm reaching
            wheel_cmds = np.zeros(4)
            tf_clamped = np.clip(tf, 0.55, 0.85)
            tl_clamped = np.clip(tl, -0.15, 0.15)
            th_clamped = np.clip(cur_target[2] - base_pos[2], 0.30, 1.15)
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

    strict_success = ee_dist_error <= 0.01
    task_success = ee_dist_error <= 0.05
    return (
        idx,
        test_type,
        start_x,
        start_y,
        start_yaw,
        cur_target,
        ee_dist_error,
        current_time,
        task_success,
        strict_success,
        max_deflection,
    )


def main():
    parser = argparse.ArgumentParser(description="Adversarial Robustness Evaluation")
    parser.add_argument("--total-tests", type=int, default=100, help="Total number of tests (split across 4 types)")
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4), help="CPU worker count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--payload-mass", type=float, default=0.5, help="Payload mass on gripper in kg (default: 0.5kg)")
    parser.add_argument("--shove-force", type=float, default=30.0, help="Shove force magnitude in N (default: 30N)")
    parser.add_argument("--displace-dist", type=float, default=0.4, help="Base displacement distance in m (default: 0.4m)")
    args = parser.parse_args()

    test_types = ["shove", "target_jump", "payload", "base_displace"]
    tests_per_type = args.total_tests // len(test_types)
    total_tests = tests_per_type * len(test_types)

    print("==================================================================")
    print("      ADVERSARIAL & PERTURBATION ROBUSTNESS BENCHMARK")
    print("==================================================================")
    print(f"Total Tests Planned          : {total_tests} ({tests_per_type} per category)")
    print(f"Parallel Workers             : {args.workers}")
    print(f"Categories Evaluated         :")
    print(f"  1. Arm Shove               : {args.shove_force:.1f} N impulse on gripper mid-reach")
    print(f"  2. Target Jump Midway      : Dynamic 0.6-1.0m target relocation en route")
    print(f"  3. Unmodeled Arm Payload   : +{args.payload_mass:.2f} kg constant downward mass")
    print(f"  4. Base Displacement       : {args.displace_dist:.2f} m lateral base displacement + closed-loop recovery")
    print("==================================================================\n")

    rng = np.random.default_rng(args.seed)
    test_cases = []
    case_idx = 0

    for t_type in test_types:
        for _ in range(tests_per_type):
            start_x = float(np.round(rng.uniform(-1.5, 1.5), 3))
            start_y = float(np.round(rng.uniform(-1.5, 1.5), 3))
            start_yaw = float(np.round(rng.uniform(-np.pi, np.pi), 3))

            angle = rng.uniform(-np.pi, np.pi)
            radius = rng.uniform(2.2, 3.8)
            tx = float(np.round(start_x + radius * np.cos(angle), 3))
            ty = float(np.round(start_y + radius * np.sin(angle), 3))
            tz = float(np.round(rng.uniform(0.4, 0.8), 3))
            target_pos = np.array([tx, ty, tz])

            # Category-specific parameters
            p_dict = {}
            if t_type == "shove":
                # Random 3D push direction with specified magnitude
                shove_dir = rng.normal(size=3)
                shove_dir /= np.linalg.norm(shove_dir)
                p_dict["shove_force"] = list(shove_dir * args.shove_force)
            elif t_type == "target_jump":
                jump_dir = rng.uniform(-1.0, 1.0, size=2)
                jump_dir = (jump_dir / np.linalg.norm(jump_dir)) * rng.uniform(0.6, 0.9)
                p_dict["jump_offset"] = np.array([jump_dir[0], jump_dir[1], rng.uniform(-0.1, 0.1)])
            elif t_type == "payload":
                p_dict["payload_mass"] = args.payload_mass
            elif t_type == "base_displace":
                disp_ang = rng.uniform(-np.pi, np.pi)
                p_dict["displace_offset"] = np.array([
                    args.displace_dist * np.cos(disp_ang),
                    args.displace_dist * np.sin(disp_ang),
                ])

            test_cases.append((case_idx, t_type, start_x, start_y, start_yaw, target_pos, p_dict))
            case_idx += 1

    start_wall_time = time.time()
    results = []

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers, initializer=init_worker) as pool:
        for count, res in enumerate(pool.imap_unordered(run_adversarial_test, test_cases, chunksize=2), 1):
            results.append(res)
            idx, t_type, sx, sy, syaw, tpos, err, sim_t, task_succ, strict_succ, max_def = res
            status = "SUCCESS (<=5cm)" if task_succ else "FAIL"
            if strict_succ:
                status += " [STRICT <=1cm]"
            extra = f" | Deflection: {max_def*100:.1f}cm" if t_type == "shove" and max_def > 0 else ""
            print(
                f"[{count:3d}/{total_tests}] Type: {t_type:<14s} | Final Err: {err*1000:5.1f} mm | "
                f"Time: {sim_t:4.1f}s | {status}{extra}"
            )

    results.sort(key=lambda r: r[0])
    wall_duration = time.time() - start_wall_time

    # Output directory
    out_dir = Path(__file__).parent / "adversarial_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = out_dir / "adversarial_results.csv"
    with csv_path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "index", "test_type", "start_x", "start_y", "start_yaw",
            "target_x", "target_y", "target_z", "final_ee_error_m",
            "sim_time_s", "task_success_5cm", "strict_success_1cm", "max_deflection_m"
        ])
        for r in results:
            writer.writerow([
                r[0], r[1], r[2], r[3], r[4],
                r[5][0], r[5][1], r[5][2],
                f"{r[6]:.5f}", f"{r[7]:.2f}", int(r[8]), int(r[9]), f"{r[10]:.5f}"
            ])

    # Category Breakdown Analysis
    print("\n" + "=" * 80)
    print("                     ADVERSARIAL BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    header = f"{'Category':<18s} | {'Count':>5s} | {'Task Succ (<=5cm)':>17s} | {'Strict (<=1cm)':>14s} | {'Median Err':>10s} | {'Mean Err':>9s}"
    print(header)
    print("-" * 80)

    summary_stats = {}
    for t_type in test_types:
        sub = [r for r in results if r[1] == t_type]
        errs = np.array([r[6] for r in sub])
        task_succ = np.array([r[8] for r in sub], dtype=bool)
        strict_succ = np.array([r[9] for r in sub], dtype=bool)

        t_rate = 100.0 * np.sum(task_succ) / len(sub)
        s_rate = 100.0 * np.sum(strict_succ) / len(sub)
        med_err = np.median(errs) * 1000.0  # mm
        mean_err = np.mean(errs) * 1000.0  # mm

        summary_stats[t_type] = {
            "count": len(sub),
            "task_rate": t_rate,
            "strict_rate": s_rate,
            "median_err": med_err,
            "mean_err": mean_err,
            "errors": errs,
        }

        print(
            f"{t_type:<18s} | {len(sub):5d} | {t_rate:15.1f}% | {s_rate:12.1f}% | "
            f"{med_err:8.1f} mm | {mean_err:7.1f} mm"
        )

    all_errs = np.array([r[6] for r in results])
    all_task = np.array([r[8] for r in results], dtype=bool)
    all_strict = np.array([r[9] for r in results], dtype=bool)
    print("-" * 80)
    print(
        f"{'OVERALL':<18s} | {len(results):5d} | {100.0 * np.sum(all_task) / len(results):15.1f}% | "
        f"{100.0 * np.sum(all_strict) / len(results):12.1f}% | "
        f"{np.median(all_errs)*1000.0:8.1f} mm | {np.mean(all_errs)*1000.0:7.1f} mm"
    )
    print("=" * 80)
    print(f"Total Wall-Clock Time: {wall_duration:.1f} seconds\n")

    # --------------------------------------------------------------------------
    # GENERATE PUBLICATION-QUALITY SUMMARY PLOTS
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    type_labels = ["Arm Shove\n(30 N pulse)", "Target Jump\n(0.6-1.0m shift)", "Payload Load\n(+0.5 kg)", "Base Displace\n(0.4m kick)"]
    x = np.arange(len(test_types))
    width = 0.35

    # Plot 1: Success Rate Bar Chart
    task_rates = [summary_stats[t]["task_rate"] for t in test_types]
    strict_rates = [summary_stats[t]["strict_rate"] for t in test_types]
    axes[0].bar(x - width / 2, task_rates, width, label="Task Success (<= 5 cm)", color="#2ca02c", alpha=0.85)
    axes[0].bar(x + width / 2, strict_rates, width, label="Strict Success (<= 1 cm)", color="#1f77b4", alpha=0.85)
    axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_xticks(x, type_labels)
    axes[0].set_title("Adversarial Robustness: Success Rate Comparison", fontsize=12, fontweight="bold")
    axes[0].axhline(100.0, color="gray", linestyle=":", alpha=0.5)
    axes[0].legend(loc="lower left")
    axes[0].grid(True, axis="y", alpha=0.3)

    # Plot 2: Final Error Distribution (Scatter / Boxplot in mm)
    for i, t in enumerate(test_types):
        errs_mm = summary_stats[t]["errors"] * 1000.0
        jitter = rng.uniform(-0.15, 0.15, size=len(errs_mm))
        axes[1].scatter(np.full_like(errs_mm, i) + jitter, errs_mm, color="navy", alpha=0.6, s=25, edgecolors="none")
        axes[1].hlines(summary_stats[t]["median_err"], i - 0.25, i + 0.25, color="crimson", linewidth=2.5)

    axes[1].axhline(10.0, color="orange", linestyle="--", linewidth=1.5, label="1 cm (Strict Threshold)")
    axes[1].axhline(50.0, color="red", linestyle="--", linewidth=1.5, label="5 cm (Task Threshold)")
    axes[1].set_ylabel("Final End-Effector Error (mm)")
    axes[1].set_xticks(x, type_labels)
    axes[1].set_title("Final EE Error Distribution under Adversaries", fontsize=12, fontweight="bold")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    plot_path = out_dir / "adversarial_summary.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"Results successfully saved to:")
    print(f"  - CSV  : {csv_path}")
    print(f"  - Plot : {plot_path}")


if __name__ == "__main__":
    main()

