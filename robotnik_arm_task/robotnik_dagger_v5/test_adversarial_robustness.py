"""
Adversarial & Perturbation Robustness Evaluation of the V6 DAgger policy
========================================================================
Port of test_adversarial_robustness.py from branch shivansh/linear_regression (written for the
decoupled linear-regression controller) to the V6 DAgger policy. The test cases (same RNG draws,
seed 42), the four perturbations, the stop rule, the success thresholds, the console output, the
CSV and the figure are the same as in that script, so the numbers of both models can be compared.

The four stress tests:
  1. Shove / Force Impulse: 30 N push on the end-effector link for 250 ms, 0.5 s into the reach.
  2. Target Jump Midway: at t = 2.0 s the target moves by 0.6-0.9 m (xy) and +-0.1 m (z).
  3. Arm Payload: a constant 0.5 kg * g downward force on the end-effector link.
  4. Base Displacement: 0.2 s into the reach, the base is teleported 0.4 m with sim.ResetPose()
     (as in the original, this also resets the arm joints and the base yaw to the start pose).

What differs from the linear version (only what the policy itself needs):
  * Control: the V6 policy (wheel net + HOLD/REACH arm nets on the 38 raw features of
    dagger_common.features) replaces the two linear-regression models. Commands are clipped to
    3.18 rad/s and 30 N*m, the limits of both the linear script and the DAgger runner.
  * Base goal: the DAgger policy's own convention (robotnik_dagger_collect.run): the goal lies
    `standoff` m short of the target on the ray from the world origin and faces along that ray,
    with the same height-dependent standoff curve. After a target jump it is recomputed the same way.
  * Navigation phase: navigation is done when the observable flag of dagger_common is set (within
    8 cm and 0.05 rad of the goal pose) instead of the linear model's reach-window test. As in
    the linear script, the test harness clears the phase after a target jump and after a base
    displacement that leaves the base > 0.2 m from its goal; the policy's latched phase
    (V6Policy.latched) is cleared with it, so the wheel net drives back.

Outputs (in --out-dir, default eval_v6/adversarial):
  - Live console progress and per-condition breakdown table.
  - adversarial_results.csv  (one row per test)
  - adversarial_summary.png  (success rates + final-error distribution per condition)

Run from robotnik_dagger_v5/:
  python test_adversarial_robustness.py [--total-tests 100] [--workers 12] [--model-dir trained_model_dagger_v6]
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
import torch

from simulation_and_control import pb, MotorCommands, PinWrapper
from robotnik_arm_control import StopCriteria, check_stop, configure_gripper_fingers, quaternion2bearing
from dagger_common import features
from robotnik_dagger_collect import FINGER_ANGLES, STANDOFF_Z, STANDOFF, load_policy

HERE = Path(__file__).resolve().parent
NAV_TIMEOUT = 45.0       # s, navigation-complete timeout of the DAgger runner (feature contract)
MAX_WHEEL_SPEED = 3.18   # rad/s
MAX_ARM_TORQUE = 30.0    # N*m

# Global worker handles (initialized once per CPU process)
sim = None
dyn_model = None
ee_frame_id = None
ee_link_idx = None
default_base_z = 0.0
policy = None


def base_goal(target):
    """Base goal (x, y) and bearing of the DAgger policy for a world-frame target."""
    standoff = float(np.interp(target[2], STANDOFF_Z, STANDOFF))
    bearing = float(np.arctan2(target[1], target[0])) if np.linalg.norm(target[:2]) > 1e-4 else 0.0
    goal = target[:2] - standoff * np.array([np.cos(bearing), np.sin(bearing)])
    return goal, bearing


def init_worker(model_dir, config):
    """Setup PyBullet, Pinocchio, and the trained policy in each worker."""
    global sim, dyn_model, ee_frame_id, ee_link_idx, default_base_z, policy
    torch.set_num_threads(1)
    policy = load_policy(model_dir, "cpu")

    sim = pb.SimInterface(config, conf_file_path_ext=str(HERE), use_gui=False)
    sim.SetFloorFriction(1.0)
    client = sim.pybullet_client
    rid = sim.bot[0].bot_pybullet
    default_base_z = float(sim.bot[0].link_floating_base_pos[2])

    for j in range(client.getNumJoints(rid)):
        if "wheel" in client.getJointInfo(rid, j)[1].decode():
            client.changeDynamics(rid, j, contactStiffness=30000, contactDamping=1000,
                                  anisotropicFriction=[1.0, 0.01, 1.0])
    sim.bot[0].servo_motor_model.set_motor_gains(400, 30)
    configure_gripper_fingers(sim, client, rid, FINGER_ANGLES)

    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    dyn_model = PinWrapper(config, "pybullet", ext_names, ["pybullet"], False, 0, str(HERE))
    ee_frame_id = dyn_model.pin_model.getFrameId("robot_j2s7s300_end_effector")

    # Find PyBullet link index for the end-effector (same lookup as the linear script)
    ee_link_idx = None
    for j in range(client.getNumJoints(rid)):
        if "end_effector" in client.getJointInfo(rid, j)[12].decode():
            ee_link_idx = j
            break
    if ee_link_idx is None:
        for j in range(client.getNumJoints(rid)):
            if "link_7" in client.getJointInfo(rid, j)[12].decode():
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

    # Reset robot to randomized initial pose (same reset as the linear script)
    sim.bot[0].link_floating_base_pos = [start_x, start_y, default_base_z]
    sim.bot[0].link_floating_base_ori = list(p.getQuaternionFromEuler([0.0, 0.0, start_yaw]))
    sim.ResetPose()
    sim.ReceiveObservation()
    policy.history.clear()
    policy.latched = False
    init_q = np.asarray(sim.GetMotorAngles(0))   # q_init features: arm angles at episode start

    cur_target = target_pos.copy()
    goal, goal_bearing = base_goal(cur_target)

    cmd = MotorCommands()
    control_modes = ["velocity"] * 4 + ["torque"] * 13
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
    note = ""

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
        dist_to_goal = float(np.linalg.norm(base_pos[:2] - goal))

        # ----------------------------------------------------------------------
        # ADVERSARIAL STRESS LOGIC (as in the linear script)
        # ----------------------------------------------------------------------
        if test_type == "shove":
            # Force pulse on the gripper for 250 ms mid-reach
            if nav_done:
                arm_phase_time += time_step
                if 0.5 <= arm_phase_time < 0.75:
                    client.applyExternalForce(
                        objectUniqueId=rid,
                        linkIndex=ee_link_idx,
                        forceObj=params.get("shove_force", [25.0, 25.0, -20.0]),
                        posObj=[0.0, 0.0, 0.0],
                        flags=p.LINK_FRAME,
                    )
                    perturbation_applied = True
                    max_deflection = max(max_deflection, ee_dist_error)

        elif test_type == "target_jump":
            # Jump the target location while cruising (at t = 2.0s)
            if not perturbation_applied and current_time >= 2.0:
                cur_target += params.get("jump_offset", np.array([0.6, -0.6, 0.1]))
                goal, goal_bearing = base_goal(cur_target)
                dist_to_goal = float(np.linalg.norm(base_pos[:2] - goal))
                nav_done = False  # Ensure base steers to new target
                perturbation_applied = True

        elif test_type == "payload":
            # Continuous downward gravity load from an unmodeled payload mass
            payload_mass = params.get("payload_mass", 0.5)  # kg
            client.applyExternalForce(
                objectUniqueId=rid,
                linkIndex=ee_link_idx,
                forceObj=[0.0, 0.0, -payload_mass * 9.81],
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
                # Closed-loop recovery: if kicked away from the goal, resume base driving
                if dist_to_goal > 0.20:
                    nav_done = False
                    arm_phase_time = 0.0

        # ----------------------------------------------------------------------
        # V6 DAgger POLICY
        # ----------------------------------------------------------------------
        try:
            x = features(base_pos, base_bearing, cur_target, ee_pos, goal, goal_bearing,
                         q_mes, qd_mes, init_q, current_time, NAV_TIMEOUT)
            if not nav_done and x[-1] >= 0.5:      # observable navigation-complete flag
                nav_done = True
            policy.latched = nav_done              # policy phase follows the harness phase
            wheel_cmds, torque_arm = policy.predict(x)
        except (ValueError, FloatingPointError) as exc:   # nonfinite state or output
            note = f"aborted: {exc}"
            break
        wheel_cmds = np.clip(wheel_cmds, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
        torque_arm = np.clip(torque_arm, -MAX_ARM_TORQUE, MAX_ARM_TORQUE)

        cmd.SetControlCmd(np.concatenate((wheel_cmds, torque_arm, np.zeros(6))), control_modes)
        sim.Step(cmd, "torque")

        if check_stop(stop_criteria, current_time, time_step, ee_dist_error, nav_done):
            break

        current_time += time_step

    strict_success = ee_dist_error <= 0.01
    task_success = ee_dist_error <= 0.05 and not note
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
        strict_success and not note,
        max_deflection,
        note,
    )


def make_test_cases(args, test_types, tests_per_type):
    """Same random draws, in the same order, as the linear script."""
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

            p_dict = {}
            if t_type == "shove":
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
    return test_cases, rng


def main():
    parser = argparse.ArgumentParser(description="Adversarial Robustness Evaluation (V6 DAgger policy)")
    parser.add_argument("--total-tests", type=int, default=100, help="Total number of tests (split across 4 types)")
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4), help="CPU worker count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--payload-mass", type=float, default=0.5, help="Payload mass on gripper in kg (default: 0.5kg)")
    parser.add_argument("--shove-force", type=float, default=30.0, help="Shove force magnitude in N (default: 30N)")
    parser.add_argument("--displace-dist", type=float, default=0.4, help="Base displacement distance in m (default: 0.4m)")
    parser.add_argument("--model-dir", default=str(HERE / "trained_model_dagger_v6"), help="V6 checkpoint folder")
    parser.add_argument("--config", default="robotnik_arm.json")
    parser.add_argument("--out-dir", default=str(HERE / "eval_v6" / "adversarial"))
    args = parser.parse_args()

    test_types = ["shove", "target_jump", "payload", "base_displace"]
    tests_per_type = args.total_tests // len(test_types)
    total_tests = tests_per_type * len(test_types)

    print("==================================================================")
    print("   ADVERSARIAL & PERTURBATION ROBUSTNESS BENCHMARK (V6 DAgger)")
    print("==================================================================")
    print(f"Model                        : {args.model_dir}")
    print(f"Total Tests Planned          : {total_tests} ({tests_per_type} per category)")
    print(f"Parallel Workers             : {args.workers}")
    print(f"Categories Evaluated         :")
    print(f"  1. Arm Shove               : {args.shove_force:.1f} N impulse on gripper mid-reach")
    print(f"  2. Target Jump Midway      : Dynamic 0.6-0.9m target relocation en route")
    print(f"  3. Unmodeled Arm Payload   : +{args.payload_mass:.2f} kg constant downward mass")
    print(f"  4. Base Displacement       : {args.displace_dist:.2f} m lateral base displacement + closed-loop recovery")
    print("==================================================================\n")

    test_cases, rng = make_test_cases(args, test_types, tests_per_type)

    start_wall_time = time.time()
    results = []

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers, initializer=init_worker, initargs=(args.model_dir, args.config)) as pool:
        for count, res in enumerate(pool.imap_unordered(run_adversarial_test, test_cases, chunksize=2), 1):
            results.append(res)
            idx, t_type, sx, sy, syaw, tpos, err, sim_t, task_succ, strict_succ, max_def, note = res
            status = "SUCCESS (<=5cm)" if task_succ else "FAIL"
            if strict_succ:
                status += " [STRICT <=1cm]"
            extra = f" | Deflection: {max_def*100:.1f}cm" if t_type == "shove" and max_def > 0 else ""
            if note:
                extra += f" | {note}"
            print(
                f"[{count:3d}/{total_tests}] Type: {t_type:<14s} | Final Err: {err*1000:5.1f} mm | "
                f"Time: {sim_t:4.1f}s | {status}{extra}", flush=True
            )

    results.sort(key=lambda r: r[0])
    wall_duration = time.time() - start_wall_time

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "adversarial_results.csv"
    with csv_path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "index", "test_type", "start_x", "start_y", "start_yaw",
            "target_x", "target_y", "target_z", "final_ee_error_m",
            "sim_time_s", "task_success_5cm", "strict_success_1cm", "max_deflection_m", "note"
        ])
        for r in results:
            writer.writerow([
                r[0], r[1], r[2], r[3], r[4],
                r[5][0], r[5][1], r[5][2],
                f"{r[6]:.5f}", f"{r[7]:.2f}", int(r[8]), int(r[9]), f"{r[10]:.5f}", r[11]
            ])

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
        t_rate = 100.0 * np.sum([r[8] for r in sub]) / len(sub)
        s_rate = 100.0 * np.sum([r[9] for r in sub]) / len(sub)
        med_err = np.median(errs) * 1000.0
        mean_err = np.mean(errs) * 1000.0
        summary_stats[t_type] = {"count": len(sub), "task_rate": t_rate, "strict_rate": s_rate,
                                 "median_err": med_err, "mean_err": mean_err, "errors": errs}
        print(
            f"{t_type:<18s} | {len(sub):5d} | {t_rate:15.1f}% | {s_rate:12.1f}% | "
            f"{med_err:8.1f} mm | {mean_err:7.1f} mm"
        )

    all_errs = np.array([r[6] for r in results])
    print("-" * 80)
    print(
        f"{'OVERALL':<18s} | {len(results):5d} | {100.0 * np.sum([r[8] for r in results]) / len(results):15.1f}% | "
        f"{100.0 * np.sum([r[9] for r in results]) / len(results):12.1f}% | "
        f"{np.median(all_errs)*1000.0:8.1f} mm | {np.mean(all_errs)*1000.0:7.1f} mm"
    )
    print("=" * 80)
    print(f"Total Wall-Clock Time: {wall_duration:.1f} seconds\n")

    # --------------------------------------------------------------------------
    # SUMMARY PLOTS (same layout as the linear script)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    type_labels = [f"Arm Shove\n({args.shove_force:.0f} N pulse)", "Target Jump\n(0.6-0.9m shift)",
                   f"Payload Load\n(+{args.payload_mass:g} kg)", f"Base Displace\n({args.displace_dist:g}m kick)"]
    x = np.arange(len(test_types))
    width = 0.35

    task_rates = [summary_stats[t]["task_rate"] for t in test_types]
    strict_rates = [summary_stats[t]["strict_rate"] for t in test_types]
    axes[0].bar(x - width / 2, task_rates, width, label="Task Success (<= 5 cm)", color="#2ca02c", alpha=0.85)
    axes[0].bar(x + width / 2, strict_rates, width, label="Strict Success (<= 1 cm)", color="#1f77b4", alpha=0.85)
    axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_xticks(x, type_labels)
    axes[0].set_title("V6 DAgger adversarial robustness: success rate", fontsize=12, fontweight="bold")
    axes[0].axhline(100.0, color="gray", linestyle=":", alpha=0.5)
    axes[0].legend(loc="lower left")
    axes[0].grid(True, axis="y", alpha=0.3)

    for i, t in enumerate(test_types):
        errs_mm = summary_stats[t]["errors"] * 1000.0
        jitter = rng.uniform(-0.15, 0.15, size=len(errs_mm))
        axes[1].scatter(np.full_like(errs_mm, i) + jitter, errs_mm, color="navy", alpha=0.6, s=25, edgecolors="none")
        axes[1].hlines(summary_stats[t]["median_err"], i - 0.25, i + 0.25, color="crimson", linewidth=2.5)

    axes[1].axhline(10.0, color="orange", linestyle="--", linewidth=1.5, label="1 cm (Strict Threshold)")
    axes[1].axhline(50.0, color="red", linestyle="--", linewidth=1.5, label="5 cm (Task Threshold)")
    axes[1].set_ylabel("Final End-Effector Error (mm)")
    axes[1].set_xticks(x, type_labels)
    axes[1].set_title("V6 DAgger: final EE error under adversaries", fontsize=12, fontweight="bold")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    plot_path = out_dir / "adversarial_summary.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print("Results successfully saved to:")
    print(f"  - CSV  : {csv_path}")
    print(f"  - Plot : {plot_path}")


if __name__ == "__main__":
    main()
