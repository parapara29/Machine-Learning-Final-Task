import os
os.environ["ROBOENV_HEADLESS"] = "1"

import csv
import multiprocessing as mp
import sys
import time
import numpy as np
import pybullet as p
import pinocchio as pin
import rf_sim   # adds robotnik_arm_task to sys.path, so it has to come before robotnik_arm_control
import test_random_spawn as base
from simulation_and_control import MotorCommands, wrap_angle
from robotnik_arm_control import StopCriteria, check_stop, quaternion2bearing
from features import wheel_features

# Robustness of E4 (RF wheels + DAgger V5.3-A arm) under the disturbances of Shivansh's
# test_adversarial_robustness.py. E4 starts at the origin with targets from the training range,
# because it already fails from far spawns without any disturbance. Every test uses the same targets.
# robotnik_arm.json must have noise_flag and collision_enable set to 0.

TESTS = ["baseline", "shove", "target_jump", "payload", "base_displace"]
NUM_PER_TEST = 25  # targets per test
SEED = 42  # seed of the targets and disturbances
SHOVE_FORCE = 30.0  # push on the gripper (N)
SHOVE_START, SHOVE_END = 0.5, 0.75  # push window after the base parks (s)
JUMP_TIME = 2.0  # time of the target jump (s)
JUMP_RANGE = (0.6, 0.9)  # horizontal target jump (m)
PAYLOAD_MASS = 0.5  # load on the gripper (kg)
DISPLACE_DIST = 0.4  # base displacement (m)
DISPLACE_TIME = 0.2  # displacement time after the base parks (s)
REACH_TOLERANCE = 0.01  # stop once the error is below this (m)
SUCCESS_TOLERANCE = 0.05  # success if the final error is below this (m)
MAX_STEPS = 40000  # step cap per episode (40 s)
NUM_WORKERS = base.NUM_WORKERS  # parallel processes

OUT_DIR = rf_sim.RUN_DIR / "eval_adversarial"  # results of this test
FIELDS = ["episode", "test", "target_x", "target_y", "target_z", "final_error", "time", "nav_time",  # CSV columns
          "goal_dist", "disturbed", "success", "stop"]

ee_link_idx = None  # PyBullet link of the gripper, set once per worker


def init_worker():
    # Same setup as the random-spawn test, plus the gripper link for the external forces.
    global ee_link_idx
    base.init_worker()
    robot = base.sim.bot[0]
    if robot.noise_flag or robot.self_collision_enabled:
        raise RuntimeError("Set noise_flag and collision_enable to 0 in robotnik_arm.json")
    client = base.sim.pybullet_client
    rid = robot.bot_pybullet
    links = [client.getJointInfo(rid, j)[12].decode() for j in range(client.getNumJoints(rid))]
    ee_link_idx = next(j for j, name in enumerate(links) if "end_effector" in name)


def base_goal(target, base_xy):
    # Parking pose facing the target from base_xy.
    delta = target[:2] - base_xy
    bearing = np.arctan2(delta[1], delta[0])
    standoff = float(np.interp(target[2], rf_sim.STANDOFF_Z_POINTS, rf_sim.STANDOFF_POINTS))
    return target[:2] - standoff * np.array([np.cos(bearing), np.sin(bearing)]), bearing


def run_episode(case):
    idx, test, target, params = case
    sim, dyn_model, ee_frame_id = base.sim, base.dyn_model, base.ee_frame_id
    rf_wheel, dagger_arm, dagger_features = base.rf_wheel, base.dagger_arm, base.dagger_features
    client = sim.pybullet_client
    rid = sim.bot[0].bot_pybullet

    # Default pose at the origin.
    sim.bot[0].link_floating_base_pos = [0.0, 0.0, base.default_base_z]
    sim.bot[0].link_floating_base_ori = list(p.getQuaternionFromEuler([0.0, 0.0, 0.0]))
    sim.ResetPose()
    sim.ReceiveObservation()
    q_init = np.array(sim.GetMotorAngles(0))
    dagger_arm.history.states.clear()
    dagger_arm.navigation_latched = False

    target = target.copy()
    goal_pos, goal_bearing = base_goal(target, np.zeros(2))
    cmd = MotorCommands()
    control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6
    stop_criteria = StopCriteria(reach_tolerance=REACH_TOLERANCE, max_steps=MAX_STEPS, min_time=1.5, stall_window=2.0)
    time_step = sim.GetTimeStep()
    current_time = 0.0
    step = 0
    nav_done = False
    nav_time = np.nan
    arm_time = 0.0  # time since the base parked
    disturbed = test in ("baseline", "payload")  # the others finish their disturbance during the episode

    def restart_stall_timer():
        stop_criteria.best_ee_error = np.inf
        stop_criteria.best_ee_error_time = current_time

    def back_to_driving():
        # The base has to drive again, so the supervisor and the arm leave the reaching phase.
        nonlocal nav_done, arm_time
        nav_done = False
        arm_time = 0.0
        dagger_arm.navigation_latched = False
        restart_stall_timer()

    while True:
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        q_mes = np.array(sim.GetMotorAngles(0))
        qd_mes = np.array(sim.GetMotorVelocities(0))

        q_pin = dyn_model.ReoderJoints2PinVec(np.concatenate((base_pos, base_ori, q_mes)), "pos")
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
        ee_dist_error = float(np.linalg.norm(target - ee_pos))

        # Supervisor (same rule as rf_sim.py)
        dist = np.linalg.norm(base_pos[:2] - goal_pos)
        ang = abs(wrap_angle(goal_bearing - base_bearing))
        nav_done = nav_done or (dist < 0.15 and ang < 0.05) or current_time > rf_sim.BASE_NAV_TIMEOUT
        if nav_done and np.isnan(nav_time):
            nav_time = current_time

        # Disturbances
        if test == "shove" and nav_done:
            arm_time += time_step
            if SHOVE_START <= arm_time < SHOVE_END:
                client.applyExternalForce(rid, ee_link_idx, params["force"], [0.0, 0.0, 0.0], p.LINK_FRAME)
            elif arm_time >= SHOVE_END and not disturbed:
                disturbed = True
                restart_stall_timer()
        elif test == "target_jump" and not disturbed and current_time >= JUMP_TIME:
            target = params["new_target"].copy()
            goal_pos, goal_bearing = base_goal(target, base_pos[:2])
            disturbed = True
            back_to_driving()
        elif test == "payload":
            client.applyExternalForce(rid, ee_link_idx, [0.0, 0.0, -PAYLOAD_MASS * 9.81], [0.0, 0.0, 0.0], p.WORLD_FRAME)
        elif test == "base_displace" and nav_done and not disturbed:
            arm_time += time_step
            if arm_time >= DISPLACE_TIME:
                # Only the base moves: sim.ResetPose() would also reset the arm and the yaw.
                position, orientation = client.getBasePositionAndOrientation(rid)
                moved = [position[0] + params["offset"][0], position[1] + params["offset"][1], position[2]]
                client.resetBasePositionAndOrientation(rid, moved, orientation)
                disturbed = True
                back_to_driving()

        # Wheels: random forest, predicted every PREDICT_EVERY steps and held in between
        if step % rf_sim.PREDICT_EVERY == 0:
            s = rf_sim.sim_state(base_pos, base_bearing, q_mes, qd_mes, target, goal_pos, goal_bearing,
                                 ee_pos, nav_done, q_init)
            wheel_cmds = rf_wheel.predict(wheel_features(s, True))[0]
            wheel_cmds = np.clip(wheel_cmds, -rf_sim.MAX_WHEEL_SPEED, rf_sim.MAX_WHEEL_SPEED)

        # Arm: DAgger MLP every step, with the supervisor's flag as its navigation input
        x = dagger_features(base_pos, base_bearing, target, ee_pos, goal_pos, goal_bearing,
                            q_mes, qd_mes, q_init, current_time)
        x[-1] = float(nav_done)
        _, torque_arm = dagger_arm.predict(x)
        torque_arm = np.clip(torque_arm, -rf_sim.MAX_ARM_TORQUE, rf_sim.MAX_ARM_TORQUE)

        commands = np.zeros(4) if nav_done else wheel_cmds
        cmd.SetControlCmd(np.concatenate((commands, torque_arm, np.zeros(6))), control_modes)
        sim.Step(cmd, "torque")

        # An episode can only end early once its disturbance has happened.
        stop_message = check_stop(stop_criteria, current_time, time_step, ee_dist_error, nav_done)
        if stop_message and (disturbed or stop_message.startswith("Reached step cap")):
            break

        current_time += time_step
        step += 1

    return {
        "episode": idx, "test": test, "target_x": target[0], "target_y": target[1], "target_z": target[2],
        "final_error": ee_dist_error, "time": current_time, "nav_time": nav_time, "goal_dist": float(dist),
        "disturbed": int(disturbed), "success": int(ee_dist_error <= SUCCESS_TOLERANCE), "stop": stop_message,
    }


def in_training_range(target):
    return 0.9 <= np.hypot(target[0], target[1]) <= 1.8 and 0.3 <= target[2] <= 0.9


def make_cases(num_per_test):
    # The same targets in every test, with the disturbance drawn per target.
    rng = np.random.default_rng(SEED)
    angle = rng.uniform(-np.pi, np.pi, NUM_PER_TEST)
    radius = rng.uniform(0.9, 1.8, NUM_PER_TEST)
    height = rng.uniform(0.3, 0.9, NUM_PER_TEST)
    targets = [np.array([r * np.cos(a), r * np.sin(a), z]) for a, r, z in zip(angle, radius, height)]

    cases = []
    for k, target in enumerate(targets):
        direction = rng.normal(size=3)
        force = list(SHOVE_FORCE * direction / np.linalg.norm(direction))
        displace_angle = rng.uniform(-np.pi, np.pi)
        offset = DISPLACE_DIST * np.array([np.cos(displace_angle), np.sin(displace_angle)])
        new_target = target
        while new_target is target or not in_training_range(new_target):
            jump_angle, jump = rng.uniform(-np.pi, np.pi), rng.uniform(*JUMP_RANGE)
            new_target = target + [jump * np.cos(jump_angle), jump * np.sin(jump_angle), rng.uniform(-0.1, 0.1)]
        params = {"force": force, "offset": offset, "new_target": new_target}
        if k < num_per_test:
            cases += [(t * NUM_PER_TEST + k, test, target, params) for t, test in enumerate(TESTS)]
    return cases


def main():
    # python test_adversarial.py     -> all targets
    # python test_adversarial.py 2   -> only the first 2 targets of every test (same cases as in the full run)
    num_per_test = int(sys.argv[1]) if len(sys.argv) > 1 else NUM_PER_TEST
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "E4.csv"
    cases = make_cases(num_per_test)
    done = {int(row["episode"]) for row in base.read_rows(path)}
    jobs = [case for case in cases if case[0] not in done]
    print(f"{len(jobs)} episodes to run, {len(done)} already done, {NUM_WORKERS} workers")

    start = time.time()
    if jobs:
        new_file = not path.exists()
        ctx = mp.get_context("spawn")
        with open(path, "a", newline="") as f, ctx.Pool(NUM_WORKERS, initializer=init_worker) as pool:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                writer.writeheader()
            for count, row in enumerate(pool.imap_unordered(run_episode, jobs), 1):
                writer.writerow(row)
                f.flush()
                print(f"[{count:3d}/{len(jobs)}] {row['test']:13s} error {row['final_error']:.4f} m | "
                      f"{row['time']:5.2f} s | {'SUCCESS' if row['success'] else 'FAIL'}")

    wanted = {case[0] for case in cases}
    rows = [row for row in base.read_rows(path) if int(row["episode"]) in wanted]
    print(f"\n{'test':14s} {'n':>3s} {'success':>8s} {'95% Wilson':>15s} {'<=1 cm':>7s} {'median':>9s} {'mean':>9s} {'parked':>7s}")
    for test in TESTS:
        sub = [row for row in rows if row["test"] == test]
        if not sub:
            continue
        error = np.array([float(row["final_error"]) for row in sub])
        success = sum(int(row["success"]) for row in sub)
        parked = np.mean([float(row["goal_dist"]) < 0.15 for row in sub])
        low, high = base.wilson(success, len(sub))
        print(f"{test:14s} {len(sub):3d} {100 * success / len(sub):7.1f}% [{low:5.1f}, {high:5.1f}]% "
              f"{100 * np.mean(error <= 0.01):6.1f}% {1000 * np.median(error):6.1f} mm {1000 * error.mean():6.1f} mm "
              f"{100 * parked:6.1f}%")
    print(f"\nWall time {time.time() - start:.1f} s. Saved results to {path}")


if __name__ == "__main__":
    main()
