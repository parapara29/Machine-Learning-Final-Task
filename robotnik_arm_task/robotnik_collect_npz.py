import os
from multiprocessing import Process
from pathlib import Path

import numpy as np

NUM_TARGETS = 500
SEED = 42
OUTPUT_DIR = Path(__file__).parent / "data_npz_noise"
# Keep every n-th simulation step (1 = every step at 1 kHz).
RECORD_EVERY = 1

# noise injection, change this to 0 to null it
ARM_NOISE_STD = 5.0
WHEEL_NOISE_STD = 0.2
NOISE_HOLD_STEPS = 100

PHASES = ["align", "drive", "orient"]


def sample_targets():
    # Same sampling as collect_data.py, so both datasets share the same targets.
    rng = np.random.default_rng(SEED)
    targets = []
    for _ in range(NUM_TARGETS):
        angle = rng.uniform(-np.pi, np.pi)
        radius = rng.uniform(0.9, 1.8)
        targets.append([radius * np.cos(angle), radius * np.sin(angle), rng.uniform(0.3, 0.9)])
    return np.array(targets)


def record(target, output, episode):
    os.environ["ROBOENV_HEADLESS"] = "1"
    import robotnik_arm_task as simulation

    simulation.TARGET_EE_POS = target
    simulation.ARM_NOISE_STD = ARM_NOISE_STD
    simulation.WHEEL_NOISE_STD = WHEEL_NOISE_STD
    simulation.NOISE_HOLD_STEPS = NOISE_HOLD_STEPS
    simulation.NOISE_SEED = SEED + episode

    base_controller = simulation.base_controller
    arm_controller = simulation.arm_controller
    check_stop = simulation.check_stop
    log = {key: [] for key in [
        "time", "base_pos", "base_bearing", "q_mes", "qd_mes", "wheel_cmds", "torque_arm",
        "q_des_arm", "qd_des_arm", "phase", "nav_done", "ee_pos", "ee_error",
    ]}
    step = {}
    step_count = 0
    stop_message = ""

    # Capture the base measurements, the wheel commands and the navigation phase that produced them.
    def record_base(state, base_pos, base_bearing, goal_pos, goal_bearing, current_time, params):
        wheel_cmds = base_controller(state, base_pos, base_bearing, goal_pos, goal_bearing, current_time, params)
        step.update(
            time=current_time, base_pos=base_pos, base_bearing=base_bearing, wheel_cmds=wheel_cmds,
            phase=PHASES.index(state.phase) if not state.done else len(PHASES),
        )
        return wheel_cmds

    # The arm controller runs next, so all values for this step are now available.
    def record_arm(model, q_pin, q_arm, qd_arm, q_des, q_hold, target_pos, ee_pos, *args, **kwargs):
        nonlocal step_count
        result = arm_controller(model, q_pin, q_arm, qd_arm, q_des, q_hold, target_pos, ee_pos, *args, **kwargs)
        if step_count % RECORD_EVERY == 0:
            step.update(
                q_mes=q_arm, qd_mes=qd_arm, torque_arm=result[0], q_des_arm=result[1], qd_des_arm=result[2],
                nav_done=kwargs["reach_active"], ee_pos=ee_pos, ee_error=np.linalg.norm(target_pos - ee_pos),
            )
            for key, value in step.items():
                log[key].append(np.copy(value))
        step_count += 1
        return result

    # Keep the reason the episode ended (reached, stalled or step cap).
    def record_stop(*args):
        nonlocal stop_message
        message = check_stop(*args)
        if message:
            stop_message = message
        return message

    simulation.base_controller = record_base
    simulation.arm_controller = record_arm
    simulation.check_stop = record_stop
    simulation.main()

    # Write to a temporary file first so an interrupted run never leaves a half-written episode.
    temp = output.with_name(output.stem + ".tmp.npz")
    np.savez_compressed(
        temp,
        arm_noise_std  =ARM_NOISE_STD,
        wheel_noise_std =WHEEL_NOISE_STD,
        noise_hold_steps =NOISE_HOLD_STEPS,
        noise_seed = SEED + episode,
        target=target,
        stop_message=stop_message,
        reached=stop_message.startswith("Target reached"),
        phase_names=PHASES + ["done"],
        **{key: np.array(value) for key, value in log.items()},
    )
    os.replace(temp, output)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = sample_targets()
    for episode, target in enumerate(targets):
        output = OUTPUT_DIR / f"episode_{episode:04d}.npz"
        # Episodes that already exist are skipped, so a stopped run can be resumed.
        if output.exists():
            print(f"Episode {episode + 1}/{NUM_TARGETS}: already recorded, skipping", flush=True)
            continue
        print(f"Episode {episode + 1}/{NUM_TARGETS}: target={target}", flush=True)

        # Run one simulation at a time, each in a fresh process.
        process = Process(target=record, args=(target, output, episode))
        process.start()
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"Episode {episode} failed")

    files = sorted(OUTPUT_DIR.glob("episode_[0-9][0-9][0-9][0-9].npz"))
    summaries = [np.load(file) for file in files]
    np.savez(
        OUTPUT_DIR / "targets.npz",
        episode=[file.stem for file in files],
        target=np.array([summary["target"] for summary in summaries]),
        reached=np.array([bool(summary["reached"]) for summary in summaries]),
        stop_message=np.array([str(summary["stop_message"]) for summary in summaries]),
        seed=SEED,
    )
    print(f"Saved {len(files)} episodes to {OUTPUT_DIR} ({sum(s['reached'] for s in summaries)} reached the target)")


if __name__ == "__main__":
    main()
