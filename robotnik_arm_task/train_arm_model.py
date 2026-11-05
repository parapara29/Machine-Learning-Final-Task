"""
Phase 2: 7-DOF Kinova Arm Linear Regression Training
=====================================================
This script trains the linear regression model for the robotic arm.
It maps 25 engineered features to the 7 arm motor joint torques (tau_arm_0..6).

Key Ideas:
- Removes gripper position (x_ee) to eliminate the positive-feedback "anti-spring" explosion.
- Uses joint angles q (7) and velocities qd (7) to learn stabilizing -Kp*q spring and -Kd*qd damping.
- Includes 3D body-frame target offsets (tf, tl, th) and quadratic cross terms to fit workspace curvature.
- Includes a low-Z hinge feature (max(0, 0.46 - th)) to bend the elbow roll joint (J3) when Shoulder 2 bottoms out.
- Uses DART state perturbation (Gaussian noise injection) so the model learns restoring forces.
- Filters out failed timeouts and the negative elbow branch to avoid averaging mirror postures to zero.
- Saves the trained weights to `trained_model/arm_linear_regression.npz`.

Run:
  uv run train_arm_model.py
"""

import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path(__file__).parent / "data" / "data_randomized"
MODEL_PATH = Path(__file__).parent.parent / "trained_model" / "arm_linear_regression.npz"
RESULTS_DIR = Path(__file__).parent / "arm_training_results"
TEST_FRACTION = 0.2
SEED = 42

# EPD gains from the expert controller code.
KP_ARM = 100.0
KD_ARM = 30.0

# Shoulder offset in the robot base frame [forward, left, up] (m) and physical reach limit (m)
SHOULDER_OFFSET = np.array([0.204, 0.0, 0.495])
ARM_REACH_LIMIT = 0.85

JOINT_POS_COLUMNS = [f"q_mes_{i}" for i in range(4, 11)]
JOINT_VEL_COLUMNS = [f"qd_mes_{i}" for i in range(4, 11)]
WHEEL_COLUMNS = [f"wheel_cmds_{i}" for i in range(4)]
OUTPUT_COLUMNS = [f"torque_arm_{i}" for i in range(7)]
STATE_COLUMNS = (
    ["base_x", "base_y", "base_z", "base_bearing", "target_x", "target_y", "target_z"]
    + JOINT_POS_COLUMNS
    + JOINT_VEL_COLUMNS
)

INPUT_COLUMNS = (
    JOINT_POS_COLUMNS
    + JOINT_VEL_COLUMNS
    + [
        "target_forward",
        "target_left",
        "target_height",
        "target_forward_sq",
        "target_left_sq",
        "target_height_sq",
        "target_forward_height",
        "target_forward_left",
        "target_left_height",
        "low_z_hinge",
        "low_z_hinge_sq",
    ]
)


def target_in_base_frame(data):
    """Transform the 3D world target into the robot base frame (forward, left, height)."""
    yaw = np.asarray(data["base_bearing"])
    c, s = np.cos(yaw), np.sin(yaw)
    dx = np.asarray(data["target_x"]) - data["base_x"]
    dy = np.asarray(data["target_y"]) - data["base_y"]
    #Same thing as train base model code, we need to convert the target position from world frame to robot's body frame, so that the model can learn to reach the target in the robot's body frame.
    target_forward = c * dx + s * dy
    target_left = -s * dx + c * dy
    target_height = np.asarray(data["target_z"]) - data["base_z"]
    return target_forward, target_left, target_height


def make_arm_features(data):
    target_forward, target_left, target_height = target_in_base_frame(data)
    q = np.column_stack([np.asarray(data[c]) for c in JOINT_POS_COLUMNS])
    qd = np.column_stack([np.asarray(data[c]) for c in JOINT_VEL_COLUMNS])
    low_z = np.maximum(0.0, 0.46 - target_height)

    return np.column_stack((
        q,
        qd,
        target_forward,
        target_left,
        target_height,
        target_forward ** 2,
        target_left ** 2,
        target_height ** 2,
        target_forward * target_height,
        target_forward * target_left,
        target_left * target_height,
        low_z,
        low_z ** 2,
    ))


def load_episode(file):
    with file.open() as fp:
        header = next(csv.reader(fp))
    needed = ["time", "ee_error", "ee_x", "ee_y", "ee_z"] + STATE_COLUMNS + WHEEL_COLUMNS + OUTPUT_COLUMNS
    indices = [header.index(col) for col in needed]
    raw = np.loadtxt(file, delimiter=",", skiprows=1, usecols=indices)
    return {col: raw[:, i] for i, col in enumerate(needed)}


def load_arm_dataset(files, rng):
    X_chunks, y_chunks = [], []
    workspace_points = []
    skipped = 0

    for file in files:
        ep = load_episode(file)

        # Throw away bad episodes:
        # If the arm didn't reach the target (error > 2cm at the end), skip it!
        # If the target is super low (Joint 2 hits its bottom limit at 0.88 rad), the arm has two mirror ways
        #   to twist its elbow (left vs right). If we train on both, linear regression gets confused and averages
        #   them to 0 (arm gets stuck). So we only keep the episodes that twisted to the positive branch (q6 >= 0.2).
        if ep["ee_error"][-1] > 0.02 or (ep["q_mes_5"][-1] < 0.88 and ep["q_mes_6"][-1] < 0.2):
            skipped += 1
            continue

        # Only look at the Arm Phase:
        # We only want rows where wheels are completely stopped.
        wheels = np.column_stack([ep[c] for c in WHEEL_COLUMNS])
        arm_phase = np.max(np.abs(wheels), axis=1) <= 1e-6
        idx = np.where(arm_phase)[0]
        if len(idx) == 0:
            skipped += 1
            continue

        # Record where the target sits in the base frame once the wheels stop
        tf, tl, th = target_in_base_frame({k: ep[k][idx[-1]] for k in ["base_x", "base_y", "base_z", "base_bearing", "target_x", "target_y", "target_z"]})
        workspace_points.append([tf, tl, th])

        # We pick 25 evenly spaced snapshots during the arm reach. To not have too much data
        step = max(1, len(idx) // 25)
        sel = idx[::step]

        # Grab the final goal pose and final gravity holding torque:
        # At the very end of the episode, the arm is stopped right at the target.
        # So tau_final is the exact torque needed to hold the arm up against gravity
        q_final = np.array([ep[c][-1] for c in JOINT_POS_COLUMNS])
        tau_final = np.array([ep[c][-1] for c in OUTPUT_COLUMNS])

        sub = {col: ep[col][sel].copy() for col in STATE_COLUMNS}
        q_clean = np.column_stack([sub[c] for c in JOINT_POS_COLUMNS])
        qd_clean = np.column_stack([sub[c] for c in JOINT_VEL_COLUMNS])

        # DART noise trick (teach the model to fight back if pushed):
        # If the arm only sees perfect smooth trajectories, it never learns how to recover from a bump.
        # So we add small random nudges to the angles (dq) and speeds (dqd).
        dq = rng.normal(0.0, 0.08, size=q_clean.shape)
        dqd = rng.normal(0.0, 0.15, size=qd_clean.shape)
        q_pert = q_clean + dq
        qd_pert = qd_clean + dqd

        for j, col in enumerate(JOINT_POS_COLUMNS):
            sub[col] = q_pert[:, j]
        for j, col in enumerate(JOINT_VEL_COLUMNS):
            sub[col] = qd_pert[:, j]

        tau_target = tau_final + KP_ARM * (q_final - q_pert) - KD_ARM * qd_pert

        # Add the features and target torques to our training pile
        X_chunks.append(make_arm_features(sub))
        y_chunks.append(tau_target)

    return np.vstack(X_chunks), np.vstack(y_chunks), np.array(workspace_points), skipped


def save_workspace_plot(workspace_points, bounds):
    tf, tl, th = workspace_points[:, 0], workspace_points[:, 1], workspace_points[:, 2]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(tf, tl, s=10, alpha=0.35, label="Reached targets (base frame)")
    axes[0].axvline(bounds["forward_min"], color="r", linestyle="--", label="Stop window")
    axes[0].axvline(bounds["forward_max"], color="r", linestyle="--")
    axes[0].axhline(-bounds["left_max"], color="r", linestyle="--")
    axes[0].axhline(bounds["left_max"], color="r", linestyle="--")
    axes[0].set_xlabel("Target Forward in Base Frame (m)")
    axes[0].set_ylabel("Target Left in Base Frame (m)")
    axes[0].set_title("Top-Down Arm Reachable Workspace")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(tf, th, s=10, alpha=0.35, color="tab:orange", label="Reached targets (base frame)")
    theta = np.linspace(-np.pi / 3, np.pi / 2, 200)
    axes[1].plot(
        SHOULDER_OFFSET[0] + ARM_REACH_LIMIT * np.cos(theta),
        SHOULDER_OFFSET[2] + ARM_REACH_LIMIT * np.sin(theta),
        "k--",
        label=f"Shoulder {ARM_REACH_LIMIT}m reach sphere",
    )
    axes[1].axvline(bounds["forward_min"], color="r", linestyle="--", label="Forward stop window")
    axes[1].axvline(bounds["forward_max"], color="r", linestyle="--")
    axes[1].set_xlabel("Target Forward in Base Frame (m)")
    axes[1].set_ylabel("Target Height above Base (m)")
    axes[1].set_title("Side-View Arm Reachable Workspace")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "arm_workspace.png", dpi=150)
    plt.close(fig)


def save_plots(metrics, weights, input_mean, input_scale, train_file, test_file):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(OUTPUT_COLUMNS))
    width = 0.35

    axes[0].bar(x - width / 2, [metrics[o]["train_rmse"] for o in OUTPUT_COLUMNS], width, label="Train")
    axes[0].bar(x + width / 2, [metrics[o]["test_rmse"] for o in OUTPUT_COLUMNS], width, label="Test")
    axes[0].set_xticks(x, OUTPUT_COLUMNS, rotation=25)
    axes[0].set_ylabel("RMSE (Nm)")
    axes[0].set_title("Arm Torque RMSE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(x - width / 2, [metrics[o]["train_r2"] for o in OUTPUT_COLUMNS], width, label="Train")
    axes[1].bar(x + width / 2, [metrics[o]["test_r2"] for o in OUTPUT_COLUMNS], width, label="Test")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xticks(x, OUTPUT_COLUMNS, rotation=25)
    axes[1].set_ylabel("R² score")
    axes[1].set_title("Arm Torque R²")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "metrics.png", dpi=150)
    plt.close(fig)

    for split, file in [("train", train_file), ("test", test_file)]:
        ep = load_episode(file)
        wheels = np.column_stack([ep[c] for c in WHEEL_COLUMNS])
        arm_phase = np.max(np.abs(wheels), axis=1) <= 1e-6
        features = (make_arm_features(ep) - input_mean) / input_scale
        predictions = np.clip(np.column_stack((np.ones(len(ep["time"])), features)) @ weights, -30.0, 30.0)

        fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
        for i, output in enumerate(OUTPUT_COLUMNS):
            ax = axes.flat[i]
            ax.plot(ep["time"][arm_phase], ep[output][arm_phase], label="Expert (arm phase)")
            ax.plot(ep["time"][arm_phase], predictions[arm_phase, i], "--", label="Predicted (arm model)")
            ax.set_title(output)
            ax.set_ylabel("Nm")
            ax.grid(True, alpha=0.3)
        axes.flat[-1].set_visible(False)
        axes.flat[0].legend()
        fig.suptitle(f"{split.capitalize()} episode (arm reaching phase): {file.name}")
        fig.supxlabel("Simulation time (s)")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"{split}_predictions.png", dpi=150)
        plt.close(fig)


def main():
    files = sorted(DATA_DIR.glob("episode_*.csv"))
    if len(files) < 2:
        raise ValueError(f"Need at least two episode CSVs in {DATA_DIR}")

    rng = np.random.default_rng(SEED)
    rng.shuffle(files)
    num_test = int(np.ceil(len(files) * TEST_FRACTION))
    test_files = files[:num_test]
    train_files = files[num_test:]

    raw_train_X, y_train, ws_train, skipped_train = load_arm_dataset(train_files, rng)
    raw_test_X, y_test, ws_test, skipped_test = load_arm_dataset(test_files, rng)
    workspace_points = np.vstack((ws_train, ws_test))

    # Empirical reachable workspace bounds in the base frame (1st to 99th percentile)
    workspace_bounds = {
        "forward_min": float(np.percentile(workspace_points[:, 0], 1)),
        "forward_max": float(np.percentile(workspace_points[:, 0], 99)),
        "left_max": float(np.percentile(np.abs(workspace_points[:, 1]), 95)),
        "height_min": float(np.min(workspace_points[:, 2])),
        "height_max": float(np.max(workspace_points[:, 2])),
    }

    print("==================================================================")
    print("Arm Reachable Workspace in Robot Base Frame (from expert data):")
    print(f"  Forward reach sweet spot : [{workspace_bounds['forward_min']:.3f}, {workspace_bounds['forward_max']:.3f}] m")
    print(f"  Lateral alignment window : [-{workspace_bounds['left_max']:.3f}, +{workspace_bounds['left_max']:.3f}] m")
    print(f"  Height above base        : [{workspace_bounds['height_min']:.3f}, {workspace_bounds['height_max']:.3f}] m")
    print(f"  Max 3D distance from shoulder {SHOULDER_OFFSET.tolist()} : {ARM_REACH_LIMIT:.2f} m")
    print("==================================================================\n")

    input_mean = raw_train_X.mean(axis=0)
    input_scale = raw_train_X.std(axis=0)
    input_scale[np.isclose(input_scale, 0)] = 1.0

    X_train = np.column_stack((np.ones(len(raw_train_X)), (raw_train_X - input_mean) / input_scale))
    X_test = np.column_stack((np.ones(len(raw_test_X)), (raw_test_X - input_mean) / input_scale))
    weights = np.linalg.pinv(X_train) @ y_train

    print(f"Training: {len(train_files) - skipped_train} valid episodes ({skipped_train} skipped), {len(y_train)} samples")
    print(f"Testing : {len(test_files) - skipped_test} valid episodes ({skipped_test} skipped), {len(y_test)} samples\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {o: {} for o in OUTPUT_COLUMNS}
    for split, X, actual in [("train", X_train, y_train), ("test", X_test, y_test)]:
        sq_err = np.sum((actual - X @ weights) ** 2, axis=0)
        tot_var = np.sum((actual - actual.mean(axis=0)) ** 2, axis=0)
        rmse = np.sqrt(sq_err / len(actual))
        r2 = 1.0 - sq_err / tot_var
        for i, col in enumerate(OUTPUT_COLUMNS):
            metrics[col][f"{split}_rmse"] = rmse[i]
            metrics[col][f"{split}_r2"] = r2[i]

    print(f"{'Output':14s} | {'Train RMSE':>10s} | {'Train R2':>9s} | {'Test RMSE':>10s} | {'Test R2':>9s}")
    print("-" * 64)
    with (RESULTS_DIR / "metrics.csv").open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["output", "train_rmse", "train_r2", "test_rmse", "test_r2"])
        for col in OUTPUT_COLUMNS:
            m = metrics[col]
            print(f"{col:14s} | {m['train_rmse']:10.4f} | {m['train_r2']:9.4f} | {m['test_rmse']:10.4f} | {m['test_r2']:9.4f}")
            writer.writerow([col, m["train_rmse"], m["train_r2"], m["test_rmse"], m["test_r2"]])

    save_workspace_plot(workspace_points, workspace_bounds)
    save_plots(metrics, weights, input_mean, input_scale, train_files[0], test_files[0])
    print(f"\nSaved workspace analysis, metrics, and plots to {RESULTS_DIR}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        MODEL_PATH,
        weights=weights,
        input_mean=input_mean,
        input_scale=input_scale,
        input_columns=INPUT_COLUMNS,
        output_columns=OUTPUT_COLUMNS,
        forward_min=workspace_bounds["forward_min"],
        forward_max=workspace_bounds["forward_max"],
        left_max=workspace_bounds["left_max"],
    )
    print(f"Saved arm model to {MODEL_PATH}")


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()

