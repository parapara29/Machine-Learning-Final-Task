"""
Phase 1: Mobile Base Linear Regression Training
================================================
This script trains the linear regression model for the mobile base.
It maps 15 engineered body-frame features to the 4 wheel velocity commands.

Where the Data Comes From:
- Data was collected by running `collect_data.py` on the expert simulation (`robotnik_arm_task.py`).
- 2,000 randomized episodes were saved as CSV files in `robotnik_arm_task/data_randomized/`.
- Each episode started with the robot at a random position between -0.25 to 0.25 and between -pi to pi and drove toward a random 3D target across the room.
- Data was recorded at 100 Hz (every 10th step of the 1,000 Hz PyBullet simulation).

Key Training & Physics Decisions:
1. Regime Separation: In 77.4% of the data, the robot is only driving its base while the arm
   is locked at home. We only train on these active navigation rows (Phase 1) so the model
   doesn't mix driving with arm reaching.
2. Body-Frame Rotation: Room coordinates (X, Y) flip sign depending on which way the robot
   faces. We rotate goal offsets into the robot's local body frame (forward, left), which
   restores a clean linear correlation (r = +0.79).
3. Distance Saturation (0.35m): We clip distances at 0.35m so targets at any distance (1m, 5m, 10m)
   cruise at the same steady top speed, and only enter linear braking within the final 35cm.
4. Alignment Gate (`drive_term`): Inspired by Aicardi et al. Lyapunov unicycle steering, forward
   thrust is strictly locked to 0 when the heading error is > 31.8 degrees. The wheels must turn
   on the spot first before moving forward.
5. The 4cm Deadband (`dist - 0.04`): Shuts off forward motor commands within the final 4cm so the
   heavy 70kg base coasts to a stop right on the parking spot without creeping or overshooting.

Run:
  uv run train_base_model.py
"""

import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Directory paths:
# DATA_DIR: Folder containing all the raw CSV episodes collected from expert runs.
# MODEL_PATH: Where the final 16x4 weight matrix and scaling constants will be saved.
# RESULTS_DIR: Where evaluation plots and metrics.csv will be saved.
DATA_DIR = Path(__file__).parent / "data_randomized"
MODEL_PATH = Path(__file__).parent.parent / "trained_model" / "base_linear_regression.npz"
RESULTS_DIR = Path(__file__).parent / "base_training_results"
TEST_FRACTION = 0.2
SEED = 42

# Columns loaded from the raw CSV files:
# STATE_COLUMNS:
#   - base_x, base_y: Robot base position in room coordinates (meters).
#   - base_bearing: Robot compass yaw angle (radians, from -pi to +pi).
#   - goal_x, goal_y: The base parking spot on the ground computed by the simulation
#     (already offset by ~0.70m standoff in front of the 3D target).
#   - goal_bearing: The compass angle the base should face when parked.
# OUTPUT_COLUMNS:
#   - wheel_cmds_0..3: Commanded wheel angular velocities (rad/s) for front-right,
#     front-left, back-left, and back-right wheels.
STATE_COLUMNS = ["base_x", "base_y", "base_bearing", "goal_x", "goal_y", "goal_bearing"]
OUTPUT_COLUMNS = ["wheel_cmds_0", "wheel_cmds_1", "wheel_cmds_2", "wheel_cmds_3"]

# The 15 final engineered input features fed into the linear regression model:
INPUT_COLUMNS = [
    "goal_forward_sat",       # 1. Forward offset to goal in robot frame, clipped to [-0.35, 0.35]m
    "goal_left_sat",          # 2. Sideways offset to goal in robot frame, clipped to [-0.35, 0.35]m
    "goal_distance_sat",      # 3. Distance to goal, saturated at 0.35m (stops runaway speed)
    "drive_term",             # 4. Gated forward drive term: max(0, dist - 0.04) * aligned
    "aligned",                # 5. Alignment gate: max(0, cos(heading_error) - 0.85)
    "near_goal",              # 6. Proximity blending weight: ramps 0->1 in the final 12cm
    "far_heading_error",      # 7. far_goal * heading_error (steers toward goal while far away)
    "far_sin_heading_error",  # 8. far_goal * sin(heading_error) (smooth angle wrap-around)
    "far_abs_heading_error",  # 9. far_goal * |heading_error| (slows wheels down during sharp turns)
    "near_bearing_error",     # 10. near_goal * bearing_error (aligns compass heading during docking)
    "near_sin_bearing_error", # 11. near_goal * sin(bearing_error)
    "near_abs_bearing_error", # 12. near_goal * |bearing_error|
    "bearing_error",          # 13. Raw bearing error (steady state orientation lock)
    "cos_heading_error",      # 14. Even function of heading error for smooth speed scaling
    "cos_bearing_error",      # 15. Even function of bearing error
]


def wrap_angle(angle):
    """Wraps any angle into [-pi, +pi] to prevent discontinuity jumps across 180 degrees."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def make_base_features(data):
    """
    Transforms raw state variables into 15 physical features for the base model:
    
    1. Body-Frame Rotation:
       In the raw CSV, base_x and base_y are room coordinates. If the robot turns 180 degrees,
       the sign flips (r = -0.01). We rotate the goal offset into the robot's heading frame:
         goal_forward = cos(yaw) * dx + sin(yaw) * dy   (distance ahead of nose)
         goal_left    = -sin(yaw) * dx + cos(yaw) * dy  (distance to the left)
       This creates a rock-solid linear relationship with wheel speeds (r = +0.79).
    
    2. Saturation at 0.35m (Distance Invariance):
       The expert controller hits its maximum speed (0.25 m/s) whenever the goal is > 0.25m away.
       If we don't clip, a target 5m away would command 5x more speed than anything in training!
       Saturating dist, forward, and left at 0.35m means any target from 1m to 50m cruises at the
       same safe top speed, and only enters linear braking when within 35cm of the goal.
       
    3. The Alignment Gate (`aligned` and `drive_term`):
       Inspired by Aicardi et al. (1995) Lyapunov unicycle steering (v = rho * cos(alpha)).
       `aligned` is strictly 0 if heading error > 31.8 degrees (cos < 0.85).
       `drive_term` multiplies distance by `aligned`, forcing forward drive to 0 when unaligned.
       The `- 0.04` deadband cuts forward thrust within 4cm of the goal so the heavy 70kg chassis
       coasts to a dead stop without creeping or overshooting.
       
    4. Regime Blending (`near_goal` and `far_goal`):
       Acts as a smooth cross-fader. Inside the final 12cm, `near_goal` ramps from 0 to 1:
       - When far (> 12cm): Steering focuses on `far_heading_error` (driving toward the spot).
       - When near (< 12cm): Steering transitions to `near_bearing_error` (aligning compass yaw).
    """
    yaw = np.asarray(data["base_bearing"])
    c, s = np.cos(yaw), np.sin(yaw)
    goal_dx = np.asarray(data["goal_x"]) - data["base_x"]
    goal_dy = np.asarray(data["goal_y"]) - data["base_y"]
    goal_distance = np.hypot(goal_dx, goal_dy)

    # Angle to the goal spot vs current robot compass heading
    heading_error = wrap_angle(np.arctan2(goal_dy, goal_dx) - yaw)
    # Required final compass angle at the parking spot vs current heading
    bearing_error = wrap_angle(np.asarray(data["goal_bearing"]) - yaw)

    # Rotate horizontal goal offset from world coordinates into the robot's body frame
    # and saturate at 0.35m so targets at any distance (1m, 5m, 10m) behave identically.
    dist_sat = np.minimum(goal_distance, 0.35)
    goal_forward = np.clip(c * goal_dx + s * goal_dy, -0.35, 0.35)
    goal_left = np.clip(-s * goal_dx + c * goal_dy, -0.35, 0.35)

    # Separate forward driving (when aligned with the goal) from turning in place near the goal
    # (Adapted from Aicardi et al., IEEE RAM 1995: Closed loop steering of unicycle vehicles)
    aligned = np.maximum(0.0, np.cos(heading_error) - 0.85)
    drive_term = np.maximum(0.0, dist_sat - 0.04) * aligned
    near_goal = np.clip((0.12 - goal_distance) / 0.12, 0.0, 1.0)
    far_goal = 1.0 - near_goal

    return np.column_stack((
        goal_forward,
        goal_left,
        dist_sat,
        drive_term,
        aligned,
        near_goal,
        far_goal * heading_error,
        far_goal * np.sin(heading_error),
        far_goal * np.abs(heading_error),
        near_goal * bearing_error,
        near_goal * np.sin(bearing_error),
        near_goal * np.abs(bearing_error),
        bearing_error,
        np.cos(heading_error),
        np.cos(bearing_error),
    ))


def load_episode(file):
    """Reads a CSV file generated by collect_data.py and extracts state and output columns."""
    with file.open() as fp:
        header = next(csv.reader(fp))
    needed = ["time", "ee_error"] + STATE_COLUMNS + OUTPUT_COLUMNS
    indices = [header.index(col) for col in needed]
    raw = np.loadtxt(file, delimiter=",", skiprows=1, usecols=indices)
    return {col: raw[:, i] for i, col in enumerate(needed)}


def load_active_nav_data(files):
    """
    Loads and filters the base navigation data across all selected episode CSVs:
    
    1. Filter failed timeout demonstrations:
       If the expert timed out and finished with end-effector error > 0.05m, skip the episode.
    2. Filter out Phase 2 rows (Regime Separation):
       In 22.6% of each episode, the wheels are stopped (0 rad/s) while the arm reaches for the ball.
       If we train on those rows, linear regression averages driving with reaching, dragging
       down the model slope. We filter for `active = |wheels| > 1e-6` so the base model only
       learns from rows where the wheels were actually driving.
    """
    chunks = {col: [] for col in STATE_COLUMNS + OUTPUT_COLUMNS}
    skipped = 0
    for file in files:
        ep = load_episode(file)
        # Skip failed episodes where the expert timed out far from the target.
        if ep["ee_error"][-1] > 0.05:
            skipped += 1
            continue
        wheels = np.column_stack([ep[c] for c in OUTPUT_COLUMNS])
        # Only keep rows where the base was actively driving
        active = np.max(np.abs(wheels), axis=1) > 1e-6
        for col in chunks:
            chunks[col].append(ep[col][active])
    return {col: np.concatenate(chunks[col]) for col in chunks}, skipped


def save_plots(metrics, weights, input_mean, input_scale, train_file, test_file):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(OUTPUT_COLUMNS))
    width = 0.35

    axes[0].bar(x - width / 2, [metrics[o]["train_rmse"] for o in OUTPUT_COLUMNS], width, label="Train")
    axes[0].bar(x + width / 2, [metrics[o]["test_rmse"] for o in OUTPUT_COLUMNS], width, label="Test")
    axes[0].set_xticks(x, OUTPUT_COLUMNS, rotation=20)
    axes[0].set_ylabel("RMSE (rad/s)")
    axes[0].set_title("Wheel Command RMSE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(x - width / 2, [metrics[o]["train_r2"] for o in OUTPUT_COLUMNS], width, label="Train")
    axes[1].bar(x + width / 2, [metrics[o]["test_r2"] for o in OUTPUT_COLUMNS], width, label="Test")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xticks(x, OUTPUT_COLUMNS, rotation=20)
    axes[1].set_ylabel("R² score")
    axes[1].set_title("Wheel Command R²")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "metrics.png", dpi=150)
    plt.close(fig)

    for split, file in [("train", train_file), ("test", test_file)]:
        ep = load_episode(file)
        wheels = np.column_stack([ep[c] for c in OUTPUT_COLUMNS])
        active = np.max(np.abs(wheels), axis=1) > 1e-6
        features = (make_base_features(ep) - input_mean) / input_scale
        predictions = np.column_stack((np.ones(len(ep["time"])), features)) @ weights
        # Zero predictions once navigation finishes, matching the runtime stop rule.
        predictions[~active] = 0.0

        fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
        for i, (ax, output) in enumerate(zip(axes.flat, OUTPUT_COLUMNS)):
            ax.plot(ep["time"], ep[output], label="Expert")
            ax.plot(ep["time"], predictions[:, i], "--", label="Predicted")
            ax.set_title(output)
            ax.set_ylabel("rad/s")
            ax.grid(True, alpha=0.3)
        axes[0, 0].legend()
        fig.suptitle(f"{split.capitalize()} episode: {file.name}")
        fig.supxlabel("Simulation time (s)")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"{split}_predictions.png", dpi=150)
        plt.close(fig)


def main():
    # 1. Load list of all CSV demonstration files
    files = sorted(DATA_DIR.glob("episode_*.csv"))

    np.random.default_rng(SEED).shuffle(files)
    num_test = int(np.ceil(len(files) * TEST_FRACTION))
    test_files = files[:num_test]
    train_files = files[num_test:]

    # 3. Load active driving rows (Phase 1 only, filtering stopped arm-reaching rows)
    train, skipped_train = load_active_nav_data(train_files)
    test, skipped_test = load_active_nav_data(test_files)

    # 4. Feature Extraction & Z-Score Normalization:
    # We compute mean and std on the TRAINING data only, then normalize both train and test.
    # This prevents features with large ranges from dominating the matrix inversion.
    features = make_base_features(train)
    input_mean = features.mean(axis=0)
    input_scale = features.std(axis=0)
    input_scale[np.isclose(input_scale, 0)] = 1.0  # Prevent divide-by-zero

    # 5. Form Design Matrix X and Target Matrix Y:
    # Prepend a column of 1.0 to learn the intercept (bias) term.
    # X_train has shape (N, 16) -> 1 bias + 15 features.
    # y_train has shape (N, 4) -> 4 wheel commands.
    X_train = np.column_stack((np.ones(len(features)), (features - input_mean) / input_scale))
    y_train = np.column_stack([train[c] for c in OUTPUT_COLUMNS])
    weights = np.linalg.pinv(X_train) @ y_train

    X_test = np.column_stack((np.ones(len(test["base_x"])), (make_base_features(test) - input_mean) / input_scale))
    y_test = np.column_stack([test[c] for c in OUTPUT_COLUMNS])

    print(f"Training: {len(train_files) - skipped_train} valid episodes ({skipped_train} skipped), {len(y_train)} active nav rows")
    print(f"Testing : {len(test_files) - skipped_test} valid episodes ({skipped_test} skipped), {len(y_test)} active nav rows\n")

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

    save_plots(metrics, weights, input_mean, input_scale, train_files[0], test_files[0])
    print(f"\nSaved metrics and plots to {RESULTS_DIR}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        MODEL_PATH,
        weights=weights,
        input_mean=input_mean,
        input_scale=input_scale,
        input_columns=INPUT_COLUMNS,
        output_columns=OUTPUT_COLUMNS,
    )
    print(f"Saved base model to {MODEL_PATH}")


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()
