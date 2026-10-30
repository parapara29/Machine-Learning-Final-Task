import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path(__file__).parent / "data" / "data_randomized"
MODEL_PATH = Path(__file__).parent.parent / "trained_model" / "base_linear_regression.npz"
RESULTS_DIR = Path(__file__).parent / "base_training_results"
TEST_FRACTION = 0.2
SEED = 42

STATE_COLUMNS = ["base_x", "base_y", "base_bearing", "goal_x", "goal_y", "goal_bearing"]
OUTPUT_COLUMNS = ["wheel_cmds_0", "wheel_cmds_1", "wheel_cmds_2", "wheel_cmds_3"]
INPUT_COLUMNS = [
    "goal_forward",
    "goal_left",
    "goal_distance",
    "heading_error",
    "sin_heading_error",
    "cos_heading_error",
    "abs_heading_error",
    "bearing_error",
    "sin_bearing_error",
    "cos_bearing_error",
    "abs_bearing_error",
]


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def make_base_features(data):
    yaw = np.asarray(data["base_bearing"])
    c, s = np.cos(yaw), np.sin(yaw)
    goal_dx = np.asarray(data["goal_x"]) - data["base_x"]
    goal_dy = np.asarray(data["goal_y"]) - data["base_y"]
    goal_distance = np.hypot(goal_dx, goal_dy)

    heading_error = wrap_angle(np.arctan2(goal_dy, goal_dx) - yaw)
    bearing_error = wrap_angle(np.asarray(data["goal_bearing"]) - yaw)

    # Rotate horizontal goal offset from world coordinates into the robot's body frame.
    goal_forward = c * goal_dx + s * goal_dy
    goal_left = -s * goal_dx + c * goal_dy

    return np.column_stack((
        goal_forward,
        goal_left,
        goal_distance,
        heading_error,
        np.sin(heading_error),
        np.cos(heading_error),
        np.abs(heading_error),
        bearing_error,
        np.sin(bearing_error),
        np.cos(bearing_error),
        np.abs(bearing_error),
    ))


def load_episode(file):
    with file.open() as fp:
        header = next(csv.reader(fp))
    needed = ["time", "ee_error"] + STATE_COLUMNS + OUTPUT_COLUMNS
    indices = [header.index(col) for col in needed]
    raw = np.loadtxt(file, delimiter=",", skiprows=1, usecols=indices)
    return {col: raw[:, i] for i, col in enumerate(needed)}


def load_active_nav_data(files):
    chunks = {col: [] for col in STATE_COLUMNS + OUTPUT_COLUMNS}
    skipped = 0
    for file in files:
        ep = load_episode(file)
        # Skip failed episodes where the expert timed out far from the target.
        if ep["ee_error"][-1] > 0.05:
            skipped += 1
            continue
        wheels = np.column_stack([ep[c] for c in OUTPUT_COLUMNS])
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
    files = sorted(DATA_DIR.glob("episode_*.csv"))
    if len(files) < 2:
        raise ValueError(f"Need at least two episode CSVs in {DATA_DIR}")

    np.random.default_rng(SEED).shuffle(files)
    num_test = int(np.ceil(len(files) * TEST_FRACTION))
    test_files = files[:num_test]
    train_files = files[num_test:]

    train, skipped_train = load_active_nav_data(train_files)
    test, skipped_test = load_active_nav_data(test_files)

    features = make_base_features(train)
    input_mean = features.mean(axis=0)
    input_scale = features.std(axis=0)
    input_scale[np.isclose(input_scale, 0)] = 1.0

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
    main()
